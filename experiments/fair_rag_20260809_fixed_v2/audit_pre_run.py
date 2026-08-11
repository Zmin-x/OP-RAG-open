from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_protocol import build_evaluation_only_assessment
from protocol import (
    ASSESSMENT_LABELS,
    CONFIGS,
    OUTPUT_DIR,
    determine_assessment_level,
    normalize_herb,
    read_jsonl,
    write_json,
)
from qwen_protocol import assemble_response, build_user_prompt, validate_response


CONTEXTS_PATH = OUTPUT_DIR / "model_contexts.jsonl"
REFERENCES_PATH = OUTPUT_DIR / "internal_reference_set.jsonl"
EVALUATIONS_PATH = OUTPUT_DIR / "evaluation_only_assessments.jsonl"
CASES_PATH = Path(__file__).resolve().parent / "inputs" / "eval_cases_visible_plan_001_050.jsonl"


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(walk_keys(nested))
    return keys


def check(condition: bool, evidence: Any) -> dict[str, Any]:
    return {"status": "pass" if condition else "fail", "evidence": evidence}


def walk_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for nested in value.values() for number in walk_numbers(nested)]
    if isinstance(value, list):
        return [number for nested in value for number in walk_numbers(nested)]
    return []


def independently_recompute_level_inputs(
    reference: dict[str, Any], context: dict[str, Any]
) -> dict[str, bool]:
    records = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    expected = {
        str(claim.get("item_id")): claim
        for claim in reference.get("expected_claims", [])
    }

    def source_supported(item_id: str | None) -> bool:
        if not item_id or item_id not in records or item_id not in expected:
            return False
        visible_sources = set(str(value) for value in records[item_id].get("source_ids") or [])
        expected_sources = set(str(value) for value in expected[item_id].get("source_ids") or [])
        return bool(visible_sources & expected_sources)

    primary_id = reference.get("primary_syndrome_id")
    syndrome_available = source_supported(
        f"syndrome:{primary_id}" if primary_id else None
    )
    syndrome_claim_ids = [
        item_id
        for item_id, claim in expected.items()
        if claim.get("layer") == "syndrome"
    ]
    all_syndromes_available = bool(
        reference.get("all_case_syndromes_resolved")
        and syndrome_claim_ids
        and all(source_supported(item_id) for item_id in syndrome_claim_ids)
    )

    formula_id = reference.get("resolved_formula_id")
    formula_item_id = f"formula:{formula_id}" if formula_id else None
    formula_available = source_supported(formula_item_id)
    formula_record = records.get(formula_item_id or "") or {}
    indication_ids = set(
        str(value)
        for value in ((formula_record.get("attributes") or {}).get("indication_syndrome_ids") or [])
        if value
    )
    case_ids = set(str(value) for value in reference.get("case_syndrome_ids") or [] if value)
    if not formula_available or not all_syndromes_available or not case_ids or not indication_ids:
        relation_status = "insufficient_evidence"
    elif case_ids & indication_ids:
        relation_status = "supported"
    else:
        relation_status = "cross_layer_inconsistency"

    if not formula_available or not syndrome_available or not primary_id or not indication_ids:
        primary_relation_status = "insufficient_evidence"
    elif str(primary_id) in indication_ids:
        primary_relation_status = "supported"
    else:
        primary_relation_status = "cross_layer_inconsistency"

    plan_herbs = set(
        normalize_herb(value)
        for value in reference.get("physician_plan", {}).get("herbs", [])
        if value
    )
    visible_mechanism_herbs = {
        normalize_herb(record.get("name"))
        for record in records.values()
        if record.get("layer") == "herb"
        and record.get("name")
        and record.get("source_ids")
    }
    mechanism_available = bool(plan_herbs & visible_mechanism_herbs)
    coverage = context.get("structured_audit", {}).get("coverage_metrics", {})
    core_coverage = (coverage.get("core_herbs") or {}).get("value")
    formula_coverage = (coverage.get("formula_composition_herbs") or {}).get("value")
    strict_coverage = bool(
        core_coverage is not None
        and formula_coverage is not None
        and core_coverage >= 0.80
        and formula_coverage >= 0.80
    )
    return {
        "contradiction": relation_status == "cross_layer_inconsistency",
        "strict_support": bool(
            syndrome_available
            and formula_available
            and mechanism_available
            and relation_status == "supported"
            and primary_relation_status == "supported"
            and strict_coverage
        ),
        "syndrome_evidence_available": syndrome_available,
        "formula_evidence_available": formula_available,
        "mechanism_evidence_available": mechanism_available,
    }


def main(report_dir: Path = OUTPUT_DIR) -> None:
    contexts = read_jsonl(CONTEXTS_PATH)
    references = read_jsonl(REFERENCES_PATH)
    evaluations = read_jsonl(EVALUATIONS_PATH)
    references_by_id = {row["case_id"]: row for row in references}
    cases = read_jsonl(CASES_PATH)
    by_key = {(row["case_id"], row["configuration"]): row["context"] for row in contexts}
    evaluations_by_key = {
        (row["case_id"], row["configuration"]): row for row in evaluations
    }
    case_ids = sorted({row["case_id"] for row in contexts})

    zero_or_negative = [
        (row["case_id"], row["configuration"], record.get("item_id"), record.get("retrieval_score"))
        for row in contexts
        for record in row["context"].get("evidence_context", [])
        if float(record.get("retrieval_score") or 0.0) <= 0.0
    ]
    irrelevant_layered_herbs = []
    for case_id in case_ids:
        context = by_key[(case_id, "layered_rag")]
        plan_herbs = {normalize_herb(value) for value in context["physician_plan"].get("herbs", [])}
        for record in context.get("evidence_context", []):
            if record.get("layer") == "herb" and normalize_herb(str(record.get("name") or "")) not in plan_herbs:
                irrelevant_layered_herbs.append((case_id, record.get("item_id")))

    excluded_source_leaks = []
    references_by_case = {row["case_id"]: row for row in references}
    for row in contexts:
        excluded = set(references_by_case[row["case_id"]].get("excluded_source_ids") or [])
        visible = {
            str(source_id)
            for record in row["context"].get("evidence_context", [])
            for source_id in (record.get("source_ids") or [])
        }
        overlap = sorted(excluded & visible)
        if overlap:
            excluded_source_leaks.append((row["case_id"], row["configuration"], overlap))

    raw_narrative_keys = []
    forbidden_keys = {"patient_text", "patient_narrative", "expected_assessment_level", "expected_level"}
    for row in contexts:
        overlap = sorted(walk_keys(row["context"]) & forbidden_keys)
        if overlap:
            raw_narrative_keys.append((row["case_id"], row["configuration"], overlap))

    budget_mismatches = []
    evidence_mismatches = []
    identical_evidence_common_output_mismatches = []
    for case_id in case_ids:
        flat = by_key[(case_id, "flat_rag")]
        layered = by_key[(case_id, "layered_rag")]
        op = by_key[(case_id, "op_rag")]
        if len(flat["evidence_context"]) != len(layered["evidence_context"]):
            budget_mismatches.append(case_id)
        if layered["evidence_context"] != op["evidence_context"]:
            evidence_mismatches.append(case_id)
        layered_audit = layered.get("structured_audit") or {}
        op_audit = op.get("structured_audit") or {}
        layered_claims = {
            claim.get("item_id") for claim in layered_audit.get("evidence_claims", [])
        }
        op_common_claims = {
            claim.get("item_id")
            for claim in op_audit.get("evidence_claims", [])
            if claim.get("item_id") and not str(claim.get("item_id")).startswith("relation:")
        }
        layered_missing = set(layered_audit.get("missing_evidence_items") or [])
        op_common_missing = {
            item
            for item in op_audit.get("missing_evidence_items") or []
            if not str(item).startswith("relation:")
        }
        if (
            layered_audit.get("coverage_metrics") != op_audit.get("coverage_metrics")
            or layered_claims != op_common_claims
            or layered_missing != op_common_missing
        ):
            identical_evidence_common_output_mismatches.append(case_id)

    qwen_only_violations = [
        case_id
        for case_id in case_ids
        if by_key[(case_id, "qwen_only")].get("evidence_context")
        or by_key[(case_id, "qwen_only")].get("rule_context") is not None
    ]
    hidden_selection = [
        row["case_id"]
        for row in cases
        if row.get("primary_syndrome_name_raw") is None and row.get("primary_syndrome_id") is not None
    ]
    structured_audit_violations = []
    independently_recomputed_input_violations = []
    system_output_boundary_violations = []
    evaluation_only_violations = []
    model_payload_violations = []
    assembled_response_violations = []
    for row in contexts:
        audit = row["context"].get("structured_audit") or {}
        required_audit_fields = {
            "generation_method",
            "audit_scope",
            "consistency_audit_applicable",
            "evidence_claims",
            "missing_evidence_items",
            "coverage_metrics",
            "assessment_level",
            "assessment_label",
            "assessment_rule_trace",
            "formula_syndrome_relation",
            "narrative_facts",
        }
        if set(audit) != required_audit_fields or audit.get("generation_method") != "deterministic_python":
            structured_audit_violations.append((row["case_id"], row["configuration"], "field contract"))
            continue
        for name, metric in audit.get("coverage_metrics", {}).items():
            numerator = metric.get("numerator")
            denominator = metric.get("denominator")
            value = metric.get("value")
            expected = numerator / denominator if denominator else None
            supported_items = metric.get("supported_items") or []
            missing_items = metric.get("missing_items") or []
            if (
                value != expected
                or numerator < 0
                or numerator > denominator
                or numerator != len(supported_items)
                or denominator != len(supported_items) + len(missing_items)
                or set(supported_items) & set(missing_items)
            ):
                structured_audit_violations.append(
                    (row["case_id"], row["configuration"], f"invalid coverage: {name}")
                )
        claim_ids = [claim.get("item_id") for claim in audit.get("evidence_claims", [])]
        if (
            len(claim_ids) != len(set(claim_ids))
            or set(claim_ids) & set(audit.get("missing_evidence_items") or [])
            or any(not claim.get("source_ids") for claim in audit.get("evidence_claims", []))
        ):
            structured_audit_violations.append(
                (row["case_id"], row["configuration"], "claims are duplicate, missing, or source-free")
            )
        if row["configuration"] == "op_rag":
            if (
                audit.get("consistency_audit_applicable") is not True
                or audit.get("audit_scope")
                != "evidence_retrieval_coverage_and_cross_layer_consistency"
                or audit.get("assessment_level") not in ASSESSMENT_LABELS
                or ASSESSMENT_LABELS.get(audit.get("assessment_level"))
                != audit.get("assessment_label")
                or not isinstance(audit.get("assessment_rule_trace"), dict)
                or not isinstance(audit.get("formula_syndrome_relation"), dict)
            ):
                system_output_boundary_violations.append(
                    (row["case_id"], row["configuration"], "OP-RAG consistency output is incomplete")
                )
            rule_trace = audit.get("assessment_rule_trace") or {}
            independently_recomputed = independently_recompute_level_inputs(
                references_by_id[row["case_id"]], row["context"]
            )
            if independently_recomputed != (rule_trace.get("inputs") or {}):
                independently_recomputed_input_violations.append(
                    {
                        "case_id": row["case_id"],
                        "configuration": row["configuration"],
                        "expected": independently_recomputed,
                        "reported": rule_trace.get("inputs") or {},
                    }
                )
            try:
                traced_level = determine_assessment_level(**(rule_trace.get("inputs") or {}))
            except (TypeError, ValueError):
                traced_level = None
            if (
                traced_level != audit.get("assessment_level")
                or rule_trace.get("triggered_rule") != f"level_{audit.get('assessment_level')}"
            ):
                structured_audit_violations.append(
                    (row["case_id"], row["configuration"], "assessment rule trace mismatch")
                )
        elif (
            audit.get("consistency_audit_applicable") is not False
            or audit.get("audit_scope") != "evidence_retrieval_and_coverage_only"
            or audit.get("assessment_level") is not None
            or audit.get("assessment_label") != "not_applicable"
            or audit.get("assessment_rule_trace") is not None
            or audit.get("formula_syndrome_relation") is not None
            or any(
                str(claim.get("item_id") or "").startswith("relation:")
                for claim in audit.get("evidence_claims", [])
            )
            or any(
                str(item).startswith("relation:")
                for item in audit.get("missing_evidence_items", [])
            )
        ):
            system_output_boundary_violations.append(
                (row["case_id"], row["configuration"], "non-OP output exposes consistency or Level")
            )
        payload = json.loads(build_user_prompt(row["context"]))
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        if (
            set(payload) != {"structured_audit"}
            or walk_numbers(payload)
            or re.search(r"[0-9０-９%％]", serialized_payload)
        ):
            model_payload_violations.append((row["case_id"], row["configuration"], payload))
        facts = audit.get("narrative_facts") or {}
        raw = {
            "assessment_summary": (
                f"{facts.get('assessment_label')}: "
                + "; ".join(facts.get("required_clauses") or [])
                + "."
            )
        }
        assembled = assemble_response(raw, row["context"])
        validation = validate_response(assembled, row["context"], raw_model_response=raw)
        if not validation["valid"]:
            assembled_response_violations.append(
                (
                    row["case_id"],
                    row["configuration"],
                    validation["errors"],
                    validation["provenance_violations"],
                )
            )

    for key in sorted(set(by_key) | set(evaluations_by_key)):
        if key not in by_key or key not in evaluations_by_key:
            evaluation_only_violations.append(
                {"key": key, "error": "context/evaluation key mismatch"}
            )
            continue
        case_id, configuration = key
        reported = evaluations_by_key[key]
        recomputed = build_evaluation_only_assessment(
            references_by_id[case_id], by_key[key]
        )
        if reported != recomputed:
            evaluation_only_violations.append(
                {
                    "key": key,
                    "error": "evaluation-only assessment differs from recomputation",
                    "reported": reported,
                    "recomputed": recomputed,
                }
            )
        if reported.get("sent_to_qwen") is not False:
            evaluation_only_violations.append(
                {"key": key, "error": "evaluation-only result is not marked as withheld from Qwen"}
            )
        independent_inputs = independently_recompute_level_inputs(
            references_by_id[case_id], by_key[key]
        )
        independent_level = determine_assessment_level(**independent_inputs)
        if (
            reported.get("assessment_rule_trace", {}).get("inputs") != independent_inputs
            or reported.get("assessment_level") != independent_level
            or reported.get("assessment_label") != ASSESSMENT_LABELS[independent_level]
        ):
            evaluation_only_violations.append(
                {
                    "key": key,
                    "error": "evaluation-only level fails independent fixed-rule check",
                    "expected_inputs": independent_inputs,
                    "expected_level": independent_level,
                    "reported": reported,
                }
            )

    checks = {
        "case_and_context_cardinality": check(
            len(cases) == 50
            and len(references) == 50
            and len(contexts) == 200
            and len(evaluations) == 200
            and len(by_key) == 200
            and len(evaluations_by_key) == 200
            and Counter(row["configuration"] for row in contexts) == Counter({name: 50 for name in CONFIGS}),
            {
                "cases": len(cases),
                "references": len(references),
                "contexts": len(contexts),
                "evaluations": len(evaluations),
                "unique_context_keys": len(by_key),
                "unique_evaluation_keys": len(evaluations_by_key),
            },
        ),
        "no_nonpositive_retrieval_scores": check(not zero_or_negative, zero_or_negative[:20]),
        "layered_herbs_are_exact_plan_items": check(not irrelevant_layered_herbs, irrelevant_layered_herbs[:20]),
        "excluded_case_sources_are_absent": check(not excluded_source_leaks, excluded_source_leaks[:20]),
        "no_patient_narrative_or_precomputed_reference_fields_outside_structured_audit": check(
            not raw_narrative_keys, raw_narrative_keys[:20]
        ),
        "flat_and_layered_record_counts_match": check(not budget_mismatches, budget_mismatches),
        "layered_and_op_use_identical_evidence": check(not evidence_mismatches, evidence_mismatches),
        "layered_and_op_share_common_evidence_outputs_before_consistency": check(
            not identical_evidence_common_output_mismatches,
            identical_evidence_common_output_mismatches,
        ),
        "qwen_only_has_no_evidence_or_rules": check(not qwen_only_violations, qwen_only_violations),
        "visible_syndrome_fields_are_resolved_or_explicitly_null": check(not hidden_selection, hidden_selection),
        "structured_audit_is_complete_and_arithmetically_consistent": check(
            not structured_audit_violations, structured_audit_violations[:20]
        ),
        "only_op_exposes_cross_layer_relation_and_system_level": check(
            not system_output_boundary_violations,
            system_output_boundary_violations[:20],
        ),
        "op_system_level_uses_independently_recomputed_inputs": check(
            not independently_recomputed_input_violations,
            independently_recomputed_input_violations[:20],
        ),
        "all_configurations_have_withheld_uniform_evaluation_only_levels": check(
            not evaluation_only_violations,
            evaluation_only_violations[:20],
        ),
        "qwen_receives_only_non_numeric_narrative_facts": check(
            not model_payload_violations, model_payload_violations[:20]
        ),
        "deterministic_fields_and_required_narrative_clauses_validate": check(
            not assembled_response_violations, assembled_response_violations[:20]
        ),
    }
    statuses = {row["status"] for row in checks.values()}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_verdict": "pass" if statuses == {"pass"} else "fail",
        "scope": "pre-API context integrity; not clinical validity",
        "checks": checks,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "PRE_RUN_AUDIT.json", report)
    lines = [
        "# Pre-run Context Audit",
        "",
        f"**Overall verdict:** {report['overall_verdict'].upper()}",
        "",
        "This audit checks model-context integrity before API requests. It does not establish clinical validity.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, row in checks.items():
        evidence = json.dumps(row["evidence"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {name} | {row['status'].upper()} | `{evidence}` |")
    (report_dir / "PRE_RUN_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.report_dir)
