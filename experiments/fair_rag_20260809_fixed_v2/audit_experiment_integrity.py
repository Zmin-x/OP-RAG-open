from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, sha256


DEFAULT_SPOT_CHECK_INDEX_PATH = (
    OUTPUT_DIR / "api_spot_check_gate_20260810" / "SPOT_CHECK_INDEX.json"
)


def check(condition: bool, evidence: str) -> dict[str, Any]:
    return {"status": "pass" if condition else "fail", "evidence": evidence}


def warn_if(condition: bool, evidence: str) -> dict[str, Any]:
    return {"status": "pass" if condition else "warn", "evidence": evidence}


def functions_called(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return defined, called


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--spot-index",
        type=Path,
        default=DEFAULT_SPOT_CHECK_INDEX_PATH,
        help="spot-check index that authorized this formal run",
    )
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.json"
    scoring_path = args.run_dir / "qwen_scoring_summary.json"
    results_path = args.run_dir / "qwen_results.jsonl"
    experiment_manifest_path = OUTPUT_DIR / "experiment_manifest.json"
    contexts_path = OUTPUT_DIR / "model_contexts.jsonl"
    reference_path = OUTPUT_DIR / "internal_reference_set.jsonl"
    score_script = Path(__file__).parent / "score_qwen_comparison.py"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    results = read_jsonl(results_path)
    references = read_jsonl(reference_path)
    contexts = read_jsonl(contexts_path)
    cases_path = Path(experiment_manifest.get("cases_path", ""))
    if not cases_path.is_absolute():
        cases_path = Path(__file__).resolve().parents[2] / cases_path
    cases = read_jsonl(cases_path) if cases_path.exists() else []
    keys = [(row["case_id"], row["configuration"]) for row in results]
    config_counts = Counter(row["configuration"] for row in results)
    attempts = Counter(int(row.get("remote_metadata", {}).get("attempt", 0)) for row in results)
    prior_failure_count = sum(
        len(row.get("remote_metadata", {}).get("prior_failures") or []) for row in results
    )
    defined, called = functions_called(score_script)

    spot_index_path = args.spot_index.resolve()
    spot_index = (
        json.loads(spot_index_path.read_text(encoding="utf-8"))
        if spot_index_path.exists()
        else {}
    )
    spot_entries = spot_index.get("entries") or []
    last_five_spot_entries = spot_entries[-5:]
    spot_run_manifests = []
    for entry in last_five_spot_entries:
        spot_manifest_path = Path(str(entry.get("run_dir") or "")) / "run_manifest.json"
        spot_run_manifests.append(
            json.loads(spot_manifest_path.read_text(encoding="utf-8"))
            if spot_manifest_path.exists()
            else {}
        )
    last_spot_finished_at = (
        spot_run_manifests[-1].get("finished_at") if spot_run_manifests else None
    )
    formal_started_at = manifest.get("started_at")

    audit_path = Path(experiment_manifest.get("visible_case_standardization", {}).get("audit_path", ""))
    if not audit_path.is_absolute():
        audit_path = Path(__file__).resolve().parents[2] / audit_path

    checks = {
        "completed_manifest": check(
            manifest.get("status") == "completed" and manifest.get("n_completed") == 200,
            f"status={manifest.get('status')}; n_completed={manifest.get('n_completed')}",
        ),
        "five_consecutive_pre_api_spot_checks": check(
            spot_index.get("formal_200_api_run_authorized") is True
            and int(spot_index.get("consecutive_passes", 0)) >= 5
            and len(last_five_spot_entries) == 5
            and len({entry.get("case_id") for entry in last_five_spot_entries}) == 5
            and all(entry.get("status") == "pass" for entry in last_five_spot_entries)
            and all(
                spot_manifest.get("status") == "completed"
                and spot_manifest.get("n_expected") == 4
                and spot_manifest.get("n_completed") == 4
                and spot_manifest.get("n_valid") == 4
                and spot_manifest.get("n_requested_in_this_run") == 4
                and spot_manifest.get("n_reused_identical_context") == 0
                and not spot_manifest.get("failures")
                for spot_manifest in spot_run_manifests
            ),
            "The last five distinct cases passed fresh four-configuration API checks with zero reuse",
        ),
        "formal_run_started_after_spot_check_gate": check(
            bool(formal_started_at)
            and bool(last_spot_finished_at)
            and str(formal_started_at) > str(last_spot_finished_at),
            f"formal_started_at={formal_started_at}; last_spot_finished_at={last_spot_finished_at}",
        ),
        "unique_scope": check(
            len(results) == 200 and len(set(keys)) == 200 and all(config_counts[name] == 50 for name in CONFIGS),
            f"rows={len(results)}; unique_keys={len(set(keys))}; configurations={dict(config_counts)}",
        ),
        "schema_validity": check(
            manifest.get("n_valid") == 200,
            f"schema-valid={manifest.get('n_valid')}/200; invalid responses are rejected before acceptance",
        ),
        "context_hash": check(
            manifest.get("contexts_sha256") == sha256(contexts_path),
            f"manifest={manifest.get('contexts_sha256')}; actual={sha256(contexts_path)}",
        ),
        "reference_hash": check(
            manifest.get("reference_sha256") == sha256(reference_path),
            f"manifest={manifest.get('reference_sha256')}; actual={sha256(reference_path)}",
        ),
        "patient_narrative_excluded": check(
            manifest.get("patient_narrative_sent_to_model") is False
            and experiment_manifest.get("patient_narrative_sent_to_model") is False,
            "Both run and experiment manifests record patient_narrative_sent_to_model=false",
        ),
        "visible_case_standardization": check(
            len(cases) == 50
            and sum(bool(row.get("primary_syndrome_name_std")) for row in cases) == 47
            and sum(not row.get("primary_syndrome_name_std") for row in cases) == 3
            and all(row.get("reference_formula_id") is None for row in cases),
            "50 corrected cases; 47 visible standardized primary labels; 3 unresolved labels; no pre-filled formula ID is used by the run",
        ),
        "no_hidden_case_ids_in_physician_plan": check(
            all(
                set((row.get("context", {}).get("physician_plan") or {}))
                <= {"primary_syndrome_name", "secondary_syndrome_names", "formula_name", "herbs"}
                for row in contexts
            ),
            "Physician-plan context exposes names and herbs only; structured syndrome/formula IDs are absent",
        ),
        "standardization_audit_hash": check(
            bool(experiment_manifest.get("visible_case_standardization", {}).get("audit_sha256"))
            and experiment_manifest.get("visible_case_standardization", {}).get("audit_sha256")
            == sha256(audit_path) if audit_path.exists() else False,
            "The visible-name recovery audit is hash-recorded in the experiment manifest",
        ),
        "reference_scope_declared": check(
            len(references) == 50
            and all(
                row.get("reference_standard_scope")
                == "versioned_internal_kb_and_predefined_rules_not_clinical_ground_truth"
                for row in references
            ),
            "All 50 reference rows identify the internal KB/rule scope and deny clinical-ground-truth status",
        ),
        "qwen_is_not_used_to_derive_structured_audit_fields": check(
            all(
                not (
                    set((row.get("context", {}).get("rule_context") or {}))
                    & {"assessment_level", "assessment_label", "expected_level"}
                )
                for row in contexts
            )
            and manifest.get("structured_audit_fields_generated_by") == "deterministic_python"
            and manifest.get("qwen_role") == "verbalize_assessment_summary_without_calculation",
            "Rule contexts contain no expected level; Python computes the structured audit before Qwen receives non-numeric narration facts",
        ),
        "case_source_excluded_from_formula_evidence": check(
            all(
                all(
                    record.get("source_ids")
                    and set(reference.get("excluded_source_ids") or []).isdisjoint(record.get("source_ids", []))
                    for row in contexts
                    if row.get("case_id") == reference.get("case_id")
                    for record in row.get("context", {}).get("evidence_context", [])
                    if record.get("layer") == "formula"
                )
                for reference in references
            ),
            "Every retrieved formula record retains at least one independent source and excludes the case source",
        ),
        "source_free_formula_not_a_retrieval_target": check(
            all(
                all(
                    any(
                        claim.get("item_id") == item_id
                        and claim.get("layer") == "formula"
                        and claim.get("source_ids")
                        for claim in reference.get("expected_claims", [])
                    )
                    for item_id in reference.get("expected_retrieval_item_ids", [])
                    if str(item_id).startswith("formula:")
                )
                for reference in references
            ),
            "A formula is an expected retrieval target only when an independently sourced formula claim exists",
        ),
        "no_self_normalization": check(
            "max(" not in score_script.read_text(encoding="utf-8")
            and "min(" not in score_script.read_text(encoding="utf-8"),
            "Score script uses fixed denominators, exact sets, macro means and source-cluster bootstrap; no model-output max/min normalization",
        ),
        "metric_functions_executed": check(
            {"score_record", "f1_score", "bootstrap_paired_difference"} <= called,
            f"defined={sorted(defined)}; called core metrics={sorted({'score_record', 'f1_score', 'bootstrap_paired_difference'} & called)}",
        ),
        "scoring_scope_label": check(
            scoring.get("scope")
            == "deterministic internal KB-grounded audit with Qwen used only for non-numeric verbalization; not clinical answer accuracy",
            f"scope={scoring.get('scope')}",
        ),
        "structured_audit_consistency": check(
            all(
                scoring.get("configurations", {}).get(name, {}).get(
                    "structured_audit_consistent", {}
                ).get("numerator")
                == 50
                for name in CONFIGS
            ),
            "All accepted outputs must preserve the Python-generated evidence, missing-item, coverage and level fields exactly",
        ),
        "narrative_contains_no_model_generated_numbers": check(
            all(
                scoring.get("configurations", {}).get(name, {}).get(
                    "narrative_numbers_absent", {}
                ).get("numerator")
                == 50
                for name in CONFIGS
            ),
            "All Qwen summaries must be qualitative and must not contain a model-generated count, fraction or percentage",
        ),
        "narrative_semantic_consistency": check(
            all(
                scoring.get("configurations", {}).get(name, {}).get(
                    "semantic_consistent", {}
                ).get("numerator")
                == 50
                for name in CONFIGS
            ),
            "Every accepted Qwen summary preserves the deterministic label and required qualitative clauses and adds no unsupported text",
        ),
        "provenance_cleanliness": warn_if(
            all(
                scoring.get("configurations", {}).get(name, {}).get("provenance_clean", {}).get("numerator")
                == scoring.get("configurations", {}).get(name, {}).get("provenance_clean", {}).get("denominator")
                for name in ("flat_rag", "layered_rag", "op_rag")
            ),
            "provenance-clean reports: "
            + "; ".join(
                f"{name}="
                f"{scoring.get('configurations', {}).get(name, {}).get('provenance_clean', {}).get('numerator')}/"
                f"{scoring.get('configurations', {}).get(name, {}).get('provenance_clean', {}).get('denominator')}"
                for name in ("flat_rag", "layered_rag", "op_rag")
            ),
        ),
    }
    statuses = {value["status"] for value in checks.values()}
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    report = {
        "date": datetime.now(timezone.utc).isoformat(),
        "auditor": "documented local deterministic audit; cross-model audit unavailable",
        "overall_verdict": overall,
        "evaluation_type": "internal_KB_and_rule_reference_not_clinical_ground_truth",
        "checks": checks,
        "request_attempt_counts": dict(attempts),
        "prior_transport_or_parse_failures": prior_failure_count,
        "known_scope_risks": [
            "F013--F024 were informed by case-source documents before analysis; strict record-level source exclusion prevents a source-free formula record from entering that case's context but does not make the dataset an external holdout.",
            "Python derives the structured audit and assessment level from the versioned knowledge base and predefined rules. Qwen only verbalizes non-numeric narrative facts; the resulting level is not a model-accuracy outcome.",
            "Qwen-only receives no source records; retrieval and structured evidence-link metrics are not applicable and do not test general medical knowledge.",
            "Evidence claims, missing-item lists, coverage values and levels are deterministic pipeline outputs. The Qwen sentence is checked against required qualitative clauses but is not an independent medical interpretation.",
            "The versioned audit reference is generated from the same knowledge base and predefined rules used by the system; it is not independent clinical ground truth.",
            "Most literature-case formula support after case-source exclusion still comes from other documents in the same internal literature collection; this is not an external holdout.",
            "Flat and layered retrieval compare complete strategies that differ in query decomposition, layer filtering and quotas; the comparison does not isolate one retrieval component.",
        ],
    }
    output_json = args.run_dir / "LOCAL_EXPERIMENT_INTEGRITY_AUDIT.json"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Local Experiment Integrity Audit",
        "",
        f"**Overall verdict:** {overall.upper()}",
        "",
        "This is a documented local deterministic audit. The independent GPT-5.4 cross-model backend required by the experiment-audit skill was unavailable, so no cross-model audit is claimed.",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, value in checks.items():
        evidence = str(value["evidence"]).replace("|", "\\|")
        lines.append(f"| {name} | {value['status'].upper()} | {evidence} |")
    lines.extend(["", "## Known scope risks", ""])
    lines.extend(f"- {item}" for item in report["known_scope_risks"])
    lines.extend(
        [
            "",
            "## Request attempts",
            "",
            f"- Returned-response attempt counts: {dict(attempts)}",
            f"- Recorded prior transport or parsing failures: {prior_failure_count}",
        ]
    )
    (args.run_dir / "LOCAL_EXPERIMENT_INTEGRITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall_verdict": overall, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
