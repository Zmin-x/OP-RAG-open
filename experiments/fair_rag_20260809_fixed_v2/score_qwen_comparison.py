from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, unique_strings, write_json


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(expected: set[str], reported: set[str]) -> float:
    if not expected and not reported:
        return 1.0
    true_positive = len(expected & reported)
    precision = safe_div(true_positive, len(reported)) or 0.0
    recall = safe_div(true_positive, len(expected)) or 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compact_key(value: Any) -> str:
    return "".join(str(value or "").strip().casefold().split())


def build_item_aliases(
    contexts: list[dict[str, Any]], references: dict[str, dict[str, Any]]
) -> dict[str, str]:
    aliases: dict[str, str] = {}

    def add(layer: str, value: Any, item_id: str) -> None:
        key = compact_key(value)
        if key:
            aliases[f"{layer}:{key}"] = item_id

    for row in contexts:
        for record in row.get("context", {}).get("evidence_context", []):
            item_id = str(record.get("item_id") or "")
            layer = str(record.get("layer") or "")
            if item_id and layer in {"syndrome", "formula", "herb"}:
                add(layer, item_id.split(":", 1)[-1], item_id)
                add(layer, record.get("name"), item_id)

    for reference in references.values():
        claims_by_layer: dict[str, list[str]] = {}
        for claim in reference.get("expected_claims", []):
            layer = str(claim.get("layer") or "")
            if layer in {"syndrome", "formula", "herb"}:
                item_id = str(claim.get("item_id") or "")
                claims_by_layer.setdefault(layer, []).append(item_id)
                add(layer, item_id.split(":", 1)[-1], item_id)
        plan = reference.get("physician_plan", {})
        formula_ids = unique_strings(claims_by_layer.get("formula", []))
        if len(formula_ids) == 1:
            add("formula", plan.get("formula_name"), formula_ids[0])
        syndrome_names = unique_strings(
            [plan.get("primary_syndrome_name"), *(plan.get("secondary_syndrome_names") or [])]
        )
        syndrome_ids = unique_strings(claims_by_layer.get("syndrome", []))
        if len(syndrome_names) == len(syndrome_ids):
            for name, item_id in zip(syndrome_names, syndrome_ids):
                add("syndrome", name, item_id)
    return aliases


def canonical_missing_item(value: Any, aliases: dict[str, str]) -> str:
    raw = str(value or "").strip()
    if ":" not in raw:
        return compact_key(raw)
    layer, suffix = raw.split(":", 1)
    layer = compact_key(layer)
    key = f"{layer}:{compact_key(suffix)}"
    return aliases.get(key, key)


def visible_claim_record(item_id: str, records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    record = records_by_id.get(item_id)
    if record is not None:
        return record
    if not item_id.startswith("relation:"):
        return None
    parts = item_id.split(":")
    formula_id = parts[1] if len(parts) >= 3 else ""
    formula_record = records_by_id.get(f"formula:{formula_id}")
    syndrome_visible = any(record.get("layer") == "syndrome" for record in records_by_id.values())
    return formula_record if formula_record is not None and syndrome_visible else None


def expected_missing_items(
    reference: dict[str, Any], context: dict[str, Any], aliases: dict[str, str]
) -> set[str]:
    structured = context.get("structured_audit") or {}
    if "missing_evidence_items" in structured:
        return {
            canonical_missing_item(value, aliases)
            for value in structured.get("missing_evidence_items", [])
        }
    return {
        canonical_missing_item(value, aliases)
        for value in reference.get("expected_missing_items", [])
    }


def score_record(
    result: dict[str, Any],
    reference: dict[str, Any],
    context: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    response = result["response"]
    configuration = result["configuration"]
    consistency_audit_applicable = configuration == "op_rag"
    schema_valid = result.get("validation", {}).get(
        "schema_valid", result.get("validation", {}).get("valid")
    ) is True
    expected_by_id = {
        claim["item_id"]: claim
        for claim in reference.get("expected_claims", [])
        if consistency_audit_applicable or claim.get("layer") != "cross_layer"
    }
    records_by_id = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    claims = response.get("evidence_claims") if isinstance(response.get("evidence_claims"), list) else []
    correct_ids: set[str] = set()
    supported_claim_count = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        supported_claim_count += 1
        item_id = str(claim.get("item_id") or "")
        expected = expected_by_id.get(item_id)
        if not expected or claim.get("support_status") != expected.get("expected_status"):
            continue
        visible_record = visible_claim_record(item_id, records_by_id)
        if visible_record is None:
            continue
        reported_sources = set(str(value) for value in (claim.get("source_ids") or []))
        expected_sources = set(str(value) for value in (expected.get("source_ids") or []))
        visible_sources = set(str(value) for value in (visible_record.get("source_ids") or []))
        source_valid = (
            bool(reported_sources)
            and reported_sources <= expected_sources
            and reported_sources <= visible_sources
        )
        if source_valid:
            correct_ids.add(item_id)
    expected_ids = set(expected_by_id)
    raw_evidence_recall = safe_div(len(correct_ids), len(expected_ids))
    raw_link_precision = safe_div(len(correct_ids), supported_claim_count)
    expected_missing = expected_missing_items(reference, context, aliases)
    reported_missing = {
        canonical_missing_item(value, aliases)
        for value in (response.get("missing_evidence_items") or [])
    }
    raw_missing_f1 = f1_score(expected_missing, reported_missing)
    deterministic_level = (
        (context.get("structured_audit") or {}).get("assessment_level")
        if consistency_audit_applicable
        else None
    )
    raw_level_agreement = (
        response.get("assessment_level") == deterministic_level
        if consistency_audit_applicable
        else None
    )
    coverage = response.get("coverage_metrics") or {}
    plan_coverage = coverage.get("physician_plan_herbs") or {}
    core_coverage = coverage.get("core_herbs") or {}
    formula_coverage = coverage.get("formula_composition_herbs") or {}

    # Qwen-only receives no retrieved records, so retrieval-grounded evidence
    # recall and link precision are not applicable rather than zero. For a
    # schema-invalid RAG response, the primary pipeline metrics are scored as
    # failures; raw partial-field scores remain available for audit only.
    if configuration == "qwen_only":
        evidence_recall = None
        structured_evidence_link_precision = None
    elif not schema_valid:
        evidence_recall = 0.0 if expected_ids else None
        structured_evidence_link_precision = 0.0
    else:
        evidence_recall = raw_evidence_recall
        structured_evidence_link_precision = raw_link_precision
    missing_f1 = 0.0 if not schema_valid else raw_missing_f1
    level_agreement = (
        None
        if not consistency_audit_applicable
        else (False if not schema_valid else raw_level_agreement)
    )
    return {
        "case_id": result["case_id"],
        "source_group": result.get("source_group"),
        "source_cluster_id": reference.get("source_cluster_id"),
        "configuration": configuration,
        "consistency_audit_applicable": consistency_audit_applicable,
        "output_schema_valid": schema_valid,
        "structured_audit_consistent": result.get("validation", {}).get(
            "structured_audit_consistent", False
        ) is True,
        "narrative_numbers_absent": result.get("validation", {}).get(
            "narrative_numbers_absent", False
        ) is True,
        "semantic_consistent": result.get("validation", {}).get(
            "semantic_consistent", False
        ) is True,
        "provenance_clean": (
            None
            if configuration == "qwen_only"
            else result.get("validation", {}).get("provenance_clean", False) is True
        ),
        "provenance_violation_count": len(result.get("validation", {}).get("provenance_violations") or []),
        "expected_evidence_item_count": len(expected_ids),
        "reported_evidence_claim_count": supported_claim_count,
        "correct_evidence_claim_count": len(correct_ids),
        "expected_missing_evidence_item_count": len(expected_missing),
        "reported_missing_evidence_item_count": len(reported_missing),
        "plan_herb_supported_count": plan_coverage.get("numerator"),
        "plan_herb_total_count": plan_coverage.get("denominator"),
        "plan_herb_coverage": plan_coverage.get("value"),
        "core_herb_supported_count": core_coverage.get("numerator"),
        "core_herb_total_count": core_coverage.get("denominator"),
        "core_herb_coverage": core_coverage.get("value"),
        "formula_herb_supported_count": formula_coverage.get("numerator"),
        "formula_herb_total_count": formula_coverage.get("denominator"),
        "formula_composition_herb_coverage": formula_coverage.get("value"),
        "evidence_recall": evidence_recall,
        "structured_evidence_link_precision": structured_evidence_link_precision,
        "missing_evidence_disclosure_f1": missing_f1,
        "assessment_level_agreement": level_agreement,
        "raw_partial_evidence_recall": raw_evidence_recall,
        "raw_partial_structured_evidence_link_precision": raw_link_precision,
        "raw_partial_missing_evidence_disclosure_f1": raw_missing_f1,
        "raw_partial_assessment_level_agreement": raw_level_agreement,
        "expected_assessment_level": deterministic_level,
        "reported_assessment_level": response.get("assessment_level"),
        "unverified_parametric_claim_count": len(response.get("unverified_parametric_claims") or []),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_paired_difference(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], metric: str, *, seed: int = 20260807
) -> dict[str, Any]:
    case_ids = sorted(
        case_id
        for case_id in set(left) & set(right)
        if left[case_id].get(metric) is not None and right[case_id].get(metric) is not None
    )
    differences_by_case = {
        case: float(right[case][metric]) - float(left[case][metric])
        for case in case_ids
    }
    differences = np.array(list(differences_by_case.values()), dtype=float)
    rng = np.random.default_rng(seed)
    if not len(differences):
        return {"n_pairs": 0, "mean_difference": None, "ci95": [None, None]}
    clusters: dict[str, list[str]] = {}
    for case in case_ids:
        cluster = str(left[case].get("source_cluster_id") or case)
        clusters.setdefault(cluster, []).append(case)
    cluster_ids = sorted(clusters)
    samples = np.empty(10000, dtype=float)
    for index in range(10000):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled = [
            differences_by_case[case]
            for cluster in sampled_clusters
            for case in clusters[str(cluster)]
        ]
        samples[index] = float(np.mean(sampled))
    return {
        "n_pairs": len(differences),
        "n_source_clusters": len(cluster_ids),
        "mean_difference": float(differences.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_DIR / "qwen_comparison")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    results = read_jsonl(args.run_dir / "qwen_results.jsonl")
    result_keys = [(row.get("case_id"), row.get("configuration")) for row in results]
    manifest_path = args.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if not args.allow_partial:
        if manifest is None or manifest.get("status") != "completed":
            raise SystemExit("Refusing final scoring: completed run_manifest.json is missing")
        if len(results) != 200 or len(set(result_keys)) != 200:
            raise SystemExit(
                f"Refusing final scoring: expected 200 unique results, got {len(results)} rows "
                f"and {len(set(result_keys))} unique keys"
            )
    references = {row["case_id"]: row for row in read_jsonl(OUTPUT_DIR / "internal_reference_set.jsonl")}
    contexts = read_jsonl(OUTPUT_DIR / "model_contexts.jsonl")
    contexts_by_key = {
        (row["case_id"], row["configuration"]): row["context"]
        for row in contexts
    }
    evaluation_rows = read_jsonl(OUTPUT_DIR / "evaluation_only_assessments.jsonl")
    evaluations_by_key = {
        (row["case_id"], row["configuration"]): row
        for row in evaluation_rows
    }
    aliases = build_item_aliases(contexts, references)
    scored = []
    for result in results:
        key = (result["case_id"], result["configuration"])
        evaluation = evaluations_by_key.get(key)
        if evaluation is None:
            raise SystemExit(f"Missing evaluation-only assessment for {key}")
        row = score_record(
            result,
            references[result["case_id"]],
            contexts_by_key[key],
            aliases,
        )
        row["evaluation_only_assessment_level"] = evaluation["assessment_level"]
        row["evaluation_only_assessment_label"] = evaluation["assessment_label"]
        row["evaluation_only_sent_to_qwen"] = evaluation["sent_to_qwen"]
        scored.append(row)
    write_csv(args.run_dir / "qwen_scored_cases.csv", scored)

    retrieval_report_metrics = (
        "evidence_recall",
        "structured_evidence_link_precision",
    )
    deterministic_qa_metrics = (
        "missing_evidence_disclosure_f1",
        "assessment_level_agreement",
    )
    summary: dict[str, Any] = {}
    by_configuration: dict[str, dict[str, dict[str, Any]]] = {}
    for configuration in CONFIGS:
        rows = [row for row in scored if row["configuration"] == configuration]
        by_configuration[configuration] = {row["case_id"]: row for row in rows}
        summary[configuration] = {
            "n_cases": len(rows),
            "output_schema_valid": {
                "numerator": sum(row["output_schema_valid"] for row in rows),
                "denominator": len(rows),
            },
            "structured_audit_consistent": {
                "numerator": sum(row["structured_audit_consistent"] for row in rows),
                "denominator": len(rows),
            },
            "narrative_numbers_absent": {
                "numerator": sum(row["narrative_numbers_absent"] for row in rows),
                "denominator": len(rows),
            },
            "semantic_consistent": {
                "numerator": sum(row["semantic_consistent"] for row in rows),
                "denominator": len(rows),
            },
            "provenance_clean": (
                None
                if configuration == "qwen_only"
                else {
                    "numerator": sum(row["provenance_clean"] is True for row in rows),
                    "denominator": len(rows),
                }
            ),
            "provenance_violation_count": sum(row["provenance_violation_count"] for row in rows),
            "deterministic_coverage": {
                "mean_plan_herb_coverage": mean(
                    [float(row["plan_herb_coverage"]) for row in rows if row["plan_herb_coverage"] is not None]
                ),
                "mean_core_herb_coverage": mean(
                    [float(row["core_herb_coverage"]) for row in rows if row["core_herb_coverage"] is not None]
                ),
                "mean_formula_composition_herb_coverage": mean(
                    [
                        float(row["formula_composition_herb_coverage"])
                        for row in rows
                        if row["formula_composition_herb_coverage"] is not None
                    ]
                ),
            },
            **{
                metric: mean([float(row[metric]) for row in rows if row[metric] is not None])
                for metric in (*retrieval_report_metrics, *deterministic_qa_metrics)
            },
            "assessment_level_counts": dict(
                Counter(
                    "N/A"
                    if row["reported_assessment_level"] is None
                    else str(row["reported_assessment_level"])
                    for row in rows
                )
            ),
            "evaluation_only_level_counts": dict(
                Counter(
                    str(row["evaluation_only_assessment_level"])
                    for row in rows
                )
            ),
        }
    comparisons = {}
    for left_name, right_name in (
        ("qwen_only", "flat_rag"),
        ("flat_rag", "layered_rag"),
        ("layered_rag", "op_rag"),
    ):
        comparisons[f"{right_name}_minus_{left_name}"] = {
            metric: bootstrap_paired_difference(
                by_configuration[left_name], by_configuration[right_name], metric
            )
            for metric in retrieval_report_metrics
        }
    retrieval_rows = read_csv(OUTPUT_DIR / "retrieval_benchmark.csv")
    retrieval_metrics = ("evidence_retrieval_precision", "evidence_retrieval_recall")
    retrieval_by_configuration: dict[str, dict[str, dict[str, Any]]] = {}
    retrieval_summary: dict[str, Any] = {}
    for configuration in CONFIGS:
        rows = [row for row in retrieval_rows if row["configuration"] == configuration]
        cast_rows = []
        for row in rows:
            cast_row = dict(row)
            for metric in retrieval_metrics:
                cast_row[metric] = float(row[metric]) if row[metric] not in {"", None} else None
            cast_rows.append(cast_row)
        retrieval_by_configuration[configuration] = {row["case_id"]: row for row in cast_rows}
        retrieval_summary[configuration] = {
            "n_cases": len(cast_rows),
            **{
                metric: mean([float(row[metric]) for row in cast_rows if row[metric] is not None])
                for metric in retrieval_metrics
            },
            "mean_evidence_record_count": mean([float(row["retrieved_item_count"]) for row in cast_rows]),
            "mean_serialized_evidence_bytes": mean(
                [float(row["serialized_evidence_bytes"]) for row in cast_rows]
            ),
        }
    retrieval_comparison = {
        metric: bootstrap_paired_difference(
            retrieval_by_configuration["flat_rag"],
            retrieval_by_configuration["layered_rag"],
            metric,
        )
        for metric in retrieval_metrics
    }

    output = {
        "scope": "deterministic internal KB-grounded audit with Qwen used only for non-numeric verbalization; not clinical answer accuracy",
        "metric_roles": {
            "evidence_recall_and_link_precision": "deterministic retrieval-and-source-linkage outputs",
            "missing_disclosure_and_level_agreement": "deterministic pipeline QA fields, not model performance",
            "evaluation_only_level": (
                "one fixed post hoc evidence-support rule applied to all configurations; "
                "withheld from Qwen and distinct from system output"
            ),
            "assessment_summary": "Qwen verbalization with numeric statements prohibited",
        },
        "run_integrity": {
            "status": (manifest or {}).get("status", "partial"),
            "n_rows": len(results),
            "n_unique_case_configuration_keys": len(set(result_keys)),
            "manifest": manifest,
        },
        "retrieval_benchmark": {
            "configurations": retrieval_summary,
            "layered_rag_minus_flat_rag": retrieval_comparison,
        },
        "configurations": summary,
        "paired_differences_with_source_cluster_bootstrap_ci95": comparisons,
    }
    write_json(args.run_dir / "qwen_scoring_summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
