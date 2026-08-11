from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_protocol import build_evaluation_only_assessment
from protocol import (
    CASES_PATH,
    CONFIGS,
    EXPERIMENT_DIR,
    HERB_INTERSECTION_PATH,
    HERB_KEGG_PATH,
    OUTPUT_DIR,
    ROOT,
    AblationRunner,
    add_formula_occurrence_provenance,
    build_case_occurrence_provenance,
    build_contexts,
    build_internal_reference,
    make_compact_records,
    qualify_mechanism_kb,
    read_json,
    read_jsonl,
    retrieval_metrics,
    sha256,
    source_ids_for_record,
    source_document_id,
    unique_strings,
    write_json,
    write_jsonl,
    load_kb,
)


REQUIRED_FIELDS = {
    "syndrome": ("syndrome_id", "name", "references", "text_description"),
    "formula": ("formula_id", "name", "indication_syndrome", "composition"),
    "herb": ("herb_name", "tcm_function"),
}


def missing_required(record: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [field for field in required if record.get(field) in (None, "", [])]


def duplicate_values(records: list[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(str(record.get(field) or "") for record in records)
    return sorted(value for value, count in counts.items() if value and count > 1)


def kb_audit(kb: dict[str, Any]) -> dict[str, Any]:
    references = read_json(ROOT / "data" / "references.json")
    reference_ids = {record.get("reference_id") for record in references}
    syndrome_ids = {record.get("syndrome_id") for record in kb["syndromes"]}
    herb_names = {record.get("herb_name") for record in kb["herbs"]}
    evidence_qualified_herb_names = {
        record.get("herb_name")
        for record in kb["herbs"]
        if record.get("mechanism_evidence_qualified")
    }
    formula_composition_herbs = {
        entry.get("herb")
        for formula in kb["formulas"]
        for entry in (formula.get("composition") or [])
        if isinstance(entry, dict) and entry.get("herb")
    }
    layers = {
        "syndrome": (kb["syndromes"], "syndrome_id"),
        "formula": (kb["formulas"], "formula_id"),
        "herb": (kb["herbs"], "herb_name"),
    }
    layer_results: dict[str, Any] = {}
    for layer, (records, id_field) in layers.items():
        missing = [
            {"record_id": record.get(id_field), "missing_fields": missing_required(record, REQUIRED_FIELDS[layer])}
            for record in records
        ]
        missing = [row for row in missing if row["missing_fields"]]
        source_complete = sum(bool(source_ids_for_record(record, layer)) for record in records)
        layer_results[layer] = {
            "record_count": len(records),
            "records_with_all_required_fields": len(records) - len(missing),
            "required_field_completeness_rate": (len(records) - len(missing)) / len(records) if records else None,
            "records_with_source_ids": source_complete,
            "source_id_completeness_rate": source_complete / len(records) if records else None,
            "missing_required_fields": missing,
            "duplicate_ids": duplicate_values(records, id_field),
            "duplicate_names": duplicate_values(records, "name" if layer != "herb" else "herb_name"),
        }
    formulas_with_curated_reference_ids = sum(bool(record.get("references")) for record in kb["formulas"])
    formulas_with_literature_case_sources = sum(bool(record.get("literature_source_ids")) for record in kb["formulas"])
    formulas_with_source_documents = sum(bool(source_ids_for_record(record, "formula")) for record in kb["formulas"])
    formulas_with_occurrence_sources = sum(bool(record.get("case_occurrence_source_ids")) for record in kb["formulas"])
    herbs_with_mechanism_sources = sum(bool(record.get("mechanism_evidence_qualified")) for record in kb["herbs"])
    layer_results["formula"]["curated_reference_id_completeness"] = {
        "numerator": formulas_with_curated_reference_ids,
        "denominator": len(kb["formulas"]),
        "rate": formulas_with_curated_reference_ids / len(kb["formulas"]),
    }
    layer_results["formula"]["literature_case_document_source_completeness"] = {
        "numerator": formulas_with_literature_case_sources,
        "denominator": len(kb["formulas"]),
        "rate": formulas_with_literature_case_sources / len(kb["formulas"]),
    }
    layer_results["formula"]["source_document_completeness"] = {
        "numerator": formulas_with_source_documents,
        "denominator": len(kb["formulas"]),
        "rate": formulas_with_source_documents / len(kb["formulas"]),
    }
    layer_results["formula"]["case_occurrence_source_completeness"] = {
        "numerator": formulas_with_occurrence_sources,
        "denominator": len(kb["formulas"]),
        "rate": formulas_with_occurrence_sources / len(kb["formulas"]),
    }
    layer_results["herb"]["pipeline_traceable_mechanism_record_completeness"] = {
        "numerator": herbs_with_mechanism_sources,
        "denominator": len(kb["herbs"]),
        "rate": herbs_with_mechanism_sources / len(kb["herbs"]),
    }
    invalid_syndrome_refs = sorted(
        {
            ref
            for record in kb["syndromes"]
            for ref in unique_strings(record.get("references") or [])
            if ref not in reference_ids
        }
    )
    invalid_formula_refs = sorted(
        {
            ref
            for record in kb["formulas"]
            for ref in unique_strings(record.get("references") or [])
            if ref not in reference_ids
        }
    )
    invalid_indications = sorted(
        {
            syndrome_id
            for record in kb["formulas"]
            for syndrome_id in unique_strings(record.get("indication_syndrome") or [])
            if syndrome_id not in syndrome_ids
        }
    )
    return {
        "audit_scope": "structural_integrity_and_provenance_completeness_not_clinical_validity",
        "layers": layer_results,
        "cross_reference_checks": {
            "invalid_syndrome_reference_ids": invalid_syndrome_refs,
            "invalid_formula_reference_ids": invalid_formula_refs,
            "invalid_formula_indication_syndrome_ids": invalid_indications,
            "formula_composition_unique_herbs": len(formula_composition_herbs),
            "formula_composition_herbs_with_mechanism_records": len(formula_composition_herbs & herb_names),
            "formula_composition_herbs_without_mechanism_records": sorted(formula_composition_herbs - herb_names),
            "formula_composition_herbs_with_traceable_mechanism_records": len(formula_composition_herbs & evidence_qualified_herb_names),
            "formula_composition_herbs_without_traceable_mechanism_records": sorted(formula_composition_herbs - evidence_qualified_herb_names),
        },
    }


def write_retrieval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "source_group",
        "source_cluster_id",
        "configuration",
        "expected_item_count",
        "retrieved_item_count",
        "correct_item_count",
        "evidence_retrieval_precision",
        "evidence_retrieval_recall",
        "serialized_evidence_bytes",
        "layered_reference_serialized_bytes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_provenance_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id",
        "case_id",
        "formula_id",
        "source_group",
        "source_document_id",
        "source_title",
        "provenance_scope",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_source_document_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_id", "source_group", "source_title", "source_reference", "provenance_scope"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_internal_case_csv(path: Path, references: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "source_group",
        "source_cluster_id",
        "primary_syndrome_resolved",
        "formula_mapped",
        "formula_source_supported_leave_one_source_out",
        "primary_formula_concordance",
        "any_formula_concordance",
        "core_herb_annotation_coverage",
        "reference_herb_annotation_coverage",
        "formula_composition_annotation_coverage",
        "any_level_closure",
        "core60_closure",
        "strict_closure",
        "assessment_level",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reference in references:
            writer.writerow(
                {
                    "case_id": reference["case_id"],
                    "source_group": reference.get("source_group"),
                    "source_cluster_id": reference.get("source_cluster_id"),
                    **reference.get("internal_case_metrics", {}),
                }
            )


def binary_summary(values: list[bool | None]) -> dict[str, Any]:
    eligible = [value for value in values if value is not None]
    numerator = sum(value is True for value in eligible)
    return {
        "numerator": numerator,
        "denominator": len(eligible),
        "rate": numerator / len(eligible) if eligible else None,
    }


def internal_case_summary(references: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [reference["internal_case_metrics"] for reference in references]
    levels = Counter(str(metric.get("assessment_level")) for metric in metrics if metric.get("assessment_level") is not None)
    core_values = [metric.get("core_herb_annotation_coverage") for metric in metrics]
    reference_values = [metric.get("reference_herb_annotation_coverage") for metric in metrics]
    formula_values = [metric.get("formula_composition_annotation_coverage") for metric in metrics]
    return {
        "n_cases": len(metrics),
        "primary_syndrome_resolution": binary_summary([metric.get("primary_syndrome_resolved") for metric in metrics]),
        "formula_mapping": binary_summary([metric.get("formula_mapped") for metric in metrics]),
        "formula_source_support_leave_one_source_out": binary_summary(
            [metric.get("formula_source_supported_leave_one_source_out") for metric in metrics]
        ),
        "primary_formula_concordance": binary_summary([metric.get("primary_formula_concordance") for metric in metrics]),
        "any_formula_concordance": binary_summary([metric.get("any_formula_concordance") for metric in metrics]),
        "mean_core_herb_annotation_coverage": mean_defined(core_values),
        "core_herb_annotation_coverage_eligible_n": sum(value is not None for value in core_values),
        "mean_reference_herb_annotation_coverage": mean_defined(reference_values),
        "reference_herb_annotation_coverage_eligible_n": sum(value is not None for value in reference_values),
        "mean_formula_composition_annotation_coverage": mean_defined(formula_values),
        "formula_composition_annotation_coverage_eligible_n": sum(value is not None for value in formula_values),
        "any_level_closure": binary_summary([metric.get("any_level_closure") for metric in metrics]),
        "core60_closure": binary_summary([metric.get("core60_closure") for metric in metrics]),
        "strict_closure": binary_summary([metric.get("strict_closure") for metric in metrics]),
        "assessment_level_counts": dict(levels),
        "scope": "internal evidence-coverage analysis using pipeline-traceable mechanism records and leave-one-source-out formula evidence; not clinical accuracy",
    }


def mean_defined(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()

    original_kb = load_kb()
    cases = read_jsonl(args.cases)
    literature_sources, occurrence_sources, provenance_rows, document_rows = build_case_occurrence_provenance(
        cases, original_kb
    )
    kb = qualify_mechanism_kb(
        add_formula_occurrence_provenance(original_kb, literature_sources, occurrence_sources)
    )
    runner = AblationRunner(kb)
    references: list[dict[str, Any]] = []
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        excluded = {source_document_id(case)} if case.get("source_group") != "hospital_real_case" else set()
        references.append(
            build_internal_reference(case, runner, kb, excluded_source_ids=excluded)
        )
        records_by_case[case["case_id"]] = make_compact_records(kb, excluded_source_ids=excluded)
    contexts: list[dict[str, Any]] = []
    evaluation_only_assessments: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for reference in references:
        by_config = build_contexts(reference, records_by_case[reference["case_id"]])
        for configuration in CONFIGS:
            context = by_config[configuration]
            metrics = retrieval_metrics(reference, context)
            contexts.append({"case_id": reference["case_id"], "configuration": configuration, "context": context})
            evaluation_only_assessments.append(
                build_evaluation_only_assessment(reference, context)
            )
            retrieval_rows.append(
                {
                    "case_id": reference["case_id"],
                    "source_group": reference.get("source_group"),
                    "source_cluster_id": reference.get("source_cluster_id"),
                    "configuration": configuration,
                    **metrics,
                }
            )

    audit = kb_audit(kb)
    audit["file_hashes"] = {
        "cases": sha256(args.cases),
        **{
            name: sha256(ROOT / "data" / name)
            for name in ("syndromes.json", "formulas.json", "herbs.json", "syndrome_formula_map.json", "references.json")
        },
        "herb_target_intersection": sha256(HERB_INTERSECTION_PATH),
        "herb_kegg": sha256(HERB_KEGG_PATH),
    }
    retrieval_summary: dict[str, Any] = {}
    for configuration in CONFIGS:
        subset = [row for row in retrieval_rows if row["configuration"] == configuration]
        retrieval_summary[configuration] = {
            "n_cases": len(subset),
            "mean_evidence_retrieval_precision": mean_defined([row["evidence_retrieval_precision"] for row in subset]),
            "mean_evidence_retrieval_recall": mean_defined([row["evidence_retrieval_recall"] for row in subset]),
            "mean_evidence_record_count": mean_defined([float(row["retrieved_item_count"]) for row in subset]),
            "mean_serialized_evidence_bytes": mean_defined([float(row["serialized_evidence_bytes"]) for row in subset]),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_cases = args.cases.resolve()
    cases_path_for_manifest = (
        resolved_cases.relative_to(ROOT).as_posix()
        if ROOT == resolved_cases or ROOT in resolved_cases.parents
        else str(resolved_cases)
    )
    release_audit = (args.cases.parent / "standardization_release_summary.json").resolve()
    if not release_audit.exists():
        release_audit = (EXPERIMENT_DIR / "inputs" / "standardization_release_summary.json").resolve()
    write_json(OUTPUT_DIR / "kb_integrity_audit.json", audit)
    write_provenance_csv(OUTPUT_DIR / "case_formula_occurrence_provenance.csv", provenance_rows)
    write_source_document_csv(OUTPUT_DIR / "formula_literature_source_index.csv", document_rows)
    write_jsonl(OUTPUT_DIR / "internal_reference_set.jsonl", references)
    write_internal_case_csv(OUTPUT_DIR / "internal_case_analysis.csv", references)
    write_json(OUTPUT_DIR / "internal_case_analysis_summary.json", internal_case_summary(references))
    write_jsonl(OUTPUT_DIR / "model_contexts.jsonl", contexts)
    write_jsonl(
        OUTPUT_DIR / "evaluation_only_assessments.jsonl",
        evaluation_only_assessments,
    )
    write_retrieval_csv(OUTPUT_DIR / "retrieval_benchmark.csv", retrieval_rows)
    write_json(OUTPUT_DIR / "retrieval_benchmark_summary.json", retrieval_summary)
    write_json(
        OUTPUT_DIR / "experiment_manifest.json",
        {
            "protocol": "fair_rag_internal_evidence_audit_v5_isolated_consistency_output",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_dir": EXPERIMENT_DIR.relative_to(ROOT).as_posix(),
            "cases_path": cases_path_for_manifest,
            "n_cases": len(cases),
            "source_group_counts": dict(Counter(case.get("source_group") for case in cases)),
            "configurations": list(CONFIGS),
            "reference_standard": "versioned audit reference derived from KB records and predefined rules; not clinical ground truth",
            "case_source_exclusion": (
                "the source document from which each literature case was extracted was removed from that case's "
                "formula sources; a formula record with no remaining independent source was removed from the "
                "case-specific retrieval index, model context, and expected retrieval targets"
            ),
            "patient_narrative_sent_to_model": False,
            "system_output_boundary": (
                "qwen_only, flat_rag, and layered_rag report retrieval evidence and coverage with "
                "cross-layer relation and Level marked not applicable; only op_rag exposes the "
                "consistency relation and Level 1-4"
            ),
            "evaluation_only_assessment": {
                "path": (OUTPUT_DIR / "evaluation_only_assessments.jsonl").relative_to(ROOT).as_posix(),
                "n_rows": len(evaluation_only_assessments),
                "sent_to_qwen": False,
                "purpose": "apply one fixed evidence-support rule to all four configurations for experiment-only comparison",
            },
            "visible_case_standardization": {
                "contract": "the public release contains standardized physician-plan fields only; patient narratives and pre-filled answer IDs are not included",
                "audit_path": release_audit.relative_to(ROOT).as_posix() if release_audit.exists() else None,
                "audit_sha256": sha256(release_audit) if release_audit.exists() else None,
            },
            "files": {
                "cases_sha256": sha256(args.cases),
                "protocol_py_sha256": sha256(EXPERIMENT_DIR / "protocol.py"),
                "prepare_py_sha256": sha256(Path(__file__)),
                "evaluation_protocol_py_sha256": sha256(EXPERIMENT_DIR / "evaluation_protocol.py"),
                "evaluation_only_assessments_sha256": sha256(
                    OUTPUT_DIR / "evaluation_only_assessments.jsonl"
                ),
            },
        },
    )
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "n_cases": len(cases), "retrieval_summary": retrieval_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
