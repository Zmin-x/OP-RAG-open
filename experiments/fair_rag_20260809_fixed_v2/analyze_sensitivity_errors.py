from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

from protocol import (
    CASES_PATH,
    HERB_INTERSECTION_PATH,
    HERB_KEGG_PATH,
    OUTPUT_DIR,
    AblationRunner,
    add_formula_occurrence_provenance,
    build_case_occurrence_provenance,
    build_internal_reference,
    evidence_item_ids,
    flat_retrieval,
    layered_retrieval,
    load_kb,
    make_compact_records,
    physician_plan,
    qualify_mechanism_kb,
    read_jsonl,
    retrieval_metrics,
    runner_input,
    serialized_bytes,
    sha256,
    source_document_id,
)


TOP_K_VALUES = (1, 2, 3, 5)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
SEED = 20260807
MM_TO_INCH = 1 / 25.4


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(values[0]) if values else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean_defined(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def proportion(values: Iterable[bool | None]) -> tuple[int, int, float | None]:
    eligible = [value for value in values if value is not None]
    numerator = sum(value is True for value in eligible)
    return numerator, len(eligible), numerator / len(eligible) if eligible else None


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 600}),
    ):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)


def build_resources(cases: list[dict[str, Any]]) -> tuple[dict[str, Any], AblationRunner]:
    original_kb = load_kb()
    literature_sources, occurrence_sources, _, _ = build_case_occurrence_provenance(cases, original_kb)
    kb = qualify_mechanism_kb(
        add_formula_occurrence_provenance(original_kb, literature_sources, occurrence_sources)
    )
    return kb, AblationRunner(kb)


def case_objects(
    cases: list[dict[str, Any]], kb: dict[str, Any], runner: AblationRunner
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        excluded = {source_document_id(case)} if case.get("source_group") != "hospital_real_case" else set()
        reference = build_internal_reference(case, runner, kb, excluded_source_ids=excluded)
        records = make_compact_records(kb, excluded_source_ids=excluded)
        local_output = runner.run_case(runner_input(case, use_llm=False), "g4")
        rows.append(
            {
                "case": case,
                "reference": reference,
                "records": records,
                "local_results": local_output.context.get("case_results", {}),
            }
        )
    return rows


def expected_layer_ids(reference: dict[str, Any], layer: str) -> set[str]:
    prefix = f"{layer}:"
    return {
        str(item_id)
        for item_id in reference.get("expected_retrieval_item_ids", [])
        if str(item_id).startswith(prefix)
    }


def layer_hit(expected: set[str], retrieved: set[str]) -> bool | None:
    return bool(expected & retrieved) if expected else None


def layer_recall(expected: set[str], retrieved: set[str]) -> float | None:
    return len(expected & retrieved) / len(expected) if expected else None


def top_k_analysis(objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for k in TOP_K_VALUES:
        for configuration in ("flat_rag", "layered_rag"):
            subset: list[dict[str, Any]] = []
            for obj in objects:
                reference = obj["reference"]
                plan = reference["physician_plan"]
                layered = layered_retrieval(
                    plan,
                    obj["records"],
                    syndrome_top_k=k,
                    formula_top_k=k,
                    herb_top_k=1,
                )
                records = (
                    layered
                    if configuration == "layered_rag"
                    else flat_retrieval(plan, obj["records"], top_k=len(layered))
                )
                context = {
                    "configuration": configuration,
                    "evidence_context": records,
                    "retrieval_budget": {
                        "evidence_record_count": len(records),
                        "serialized_evidence_bytes": serialized_bytes(records),
                        "layered_reference_record_count": len(layered),
                        "layered_reference_serialized_bytes": serialized_bytes(layered),
                    },
                }
                metrics = retrieval_metrics(reference, context)
                retrieved = evidence_item_ids(records)
                syndrome_ids = expected_layer_ids(reference, "syndrome")
                formula_ids = expected_layer_ids(reference, "formula")
                herb_ids = expected_layer_ids(reference, "herb")
                row = {
                    "case_id": reference["case_id"],
                    "source_group": reference.get("source_group"),
                    "k": k,
                    "configuration": configuration,
                    "retrieval_precision": metrics["evidence_retrieval_precision"],
                    "retrieval_recall": metrics["evidence_retrieval_recall"],
                    "syndrome_hit": layer_hit(syndrome_ids, retrieved),
                    "formula_hit": layer_hit(formula_ids, retrieved),
                    "herb_recall": layer_recall(herb_ids, retrieved),
                    "record_count": len(records),
                    "serialized_evidence_bytes": serialized_bytes(records),
                }
                subset.append(row)
                case_rows.append(row)
            syndrome_n, syndrome_d, syndrome_rate = proportion(row["syndrome_hit"] for row in subset)
            formula_n, formula_d, formula_rate = proportion(row["formula_hit"] for row in subset)
            summary_rows.append(
                {
                    "k": k,
                    "configuration": configuration,
                    "n_cases": len(subset),
                    "mean_retrieval_precision": mean_defined(row["retrieval_precision"] for row in subset),
                    "mean_retrieval_recall": mean_defined(row["retrieval_recall"] for row in subset),
                    "syndrome_hit_numerator": syndrome_n,
                    "syndrome_hit_denominator": syndrome_d,
                    "syndrome_hit_rate": syndrome_rate,
                    "formula_hit_numerator": formula_n,
                    "formula_hit_denominator": formula_d,
                    "formula_hit_rate": formula_rate,
                    "mean_herb_recall": mean_defined(row["herb_recall"] for row in subset),
                    "mean_record_count": mean_defined(row["record_count"] for row in subset),
                    "mean_serialized_evidence_bytes": mean_defined(
                        row["serialized_evidence_bytes"] for row in subset
                    ),
                }
            )
    return case_rows, summary_rows


def threshold_analysis(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        core_states: list[bool | None] = []
        strict_states: list[bool | None] = []
        for obj in objects:
            metric = obj["reference"]["internal_case_metrics"]
            any_chain = metric.get("any_level_closure") is True
            core = metric.get("core_herb_annotation_coverage")
            formula = metric.get("formula_composition_annotation_coverage")
            primary = metric.get("primary_formula_concordance")
            core_states.append(bool(core >= threshold) if any_chain and core is not None else None)
            strict_states.append(
                bool(primary is True and core >= threshold and formula >= threshold)
                if any_chain and core is not None and formula is not None and primary is not None
                else None
            )
        core_n, core_d, core_rate = proportion(core_states)
        strict_n, strict_d, strict_rate = proportion(strict_states)
        rows.append(
            {
                "threshold": threshold,
                "core_numerator": core_n,
                "core_denominator": core_d,
                "core_conditional_rate": core_rate,
                "core_overall_rate": core_n / len(objects),
                "strict_numerator": strict_n,
                "strict_denominator": strict_d,
                "strict_conditional_rate": strict_rate,
                "strict_overall_rate": strict_n / len(objects),
            }
        )
    return rows


def failure_reasons(metric: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not metric.get("primary_syndrome_resolved"):
        reasons.append("primary_syndrome_unresolved")
    if not metric.get("formula_mapped"):
        reasons.append("formula_unmapped")
    elif not metric.get("formula_source_supported_leave_one_source_out"):
        reasons.append("formula_without_independent_source")
    if metric.get("primary_formula_concordance") is False:
        reasons.append("primary_syndrome_formula_contradiction")
    core = metric.get("core_herb_annotation_coverage")
    formula = metric.get("formula_composition_annotation_coverage")
    if core is None:
        reasons.append("core_herb_coverage_not_evaluable")
    elif core < 0.60:
        reasons.append("core_herb_coverage_below_0.60")
    elif core < 0.80:
        reasons.append("core_herb_coverage_below_0.80")
    if formula is None:
        reasons.append("formula_coverage_not_evaluable")
    elif formula < 0.80:
        reasons.append("formula_coverage_below_0.80")
    return reasons


def primary_failure(metric: dict[str, Any], reasons: list[str]) -> str:
    if metric.get("assessment_level") == 4:
        return "explicit_cross_layer_contradiction"
    priority = (
        "primary_syndrome_unresolved",
        "formula_unmapped",
        "formula_without_independent_source",
        "primary_syndrome_formula_contradiction",
        "core_herb_coverage_not_evaluable",
        "core_herb_coverage_below_0.60",
        "formula_coverage_not_evaluable",
        "formula_coverage_below_0.80",
        "core_herb_coverage_below_0.80",
    )
    return next((reason for reason in priority if reason in reasons), "complete_under_predefined_rules")


def formula_group(formula_id: str | None) -> str:
    if not formula_id:
        return "unmapped"
    number = int(formula_id[1:]) if formula_id.startswith("F") and formula_id[1:].isdigit() else None
    if number is not None and 1 <= number <= 12:
        return "initial_F001_F012"
    if formula_id == "F024":
        return "F024_main_report"
    if number is not None and 13 <= number <= 23:
        return "expanded_F013_F023"
    return "other"


def case_error_analysis(
    objects: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    for obj in objects:
        reference = obj["reference"]
        metric = reference["internal_case_metrics"]
        local = obj["local_results"]
        reasons = failure_reasons(metric)
        formula_id = local.get("resolved_formula_id")
        case_rows.append(
            {
                "case_id": reference["case_id"],
                "source_group": reference.get("source_group"),
                "recorded_formula": reference["physician_plan"].get("formula_name"),
                "resolved_formula_id": formula_id,
                "formula_group": formula_group(formula_id),
                "assessment_level": metric.get("assessment_level"),
                "primary_failure_category": primary_failure(metric, reasons),
                "all_failure_reasons": ";".join(reasons) if reasons else "none",
                "syndrome_resolved": metric.get("primary_syndrome_resolved"),
                "formula_mapped": metric.get("formula_mapped"),
                "independent_formula_source": metric.get("formula_source_supported_leave_one_source_out"),
                "primary_formula_concordance": metric.get("primary_formula_concordance"),
                "core_herb_coverage": metric.get("core_herb_annotation_coverage"),
                "formula_composition_coverage": metric.get("formula_composition_annotation_coverage"),
                "any_chain": metric.get("any_level_closure"),
                "core60": metric.get("core60_closure"),
                "strict": metric.get("strict_closure"),
            }
        )

    categories = Counter(row["primary_failure_category"] for row in case_rows)
    summary_rows = [
        {
            "category": category,
            "n_cases": count,
            "percentage_of_50": count / len(case_rows),
            "hospital_cases": sum(
                row["source_group"] == "hospital_real_case" and row["primary_failure_category"] == category
                for row in case_rows
            ),
            "literature_cases": sum(
                row["source_group"] != "hospital_real_case" and row["primary_failure_category"] == category
                for row in case_rows
            ),
        }
        for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0]))
    ]

    mitigation = {
        "primary_syndrome_unresolved": "expand and independently adjudicate the syndrome alias registry",
        "formula_unmapped": "curate the formula only after independent source verification",
        "formula_without_independent_source": "obtain an independent formula source before treating the record as retrievable evidence",
        "explicit_cross_layer_contradiction": "review the syndrome-formula relation and retain the contradiction flag unless adjudicated",
        "primary_syndrome_formula_contradiction": "review the primary relation and secondary-syndrome mapping",
        "core_herb_coverage_not_evaluable": "curate traceable herb-mechanism records or retain the missing-evidence state",
        "core_herb_coverage_below_0.60": "expand traceable mechanism curation without interpreting absence as inefficacy",
        "formula_coverage_not_evaluable": "standardize formula composition and mechanism annotations",
        "formula_coverage_below_0.80": "extend source-linked composition coverage and rerun the same threshold analysis",
        "core_herb_coverage_below_0.80": "retain partial support and report the exact uncovered herbs",
    }
    representative_rows: list[dict[str, Any]] = []
    wanted = (
        "primary_syndrome_unresolved",
        "formula_unmapped",
        "formula_without_independent_source",
        "core_herb_coverage_below_0.60",
        "formula_coverage_below_0.80",
        "explicit_cross_layer_contradiction",
    )
    for category in wanted:
        selected = next((row for row in case_rows if row["primary_failure_category"] == category), None)
        if not selected:
            continue
        representative_rows.append(
            {
                "case_id": selected["case_id"],
                "source_group": selected["source_group"],
                "recorded_formula": selected["recorded_formula"],
                "failure_category": category,
                "observed_evidence": (
                    f"level={selected['assessment_level']}; core={selected['core_herb_coverage']}; "
                    f"formula={selected['formula_composition_coverage']}; "
                    f"independent_formula_source={selected['independent_formula_source']}"
                ),
                "downstream_effect": "prevents complete evidence-chain support under the predefined rules",
                "mitigation": mitigation[category],
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group in ("initial_F001_F012", "F024_main_report", "expanded_F013_F023", "unmapped"):
        subset = [row for row in case_rows if row["formula_group"] == group]
        if not subset:
            continue
        source_n, source_d, source_rate = proportion(row["independent_formula_source"] for row in subset)
        any_n, any_d, any_rate = proportion(row["any_chain"] for row in subset)
        core_n, core_d, core_rate = proportion(row["core60"] for row in subset)
        strict_n, strict_d, strict_rate = proportion(row["strict"] for row in subset)
        level_counts = Counter(str(row["assessment_level"]) for row in subset)
        group_rows.append(
            {
                "formula_group": group,
                "n_cases": len(subset),
                "independent_source_numerator": source_n,
                "independent_source_denominator": source_d,
                "independent_source_rate": source_rate,
                "any_chain_numerator": any_n,
                "any_chain_denominator": any_d,
                "any_chain_rate": any_rate,
                "core60_numerator": core_n,
                "core60_denominator": core_d,
                "core60_rate": core_rate,
                "strict_numerator": strict_n,
                "strict_denominator": strict_d,
                "strict_rate": strict_rate,
                "mean_core_coverage": mean_defined(row["core_herb_coverage"] for row in subset),
                "mean_formula_coverage": mean_defined(
                    row["formula_composition_coverage"] for row in subset
                ),
                "level_1_2_3_4": "/".join(str(level_counts.get(str(level), 0)) for level in (1, 2, 3, 4)),
            }
        )
    return case_rows, summary_rows, representative_rows, group_rows


def plot_threshold_sensitivity(rows: list[dict[str, Any]], out_base: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(89 * MM_TO_INCH, 70 * MM_TO_INCH))
    thresholds = [row["threshold"] for row in rows]
    ax.plot(
        thresholds,
        [row["core_conditional_rate"] for row in rows],
        color="#3f3f3f",
        marker="o",
        linewidth=1.2,
        markersize=4,
        label="Core coverage criterion",
    )
    ax.plot(
        thresholds,
        [row["strict_conditional_rate"] for row in rows],
        color="#8f8f8f",
        marker="s",
        linewidth=1.2,
        markersize=4,
        label="Strict dual-coverage criterion",
    )
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Exploratory annotation-coverage threshold")
    ax.set_ylabel("Conditional completion rate")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "sensitivity_error_analysis")
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    kb, runner = build_resources(cases)
    objects = case_objects(cases, kb, runner)
    topk_cases, topk_summary = top_k_analysis(objects)
    threshold_rows = threshold_analysis(objects)
    case_rows, taxonomy_rows, representative_rows, formula_rows = case_error_analysis(objects)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "topk_sensitivity_case_level.csv", topk_cases)
    write_csv(args.output_dir / "topk_sensitivity_summary.csv", topk_summary)
    write_csv(args.output_dir / "threshold_sensitivity.csv", threshold_rows)
    write_csv(args.output_dir / "case_failure_taxonomy.csv", case_rows)
    write_csv(args.output_dir / "failure_taxonomy_summary.csv", taxonomy_rows)
    write_csv(args.output_dir / "representative_failure_cases.csv", representative_rows)
    write_csv(args.output_dir / "formula_group_analysis.csv", formula_rows)
    plot_threshold_sensitivity(threshold_rows, args.output_dir / "FigureS1_threshold_sensitivity")
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "strict-record-source-exclusion local sensitivity and error analysis",
            "n_cases": len(cases),
            "top_k_values": list(TOP_K_VALUES),
            "top_k_scope": "syndrome and formula retrieval; herb lookup fixed at top-1; Flat RAG matched to the Layered record count",
            "thresholds": list(THRESHOLDS),
            "threshold_scope": "exploratory engineering criteria, not clinical cutoffs",
            "cases_sha256": sha256(args.cases),
            "herb_intersection_sha256": sha256(HERB_INTERSECTION_PATH),
            "herb_kegg_sha256": sha256(HERB_KEGG_PATH),
            "analysis_script_sha256": sha256(Path(__file__)),
            "seed": SEED,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_cases": len(cases),
                "topk_rows": len(topk_summary),
                "threshold_rows": len(threshold_rows),
                "taxonomy": taxonomy_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
