from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocol import OUTPUT_DIR, sha256


MM_TO_INCH = 1 / 25.4
SEED = 20260807
CONFIG_LABELS = {
    "qwen_only": "Qwen only",
    "flat_rag": "Flat RAG",
    "layered_rag": "Layered RAG",
    "op_rag": "OP-RAG",
}
REPORT_METRICS = (
    ("evidence_recall", "Evidence recall"),
    ("structured_evidence_link_precision", "Structured evidence-link\nprecision"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: np.ndarray, clusters: np.ndarray, *, repetitions: int = 10_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    rng = np.random.default_rng(SEED)
    cluster_ids = np.unique(clusters)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_values = np.concatenate([values[clusters == cluster] for cluster in sampled_clusters])
        means[index] = sampled_values.mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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


def build_kb_table(kb: dict[str, Any]) -> list[dict[str, Any]]:
    layers = kb["layers"]
    syndrome = layers["syndrome"]
    formula = layers["formula"]
    herb = layers["herb"]
    formula_curated = formula["curated_reference_id_completeness"]
    formula_literature = formula["literature_case_document_source_completeness"]
    formula_sources = formula["source_document_completeness"]
    herb_mechanism = herb["pipeline_traceable_mechanism_record_completeness"]
    return [
        {
            "layer": "Syndrome",
            "records": syndrome["record_count"],
            "required_fields_complete": (
                f"{syndrome['records_with_all_required_fields']}/{syndrome['record_count']}"
            ),
            "bibliographic_or_traceable_source": (
                f"{syndrome['records_with_source_ids']}/{syndrome['record_count']}"
            ),
            "qualified_secondary_field": "NA",
        },
        {
            "layer": "Formula",
            "records": formula["record_count"],
            "required_fields_complete": (
                f"{formula['records_with_all_required_fields']}/{formula['record_count']}"
            ),
            "bibliographic_or_traceable_source": (
                f"{formula_sources['numerator']}/{formula_sources['denominator']} source documents"
            ),
            "qualified_secondary_field": (
                f"{formula_curated['numerator']} curated-reference; "
                f"{formula_literature['numerator']} literature-case source records (overlap possible)"
            ),
        },
        {
            "layer": "Herb--target--pathway",
            "records": herb["record_count"],
            "required_fields_complete": (
                f"{herb['records_with_all_required_fields']}/{herb['record_count']}"
            ),
            "bibliographic_or_traceable_source": (
                f"{herb_mechanism['numerator']}/{herb_mechanism['denominator']} "
                "pipeline-traceable mechanism records"
            ),
            "qualified_secondary_field": (
                f"{herb['record_count']}/{herb['record_count']} retained for terminology normalization"
            ),
        },
    ]


def fraction_text(metric: dict[str, Any]) -> str:
    numerator = metric["numerator"]
    denominator = metric["denominator"]
    return f"{numerator}/{denominator}"


def build_internal_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    syndrome = summary["primary_syndrome_resolution"]
    formula = summary["formula_mapping"]
    source = summary["formula_source_support_leave_one_source_out"]
    primary = summary["primary_formula_concordance"]
    any_relation = summary["any_formula_concordance"]
    any_chain = summary["any_level_closure"]
    core60 = summary["core60_closure"]
    strict = summary["strict_closure"]
    levels = summary["assessment_level_counts"]
    return [
        {"metric": "Primary syndrome resolution", "estimate": fraction_text(syndrome), "rate": syndrome["rate"], "denominator_scope": "all records"},
        {"metric": "Formula mapping", "estimate": fraction_text(formula), "rate": formula["rate"], "denominator_scope": "all records"},
        {"metric": "Formula source support after case-source exclusion", "estimate": fraction_text(source), "rate": source["rate"], "denominator_scope": "all records"},
        {"metric": "Primary formula concordance", "estimate": fraction_text(primary), "rate": primary["rate"], "denominator_scope": "mapped formulas"},
        {"metric": "Any-syndrome formula concordance", "estimate": fraction_text(any_relation), "rate": any_relation["rate"], "denominator_scope": "mapped formulas"},
        {
            "metric": "Mean core-herb mechanism coverage",
            "estimate": f"{summary['mean_core_herb_annotation_coverage']:.3f}",
            "rate": summary["mean_core_herb_annotation_coverage"],
            "denominator_scope": f"{summary['core_herb_annotation_coverage_eligible_n']} eligible records",
        },
        {
            "metric": "Mean prescription-herb mechanism coverage",
            "estimate": f"{summary['mean_reference_herb_annotation_coverage']:.3f}",
            "rate": summary["mean_reference_herb_annotation_coverage"],
            "denominator_scope": f"{summary['reference_herb_annotation_coverage_eligible_n']} eligible records",
        },
        {"metric": "Any-level closure", "estimate": fraction_text(any_chain), "rate": any_chain["rate"], "denominator_scope": "all records"},
        {"metric": "Core60 closure", "estimate": fraction_text(core60), "rate": core60["rate"], "denominator_scope": "any-level-evaluable records"},
        {"metric": "Strict closure", "estimate": fraction_text(strict), "rate": strict["rate"], "denominator_scope": "strict-evaluable records"},
        {"metric": "Four-level counts", "estimate": "/".join(str(levels.get(str(level), 0)) for level in (1, 2, 3, 4)), "rate": "", "denominator_scope": "levels 1/2/3/4"},
    ]


def build_comparison_table(scoring: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = scoring["retrieval_benchmark"]["configurations"]
    report = scoring["configurations"]
    rows: list[dict[str, Any]] = []
    for configuration in ("qwen_only", "flat_rag", "layered_rag", "op_rag"):
        provenance = report[configuration]["provenance_clean"]
        row = {
            "configuration": CONFIG_LABELS[configuration],
            "retrieval_precision": (
                "NA"
                if retrieval[configuration]["evidence_retrieval_precision"] is None
                else retrieval[configuration]["evidence_retrieval_precision"]
            ),
            "retrieval_recall": (
                "NA"
                if retrieval[configuration]["evidence_retrieval_recall"] is None
                else retrieval[configuration]["evidence_retrieval_recall"]
            ),
            "evidence_recall": (
                "NA"
                if report[configuration]["evidence_recall"] is None
                else report[configuration]["evidence_recall"]
            ),
            "structured_evidence_link_precision": (
                "NA"
                if report[configuration]["structured_evidence_link_precision"] is None
                else report[configuration]["structured_evidence_link_precision"]
            ),
            "structured_audit_consistent": (
                f"{report[configuration]['structured_audit_consistent']['numerator']}/"
                f"{report[configuration]['structured_audit_consistent']['denominator']}"
            ),
            "narrative_numbers_absent": (
                f"{report[configuration]['narrative_numbers_absent']['numerator']}/"
                f"{report[configuration]['narrative_numbers_absent']['denominator']}"
            ),
            "semantic_consistent": (
                f"{report[configuration]['semantic_consistent']['numerator']}/"
                f"{report[configuration]['semantic_consistent']['denominator']}"
            ),
            "provenance_clean": (
                "NA"
                if provenance is None
                else f"{provenance['numerator']}/{provenance['denominator']}"
            ),
            "system_assessment_levels": (
                "NA"
                if report[configuration]["assessment_level_counts"] == {"N/A": 50}
                else json.dumps(
                    report[configuration]["assessment_level_counts"],
                    sort_keys=True,
                )
            ),
            "evaluation_only_levels": json.dumps(
                report[configuration]["evaluation_only_level_counts"],
                sort_keys=True,
            ),
        }
        rows.append(row)
    return rows


def plot_figure4(scoring: dict[str, Any], retrieval_cases: pd.DataFrame, out_base: Path) -> None:
    configure_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(174 * MM_TO_INCH, 94 * MM_TO_INCH), gridspec_kw={"width_ratios": [0.9, 1.35]})

    metric_columns = ["evidence_retrieval_precision", "evidence_retrieval_recall"]
    labels = ["Precision", "Recall"]
    x = np.arange(len(labels))
    width = 0.34
    for offset, (configuration, color, hatch) in enumerate(
        (("flat_rag", "#d9d9d9", "//"), ("layered_rag", "#5c5c5c", ""))
    ):
        means, lower, upper = [], [], []
        subset = retrieval_cases[retrieval_cases["configuration"] == configuration]
        for metric in metric_columns:
            valid = subset[[metric, "source_cluster_id"]].dropna()
            mean_value, low, high = bootstrap_mean_ci(
                valid[metric].to_numpy(float), valid["source_cluster_id"].to_numpy(str)
            )
            means.append(mean_value)
            lower.append(mean_value - low)
            upper.append(high - mean_value)
        positions = x + (offset - 0.5) * width
        bars = ax_a.bar(
            positions,
            means,
            width,
            color=color,
            edgecolor="#333333",
            linewidth=0.7,
            hatch=hatch,
            label=CONFIG_LABELS[configuration],
            yerr=np.array([lower, upper]),
            capsize=2,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
        for bar, value in zip(bars, means):
            ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    ax_a.set_xticks(x, labels)
    ax_a.set_ylim(0, 1.08)
    ax_a.set_ylabel("Macro-averaged metric")
    ax_a.set_title("a  Retrieval under an equal record budget", loc="left", y=1.18, fontweight="bold")
    ax_a.legend(loc="lower right")
    ax_a.grid(False)
    flat_budget = scoring["retrieval_benchmark"]["configurations"]["flat_rag"]["mean_evidence_record_count"]
    layered_budget = scoring["retrieval_benchmark"]["configurations"]["layered_rag"]["mean_evidence_record_count"]
    ax_a.text(
        0.02,
        0.98,
        f"Mean records/case: {flat_budget:.2f} flat, {layered_budget:.2f} layered",
        transform=ax_a.transAxes,
        va="top",
        fontsize=6.2,
    )

    comparisons = scoring["paired_differences_with_source_cluster_bootstrap_ci95"]
    comparison_specs = (
        ("layered_rag_minus_flat_rag", "Layered RAG - Flat RAG", "#3f3f3f", "o", -0.10),
        ("op_rag_minus_layered_rag", "OP-RAG - Layered RAG", "#8f8f8f", "s", 0.10),
    )
    y = np.arange(len(REPORT_METRICS))[::-1]
    for key, label, color, marker, offset in comparison_specs:
        means, lo, hi = [], [], []
        for metric, _ in REPORT_METRICS:
            result = comparisons[key][metric]
            means.append(result["mean_difference"])
            lo.append(result["ci95"][0])
            hi.append(result["ci95"][1])
        means_arr = np.asarray(means)
        ax_b.errorbar(
            means_arr,
            y + offset,
            xerr=np.vstack([means_arr - np.asarray(lo), np.asarray(hi) - means_arr]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            markersize=4,
            label=label,
        )
    ax_b.axvline(0, color="#222222", linewidth=0.8)
    ax_b.set_yticks(y, [label for _, label in REPORT_METRICS])
    ax_b.set_xlabel("Paired mean difference (95% source-cluster bootstrap CI)")
    ax_b.set_title("b  Paired differences in report metrics", loc="left", y=1.18, fontweight="bold")
    ax_b.grid(False)
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        fontsize=6,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.18, wspace=0.55)
    save_figure(fig, out_base)
    plt.close(fig)


def plot_figure5(internal_cases: pd.DataFrame, summary: dict[str, Any], out_base: Path) -> None:
    configure_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(174 * MM_TO_INCH, 84 * MM_TO_INCH), gridspec_kw={"width_ratios": [1.12, 0.88]})

    labels = [
        "Syndrome\nresolved",
        "Formula\nmapped",
        "Formula\nsource",
        "Any-chain\ncompletion",
        "Core60",
        "Strict",
    ]
    metrics = [
        summary["primary_syndrome_resolution"],
        summary["formula_mapping"],
        summary["formula_source_support_leave_one_source_out"],
        summary["any_level_closure"],
        summary["core60_closure"],
        summary["strict_closure"],
    ]
    values = [metric["rate"] for metric in metrics]
    counts = [fraction_text(metric) for metric in metrics]
    bars = ax_a.bar(
        np.arange(len(values)),
        values,
        color=["#3f3f3f", "#5f5f5f", "#7f7f7f", "#9f9f9f", "#bfbfbf", "#dfdfdf"],
        edgecolor="#333333",
        linewidth=0.7,
    )
    for bar, value, count in zip(bars, values, counts):
        ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{count}\n({value:.1%})", ha="center", va="bottom", fontsize=6)
    ax_a.set_xticks(np.arange(len(labels)), labels, fontsize=6)
    ax_a.set_ylim(0, 1.13)
    ax_a.set_ylabel("Observed proportion")
    ax_a.set_title("a  Internal evidence coverage", loc="left", fontweight="bold")
    ax_a.grid(False)
    ax_a.text(
        0.02,
        0.98,
        f"Core60 and strict rates use {summary['core60_closure']['denominator']} evaluable records",
        transform=ax_a.transAxes,
        va="top",
        fontsize=6.2,
    )

    source = internal_cases["source_group"].replace({"literature_binli2": "Literature", "literature_binli3": "Literature", "hospital_real_case": "Hospital"})
    level = internal_cases["assessment_level"].astype(int)
    groups = ["Hospital", "Literature"]
    level_counts = {group: Counter(level[source == group]) for group in groups}
    bottoms = np.zeros(len(groups))
    colors = ["#2f2f2f", "#7a7a7a", "#c2c2c2", "#ffffff"]
    hatches = ["", "//", "..", "xx"]
    for assessment_level, color, hatch in zip((1, 2, 3, 4), colors, hatches):
        counts_for_level = np.array([level_counts[group].get(assessment_level, 0) for group in groups])
        ax_b.bar(groups, counts_for_level, bottom=bottoms, color=color, edgecolor="#333333", linewidth=0.7, hatch=hatch, label=f"Level {assessment_level}")
        for index, (count, bottom) in enumerate(zip(counts_for_level, bottoms)):
            if count:
                ax_b.text(index, bottom + count / 2, str(int(count)), ha="center", va="center", fontsize=6, color="white" if assessment_level == 1 else "black")
        bottoms += counts_for_level
    ax_b.set_ylabel("Number of records")
    ax_b.set_ylim(0, max(bottoms) * 1.08)
    ax_b.set_title("b  Four-level internal results", loc="left", fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=6)
    ax_b.grid(False)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.20, wspace=0.42)
    save_figure(fig, out_base)
    plt.close(fig)


def copy_reproducibility_artifacts(
    run_dir: Path,
    analysis_dir: Path,
    package_dir: Path,
    experiment_root: Path,
    pre_api_spot_check_dir: Path,
    post_run_audit_dir: Path,
) -> None:
    supplementary = package_dir / "supplementary_data"
    scripts = package_dir / "scripts"
    supplementary.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    for source in (
        OUTPUT_DIR / "kb_integrity_audit.json",
        OUTPUT_DIR / "internal_case_analysis.csv",
        OUTPUT_DIR / "internal_case_analysis_summary.json",
        OUTPUT_DIR / "case_formula_occurrence_provenance.csv",
        OUTPUT_DIR / "formula_literature_source_index.csv",
        OUTPUT_DIR / "retrieval_benchmark.csv",
        OUTPUT_DIR / "retrieval_benchmark_summary.json",
        OUTPUT_DIR / "internal_reference_set.jsonl",
        OUTPUT_DIR / "model_contexts.jsonl",
        OUTPUT_DIR / "evaluation_only_assessments.jsonl",
        run_dir / "qwen_results.jsonl",
        run_dir / "qwen_scored_cases.csv",
        run_dir / "qwen_scoring_summary.json",
        run_dir / "run_manifest.json",
        run_dir / "failures.json",
        run_dir / "LOCAL_EXPERIMENT_INTEGRITY_AUDIT.json",
        run_dir / "LOCAL_EXPERIMENT_INTEGRITY_AUDIT.md",
        run_dir / "response_consistency_audit.csv",
        run_dir / "response_consistency_audit_summary.json",
        run_dir / "RESPONSE_CONSISTENCY_AUDIT.md",
        analysis_dir / "analysis_manifest.json",
        analysis_dir / "topk_sensitivity_case_level.csv",
        analysis_dir / "topk_sensitivity_summary.csv",
        analysis_dir / "threshold_sensitivity.csv",
        analysis_dir / "case_failure_taxonomy.csv",
        analysis_dir / "failure_taxonomy_summary.csv",
        analysis_dir / "representative_failure_cases.csv",
        analysis_dir / "formula_group_analysis.csv",
    ):
        shutil.copy2(source, supplementary / source.name)

    pre_spot_destination = supplementary / "pre_api_spot_checks"
    pre_spot_destination.mkdir(parents=True, exist_ok=True)
    for source in [pre_api_spot_check_dir / "SPOT_CHECK_INDEX.json", *sorted(pre_api_spot_check_dir.glob("spot_check_*.md"))]:
        shutil.copy2(source, pre_spot_destination / source.name)

    post_run_source = post_run_audit_dir
    post_run_destination = supplementary / "post_formal_run_audit"
    post_run_destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(post_run_source.glob("*.md")) + sorted(
        post_run_source.glob("*.json")
    ):
        shutil.copy2(source, post_run_destination / source.name)

    experiment_manifest = read_json(OUTPUT_DIR / "experiment_manifest.json")
    relative_experiment_dir = f"experiments/{experiment_root.name}"
    experiment_manifest["experiment_dir"] = relative_experiment_dir
    experiment_manifest["cases_path"] = (
        f"{relative_experiment_dir}/inputs/eval_cases_visible_plan_001_050.jsonl"
    )
    (supplementary / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for source in (
        Path(__file__).parent / "protocol.py",
        Path(__file__).parent / "evaluation_protocol.py",
        Path(__file__).parent / "prepare_experiment.py",
        Path(__file__).parent / "qwen_protocol.py",
        Path(__file__).parent / "run_qwen_comparison.py",
        Path(__file__).parent / "score_qwen_comparison.py",
        Path(__file__),
        Path(__file__).parent / "audit_experiment_integrity.py",
        Path(__file__).parent / "audit_pre_run.py",
        Path(__file__).parent / "audit_generated_responses.py",
        Path(__file__).parent / "audit_publication_inputs.py",
        Path(__file__).parent / "run_api_spot_check.py",
        Path(__file__).parent / "audit_post_run_samples.py",
        Path(__file__).parent / "finalize_qwen_manifest.py",
        Path(__file__).parent / "run_full_v4_pipeline.py",
        Path(__file__).parent / "test_scoring.py",
        Path(__file__).parent / "test_regressions.py",
        Path(__file__).parent / "test_structured_reporting.py",
        Path(__file__).parent / "test_reporting_integration.py",
        Path(__file__).parent / "analyze_sensitivity_errors.py",
    ):
        shutil.copy2(source, scripts / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=OUTPUT_DIR / "sensitivity_error_analysis")
    parser.add_argument("--generated-dir", type=Path, default=OUTPUT_DIR / "publication_v4_final")
    parser.add_argument(
        "--pre-api-spot-check-dir",
        type=Path,
        default=OUTPUT_DIR / "api_spot_check_gate_20260810",
    )
    parser.add_argument(
        "--post-run-audit-dir",
        type=Path,
        default=OUTPUT_DIR / "api_spot_check_gate_20260810" / "post_formal_run_audit",
    )
    args = parser.parse_args()

    scoring = read_json(args.run_dir / "qwen_scoring_summary.json")
    run_integrity = scoring.get("run_integrity", {})
    if run_integrity.get("status") != "completed" or run_integrity.get("n_unique_case_configuration_keys") != 200:
        raise SystemExit("Publication outputs require a completed run with 200 unique case-configuration keys")

    kb = read_json(OUTPUT_DIR / "kb_integrity_audit.json")
    internal_summary = read_json(OUTPUT_DIR / "internal_case_analysis_summary.json")
    internal_cases = pd.read_csv(OUTPUT_DIR / "internal_case_analysis.csv")
    retrieval_cases = pd.read_csv(OUTPUT_DIR / "retrieval_benchmark.csv")

    generated = args.generated_dir
    generated.mkdir(parents=True, exist_ok=True)
    write_csv(generated / "table_kb_quality.csv", build_kb_table(kb))
    write_csv(generated / "table_internal_application.csv", build_internal_table(internal_summary))
    write_csv(generated / "table_rag_comparison.csv", build_comparison_table(scoring))
    value_registry = {
        "scope": "single source of values for later manuscript, table, and figure synchronization",
        "run_results_sha256": sha256(args.run_dir / "qwen_results.jsonl"),
        "scoring_summary_sha256": sha256(args.run_dir / "qwen_scoring_summary.json"),
        "retrieval_benchmark": scoring["retrieval_benchmark"],
        "deterministic_internal_case_analysis": internal_summary,
        "reporting_quality_by_configuration": {
            name: {
                "n_cases": values["n_cases"],
                "output_schema_valid": values["output_schema_valid"],
                "structured_audit_consistent": values["structured_audit_consistent"],
                "narrative_numbers_absent": values["narrative_numbers_absent"],
                "semantic_consistent": values["semantic_consistent"],
                "provenance_clean": values["provenance_clean"],
                "deterministic_coverage": values["deterministic_coverage"],
                "assessment_level_counts": values["assessment_level_counts"],
                "evaluation_only_level_counts": values["evaluation_only_level_counts"],
            }
            for name, values in scoring["configurations"].items()
        },
    }
    (generated / "publication_value_registry.json").write_text(
        json.dumps(value_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_figure4(scoring, retrieval_cases, generated / "Figure4_fair_rag_comparison")
    plot_figure5(internal_cases, internal_summary, generated / "Figure5_internal_application")

    figures = args.package_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    supplementary_data = args.package_dir / "supplementary_data"
    supplementary_data.mkdir(parents=True, exist_ok=True)
    for stem in ("Figure4_fair_rag_comparison", "Figure5_internal_application"):
        for suffix in (".svg", ".pdf", ".png", ".tiff"):
            shutil.copy2(generated / f"{stem}{suffix}", figures / f"{stem}{suffix}")
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        shutil.copy2(
            args.analysis_dir / f"FigureS1_threshold_sensitivity{suffix}",
            figures / f"FigureS1_threshold_sensitivity{suffix}",
        )
    for source in generated.glob("*.csv"):
        shutil.copy2(source, supplementary_data / source.name)
    shutil.copy2(
        generated / "publication_value_registry.json",
        supplementary_data / "publication_value_registry.json",
    )
    copy_reproducibility_artifacts(
        args.run_dir,
        args.analysis_dir,
        args.package_dir,
        Path(__file__).parent,
        args.pre_api_spot_check_dir,
        args.post_run_audit_dir,
    )

    qa = {
        "figure_contract": {
            "Figure4": "Layered retrieval is compared with flat retrieval under an equal record budget; code-generated missing-item, coverage, and assessment-level fields are excluded from model-performance comparisons.",
            "Figure5": "Internal coverage and four-level results are shown with explicit denominators and source groups.",
        },
        "backend": "Python/matplotlib",
        "exports": ["SVG", "PDF", "600-dpi TIFF", "600-dpi PNG"],
        "gridlines": "disabled in Figures 4, 5, and S1",
        "grayscale_readable": True,
        "source_data_files": [source.name for source in generated.glob("*.csv")],
        "scope": "deterministic internal KB-grounded audit with Qwen used only for non-numeric verbalization; not clinical accuracy or efficacy",
    }
    (generated / "FIGURE_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(generated / "FIGURE_QA.json", supplementary_data / "FIGURE_QA.json")
    print(json.dumps({"generated_dir": str(generated), "package_dir": str(args.package_dir)}, indent=2))


if __name__ == "__main__":
    main()
