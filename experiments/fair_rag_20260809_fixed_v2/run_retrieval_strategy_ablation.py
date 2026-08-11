from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from protocol import (
    CASES_PATH,
    OUTPUT_DIR,
    AblationRunner,
    RecordRetriever,
    add_formula_occurrence_provenance,
    build_case_occurrence_provenance,
    build_internal_reference,
    clean_record,
    dedupe_records,
    evidence_item_ids,
    flat_retrieval,
    layered_retrieval,
    load_kb,
    make_compact_records,
    normalize_herb,
    qualify_mechanism_kb,
    read_jsonl,
    retrieval_metrics,
    serialized_bytes,
    sha256,
    source_document_id,
    unique_strings,
)
from score_qwen_comparison import bootstrap_paired_difference


STRATEGIES = (
    "flat_tfidf",
    "layered_tfidf",
    "layered_hybrid_exact_herb",
)
METRICS = (
    "evidence_retrieval_precision",
    "evidence_retrieval_recall",
    "syndrome_precision",
    "syndrome_recall",
    "formula_precision",
    "formula_recall",
    "herb_precision",
    "herb_recall",
)
COMPARISON_METRICS = (
    *METRICS,
    "retrieved_item_count",
    "serialized_evidence_bytes",
)


def layered_tfidf_retrieval(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    syndrome_top_k: int = 2,
    formula_top_k: int = 2,
    herb_top_k: int = 1,
) -> list[dict[str, Any]]:
    """Retrieve every structured field from its own layer using TF-IDF."""
    layers = {
        layer: RecordRetriever([record for record in records if record["layer"] == layer])
        for layer in ("syndrome", "formula", "herb")
    }
    selected: list[dict[str, Any]] = []
    syndrome_queries = unique_strings(
        [plan.get("primary_syndrome_name"), *(plan.get("secondary_syndrome_names") or [])]
    )
    for query in syndrome_queries:
        selected.extend(layers["syndrome"].search(query, top_k=syndrome_top_k))
    formula_query = str(plan.get("formula_name") or "").strip()
    if formula_query:
        selected.extend(layers["formula"].search(formula_query, top_k=formula_top_k))
    for herb in unique_strings(plan.get("herbs") or []):
        if herb_top_k <= 0:
            continue
        selected.extend(
            layers["herb"].search(normalize_herb(herb), top_k=herb_top_k)
        )
    return [clean_record(record) for record in dedupe_records(selected)]


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def layer_ids(item_ids: set[str], layer: str) -> set[str]:
    prefix = f"{layer}:"
    return {item_id for item_id in item_ids if item_id.startswith(prefix)}


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def layer_metrics(
    expected_ids: set[str], retrieved_ids: set[str], layer: str
) -> dict[str, Any]:
    expected = layer_ids(expected_ids, layer)
    retrieved = layer_ids(retrieved_ids, layer)
    correct = expected & retrieved
    return {
        f"{layer}_expected_count": len(expected),
        f"{layer}_retrieved_count": len(retrieved),
        f"{layer}_correct_count": len(correct),
        f"{layer}_extra_count": len(retrieved - expected),
        f"{layer}_missing_count": len(expected - retrieved),
        f"{layer}_precision": safe_ratio(len(correct), len(retrieved)),
        f"{layer}_recall": safe_ratio(len(correct), len(expected)),
    }


def case_row(
    reference: dict[str, Any], strategy: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    context = {
        "configuration": strategy,
        "evidence_context": records,
        "retrieval_budget": {
            "evidence_record_count": len(records),
            "serialized_evidence_bytes": serialized_bytes(records),
        },
    }
    metrics = retrieval_metrics(reference, context)
    expected = set(reference.get("expected_retrieval_item_ids", []))
    retrieved = evidence_item_ids(records)
    return {
        "case_id": reference["case_id"],
        "source_group": reference.get("source_group"),
        "source_cluster_id": reference.get("source_cluster_id"),
        "strategy": strategy,
        **metrics,
        **layer_metrics(expected, retrieved, "syndrome"),
        **layer_metrics(expected, retrieved, "formula"),
        **layer_metrics(expected, retrieved, "herb"),
    }


def mean_defined(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return mean(values) if values else None


def strategy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cases": len(rows),
        **{metric: mean_defined(rows, metric) for metric in METRICS},
        "mean_evidence_record_count": mean_defined(rows, "retrieved_item_count"),
        "mean_serialized_evidence_bytes": mean_defined(
            rows, "serialized_evidence_bytes"
        ),
        "total_extra_herb_records": sum(int(row["herb_extra_count"]) for row in rows),
        "total_missing_herb_records": sum(
            int(row["herb_missing_count"]) for row in rows
        ),
    }


def paired_comparison(
    rows_by_strategy: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "interpretation": f"{right} minus {left}",
        "metrics": {
            metric: bootstrap_paired_difference(
                rows_by_strategy[left], rows_by_strategy[right], metric
            )
            for metric in COMPARISON_METRICS
        },
    }


def report_text(summary: dict[str, Any]) -> str:
    strategies = summary["strategies"]

    def value(strategy: str, metric: str) -> str:
        result = strategies[strategy].get(metric)
        return "NA" if result is None else f"{result:.3f}"

    return f"""# OP-RAG确定性检索消融实验

## 实验范围

- 病例数：{summary['n_cases']}例。
- 本实验仅运行本地确定性检索，不调用千问API。
- 所有策略使用同一知识库、同一病例标准化结果、同一病例来源排除规则和同一版本化审计参考。

## 比较一：分层与字段路由

- Flat-TFIDF：证型、方剂和药材合并为一次查询，在混合索引中检索。
- Layered-TFIDF：证型、方剂和每味药材分别进入对应层，三层均使用字符级TF-IDF。
- 两种方法逐病例使用相同返回记录预算。

| 策略 | 精确率 | 召回率 | 药材精确率 | 药材召回率 | 平均记录数 |
|---|---:|---:|---:|---:|---:|
| Flat-TFIDF | {value('flat_tfidf', 'evidence_retrieval_precision')} | {value('flat_tfidf', 'evidence_retrieval_recall')} | {value('flat_tfidf', 'herb_precision')} | {value('flat_tfidf', 'herb_recall')} | {value('flat_tfidf', 'mean_evidence_record_count')} |
| Layered-TFIDF | {value('layered_tfidf', 'evidence_retrieval_precision')} | {value('layered_tfidf', 'evidence_retrieval_recall')} | {value('layered_tfidf', 'herb_precision')} | {value('layered_tfidf', 'herb_recall')} | {value('layered_tfidf', 'mean_evidence_record_count')} |

配对差值及24个来源簇Bootstrap置信区间见`summary.json`中的`layer_routing_effect`。本比较检验分层和字段路由的整体贡献，不包含药材精确匹配。

## 比较二：药材精确匹配

- Layered-TFIDF：三层均使用TF-IDF。
- Layered-Hybrid：证型和方剂使用TF-IDF，药材使用标准化名称精确匹配。
- 两种方法均使用证型top-2、方剂top-2和每味药材top-1的最大查询规则。精确匹配在知识库无对应药材时不返回相似药材，因此实际记录数可以更少；这是方法效率结果，不是人为删减。

| 策略 | 精确率 | 召回率 | 药材精确率 | 药材召回率 | 平均记录数 |
|---|---:|---:|---:|---:|---:|
| Layered-TFIDF | {value('layered_tfidf', 'evidence_retrieval_precision')} | {value('layered_tfidf', 'evidence_retrieval_recall')} | {value('layered_tfidf', 'herb_precision')} | {value('layered_tfidf', 'herb_recall')} | {value('layered_tfidf', 'mean_evidence_record_count')} |
| Layered-Hybrid | {value('layered_hybrid_exact_herb', 'evidence_retrieval_precision')} | {value('layered_hybrid_exact_herb', 'evidence_retrieval_recall')} | {value('layered_hybrid_exact_herb', 'herb_precision')} | {value('layered_hybrid_exact_herb', 'herb_recall')} | {value('layered_hybrid_exact_herb', 'mean_evidence_record_count')} |

配对差值见`summary.json`中的`exact_herb_effect_same_query_limits`。召回率、精确率和实际记录数分别报告，不把减少无关记录误写成召回率提升。

## 边界

这些结果评价版本化知识库中的记录检索，不评价诊断准确性、处方合理性、治疗有效性或千问能力。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "deterministic_retrieval_ablation_20260810",
    )
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    original_kb = load_kb()
    literature_sources, occurrence_sources, _, _ = build_case_occurrence_provenance(
        cases, original_kb
    )
    kb = qualify_mechanism_kb(
        add_formula_occurrence_provenance(
            original_kb, literature_sources, occurrence_sources
        )
    )
    runner = AblationRunner(kb)
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        excluded = (
            {source_document_id(case)}
            if case.get("source_group") != "hospital_real_case"
            else set()
        )
        reference = build_internal_reference(
            case, runner, kb, excluded_source_ids=excluded
        )
        records = make_compact_records(kb, excluded_source_ids=excluded)
        plan = reference["physician_plan"]

        layered_tfidf = layered_tfidf_retrieval(plan, records)
        flat_tfidf = flat_retrieval(plan, records, top_k=len(layered_tfidf))
        layered_hybrid = layered_retrieval(plan, records)

        strategy_records = {
            "flat_tfidf": flat_tfidf,
            "layered_tfidf": layered_tfidf,
            "layered_hybrid_exact_herb": layered_hybrid,
        }
        if len(flat_tfidf) != len(layered_tfidf):
            raise RuntimeError(
                f"Matched-budget failure for {reference['case_id']}: "
                f"flat={len(flat_tfidf)}, layered={len(layered_tfidf)}"
            )
        for strategy in STRATEGIES:
            item_ids = [record["item_id"] for record in strategy_records[strategy]]
            if len(item_ids) != len(set(item_ids)):
                raise RuntimeError(
                    f"Duplicate retrieved ID for {reference['case_id']} {strategy}"
                )
            if any(float(record.get("retrieval_score") or 0.0) <= 0.0 for record in strategy_records[strategy]):
                raise RuntimeError(
                    f"Non-positive retrieval score for {reference['case_id']} {strategy}"
                )
            case_rows.append(case_row(reference, strategy, strategy_records[strategy]))

    case_ids = {case["case_id"] for case in cases}
    source_clusters = {
        str(row.get("source_cluster_id"))
        for row in case_rows
        if row.get("source_cluster_id")
    }
    integrity_checks = {
        "expected_50_unique_cases": len(cases) == 50 and len(case_ids) == 50,
        "expected_150_case_strategy_rows": len(case_rows) == 50 * len(STRATEGIES),
        "routing_budget_matched_50_of_50": all(
            next(
                row["retrieved_item_count"]
                for row in case_rows
                if row["case_id"] == case_id and row["strategy"] == "flat_tfidf"
            )
            == next(
                row["retrieved_item_count"]
                for row in case_rows
                if row["case_id"] == case_id and row["strategy"] == "layered_tfidf"
            )
            for case_id in case_ids
        ),
        "expected_24_source_clusters": len(source_clusters) == 24,
        "api_calls": 0,
    }
    if not all(value is True or key == "api_calls" for key, value in integrity_checks.items()):
        raise RuntimeError(f"Retrieval-ablation integrity gate failed: {integrity_checks}")

    rows_by_strategy = {
        strategy: {
            row["case_id"]: row
            for row in case_rows
            if row["strategy"] == strategy
        }
        for strategy in STRATEGIES
    }
    summary = {
        "scope": "deterministic retrieval ablation; no Qwen API calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "integrity_checks": integrity_checks,
        "strategies": {
            strategy: strategy_summary(list(rows.values()))
            for strategy, rows in rows_by_strategy.items()
        },
        "comparisons": {
            "layer_routing_effect": paired_comparison(
                rows_by_strategy, "flat_tfidf", "layered_tfidf"
            ),
            "exact_herb_effect_same_query_limits": paired_comparison(
                rows_by_strategy,
                "layered_tfidf",
                "layered_hybrid_exact_herb",
            ),
        },
        "query_settings": {
            "syndrome_top_k": 2,
            "formula_top_k": 2,
            "herb_top_k_per_input_herb": 1,
            "minimum_tfidf_score_exclusive": 0.0,
        },
        "interpretation_limits": [
            "The versioned audit reference is derived from the locked resource and predefined rules, not independent clinical ground truth.",
            "The layer-routing comparison holds the per-case record count equal.",
            "The herb-matcher comparison uses identical per-query top-k limits but allows realized record counts to differ when exact lookup has no match.",
            "Qwen does not participate in retrieval or metric calculation.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "case_level.csv", case_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "status": "completed",
            "created_at": summary["created_at"],
            "n_cases": len(cases),
            "n_case_strategy_rows": len(case_rows),
            "strategies": list(STRATEGIES),
            "api_calls": 0,
            "integrity_checks": integrity_checks,
            "input_hashes": {
                "cases": sha256(args.cases),
                "protocol.py": sha256(Path(__file__).resolve().parent / "protocol.py"),
                "run_retrieval_strategy_ablation.py": sha256(Path(__file__).resolve()),
            },
        },
    )
    (args.output_dir / "REPORT_zh-CN.md").write_text(
        report_text(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
