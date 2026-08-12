from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for import_path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, sha256, write_json  # noqa: E402
from qwen_protocol import build_user_prompt, validate_response  # noqa: E402
from run_qwen_comparison import context_hash  # noqa: E402
from score_qwen_comparison import (  # noqa: E402
    build_item_aliases,
    score_record,
)


SELECTION_SEED = 20260810
DEFAULT_SPOT_ROOT = OUTPUT_DIR / "api_spot_check_gate_20260810"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def values_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_details(expected: set[str], reported: set[str]) -> dict[str, Any]:
    true_positive = len(expected & reported)
    precision = ratio(true_positive, len(reported))
    recall = ratio(true_positive, len(expected))
    if not expected and not reported:
        value = 1.0
    else:
        p = precision or 0.0
        r = recall or 0.0
        value = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "true_positive": true_positive,
        "expected_count": len(expected),
        "reported_count": len(reported),
        "precision": precision,
        "recall": recall,
        "f1": value,
    }


def independent_level(rule_inputs: dict[str, Any]) -> int:
    if bool(rule_inputs.get("contradiction")):
        return 4
    if bool(rule_inputs.get("strict_support")):
        return 1
    if all(
        bool(rule_inputs.get(name))
        for name in (
            "syndrome_evidence_available",
            "formula_evidence_available",
            "mechanism_evidence_available",
        )
    ):
        return 2
    return 3


def visible_record(
    item_id: str, records_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    direct = records_by_id.get(item_id)
    if direct is not None:
        return direct
    if not item_id.startswith("relation:"):
        return None
    parts = item_id.split(":")
    formula_id = parts[1] if len(parts) >= 3 else ""
    formula_record = records_by_id.get(f"formula:{formula_id}")
    syndrome_visible = any(
        record.get("layer") == "syndrome" for record in records_by_id.values()
    )
    return formula_record if formula_record is not None and syndrome_visible else None


def independent_claim_counts(
    reference: dict[str, Any], context: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    consistency_applicable = context.get("configuration") == "op_rag"
    expected_by_id = {
        str(claim["item_id"]): claim
        for claim in reference.get("expected_claims", [])
        if consistency_applicable or claim.get("layer") != "cross_layer"
    }
    records_by_id = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    reported_claims = [
        claim
        for claim in (response.get("evidence_claims") or [])
        if isinstance(claim, dict)
    ]
    correct_ids: set[str] = set()
    for claim in reported_claims:
        item_id = str(claim.get("item_id") or "")
        expected = expected_by_id.get(item_id)
        if expected is None or claim.get("support_status") != expected.get("expected_status"):
            continue
        record = visible_record(item_id, records_by_id)
        if record is None:
            continue
        reported_sources = {str(value) for value in (claim.get("source_ids") or [])}
        expected_sources = {str(value) for value in (expected.get("source_ids") or [])}
        visible_sources = {str(value) for value in (record.get("source_ids") or [])}
        if reported_sources and reported_sources <= expected_sources and reported_sources <= visible_sources:
            correct_ids.add(item_id)
    expected_count = len(expected_by_id)
    reported_count = len(reported_claims)
    correct_count = len(correct_ids)
    return {
        "expected_count": expected_count,
        "reported_count": reported_count,
        "correct_count": correct_count,
        "correct_item_ids": sorted(correct_ids),
        "evidence_recall": ratio(correct_count, expected_count),
        "link_precision": ratio(correct_count, reported_count),
    }


def compact_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": record.get("item_id"),
            "layer": record.get("layer"),
            "name": record.get("name"),
            "source_ids": record.get("source_ids"),
            "retrieval_score": record.get("retrieval_score"),
        }
        for record in records
    ]


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def display(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.12g}"


def selected_case_order() -> list[str]:
    case_ids = sorted(
        row["case_id"]
        for row in read_jsonl(
            EXPERIMENT_DIR / "inputs" / "eval_cases_visible_plan_001_050.jsonl"
        )
    )
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(case_ids)
    return case_ids


def load_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "selection_seed": SELECTION_SEED,
            "selection_method": "Python random.Random(seed).shuffle over sorted case IDs; no replacement",
            "entries": [],
            "consecutive_passes": 0,
            "formal_200_api_run_authorized": False,
        }
    return json.loads(index_path.read_text(encoding="utf-8"))


def validate_existing_index(index: dict[str, Any]) -> None:
    seen_cases: set[str] = set()
    consecutive = 0
    for expected_number, entry in enumerate(index.get("entries", []), start=1):
        if entry.get("check_number") != expected_number:
            raise SystemExit("Spot-check index numbers are not sequential")
        case_id = str(entry.get("case_id") or "")
        if not case_id or case_id in seen_cases:
            raise SystemExit("Spot-check index contains a missing or repeated case ID")
        seen_cases.add(case_id)
        status = str(entry.get("status") or "").casefold()
        if status not in {"pass", "fail"}:
            raise SystemExit("Spot-check index contains an invalid status")
        report_path = Path(str(entry.get("report_path") or ""))
        if not report_path.exists():
            raise SystemExit(f"Spot-check report is missing: {report_path}")
        report_text = report_path.read_text(encoding="utf-8")
        if f"**最终判定：{status.upper()}**" not in report_text:
            raise SystemExit(
                f"Spot-check index and Markdown verdict disagree for check {expected_number}"
            )
        consecutive = consecutive + 1 if status == "pass" else 0
        if entry.get("consecutive_passes_after_check") != consecutive:
            raise SystemExit(
                f"Stored consecutive-pass count is inconsistent at check {expected_number}"
            )
    if index.get("consecutive_passes") != consecutive:
        raise SystemExit("Top-level consecutive-pass count is inconsistent")
    if bool(index.get("formal_200_api_run_authorized")) != (consecutive >= 5):
        raise SystemExit("Formal-run authorization disagrees with the consecutive-pass count")


def validate_configuration(
    *,
    configuration: str,
    context: dict[str, Any],
    reference: dict[str, Any],
    result: dict[str, Any],
    retrieval_row: dict[str, str],
    scored: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    response = result["response"]
    raw = {"assessment_summary": response.get("assessment_summary")}
    validation = validate_response(response, context, raw_model_response=raw)
    if not validation["valid"]:
        errors.append(f"response validation failed: {validation}")

    prompt = build_user_prompt(context)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if result.get("context_sha256") != context_hash(context):
        errors.append("stored context hash differs from the current context")
    if result.get("remote_metadata", {}).get("user_prompt_sha256") != prompt_sha:
        errors.append("stored user-prompt hash differs from the reconstructed prompt")
    if result.get("remote_metadata", {}).get("model_output_fields") != ["assessment_summary"]:
        errors.append("the model returned fields other than assessment_summary")

    coverage_checks: dict[str, Any] = {}
    for metric_name, record in response.get("coverage_metrics", {}).items():
        supported = set(record.get("supported_items") or [])
        missing = set(record.get("missing_items") or [])
        numerator = len(supported)
        denominator = len(supported | missing)
        value = ratio(numerator, denominator)
        check = {
            "supported_count_recomputed": numerator,
            "denominator_recomputed": denominator,
            "value_recomputed": value,
            "stored_numerator": record.get("numerator"),
            "stored_denominator": record.get("denominator"),
            "stored_value": record.get("value"),
            "sets_disjoint": not bool(supported & missing),
        }
        coverage_checks[metric_name] = check
        if numerator != record.get("numerator"):
            errors.append(f"{metric_name}: numerator is not the supported-item count")
        if denominator != record.get("denominator"):
            errors.append(f"{metric_name}: denominator does not equal the item universe")
        if not values_equal(value, record.get("value")):
            errors.append(f"{metric_name}: coverage value is arithmetically inconsistent")
        if supported & missing:
            errors.append(f"{metric_name}: supported and missing item sets overlap")

    expected_retrieval = set(reference.get("expected_retrieval_item_ids") or [])
    retrieved = {
        str(record.get("item_id"))
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    retrieval_correct = expected_retrieval & retrieved
    if configuration == "qwen_only":
        retrieval_precision = None
        retrieval_recall = None
    else:
        retrieval_precision = ratio(len(retrieval_correct), len(retrieved))
        retrieval_recall = ratio(len(retrieval_correct), len(expected_retrieval))
    retrieval_check = {
        "expected_ids": sorted(expected_retrieval),
        "retrieved_ids": sorted(retrieved),
        "correct_ids": sorted(retrieval_correct),
        "precision_recomputed": retrieval_precision,
        "recall_recomputed": retrieval_recall,
    }
    if int(retrieval_row["expected_item_count"]) != len(expected_retrieval):
        errors.append("retrieval expected-item count mismatch")
    if int(retrieval_row["retrieved_item_count"]) != len(retrieved):
        errors.append("retrieval record count mismatch")
    if int(retrieval_row["correct_item_count"]) != len(retrieval_correct):
        errors.append("retrieval correct-item count mismatch")
    if not values_equal(
        parse_optional_float(retrieval_row["evidence_retrieval_precision"]),
        retrieval_precision,
    ):
        errors.append("retrieval precision mismatch")
    if not values_equal(
        parse_optional_float(retrieval_row["evidence_retrieval_recall"]),
        retrieval_recall,
    ):
        errors.append("retrieval recall mismatch")

    claims = independent_claim_counts(reference, context, response)
    scored_recall = scored.get("evidence_recall")
    scored_precision = scored.get("structured_evidence_link_precision")
    expected_scored_recall = None if configuration == "qwen_only" else claims["evidence_recall"]
    expected_scored_precision = None if configuration == "qwen_only" else claims["link_precision"]
    if not values_equal(scored_recall, expected_scored_recall):
        errors.append("reported evidence recall differs from independent substitution")
    if not values_equal(scored_precision, expected_scored_precision):
        errors.append("structured evidence-link precision differs from independent substitution")

    expected_missing = set(context["structured_audit"].get("missing_evidence_items") or [])
    reported_missing = set(response.get("missing_evidence_items") or [])
    missing_f1 = f1_details(expected_missing, reported_missing)
    if not values_equal(scored.get("missing_evidence_disclosure_f1"), missing_f1["f1"]):
        errors.append("missing-evidence F1 differs from independent substitution")

    consistency_applicable = response.get("consistency_audit_applicable") is True
    stored_level = response.get("assessment_level")
    if consistency_applicable:
        rule_inputs = response.get("assessment_rule_trace", {}).get("inputs") or {}
        recalculated_level = independent_level(rule_inputs)
        if recalculated_level != stored_level:
            errors.append("assessment level differs from independent rule precedence")
        level_agreement = stored_level == context["structured_audit"].get("assessment_level")
        if scored.get("assessment_level_agreement") is not level_agreement:
            errors.append("assessment-level agreement field is inconsistent")
    else:
        rule_inputs = {}
        recalculated_level = None
        level_agreement = None
        if (
            stored_level is not None
            or response.get("assessment_rule_trace") is not None
            or response.get("formula_syndrome_relation") is not None
            or scored.get("assessment_level_agreement") is not None
        ):
            errors.append("non-OP configuration exposes a consistency result")

    evaluation_inputs = evaluation.get("assessment_rule_trace", {}).get("inputs") or {}
    evaluation_level = independent_level(evaluation_inputs)
    if evaluation.get("sent_to_qwen") is not False:
        errors.append("evaluation-only level is not marked as withheld from Qwen")
    if evaluation_level != evaluation.get("assessment_level"):
        errors.append("evaluation-only level differs from fixed-rule recomputation")
    if scored.get("evaluation_only_assessment_level") != evaluation_level:
        errors.append("scored evaluation-only level differs from the evaluation artifact")
    if scored.get("evaluation_only_sent_to_qwen") is not False:
        errors.append("scored row incorrectly marks the evaluation-only level as sent to Qwen")

    calculations = {
        "coverage": coverage_checks,
        "retrieval": retrieval_check,
        "claims": claims,
        "missing_evidence": missing_f1,
        "assessment": {
            "applicable": consistency_applicable,
            "rule_inputs": rule_inputs,
            "level_recomputed": recalculated_level,
            "level_stored": stored_level,
            "agreement": level_agreement,
        },
        "evaluation_only_assessment": {
            "sent_to_qwen": evaluation.get("sent_to_qwen"),
            "rule_inputs": evaluation_inputs,
            "level_recomputed": evaluation_level,
            "level_stored": evaluation.get("assessment_level"),
        },
    }
    return errors, calculations


def build_report(
    *,
    check_number: int,
    case_id: str,
    run_dir: Path,
    report_path: Path,
    audit_phase: str = "pre_api_spot_check",
    selection_seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    context_rows = [
        row for row in read_jsonl(OUTPUT_DIR / "model_contexts.jsonl") if row["case_id"] == case_id
    ]
    contexts = {row["configuration"]: row["context"] for row in context_rows}
    references = {
        row["case_id"]: row for row in read_jsonl(OUTPUT_DIR / "internal_reference_set.jsonl")
    }
    reference = references[case_id]
    results = {
        row["configuration"]: row
        for row in read_jsonl(run_dir / "qwen_results.jsonl")
        if row["case_id"] == case_id
    }
    retrieval_rows = {
        row["configuration"]: row
        for row in read_csv(OUTPUT_DIR / "retrieval_benchmark.csv")
        if row["case_id"] == case_id
    }
    evaluations = {
        row["configuration"]: row
        for row in read_jsonl(OUTPUT_DIR / "evaluation_only_assessments.jsonl")
        if row["case_id"] == case_id
    }
    all_context_rows = read_jsonl(OUTPUT_DIR / "model_contexts.jsonl")
    aliases = build_item_aliases(all_context_rows, references)
    scored = {}
    for name in CONFIGS:
        if name not in results or name not in contexts or name not in evaluations:
            continue
        row = score_record(results[name], reference, contexts[name], aliases)
        row["evaluation_only_assessment_level"] = evaluations[name]["assessment_level"]
        row["evaluation_only_assessment_label"] = evaluations[name]["assessment_label"]
        row["evaluation_only_sent_to_qwen"] = evaluations[name]["sent_to_qwen"]
        scored[name] = row
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    expected_configs = set(CONFIGS)
    if set(contexts) != expected_configs:
        errors.append(f"context configurations differ: {sorted(contexts)}")
    if set(results) != expected_configs:
        errors.append(f"result configurations differ: {sorted(results)}")
    if set(retrieval_rows) != expected_configs:
        errors.append(f"retrieval rows differ: {sorted(retrieval_rows)}")
    if set(evaluations) != expected_configs:
        errors.append(f"evaluation-only rows differ: {sorted(evaluations)}")
    if audit_phase == "pre_api_spot_check":
        manifest_valid = (
            manifest.get("status") == "completed"
            and manifest.get("n_expected") == 4
            and manifest.get("n_completed") == 4
            and manifest.get("n_valid") == 4
            and manifest.get("n_requested_in_this_run") == 4
            and manifest.get("n_reused_identical_context") == 0
            and not manifest.get("failures")
        )
    elif audit_phase == "post_formal_run_audit":
        manifest_valid = (
            manifest.get("status") == "completed"
            and manifest.get("n_expected") == 200
            and manifest.get("n_completed") == 200
            and manifest.get("n_valid") == 200
            and manifest.get("n_requested_in_this_run") == 200
            and manifest.get("n_unique_case_configuration_keys") == 200
            and manifest.get("n_reused_identical_context") == 0
            and manifest.get("no_old_result_reuse") is True
            and manifest.get("configuration_counts")
            == {name: 50 for name in sorted(CONFIGS)}
            and manifest.get("results_sha256") == sha256(run_dir / "qwen_results.jsonl")
            and not manifest.get("failures")
        )
    else:
        raise ValueError(f"Unsupported audit phase: {audit_phase}")
    if not manifest_valid:
        errors.append(f"{audit_phase} run manifest failed: {manifest}")

    calculations: dict[str, Any] = {}
    for configuration in CONFIGS:
        if not all(
            configuration in collection
            for collection in (contexts, results, retrieval_rows, scored, evaluations)
        ):
            continue
        config_errors, config_calculations = validate_configuration(
            configuration=configuration,
            context=contexts[configuration],
            reference=reference,
            result=results[configuration],
            retrieval_row=retrieval_rows[configuration],
            scored=scored[configuration],
            evaluation=evaluations[configuration],
        )
        errors.extend(f"{configuration}: {error}" for error in config_errors)
        calculations[configuration] = config_calculations

    passed = not errors
    lines = [
        (
            f"# OP-RAG API抽查测试集：第 {check_number} 次"
            if audit_phase == "pre_api_spot_check"
            else f"# OP-RAG 正式运行后抽查：第 {check_number} 例"
        ),
        "",
        f"- 抽查病例：`{case_id}`",
        f"- 选择种子：`{selection_seed}`（排序后的候选病例无放回打乱）",
        f"- 本次结论：**{'PASS' if passed else 'FAIL'}**",
        "- 实验范围：同一病例依次检查 `qwen_only`、`flat_rag`、`layered_rag`、`op_rag` 四个配置。",
        (
            "- 数据阶段：正式 200 次 API 运行前的独立单病例 API 抽查。"
            if audit_phase == "pre_api_spot_check"
            else "- 数据阶段：从已完成的正式 200 次 API 运行中抽取，不重新请求或替换模型响应。"
        ),
        "- 边界：这是内部工程与一致性抽查，不是临床正确性、诊断准确性或疗效验证。",
        f"- API 结果文件：`{run_dir / 'qwen_results.jsonl'}`",
        f"- 上下文 SHA-256：`{sha256(OUTPUT_DIR / 'model_contexts.jsonl')}`",
        f"- 版本化审计参考 SHA-256：`{sha256(OUTPUT_DIR / 'internal_reference_set.jsonl')}`",
        "",
        "## 先区分四类容易混淆的量",
        "",
        "1. **证据声明总数**：输出中 syndrome、formula、herb 和 cross-layer claim 的总条数。它不是药材数。",
        "2. **有机制证据的药材数**：某个药材集合中被机制记录支持的唯一药材数，即 coverage 的分子。",
        "3. **核心药材覆盖率**：有机制证据的核心药材数 / 该病例核心药材总数。",
        "4. **方剂组成覆盖率**：有机制证据的标准方剂组成药材数 / 标准方剂组成药材总数。",
        "",
        "## 公式",
        "",
        "- 检索精确率：`P_ret = |E_expected ∩ E_retrieved| / |E_retrieved|`。",
        "- 检索召回率：`R_ret = |E_expected ∩ E_retrieved| / |E_expected|`。",
        "- 输出证据召回率：`R_claim = N_correct_claims / N_expected_claims`。",
        "- 来源链接精确率：`P_link = N_correct_claims / N_reported_claims`。",
        "- 药材覆盖率：`C = |H_supported| / |H_supported ∪ H_missing|`。",
        "- 缺失证据 F1：`P_m = TP_m/N_reported_missing`，`R_m = TP_m/N_expected_missing`，`F1_m = 2P_mR_m/(P_m+R_m)`。",
        "- 审计等级：矛盾优先为 Level 4；否则 strict support 为 Level 1；否则 syndrome、formula、mechanism 均可用为 Level 2；其余为 Level 3。",
        "- `qwen_only` 没有外部证据输入，因此检索精确率、检索召回率、输出证据召回率和来源链接精确率均记为 N/A，而不是 0。",
        "",
        "## 流程输入（系统可见，非患者原始病历）",
        "",
        json_block(reference.get("physician_plan")),
        "",
        "## 版本化审计参考（评分边界，不是临床金标准）",
        "",
        json_block(
            {
                "case_id": case_id,
                "source_group": reference.get("source_group"),
                "source_cluster_id": reference.get("source_cluster_id"),
                "excluded_source_ids": reference.get("excluded_source_ids"),
                "expected_retrieval_item_ids": reference.get("expected_retrieval_item_ids"),
                "expected_claims": reference.get("expected_claims"),
                "core_herbs": reference.get("core_herbs"),
                "formula_composition_herbs": reference.get("formula_composition_herbs"),
            }
        ),
        "",
        "## 四配置总览",
        "",
        "| 配置 | 检索记录 | API尝试 | 结构字段一致 | 无模型数字 | 语义一致 | 来源合法 | 系统等级 | 实验评价器等级 |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for configuration in CONFIGS:
        result = results.get(configuration, {})
        validation = result.get("validation", {})
        lines.append(
            "| {name} | {retrieved} | {attempt} | {structured} | {numbers} | {semantic} | {provenance} | {level} | {evaluation_level} |".format(
                name=configuration,
                retrieved=len(contexts.get(configuration, {}).get("evidence_context", [])),
                attempt=result.get("remote_metadata", {}).get("attempt"),
                structured=validation.get("structured_audit_consistent"),
                numbers=validation.get("narrative_numbers_absent"),
                semantic=validation.get("semantic_consistent"),
                provenance=("N/A" if configuration == "qwen_only" else validation.get("provenance_clean")),
                level=(
                    result.get("response", {}).get("assessment_level")
                    if configuration == "op_rag"
                    else "N/A"
                ),
                evaluation_level=evaluations.get(configuration, {}).get(
                    "assessment_level"
                ),
            )
        )

    for configuration in CONFIGS:
        if configuration not in calculations:
            continue
        context = contexts[configuration]
        result = results[configuration]
        response = result["response"]
        calc = calculations[configuration]
        ret = calc["retrieval"]
        claim = calc["claims"]
        missing = calc["missing_evidence"]
        lines.extend(
            [
                "",
                f"## 配置：{configuration}",
                "",
                "### 经历的步骤",
                "",
                "1. 读取同一个 physician-recorded plan。",
                f"2. 按该配置检索出 {len(context.get('evidence_context', []))} 条记录。",
                (
                    "3. Python 根据检索记录生成 evidence claims、三种 coverage 和 missing items；仅 OP-RAG 继续计算跨层关系和系统 Level。"
                ),
                "4. 仅把无数字的 narrative_facts 发送给千问。",
                "5. 千问只返回 assessment_summary；Python 把它与原结构化结果组装。",
                "6. 校验器再次核对字段锁定、数字、语义、来源和规则，随后独立代入公式。",
                "",
                "### 检索记录清单",
                "",
                json_block(compact_evidence(context.get("evidence_context", []))),
                "",
                "### 千问实际输入",
                "",
                json_block(json.loads(build_user_prompt(context))),
                "",
                "### 千问实际输出（模型唯一允许字段）",
                "",
                json_block({"assessment_summary": response.get("assessment_summary")}),
                "",
                "### API与校验元数据",
                "",
                json_block(
                    {
                        "remote_metadata": result.get("remote_metadata"),
                        "validation": result.get("validation"),
                        "context_sha256": result.get("context_sha256"),
                    }
                ),
                "",
                "### Python组装的完整输出",
                "",
                json_block(response),
                "",
                "### 指标逐项代入",
                "",
                f"- 检索精确率：`{len(ret['correct_ids'])}/{len(ret['retrieved_ids'])}` = `{display(ret['precision_recomputed'])}`。",
                f"- 检索召回率：`{len(ret['correct_ids'])}/{len(ret['expected_ids'])}` = `{display(ret['recall_recomputed'])}`。",
                f"- 输出证据召回率：`{claim['correct_count']}/{claim['expected_count']}` = `{display(None if configuration == 'qwen_only' else claim['evidence_recall'])}`。",
                f"- 来源链接精确率：`{claim['correct_count']}/{claim['reported_count']}` = `{display(None if configuration == 'qwen_only' else claim['link_precision'])}`。",
                f"- 缺失证据 precision：`{missing['true_positive']}/{missing['reported_count']}` = `{display(missing['precision'])}`。",
                f"- 缺失证据 recall：`{missing['true_positive']}/{missing['expected_count']}` = `{display(missing['recall'])}`。",
                f"- 缺失证据 F1：`{display(missing['f1'])}`。这是 Python 输出自洽性 QA，不是千问能力指标。",
            ]
        )
        for metric_name, cov in calc["coverage"].items():
            lines.append(
                f"- `{metric_name}`：`{cov['supported_count_recomputed']}/{cov['denominator_recomputed']}` = `{display(cov['value_recomputed'])}`。"
            )
        assessment = calc["assessment"]
        evaluation_assessment = calc["evaluation_only_assessment"]
        lines.extend(
            [
                (
                    f"- 系统一致性等级：适用 = `{assessment['applicable']}`；规则输入 "
                    f"`{json.dumps(assessment['rule_inputs'], ensure_ascii=False, sort_keys=True)}`；"
                    f"重算 Level `{assessment['level_recomputed']}`；存储值 Level "
                    f"`{assessment['level_stored']}`；一致 = `{assessment['agreement']}`。"
                ),
                (
                    f"- 实验外部评价器：未发送给千问 = `{not evaluation_assessment['sent_to_qwen']}`；"
                    f"统一规则输入 `{json.dumps(evaluation_assessment['rule_inputs'], ensure_ascii=False, sort_keys=True)}`；"
                    f"重算 Level `{evaluation_assessment['level_recomputed']}`；存储值 Level "
                    f"`{evaluation_assessment['level_stored']}`。"
                ),
            ]
        )

    lines.extend(["", "## 本次漏洞检查", ""])
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors)
    else:
        lines.extend(
            [
                (
                    "- 四个配置均为本次新 API 请求，未复用旧响应。"
                    if audit_phase == "pre_api_spot_check"
                    else "- 四个配置均来自正式 200 次 API 运行；正式运行清单证明未复用旧响应。"
                ),
                "- 千问只输出定性 summary，未生成或覆盖任何数量、覆盖率、缺失项或等级。",
                "- 所有结构化字段与 Python source of truth 完全一致。",
                "- 检索、证据声明、三种覆盖率、缺失证据 F1、OP-RAG 系统等级和四配置实验评价器等级均已独立代入复算。",
                "- 未发现数值、语义、来源、规则或字段所有权漏洞。",
            ]
        )
    lines.extend(["", f"**最终判定：{'PASS' if passed else 'FAIL'}**", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "check_number": check_number,
        "case_id": case_id,
        "status": "pass" if passed else "fail",
        "report_path": str(report_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "audit_phase": audit_phase,
        "errors": errors,
        "result_sha256": sha256(run_dir / "qwen_results.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one fresh four-configuration Qwen API spot check and write a full Markdown audit."
    )
    parser.add_argument("--check-number", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--spot-root",
        type=Path,
        default=DEFAULT_SPOT_ROOT,
        help="directory for this gate; use a new directory when prior outputs are stale",
    )
    args = parser.parse_args()
    if args.check_number < 1:
        raise SystemExit("--check-number must be positive")

    order = selected_case_order()
    if args.check_number > len(order):
        raise SystemExit("The fixed no-replacement selection order has been exhausted")
    case_id = order[args.check_number - 1]
    spot_root = args.spot_root.resolve()
    index_path = spot_root / "SPOT_CHECK_INDEX.json"
    spot_root.mkdir(parents=True, exist_ok=True)
    index = load_index(index_path)
    validate_existing_index(index)
    entries = index.get("entries", [])
    if args.check_number != len(entries) + 1:
        raise SystemExit(
            f"Expected check number {len(entries) + 1}; refusing overwrite or out-of-order execution"
        )
    if case_id in {entry.get("case_id") for entry in entries}:
        raise SystemExit(f"Case {case_id} was already used by an earlier spot check")

    run_dir = spot_root / "runs" / f"spot_check_{args.check_number:03d}_{case_id}"
    report_path = spot_root / f"spot_check_{args.check_number:03d}_{case_id}.md"
    if run_dir.exists() or report_path.exists():
        raise SystemExit("Spot-check output already exists; refusing to reuse or overwrite it")

    run(
        str(EXPERIMENT_DIR / "run_qwen_comparison.py"),
        "--output-dir",
        str(run_dir),
        "--only-case",
        case_id,
        "--workers",
        str(args.workers),
        "--max-attempts",
        str(args.max_attempts),
    )
    entry = build_report(
        check_number=args.check_number,
        case_id=case_id,
        run_dir=run_dir,
        report_path=report_path,
    )
    previous = int(index.get("consecutive_passes", 0))
    consecutive = previous + 1 if entry["status"] == "pass" else 0
    entry["consecutive_passes_after_check"] = consecutive
    entries.append(entry)
    index["entries"] = entries
    index["consecutive_passes"] = consecutive
    index["formal_200_api_run_authorized"] = consecutive >= 5
    index["next_case_id"] = order[len(entries)] if len(entries) < len(order) else None
    write_json(index_path, index)
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    if entry["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
