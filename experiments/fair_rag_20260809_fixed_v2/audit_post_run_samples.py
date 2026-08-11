from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for import_path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from protocol import OUTPUT_DIR, read_jsonl, sha256, write_json  # noqa: E402
from run_api_spot_check import build_report  # noqa: E402


SELECTION_SEED = 2026081010
REQUIRED_CASES = 10
DEFAULT_FORMAL_RUN_DIR = OUTPUT_DIR / "qwen_comparison_v5_post_spotcheck_20260810"
DEFAULT_PRE_RUN_INDEX = OUTPUT_DIR / "api_spot_check_gate_20260810" / "SPOT_CHECK_INDEX.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "api_spot_check_gate_20260810" / "post_formal_run_audit"
INDEX_NAME = "POST_RUN_AUDIT_INDEX.json"
SUMMARY_NAME = "POST_RUN_10_CASE_AUDIT.md"


def selected_cases(
    pre_run_index: Path = DEFAULT_PRE_RUN_INDEX,
) -> tuple[list[str], list[str]]:
    all_case_ids = sorted(
        row["case_id"]
        for row in read_jsonl(
            EXPERIMENT_DIR / "inputs" / "eval_cases_visible_plan_001_050.jsonl"
        )
    )
    pre_index = json.loads(pre_run_index.read_text(encoding="utf-8"))
    pre_used = [str(entry["case_id"]) for entry in pre_index.get("entries", [])]
    candidates = [case_id for case_id in all_case_ids if case_id not in set(pre_used)]
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(candidates)
    selected = candidates[:REQUIRED_CASES]
    if len(all_case_ids) != 50:
        raise SystemExit(f"Expected 50 cases, found {len(all_case_ids)}")
    if len(selected) != REQUIRED_CASES or len(set(selected)) != REQUIRED_CASES:
        raise SystemExit("Post-run selection is incomplete or contains duplicates")
    return selected, pre_used


def build_summary(index: dict[str, Any]) -> str:
    lines = [
        "# OP-RAG 正式 200 次 API 运行后 10 例抽查",
        "",
        f"- 总体结论：**{str(index['overall_status']).upper()}**",
        f"- 固定随机种子：`{index['selection_seed']}`",
        "- 抽样方法：50 个病例按编号排序，排除正式运行前已抽查病例后，使用固定种子无放回打乱并取前 10 例。",
        "- 数据来源：同一批正式 200 次 API 输出；本阶段没有重新调用千问，也没有替换任何响应。",
        "- 检查单位：每个病例均完整复核 Qwen-only、Flat RAG、Layered RAG 和 OP-RAG 四个配置。",
        "- 每份明细均记录输入字段、检索记录、千问实际输入与输出、Python 完整输出、验证元数据、公式和逐项代入计算。",
        "- 解释边界：这是内部工程一致性复核，不是诊断、处方、疗效或临床有效性验证。",
        f"- 正式结果 SHA-256：`{index['formal_results_sha256']}`",
        f"- 正式运行清单 SHA-256：`{index['formal_manifest_sha256']}`",
        "",
        "| 序号 | 病例 | 四配置 | 结论 | 详细记录 |",
        "|---:|---|---:|---|---|",
    ]
    for entry in index["entries"]:
        lines.append(
            f"| {entry['check_number']} | `{entry['case_id']}` | 4 | "
            f"{entry['status'].upper()} | [{entry['report_path']}]({entry['report_path']}) |"
        )
    lines.extend(
        [
            "",
            "## 完成条件核对",
            "",
            f"- 10 个病例互不重复：`{index['ten_unique_cases']}`",
            f"- 40 个病例-配置结果全部存在：`{index['forty_configuration_records_present']}`",
            f"- 10 例逐例详细复核全部通过：`{index['all_ten_passed']}`",
            f"- 正式运行未复用旧响应：`{index['formal_run_reused_zero_responses']}`",
            f"- 正式结果文件哈希与清单一致：`{index['formal_results_hash_matches_manifest']}`",
            "",
        ]
    )
    return "\n".join(lines)


def generate(
    output_dir: Path,
    *,
    formal_run_dir: Path = DEFAULT_FORMAL_RUN_DIR,
    pre_run_index: Path = DEFAULT_PRE_RUN_INDEX,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty; refusing to overwrite audit evidence: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, pre_used = selected_cases(pre_run_index)
    manifest_path = formal_run_dir / "run_manifest.json"
    results_path = formal_run_dir / "qwen_results.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entries: list[dict[str, Any]] = []
    for number, case_id in enumerate(selected, start=1):
        report_name = f"post_run_check_{number:02d}_{case_id}.md"
        report_path = output_dir / report_name
        entry = build_report(
            check_number=number,
            case_id=case_id,
            run_dir=formal_run_dir,
            report_path=report_path,
            audit_phase="post_formal_run_audit",
            selection_seed=SELECTION_SEED,
        )
        entry["report_path"] = report_name
        entry["report_sha256"] = sha256(report_path)
        entries.append(entry)

    selected_result_rows = [
        row
        for row in read_jsonl(results_path)
        if row.get("case_id") in set(selected)
    ]
    ten_unique = len(set(selected)) == REQUIRED_CASES
    forty_present = (
        len(selected_result_rows) == REQUIRED_CASES * 4
        and len(
            {
                (row.get("case_id"), row.get("configuration"))
                for row in selected_result_rows
            }
        )
        == REQUIRED_CASES * 4
    )
    all_passed = len(entries) == REQUIRED_CASES and all(
        entry["status"] == "pass" for entry in entries
    )
    hash_matches = manifest.get("results_sha256") == sha256(results_path)
    reused_zero = (
        manifest.get("n_reused_identical_context") == 0
        and manifest.get("no_old_result_reuse") is True
    )
    overall_pass = all(
        (ten_unique, forty_present, all_passed, hash_matches, reused_zero)
    )
    index: dict[str, Any] = {
        "scope": "post-formal-run case-level consistency audit; not clinical validation",
        "selection_seed": SELECTION_SEED,
        "selection_method": (
            "sort all 50 case IDs; exclude pre-run spot-check cases; "
            "random.Random(seed).shuffle; take first 10 without replacement"
        ),
        "pre_run_spot_check_cases_excluded": pre_used,
        "selected_case_ids": selected,
        "required_case_count": REQUIRED_CASES,
        "required_configuration_count_per_case": 4,
        "formal_run_dir": str(formal_run_dir.resolve()),
        "formal_results_sha256": sha256(results_path),
        "formal_manifest_sha256": sha256(manifest_path),
        "formal_manifest_recorded_results_sha256": manifest.get("results_sha256"),
        "ten_unique_cases": ten_unique,
        "forty_configuration_records_present": forty_present,
        "all_ten_passed": all_passed,
        "formal_run_reused_zero_responses": reused_zero,
        "formal_results_hash_matches_manifest": hash_matches,
        "entries": entries,
        "overall_status": "pass" if overall_pass else "fail",
    }
    write_json(output_dir / INDEX_NAME, index)
    (output_dir / SUMMARY_NAME).write_text(build_summary(index), encoding="utf-8")
    return index


def verify_existing(
    output_dir: Path,
    *,
    formal_run_dir: Path = DEFAULT_FORMAL_RUN_DIR,
    pre_run_index: Path = DEFAULT_PRE_RUN_INDEX,
) -> dict[str, Any]:
    index_path = output_dir / INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_selected, expected_pre_used = selected_cases(pre_run_index)
    errors: list[str] = []
    if index.get("selected_case_ids") != expected_selected:
        errors.append("stored case selection differs from deterministic selection")
    if index.get("pre_run_spot_check_cases_excluded") != expected_pre_used:
        errors.append("stored pre-run exclusion list differs from current index")
    entries = index.get("entries", [])
    if len(entries) != REQUIRED_CASES:
        errors.append("post-run audit does not contain exactly 10 entries")
    seen: set[str] = set()
    for number, entry in enumerate(entries, start=1):
        if entry.get("check_number") != number:
            errors.append(f"entry {number} has the wrong sequential number")
        case_id = str(entry.get("case_id") or "")
        if not case_id or case_id in seen:
            errors.append(f"entry {number} has a missing or duplicate case ID")
        seen.add(case_id)
        report_path = output_dir / str(entry.get("report_path") or "")
        if not report_path.is_file():
            errors.append(f"entry {number} report is missing")
            continue
        if entry.get("report_sha256") != sha256(report_path):
            errors.append(f"entry {number} report hash mismatch")
        if "**最终判定：PASS**" not in report_path.read_text(encoding="utf-8"):
            errors.append(f"entry {number} Markdown does not end in PASS")
    results_path = formal_run_dir / "qwen_results.jsonl"
    manifest = json.loads((formal_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if index.get("formal_results_sha256") != sha256(results_path):
        errors.append("formal result hash changed after the post-run audit")
    if manifest.get("results_sha256") != sha256(results_path):
        errors.append("formal result hash no longer agrees with its run manifest")
    if index.get("overall_status") != "pass":
        errors.append("stored post-run overall status is not pass")
    return {
        "status": "pass" if not errors else "fail",
        "selected_case_ids": expected_selected,
        "n_entries": len(entries),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify a deterministic 10-case audit from the retained "
            "formal 200-response Qwen run without making API calls."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formal-run-dir", type=Path, default=DEFAULT_FORMAL_RUN_DIR)
    parser.add_argument("--pre-run-index", type=Path, default=DEFAULT_PRE_RUN_INDEX)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    result = (
        verify_existing(
            args.output_dir,
            formal_run_dir=args.formal_run_dir,
            pre_run_index=args.pre_run_index,
        )
        if args.verify_existing
        else generate(
            args.output_dir,
            formal_run_dir=args.formal_run_dir,
            pre_run_index=args.pre_run_index,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status", result.get("overall_status")) != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
