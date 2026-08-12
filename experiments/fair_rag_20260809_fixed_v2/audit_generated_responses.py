from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, write_json
from qwen_protocol import validate_response


def record_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("configuration"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revalidate every generated report against its deterministic structured audit."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OUTPUT_DIR / "qwen_comparison_v4_final_20260810",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    contexts = {
        record_key(row): row["context"]
        for row in read_jsonl(OUTPUT_DIR / "model_contexts.jsonl")
    }
    manifest_path = args.run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    audit_timestamp = (
        manifest.get("manifest_finalized_at")
        or manifest.get("finished_at")
        or "not_recorded"
    )
    results = read_jsonl(args.run_dir / "qwen_results.jsonl")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicate_keys: list[tuple[str, str]] = []
    for result in results:
        key = record_key(result)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        context = contexts.get(key)
        response = result.get("response", {})
        if context is None:
            validation = {
                "valid": False,
                "structured_audit_consistent": False,
                "narrative_numbers_absent": False,
                "provenance_clean": False,
                "errors": ["matching context is missing"],
                "provenance_violations": [],
            }
        else:
            raw = {"assessment_summary": response.get("assessment_summary")}
            validation = validate_response(response, context, raw_model_response=raw)
        coverage = response.get("coverage_metrics") or {}
        rows.append(
            {
                "case_id": key[0],
                "configuration": key[1],
                "valid": validation.get("valid") is True,
                "structured_audit_consistent": validation.get("structured_audit_consistent") is True,
                "narrative_numbers_absent": validation.get("narrative_numbers_absent") is True,
                "semantic_consistent": validation.get("semantic_consistent") is True,
                "provenance_clean": validation.get("provenance_clean") is True,
                "assessment_level": response.get("assessment_level"),
                "plan_herb_numerator": (coverage.get("physician_plan_herbs") or {}).get("numerator"),
                "plan_herb_denominator": (coverage.get("physician_plan_herbs") or {}).get("denominator"),
                "core_herb_numerator": (coverage.get("core_herbs") or {}).get("numerator"),
                "core_herb_denominator": (coverage.get("core_herbs") or {}).get("denominator"),
                "formula_herb_numerator": (coverage.get("formula_composition_herbs") or {}).get("numerator"),
                "formula_herb_denominator": (coverage.get("formula_composition_herbs") or {}).get("denominator"),
                "errors": " | ".join(validation.get("errors") or []),
                "provenance_violations": " | ".join(
                    validation.get("provenance_violations") or []
                ),
                "assessment_summary": response.get("assessment_summary"),
            }
        )

    expected_keys = set(contexts)
    missing_keys = sorted(expected_keys - seen)
    unexpected_keys = sorted(seen - expected_keys)
    invalid_rows = [row for row in rows if not row["valid"]]
    complete = (
        len(rows) == 200
        and len(seen) == 200
        and not duplicate_keys
        and not missing_keys
        and not unexpected_keys
    )
    passed = not invalid_rows and (complete or args.allow_partial)
    summary = {
        "created_at": audit_timestamp,
        "scope": "report-to-structured-audit consistency; not independent medical fact verification",
        "status": "pass" if passed else "fail",
        "complete_200_output_run": complete,
        "n_rows": len(rows),
        "n_unique_keys": len(seen),
        "n_valid": sum(row["valid"] for row in rows),
        "n_invalid": len(invalid_rows),
        "n_semantically_consistent": sum(row["semantic_consistent"] for row in rows),
        "configuration_counts": dict(Counter(row["configuration"] for row in rows)),
        "duplicate_keys": duplicate_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "invalid_keys": [
            [row["case_id"], row["configuration"]] for row in invalid_rows
        ],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "response_consistency_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.run_dir / "response_consistency_audit_summary.json", summary)
    lines = [
        "# Generated-response consistency audit",
        "",
        f"- Status: **{summary['status'].upper()}**",
        f"- Outputs: {summary['n_rows']} rows / {summary['n_unique_keys']} unique case-configuration keys",
        f"- Valid: {summary['n_valid']}; invalid: {summary['n_invalid']}",
        f"- Complete 50-case x 4-configuration run: {summary['complete_200_output_run']}",
        "- Scope: deterministic field and narration consistency only; this is not an independent clinical-validity check.",
    ]
    (args.run_dir / "RESPONSE_CONSISTENCY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
