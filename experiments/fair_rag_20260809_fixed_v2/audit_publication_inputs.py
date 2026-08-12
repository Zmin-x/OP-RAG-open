from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_publication_outputs import (
    build_comparison_table,
    build_internal_table,
    build_kb_table,
    read_json,
)
from protocol import OUTPUT_DIR, sha256, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that generated publication inputs match one completed experiment run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--publication-package-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, default=OUTPUT_DIR / "publication_v4_final")
    args = parser.parse_args()

    generated = args.generated_dir
    scoring = read_json(args.run_dir / "qwen_scoring_summary.json")
    kb = read_json(OUTPUT_DIR / "kb_integrity_audit.json")
    internal = read_json(OUTPUT_DIR / "internal_case_analysis_summary.json")
    registry_path = generated / "publication_value_registry.json"
    registry = read_json(registry_path)

    expected_tables = {
        "table_kb_quality.csv": normalized(build_kb_table(kb)),
        "table_internal_application.csv": normalized(build_internal_table(internal)),
        "table_rag_comparison.csv": normalized(build_comparison_table(scoring)),
    }
    table_checks = {}
    for name, expected in expected_tables.items():
        actual = read_csv(generated / name)
        table_checks[name] = {
            "match": actual == expected,
            "rows": len(actual),
            "sha256": sha256(generated / name),
        }

    figure_checks = {}
    for stem in (
        "Figure4_fair_rag_comparison",
        "Figure5_internal_application",
    ):
        for suffix in (".svg", ".pdf", ".png", ".tiff"):
            path = generated / f"{stem}{suffix}"
            figure_checks[path.name] = {
                "exists": path.exists(),
                "nonempty": path.exists() and path.stat().st_size > 0,
                "sha256": sha256(path) if path.exists() else None,
            }

    registry_checks = {
        "run_results_hash_matches": registry.get("run_results_sha256")
        == sha256(args.run_dir / "qwen_results.jsonl"),
        "scoring_hash_matches": registry.get("scoring_summary_sha256")
        == sha256(args.run_dir / "qwen_scoring_summary.json"),
        "retrieval_values_match": registry.get("retrieval_benchmark")
        == scoring.get("retrieval_benchmark"),
        "internal_values_match": registry.get("deterministic_internal_case_analysis") == internal,
    }
    passed = (
        all(check["match"] for check in table_checks.values())
        and all(check["exists"] and check["nonempty"] for check in figure_checks.values())
        and all(registry_checks.values())
    )
    report = {
        "status": "pass" if passed else "fail",
        "scope": (
            "generated table/figure input synchronization only; the manuscript is not read or modified"
        ),
        "run_dir": str(args.run_dir.resolve()),
        "registry_sha256": sha256(registry_path),
        "table_checks": table_checks,
        "figure_checks": figure_checks,
        "registry_checks": registry_checks,
        "manuscript_modified": False,
    }
    output_path = args.run_dir / "PUBLICATION_INPUT_CONSISTENCY_AUDIT.json"
    write_json(output_path, report)
    supplementary = args.publication_package_dir / "supplementary_data"
    supplementary.mkdir(parents=True, exist_ok=True)
    write_json(supplementary / output_path.name, report)
    lines = [
        "# Publication-input consistency audit",
        "",
        f"- Status: **{report['status'].upper()}**",
        "- Scope: generated tables, figure files, result hashes, and the publication value registry.",
        "- The manuscript was not read or modified.",
    ]
    markdown = "\n".join(lines) + "\n"
    (args.run_dir / "PUBLICATION_INPUT_CONSISTENCY_AUDIT.md").write_text(
        markdown, encoding="utf-8"
    )
    (supplementary / "PUBLICATION_INPUT_CONSISTENCY_AUDIT.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
