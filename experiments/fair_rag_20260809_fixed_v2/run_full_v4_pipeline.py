from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from protocol import EXPERIMENT_DIR, OUTPUT_DIR, ROOT, sha256, write_json
from run_api_spot_check import INDEX_PATH as SPOT_CHECK_INDEX_PATH
from run_api_spot_check import validate_existing_index


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete v4 experiment only after code tests and the pre-API audit pass. "
            "This script writes experiment artifacts and a separate publication-value package; "
            "it does not edit a manuscript."
        )
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OUTPUT_DIR / "qwen_comparison_v4_final_20260810",
    )
    parser.add_argument(
        "--publication-package-dir",
        type=Path,
        default=OUTPUT_DIR / "publication_sync_package_v4_20260810",
    )
    parser.add_argument(
        "--generated-publication-dir",
        type=Path,
        default=OUTPUT_DIR / "publication_v4_final",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=OUTPUT_DIR / "sensitivity_error_analysis_fixed_v2_20260809",
    )
    args = parser.parse_args()

    if not SPOT_CHECK_INDEX_PATH.exists():
        raise SystemExit("The mandatory pre-API spot-check index is missing")
    spot_check_index = json.loads(SPOT_CHECK_INDEX_PATH.read_text(encoding="utf-8"))
    validate_existing_index(spot_check_index)
    if not spot_check_index.get("formal_200_api_run_authorized"):
        raise SystemExit(
            "The final 200-context API run is prohibited until five different cases pass consecutively"
        )

    results_path = args.run_dir / "qwen_results.jsonl"
    if results_path.exists() and not args.resume:
        raise SystemExit(
            f"Run directory already contains results: {args.run_dir}. "
            "Use --resume only when intentional, or select a fresh --run-dir."
        )

    cases_path = EXPERIMENT_DIR / "inputs" / "eval_cases_visible_plan_001_050.jsonl"
    pre_run_dir = OUTPUT_DIR / "pre_run_v4_final"
    run(
        "-m",
        "unittest",
        str(EXPERIMENT_DIR / "test_scoring.py"),
        str(EXPERIMENT_DIR / "test_regressions.py"),
        str(EXPERIMENT_DIR / "test_structured_reporting.py"),
        str(EXPERIMENT_DIR / "test_reporting_integration.py"),
    )
    run(str(EXPERIMENT_DIR / "prepare_experiment.py"), "--cases", str(cases_path))
    run(str(EXPERIMENT_DIR / "audit_pre_run.py"), "--report-dir", str(pre_run_dir))

    qwen_args = [
        str(EXPERIMENT_DIR / "run_qwen_comparison.py"),
        "--output-dir",
        str(args.run_dir),
        "--workers",
        str(args.workers),
        "--max-attempts",
        str(args.max_attempts),
    ]
    if args.resume:
        qwen_args.append("--resume")
    run(*qwen_args)
    run(str(EXPERIMENT_DIR / "finalize_qwen_manifest.py"), "--run-dir", str(args.run_dir))
    run(str(EXPERIMENT_DIR / "score_qwen_comparison.py"), "--run-dir", str(args.run_dir))
    run(str(EXPERIMENT_DIR / "audit_generated_responses.py"), "--run-dir", str(args.run_dir))
    run(str(EXPERIMENT_DIR / "audit_experiment_integrity.py"), "--run-dir", str(args.run_dir))
    run(
        str(EXPERIMENT_DIR / "build_publication_outputs.py"),
        "--run-dir",
        str(args.run_dir),
        "--package-dir",
        str(args.publication_package_dir),
        "--analysis-dir",
        str(args.analysis_dir),
        "--generated-dir",
        str(args.generated_publication_dir),
    )
    run(
        str(EXPERIMENT_DIR / "audit_publication_inputs.py"),
        "--run-dir",
        str(args.run_dir),
        "--publication-package-dir",
        str(args.publication_package_dir),
        "--generated-dir",
        str(args.generated_publication_dir),
    )

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "run_dir": str(args.run_dir.resolve()),
        "publication_value_package": str(args.publication_package_dir.resolve()),
        "generated_publication_dir": str(args.generated_publication_dir.resolve()),
        "manuscript_modified": False,
        "pre_api_spot_check_gate": {
            "index_path": str(SPOT_CHECK_INDEX_PATH.resolve()),
            "index_sha256": sha256(SPOT_CHECK_INDEX_PATH),
            "consecutive_passes": spot_check_index.get("consecutive_passes"),
            "formal_200_api_run_authorized": spot_check_index.get(
                "formal_200_api_run_authorized"
            ),
        },
        "required_gates": [
            "unit_and_regression_tests",
            "pre_api_context_audit",
            "completed_200_output_manifest",
            "deterministic_rescoring",
            "response_consistency_audit",
            "local_experiment_integrity_audit",
            "publication_value_registry",
            "publication_input_consistency_audit",
        ],
    }
    write_json(args.run_dir / "V4_PIPELINE_COMPLETION.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
