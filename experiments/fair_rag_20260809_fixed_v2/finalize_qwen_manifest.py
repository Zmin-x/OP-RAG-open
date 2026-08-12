from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a completed Qwen narration-run manifest without changing results."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OUTPUT_DIR / "qwen_comparison_v4_final_20260810",
    )
    args = parser.parse_args()
    results_path = args.run_dir / "qwen_results.jsonl"
    manifest_path = args.run_dir / "run_manifest.json"
    rows = read_jsonl(results_path)
    keys = {(row.get("case_id"), row.get("configuration")) for row in rows}
    counts = Counter(str(row.get("configuration")) for row in rows)
    if (
        len(rows) != 200
        or len(keys) != 200
        or any(counts[name] != 50 for name in CONFIGS)
    ):
        raise SystemExit(
            f"Refusing to finalize an incomplete run: rows={len(rows)}, "
            f"unique={len(keys)}, configurations={dict(counts)}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("n_valid") != 200:
        raise SystemExit("Refusing to finalize a run that is incomplete or contains invalid outputs")
    manifest["n_unique_case_configuration_keys"] = len(keys)
    manifest["configuration_counts"] = dict(counts)
    manifest["results_sha256"] = sha256(results_path)
    manifest["manifest_finalized_at"] = datetime.now(timezone.utc).isoformat()
    manifest["no_old_result_reuse"] = manifest.get("n_reused_identical_context", 0) == 0
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_unique_case_configuration_keys": len(keys),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
