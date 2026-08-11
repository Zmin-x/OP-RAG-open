from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from statistics import mean


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "deterministic_retrieval_ablation_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RetrievalAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required_files = (
            OUTPUT_DIR / "case_level.csv",
            OUTPUT_DIR / "summary.json",
            OUTPUT_DIR / "manifest.json",
        )
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                "Paper-internal retrieval outputs are not distributed (missing: "
                + ", ".join(missing)
                + "); run this suite with the authorized audit package."
            )
        with (OUTPUT_DIR / "case_level.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.summary = json.loads(
            (OUTPUT_DIR / "summary.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8")
        )

    def test_release_contains_50_cases_for_each_strategy(self) -> None:
        counts = Counter(row["strategy"] for row in self.rows)
        self.assertEqual(len(self.rows), 150)
        self.assertEqual(len({(row["case_id"], row["strategy"]) for row in self.rows}), 150)
        self.assertEqual(
            counts,
            Counter(
                {
                    "flat_tfidf": 50,
                    "layered_tfidf": 50,
                    "layered_hybrid_exact_herb": 50,
                }
            ),
        )

    def test_flat_and_layered_tfidf_have_equal_case_budgets(self) -> None:
        by_key = {(row["case_id"], row["strategy"]): row for row in self.rows}
        case_ids = sorted({row["case_id"] for row in self.rows})
        self.assertEqual(len(case_ids), 50)
        for case_id in case_ids:
            self.assertEqual(
                int(by_key[(case_id, "flat_tfidf")]["retrieved_item_count"]),
                int(by_key[(case_id, "layered_tfidf")]["retrieved_item_count"]),
            )

    def test_summary_means_match_case_level_rows(self) -> None:
        for strategy, expected in self.summary["strategies"].items():
            rows = [row for row in self.rows if row["strategy"] == strategy]
            for metric in (
                "evidence_retrieval_precision",
                "evidence_retrieval_recall",
                "herb_precision",
                "herb_recall",
                "retrieved_item_count",
            ):
                values = [float(row[metric]) for row in rows if row[metric] != ""]
                summary_key = (
                    "mean_evidence_record_count"
                    if metric == "retrieved_item_count"
                    else metric
                )
                self.assertAlmostEqual(mean(values), float(expected[summary_key]), places=12)

    def test_manifest_proves_local_only_run_and_current_scripts(self) -> None:
        self.assertEqual(self.manifest["status"], "completed")
        self.assertEqual(self.manifest["api_calls"], 0)
        self.assertEqual(self.manifest["integrity_checks"]["api_calls"], 0)
        self.assertTrue(
            all(
                value is True
                for key, value in self.manifest["integrity_checks"].items()
                if key != "api_calls"
            )
        )
        for name in ("protocol.py", "run_retrieval_strategy_ablation.py"):
            self.assertEqual(
                sha256(EXPERIMENT_DIR / name),
                self.manifest["input_hashes"][name],
            )


if __name__ == "__main__":
    unittest.main()
