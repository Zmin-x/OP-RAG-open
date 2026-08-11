from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_corrected_cases import visible_names
from protocol import (
    RecordRetriever,
    layered_retrieval,
    read_jsonl,
)
from run_retrieval_strategy_ablation import layered_tfidf_retrieval
from src.ablation_runner import AblationCaseInput, AblationRunner


class RegressionTests(unittest.TestCase):
    @staticmethod
    def resolve_formula(raw_name: str) -> dict[str, object]:
        formulas = [
            {"formula_id": "F004", "name": "左归丸"},
            {"formula_id": "F014", "name": "左归丸合参苓白术散加减"},
            {"formula_id": "F016", "name": "参苓白术散合补肾壮骨方"},
        ]
        runner = object.__new__(AblationRunner)
        runner.kb = {"formulas": formulas}
        runner.formula_by_id = {row["formula_id"]: row for row in formulas}
        case = AblationCaseInput(
            case_id="case_test",
            patient_text="visible plan",
            reference_main_formula_name=raw_name,
        )
        return runner._resolve_doctor_formula(case)

    def test_exact_long_formula_precedes_short_substring(self) -> None:
        resolved = self.resolve_formula("左归丸合参苓白术散加减")
        self.assertEqual(resolved["id"], "F014")

    def test_nonidentical_shenling_combination_is_not_forced(self) -> None:
        resolved = self.resolve_formula("参苓白术散加减")
        self.assertIsNone(resolved["id"])
        self.assertEqual(resolved["kb_status"], "unmapped")

    def test_partial_formula_still_resolves_when_unambiguous(self) -> None:
        resolved = self.resolve_formula("左归丸加减")
        self.assertEqual(resolved["id"], "F004")

    def test_zero_similarity_retrieval_returns_no_record(self) -> None:
        retriever = RecordRetriever(
            [
                {
                    "item_id": "herb:黄芪",
                    "layer": "herb",
                    "name": "黄芪",
                    "search_text": "herb:黄芪 黄芪 补气升阳",
                }
            ]
        )
        self.assertEqual(retriever.search("地龙", top_k=1), [])

    def test_layered_tfidf_and_exact_herb_modes_are_distinct(self) -> None:
        records = [
            {
                "item_id": "syndrome:S1",
                "layer": "syndrome",
                "name": "kidney yang deficiency",
                "search_text": "syndrome:S1 kidney yang deficiency",
            },
            {
                "item_id": "formula:F001",
                "layer": "formula",
                "name": "Bu Sui Dan",
                "search_text": "formula:F001 Bu Sui Dan",
            },
            {
                "item_id": "herb:test",
                "layer": "herb",
                "name": "testherb",
                "search_text": "herb:test testherb",
            },
        ]
        plan = {
            "primary_syndrome_name": "",
            "secondary_syndrome_names": [],
            "formula_name": "",
            "herbs": ["testherbextra"],
        }
        tfidf_ids = {
            row["item_id"] for row in layered_tfidf_retrieval(plan, records)
        }
        exact_ids = {row["item_id"] for row in layered_retrieval(plan, records)}
        self.assertEqual(tfidf_ids, {"herb:test"})
        self.assertEqual(exact_ids, set())

    def test_visible_syndrome_order_does_not_use_hidden_id(self) -> None:
        case = {
            "patient_text": "中医诊断：骨痿，证属肾阴阳两虚，脾阳不足，瘀血阻络。",
            "primary_syndrome_id": "S4",
            "primary_syndrome_name_raw": None,
        }
        primary, secondary, _ = visible_names(case)
        self.assertEqual(primary, "肾阴阳两虚")
        self.assertEqual(secondary, ["瘀血阻络"])

    def test_jsonl_reader_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text('{"case_id":"case_test"}\n', encoding="utf-8-sig")
            self.assertEqual(read_jsonl(path), [{"case_id": "case_test"}])


if __name__ == "__main__":
    unittest.main()
