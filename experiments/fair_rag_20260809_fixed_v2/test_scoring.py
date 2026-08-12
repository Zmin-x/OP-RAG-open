from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from qwen_protocol import BOUNDARY_STATEMENT, assemble_response, validate_response
from protocol import build_internal_reference, build_structured_audit, make_compact_records, retrieval_metrics
from score_qwen_comparison import (
    bootstrap_paired_difference,
    build_item_aliases,
    expected_missing_items,
    score_record,
)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {
            "case_id": "case_test",
            "source_cluster_id": "cluster_test",
            "primary_syndrome_id": "S6",
            "secondary_syndrome_ids": [],
            "case_syndrome_ids": ["S6"],
            "resolved_formula_id": "F024",
            "physician_plan": {
                "primary_syndrome_name": "syndrome name",
                "secondary_syndrome_names": [],
                "formula_name": "formula name",
                "herbs": ["herb_a", "herb_missing"],
            },
            "expected_claims": [
                {
                    "item_id": "syndrome:S6",
                    "layer": "syndrome",
                    "expected_status": "supported",
                    "source_ids": ["REF:S6"],
                },
                {
                    "item_id": "formula:F024",
                    "layer": "formula",
                    "expected_status": "supported",
                    "source_ids": ["REF:F024"],
                },
                {
                    "item_id": "herb:herb_a",
                    "layer": "herb",
                    "expected_status": "supported",
                    "source_ids": ["PIPELINE:1"],
                },
            ],
            "expected_missing_items": ["herb:herb_missing"],
            "expected_assessment_level": 2,
            "core_herbs": ["herb_a", "herb_missing"],
            "formula_composition_herbs": ["herb_a", "herb_missing"],
            "mechanism_supported_plan_herbs": ["herb_a"],
            "mechanism_supported_formula_herbs": ["herb_a"],
        }
        self.layered_context = {
            "configuration": "layered_rag",
            "evidence_context": [
                {"item_id": "syndrome:S6", "layer": "syndrome", "name": "syndrome name", "source_ids": ["REF:S6"]},
                {
                    "item_id": "formula:F024",
                    "layer": "formula",
                    "name": "formula name",
                    "source_ids": ["REF:F024"],
                    "attributes": {"indication_syndrome_ids": ["S6"]},
                },
                {"item_id": "herb:herb_a", "layer": "herb", "name": "herb_a", "source_ids": ["PIPELINE:1"]},
            ],
        }
        self.context_rows = [
            {"case_id": "case_test", "configuration": "layered_rag", "context": self.layered_context},
            {
                "case_id": "case_test",
                "configuration": "qwen_only",
                "context": {"configuration": "qwen_only", "evidence_context": []},
            },
        ]
        for row in self.context_rows:
            row["context"]["structured_audit"] = build_structured_audit(
                self.reference, row["context"]
            )
        self.aliases = build_item_aliases(self.context_rows, {"case_test": self.reference})

    def test_missing_target_is_computed_from_each_configuration(self) -> None:
        targets = [
            expected_missing_items(self.reference, context, self.aliases)
            for context in (
                self.context_rows[1]["context"],
                self.layered_context,
            )
        ]
        self.assertEqual(
            targets[0],
            {"syndrome:S6", "formula:F024", "herb:herb_a", "herb:herb_missing"},
        )
        self.assertEqual(targets[1], {"herb:herb_missing"})

    def test_invisible_claim_cannot_score_as_structured_link(self) -> None:
        context = {
            "configuration": "flat_rag",
            "evidence_context": self.layered_context["evidence_context"][:2],
        }
        result = {
            "case_id": "case_test",
            "configuration": "flat_rag",
            "response": {
                "evidence_claims": [
                    {
                        "item_id": "herb:herb_a",
                        "support_status": "supported",
                        "source_ids": ["PIPELINE:1"],
                        "statement": "not visible",
                    }
                ],
                "missing_evidence_items": [],
                "unverified_parametric_claims": [],
                "assessment_level": 2,
            },
            "validation": {"schema_valid": True, "provenance_clean": False, "provenance_violations": ["invisible"]},
        }
        scored = score_record(result, self.reference, context, self.aliases)
        self.assertEqual(scored["correct_evidence_claim_count"], 0)
        self.assertEqual(scored["structured_evidence_link_precision"], 0.0)

    def test_precision_is_undefined_without_reported_claims(self) -> None:
        result = {
            "case_id": "case_test",
            "configuration": "qwen_only",
            "response": {
                "evidence_claims": [],
                "missing_evidence_items": [],
                "unverified_parametric_claims": [],
                "assessment_level": 3,
            },
            "validation": {"schema_valid": True, "provenance_clean": True, "provenance_violations": []},
        }
        scored = score_record(
            result,
            self.reference,
            {"configuration": "qwen_only", "evidence_context": []},
            self.aliases,
        )
        self.assertIsNone(scored["structured_evidence_link_precision"])
        self.assertIsNone(scored["provenance_clean"])

    def test_cluster_bootstrap_excludes_undefined_pairs(self) -> None:
        left = {
            "case_1": {"metric": None, "source_cluster_id": "cluster_1"},
            "case_2": {"metric": 0.5, "source_cluster_id": "cluster_2"},
        }
        right = {
            "case_1": {"metric": 1.0, "source_cluster_id": "cluster_1"},
            "case_2": {"metric": 0.75, "source_cluster_id": "cluster_2"},
        }
        result = bootstrap_paired_difference(left, right, "metric")
        self.assertEqual(result["n_pairs"], 1)
        self.assertEqual(result["n_source_clusters"], 1)
        self.assertEqual(result["mean_difference"], 0.25)

    def test_validator_rejects_model_generated_numbers(self) -> None:
        context = self.layered_context
        response = assemble_response(
            {"assessment_summary": "partial evidence support: 1 herb is missing."},
            context,
        )
        validation = validate_response(
            response,
            context,
            raw_model_response={"assessment_summary": response["assessment_summary"]},
        )
        self.assertFalse(validation["schema_valid"])
        self.assertTrue(any("model-generated numbers" in value for value in validation["errors"]))

    def test_validator_accepts_narration_only_response(self) -> None:
        context = self.layered_context
        facts = context["structured_audit"]["narrative_facts"]
        label = facts["assessment_label"]
        clauses = facts["required_clauses"]
        raw = {
            "assessment_summary": f"{label}: " + "; ".join(clauses) + "."
        }
        response = assemble_response(raw, context)
        validation = validate_response(response, context, raw_model_response=raw)
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["structured_audit_consistent"])

    def test_source_free_formula_record_is_removed(self) -> None:
        kb = {
            "syndromes": [],
            "formulas": [
                {
                    "formula_id": "F014",
                    "name": "formula name",
                    "references": [],
                    "literature_source_ids": ["LITDOC:self"],
                    "indication_syndrome": ["S2"],
                    "composition": [{"herb": "herb_a"}],
                    "text_description": "case-derived formula content",
                }
            ],
            "herbs": [],
        }
        records = make_compact_records(kb, excluded_source_ids={"LITDOC:self"})
        self.assertFalse(any(row.get("item_id") == "formula:F014" for row in records))

    def test_source_free_formula_is_not_expected_retrieval_target(self) -> None:
        class StubRunner:
            def run_case(self, _case, _mode):
                return SimpleNamespace(
                    context={
                        "case_results": {
                            "resolved_formula_id": "F014",
                            "formula_kb_status": "mapped",
                            "primary_syndrome_id": "S2",
                            "formula_consistent_with_primary": True,
                            "formula_consistent_with_any_syndrome": True,
                            "core_herb_mechanism_coverage": None,
                            "mechanism_coverage_over_reference_herbs": None,
                            "formula_composition_mechanism_coverage": None,
                        }
                    }
                )

        kb = {
            "syndromes": [{"syndrome_id": "S2", "name": "syndrome", "references": ["REF:S2"]}],
            "formulas": [
                {
                    "formula_id": "F014",
                    "name": "formula name",
                    "references": [],
                    "literature_source_ids": ["LITDOC:self"],
                    "indication_syndrome": ["S2"],
                }
            ],
            "herbs": [],
        }
        case = {
            "case_id": "case_test",
            "patient_text": "",
            "source_group": "literature",
            "primary_syndrome_id": "S2",
            "primary_syndrome_name_raw": "syndrome",
            "secondary_syndrome_ids": [],
            "secondary_syndrome_names_raw": [],
            "reference_main_formula_name": "formula name",
            "reference_herb_set": [],
        }
        reference = build_internal_reference(
            case,
            kb,
            excluded_source_ids={"LITDOC:self"},
        )
        self.assertNotIn("formula:F014", reference["expected_retrieval_item_ids"])
        self.assertIn("formula:F014", reference["expected_missing_items"])

    def test_qwen_only_retrieval_metrics_are_not_applicable(self) -> None:
        metrics = retrieval_metrics(
            {"expected_retrieval_item_ids": ["syndrome:S6"]},
            {"configuration": "qwen_only", "evidence_context": [], "retrieval_budget": {}},
        )
        self.assertIsNone(metrics["evidence_retrieval_precision"])
        self.assertIsNone(metrics["evidence_retrieval_recall"])


if __name__ == "__main__":
    unittest.main()
