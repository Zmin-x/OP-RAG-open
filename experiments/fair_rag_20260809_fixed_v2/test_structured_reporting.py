from __future__ import annotations

import json
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from protocol import (  # noqa: E402
    OUTPUT_DIR,
    RELATION_INCONSISTENT,
    RELATION_INSUFFICIENT,
    RELATION_SUPPORTED,
    build_assessment_inputs,
    build_structured_audit,
    classify_formula_syndrome_relation,
    determine_assessment_level,
    read_jsonl,
)
from qwen_protocol import assemble_response, build_user_prompt, validate_response  # noqa: E402
from evaluation_protocol import build_evaluation_only_assessment  # noqa: E402


class StructuredReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required_files = (
            OUTPUT_DIR / "model_contexts.jsonl",
            OUTPUT_DIR / "evaluation_only_assessments.jsonl",
        )
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                "Paper-internal structured audit outputs are not distributed "
                "(missing: " + ", ".join(missing) + "); run this suite with "
                "the authorized audit package."
            )
        cls.context_rows = read_jsonl(required_files[0])
        cls.contexts = {
            (row["case_id"], row["configuration"]): row["context"]
            for row in cls.context_rows
        }
        cls.evaluation_rows = read_jsonl(
            OUTPUT_DIR / "evaluation_only_assessments.jsonl"
        )
        cls.evaluations = {
            (row["case_id"], row["configuration"]): row
            for row in cls.evaluation_rows
        }

    @staticmethod
    def valid_raw(context: dict[str, object]) -> dict[str, str]:
        facts = context["structured_audit"]["narrative_facts"]
        return {
            "assessment_summary": (
                f"{facts['assessment_label']}: "
                + "; ".join(facts["required_clauses"])
                + "."
            )
        }

    def test_all_200_contexts_have_deterministic_structured_audits(self) -> None:
        self.assertEqual(len(self.context_rows), 200)
        self.assertEqual(len(self.contexts), 200)
        for context in self.contexts.values():
            audit = context["structured_audit"]
            self.assertEqual(audit["generation_method"], "deterministic_python")
            if context["configuration"] == "op_rag":
                self.assertTrue(audit["consistency_audit_applicable"])
                self.assertIn(audit["assessment_level"], {1, 2, 3, 4})
                self.assertEqual(
                    determine_assessment_level(**audit["assessment_rule_trace"]["inputs"]),
                    audit["assessment_level"],
                )
                self.assertEqual(
                    audit["assessment_rule_trace"]["triggered_rule"],
                    f"level_{audit['assessment_level']}",
                )
                self.assertIsInstance(audit["formula_syndrome_relation"], dict)
            else:
                self.assertFalse(audit["consistency_audit_applicable"])
                self.assertIsNone(audit["assessment_level"])
                self.assertEqual(audit["assessment_label"], "not_applicable")
                self.assertIsNone(audit["assessment_rule_trace"])
                self.assertIsNone(audit["formula_syndrome_relation"])
            for metric in audit["coverage_metrics"].values():
                self.assertLessEqual(metric["numerator"], metric["denominator"])
                expected = (
                    metric["numerator"] / metric["denominator"]
                    if metric["denominator"]
                    else None
                )
                self.assertEqual(metric["value"], expected)
                self.assertEqual(metric["numerator"], len(metric["supported_items"]))
                self.assertEqual(
                    metric["denominator"],
                    len(metric["supported_items"]) + len(metric["missing_items"]),
                )
                self.assertFalse(set(metric["supported_items"]) & set(metric["missing_items"]))
            claim_ids = [claim["item_id"] for claim in audit["evidence_claims"]]
            self.assertEqual(len(claim_ids), len(set(claim_ids)))
            self.assertFalse(set(claim_ids) & set(audit["missing_evidence_items"]))
            self.assertTrue(all(claim["source_ids"] for claim in audit["evidence_claims"]))

    def test_case_018_op_rag_counts_and_level_are_fixed_by_code(self) -> None:
        audit = self.contexts[("case_018", "op_rag")]["structured_audit"]
        self.assertEqual(
            (audit["coverage_metrics"]["physician_plan_herbs"]["numerator"],
             audit["coverage_metrics"]["physician_plan_herbs"]["denominator"]),
            (9, 13),
        )
        self.assertEqual(
            (audit["coverage_metrics"]["core_herbs"]["numerator"],
             audit["coverage_metrics"]["core_herbs"]["denominator"]),
            (5, 8),
        )
        self.assertEqual(
            (audit["coverage_metrics"]["formula_composition_herbs"]["numerator"],
             audit["coverage_metrics"]["formula_composition_herbs"]["denominator"]),
            (9, 13),
        )
        self.assertEqual(audit["assessment_level"], 3)
        self.assertEqual(len(audit["evidence_claims"]), 10)
        self.assertEqual(len(audit["missing_evidence_items"]), 6)

    def test_layered_and_op_share_evidence_but_only_op_exposes_consistency(self) -> None:
        case_ids = sorted({case_id for case_id, _ in self.contexts})
        self.assertEqual(len(case_ids), 50)
        for case_id in case_ids:
            with self.subTest(case_id=case_id):
                layered = self.contexts[(case_id, "layered_rag")]
                op_rag = self.contexts[(case_id, "op_rag")]
                self.assertEqual(layered["evidence_context"], op_rag["evidence_context"])
                self.assertEqual(
                    layered["structured_audit"]["coverage_metrics"],
                    op_rag["structured_audit"]["coverage_metrics"],
                )
                self.assertIsNone(layered["structured_audit"]["assessment_level"])
                self.assertIsNone(layered["structured_audit"]["formula_syndrome_relation"])
                self.assertIn(op_rag["structured_audit"]["assessment_level"], {1, 2, 3, 4})
                self.assertIsInstance(
                    op_rag["structured_audit"]["formula_syndrome_relation"], dict
                )

    def test_all_four_configurations_have_withheld_uniform_evaluations(self) -> None:
        self.assertEqual(len(self.evaluation_rows), 200)
        self.assertEqual(len(self.evaluations), 200)
        for key, context in self.contexts.items():
            evaluation = self.evaluations[key]
            self.assertFalse(evaluation["sent_to_qwen"])
            self.assertIn(evaluation["assessment_level"], {1, 2, 3, 4})
            self.assertEqual(
                determine_assessment_level(
                    **evaluation["assessment_rule_trace"]["inputs"]
                ),
                evaluation["assessment_level"],
            )
            reference = next(
                row
                for row in read_jsonl(OUTPUT_DIR / "internal_reference_set.jsonl")
                if row["case_id"] == key[0]
            )
            self.assertEqual(
                evaluation,
                build_evaluation_only_assessment(reference, context),
            )

        for case_id in sorted({case_id for case_id, _ in self.contexts}):
            self.assertEqual(
                self.evaluations[(case_id, "layered_rag")]["assessment_level"],
                self.evaluations[(case_id, "op_rag")]["assessment_level"],
            )

    def test_relation_classifier_distinguishes_all_three_states(self) -> None:
        cases = [
            ({"S1"}, {"S1", "S2"}, True, RELATION_SUPPORTED),
            ({"S1", "S2"}, {"S2"}, True, RELATION_SUPPORTED),
            ({"S1"}, {"S3"}, True, RELATION_INCONSISTENT),
            ({"S1"}, {"S1"}, False, RELATION_INSUFFICIENT),
            (set(), {"S1"}, True, RELATION_INSUFFICIENT),
            ({"S1"}, set(), True, RELATION_INSUFFICIENT),
        ]
        for case_ids, indication_ids, evidence_complete, expected in cases:
            with self.subTest(
                case_ids=case_ids,
                indication_ids=indication_ids,
                evidence_complete=evidence_complete,
            ):
                self.assertEqual(
                    classify_formula_syndrome_relation(
                        case_syndrome_ids=case_ids,
                        formula_indication_ids=indication_ids,
                        evidence_complete=evidence_complete,
                    ),
                    expected,
                )

    def test_secondary_syndrome_support_is_not_a_contradiction(self) -> None:
        inputs = build_assessment_inputs(
            relation_status=RELATION_SUPPORTED,
            primary_relation_status=RELATION_INCONSISTENT,
            syndrome_evidence_available=True,
            formula_evidence_available=True,
            mechanism_evidence_available=True,
            core_herb_coverage=0.90,
            formula_composition_coverage=0.90,
        )
        self.assertFalse(inputs["contradiction"])
        self.assertFalse(inputs["strict_support"])
        self.assertEqual(determine_assessment_level(**inputs), 2)

    def test_complete_disjoint_relation_has_level_four_precedence(self) -> None:
        inputs = build_assessment_inputs(
            relation_status=RELATION_INCONSISTENT,
            primary_relation_status=RELATION_INCONSISTENT,
            syndrome_evidence_available=True,
            formula_evidence_available=True,
            mechanism_evidence_available=True,
            core_herb_coverage=1.0,
            formula_composition_coverage=1.0,
        )
        self.assertTrue(inputs["contradiction"])
        self.assertFalse(inputs["strict_support"])
        self.assertEqual(determine_assessment_level(**inputs), 4)

    def test_qwen_payload_contains_only_non_numeric_narrative_facts(self) -> None:
        for context in self.contexts.values():
            payload = json.loads(build_user_prompt(context))
            self.assertEqual(set(payload), {"structured_audit"})
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertIsNone(re.search(r"[0-9０-９%％]", serialized))
            self.assertNotIn("evidence_context", serialized)
            self.assertNotIn("coverage_metrics", serialized)
            self.assertNotIn("physician_plan", serialized)

    def test_all_code_assembled_outputs_pass_before_api_use(self) -> None:
        for context in self.contexts.values():
            raw = self.valid_raw(context)
            response = assemble_response(raw, context)
            validation = validate_response(response, context, raw_model_response=raw)
            self.assertTrue(validation["valid"], validation)
            self.assertTrue(validation["semantic_consistent"], validation)

    def test_model_cannot_overwrite_structured_fields(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        raw = {
            **self.valid_raw(context),
            "assessment_level": 1,
            "coverage_metrics": {"physician_plan_herbs": {"value": 1.0}},
        }
        response = assemble_response(raw, context)
        self.assertEqual(response["assessment_level"], 3)
        self.assertEqual(response["coverage_metrics"]["physician_plan_herbs"]["value"], 9 / 13)
        validation = validate_response(response, context, raw_model_response=raw)
        self.assertFalse(validation["valid"])
        self.assertIn("model may output only assessment_summary", validation["errors"])

    def test_non_op_system_level_injection_is_rejected(self) -> None:
        context = self.contexts[("case_018", "flat_rag")]
        response = assemble_response(self.valid_raw(context), context)
        response["assessment_level"] = 2
        validation = validate_response(
            response,
            context,
            raw_model_response=self.valid_raw(context),
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("assessment_level differs from deterministic" in error for error in validation["errors"])
        )

    def test_evaluation_only_artifact_is_not_sent_to_model(self) -> None:
        context = self.contexts[("case_018", "qwen_only")]
        context_with_external_artifact = deepcopy(context)
        context_with_external_artifact["evaluation_only_assessment"] = {
            "assessment_level": 1,
            "sent_to_qwen": False,
        }
        payload = json.loads(build_user_prompt(context_with_external_artifact))
        self.assertNotIn("evaluation_only_assessment", json.dumps(payload))

    def test_any_structured_result_mutation_is_rejected(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        raw = self.valid_raw(context)
        baseline = assemble_response(raw, context)
        mutations = {
            "evidence_claims": [],
            "missing_evidence_items": [],
            "coverage_metrics": {
                **deepcopy(baseline["coverage_metrics"]),
                "physician_plan_herbs": {
                    **deepcopy(baseline["coverage_metrics"]["physician_plan_herbs"]),
                    "numerator": 11,
                    "value": 11 / 13,
                },
            },
            "assessment_level": 1,
            "assessment_label": "complete_support",
            "assessment_rule_trace": {
                "inputs": {},
                "triggered_rule": "level_1",
            },
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                changed = deepcopy(baseline)
                changed[field] = replacement
                validation = validate_response(
                    changed, context, raw_model_response=raw
                )
                self.assertFalse(validation["valid"], validation)
                self.assertFalse(validation["structured_audit_consistent"], validation)

    def test_missing_required_clause_is_rejected(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        label = context["structured_audit"]["narrative_facts"]["assessment_label"]
        raw = {"assessment_summary": f"{label}: some evidence remains missing."}
        validation = validate_response(
            assemble_response(raw, context), context, raw_model_response=raw
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("omits required clause" in error for error in validation["errors"]))

    def test_spelled_out_number_is_rejected(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        raw = self.valid_raw(context)
        raw["assessment_summary"] = raw["assessment_summary"][:-1] + " and nine herbs."
        validation = validate_response(
            assemble_response(raw, context), context, raw_model_response=raw
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("spell out" in error for error in validation["errors"]))

    def test_unicode_numeric_symbols_are_rejected(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        for numeric_text in ("½", "Ⅸ", "９", "69.2%"):
            with self.subTest(numeric_text=numeric_text):
                raw = self.valid_raw(context)
                raw["assessment_summary"] = (
                    raw["assessment_summary"][:-1] + f" and {numeric_text}."
                )
                validation = validate_response(
                    assemble_response(raw, context), context, raw_model_response=raw
                )
                self.assertFalse(validation["valid"], validation)
                self.assertFalse(validation["narrative_numbers_absent"], validation)

    def test_opposite_missing_evidence_statement_is_rejected(self) -> None:
        context = self.contexts[("case_018", "op_rag")]
        raw = self.valid_raw(context)
        raw["assessment_summary"] = raw["assessment_summary"].replace(
            "some evidence remains missing", "no evidence item remains missing"
        )
        validation = validate_response(
            assemble_response(raw, context), context, raw_model_response=raw
        )
        self.assertFalse(validation["valid"], validation)
        self.assertFalse(validation["semantic_consistent"], validation)

    def test_assessment_rule_precedence(self) -> None:
        common = {
            "syndrome_evidence_available": True,
            "formula_evidence_available": True,
            "mechanism_evidence_available": True,
        }
        self.assertEqual(
            determine_assessment_level(
                contradiction=True, strict_support=True, **common
            ),
            4,
        )
        self.assertEqual(
            determine_assessment_level(
                contradiction=False, strict_support=True, **common
            ),
            1,
        )
        self.assertEqual(
            determine_assessment_level(
                contradiction=False, strict_support=False, **common
            ),
            2,
        )
        self.assertEqual(
            determine_assessment_level(
                contradiction=False,
                strict_support=False,
                syndrome_evidence_available=True,
                formula_evidence_available=False,
                mechanism_evidence_available=True,
            ),
            3,
        )

    def test_relation_requires_the_expected_syndrome_record(self) -> None:
        reference = {
            "physician_plan": {"herbs": []},
            "primary_syndrome_id": "S1",
            "case_syndrome_ids": ["S1"],
            "resolved_formula_id": "F1",
            "expected_claims": [
                {
                    "item_id": "syndrome:S1",
                    "layer": "syndrome",
                    "expected_status": "supported",
                    "source_ids": ["REF:S1"],
                },
                {
                    "item_id": "formula:F1",
                    "layer": "formula",
                    "expected_status": "supported",
                    "source_ids": ["REF:F1"],
                },
                {
                    "item_id": "relation:F1:syndrome",
                    "layer": "cross_layer",
                    "expected_status": "supported",
                    "source_ids": ["REF:F1"],
                },
            ],
            "expected_missing_items": [],
            "expected_assessment_level": 2,
            "core_herbs": [],
            "formula_composition_herbs": [],
            "mechanism_supported_plan_herbs": [],
            "mechanism_supported_formula_herbs": [],
        }
        context = {
            "configuration": "flat_rag",
            "evidence_context": [
                {"item_id": "syndrome:S2", "layer": "syndrome", "source_ids": ["REF:S2"]},
                {"item_id": "formula:F1", "layer": "formula", "source_ids": ["REF:F1"]},
            ],
        }
        audit = build_structured_audit(reference, context)
        claim_ids = {claim["item_id"] for claim in audit["evidence_claims"]}
        self.assertNotIn("relation:F1:syndrome", claim_ids)
        self.assertNotIn("relation:F1:syndrome", audit["missing_evidence_items"])


if __name__ == "__main__":
    unittest.main()
