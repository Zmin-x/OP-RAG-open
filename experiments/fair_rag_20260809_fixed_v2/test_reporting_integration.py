from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from protocol import OUTPUT_DIR, read_jsonl, write_json, write_jsonl  # noqa: E402
from qwen_protocol import assemble_response, validate_response  # noqa: E402
from run_qwen_comparison import context_hash  # noqa: E402


class ReportingIntegrationTests(unittest.TestCase):
    def test_all_200_code_owned_outputs_rescore_and_revalidate(self) -> None:
        contexts_path = OUTPUT_DIR / "model_contexts.jsonl"
        if not contexts_path.exists():
            self.skipTest(
                "Paper-internal model_contexts.jsonl is not distributed; "
                "run this test only with the authorized 50-case audit package."
            )
        contexts = read_jsonl(contexts_path)
        results = []
        for row in contexts:
            facts = row["context"]["structured_audit"]["narrative_facts"]
            raw = {
                "assessment_summary": (
                    f"{facts['assessment_label']}: "
                    + "; ".join(facts["required_clauses"])
                    + "."
                )
            }
            response = assemble_response(raw, row["context"])
            validation = validate_response(
                response, row["context"], raw_model_response=raw
            )
            self.assertTrue(validation["valid"], validation)
            results.append(
                {
                    "case_id": row["case_id"],
                    "source_group": "integration_test",
                    "configuration": row["configuration"],
                    "response": response,
                    "validation": validation,
                    "remote_metadata": {
                        "model": "deterministic_test_double_not_qwen",
                        "request_id": None,
                        "attempt": 1,
                        "prior_failures": [],
                    },
                    "context_sha256": context_hash(row["context"]),
                }
            )

        with tempfile.TemporaryDirectory(prefix="oprag_v4_reporting_test_") as temp_name:
            run_dir = Path(temp_name)
            write_jsonl(run_dir / "qwen_results.jsonl", results)
            write_json(
                run_dir / "run_manifest.json",
                {
                    "status": "completed",
                    "n_expected": 200,
                    "n_completed": 200,
                    "n_valid": 200,
                    "test_double": True,
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_DIR / "score_qwen_comparison.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_DIR / "audit_generated_responses.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            first_audit_summary = (
                run_dir / "response_consistency_audit_summary.json"
            ).read_bytes()
            subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_DIR / "audit_generated_responses.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            self.assertEqual(
                first_audit_summary,
                (run_dir / "response_consistency_audit_summary.json").read_bytes(),
            )
            scoring = json.loads(
                (run_dir / "qwen_scoring_summary.json").read_text(encoding="utf-8")
            )
            audit = json.loads(
                (run_dir / "response_consistency_audit_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["n_valid"], 200)
            for configuration, values in scoring["configurations"].items():
                self.assertEqual(values["structured_audit_consistent"]["numerator"], 50)
                self.assertEqual(values["narrative_numbers_absent"]["numerator"], 50)
                self.assertEqual(values["semantic_consistent"]["numerator"], 50)
                self.assertEqual(values["missing_evidence_disclosure_f1"], 1.0)
                if configuration == "op_rag":
                    self.assertEqual(values["assessment_level_agreement"], 1.0)
                else:
                    self.assertIsNone(values["assessment_level_agreement"])
                    self.assertEqual(values["assessment_level_counts"], {"N/A": 50})
                self.assertEqual(
                    sum(values["evaluation_only_level_counts"].values()), 50
                )
            for comparison in scoring[
                "paired_differences_with_source_cluster_bootstrap_ci95"
            ].values():
                self.assertEqual(
                    set(comparison),
                    {"evidence_recall", "structured_evidence_link_precision"},
                )


if __name__ == "__main__":
    unittest.main()
