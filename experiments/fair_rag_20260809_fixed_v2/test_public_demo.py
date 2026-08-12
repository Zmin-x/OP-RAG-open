from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = ROOT / "scripts" / "run_public_demo.py"
SPEC = importlib.util.spec_from_file_location("run_public_demo", DEMO_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load public demo: {DEMO_SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicDemoTests(unittest.TestCase):
    def test_public_demo_is_synthetic_deterministic_and_api_free(self) -> None:
        first = MODULE.run_demo()
        second = MODULE.run_demo()
        self.assertEqual(first, second)
        self.assertEqual(first["case_count"], 4)
        self.assertFalse(first["uses_qwen_api"])
        self.assertEqual(
            first["scope"],
            "public_synthetic_method_demonstration_not_manuscript_results",
        )
        self.assertEqual(
            first["configurations"],
            ["qwen_only", "flat_rag", "layered_rag", "op_rag"],
        )
        for row in first["results"]:
            self.assertEqual(row["data_scope"], "synthetic_non_patient_plan")
            self.assertNotIn("patient_text", row)
            self.assertNotIn("primary_syndrome_id", row["physician_plan"])
            self.assertNotIn("reference_formula_id", row["physician_plan"])
            self.assertEqual(set(row["configurations"]), set(first["configurations"]))


if __name__ == "__main__":
    unittest.main()
