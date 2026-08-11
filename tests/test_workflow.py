from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from op_rag.ablation_runner import AblationCaseInput, AblationRunner
from op_rag.loader import load_kb


def test_knowledge_base_has_three_layers() -> None:
    kb = load_kb()
    assert kb["syndromes"]
    assert kb["formulas"]
    assert kb["herbs"]
    assert kb["syndrome_formula_map"]


def test_g4_reports_closure_fields_for_synthetic_case() -> None:
    case = AblationCaseInput(
        case_id="test",
        patient_text="Synthetic test case.",
        primary_syndrome_id="S1",
        reference_formula_id="F002",
        reference_main_formula_name="右归丸 / 加味右归丸",
        reference_herb_set=["熟地黄", "附子", "肉桂"],
        core_herbs=["熟地黄", "附子"],
        use_llm=False,
    )
    output = AblationRunner(load_kb()).run_case(case, "g4")
    assert output.context["case_results"]["formula_kb_status"] == "mapped"
    assert "chain_closed_core60" in output.metrics
