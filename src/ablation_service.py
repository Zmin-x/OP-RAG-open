from __future__ import annotations

from .ablation_runner import AblationCaseInput, AblationRunner


class AblationService:
    MODES = AblationRunner.MODES

    def __init__(self, kb: dict, pipeline) -> None:
        self.runner = AblationRunner(kb, pipeline)

    def run_ablation_case(self, query: str, mode: str) -> dict:
        case = AblationCaseInput(case_id="web_case", patient_text=query)
        return self.runner.run_case(case, mode).to_dict()

    def run_all_ablation(self, query: str) -> dict:
        case = AblationCaseInput(case_id="web_case", patient_text=query)
        return self.runner.run_all(case)
