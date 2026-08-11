from __future__ import annotations

from .config import DEFAULT_TOP_K_FORMULAS, DEFAULT_TOP_K_SYNDROMES, HERB_ALIAS_MAP
from .llm_client import QwenClient
from .pipeline_trace import trace_herb_lookup
from .reflection import check_formula_consistency, choose_consistent_formula
from .retriever import TfidfRetriever, filter_formulas_by_syndrome


class OPRagPipeline:
    def __init__(self, kb: dict) -> None:
        self.kb = kb
        self.syndrome_retriever = TfidfRetriever(kb["syndromes"])
        self.herb_by_name = {herb["herb_name"]: herb for herb in kb["herbs"]}
        self.llm_client = QwenClient()

    def run(self, patient_text: str, top_k_syndromes: int = DEFAULT_TOP_K_SYNDROMES, top_k_formulas: int = DEFAULT_TOP_K_FORMULAS, use_llm: bool = True) -> dict:
        syndrome_results = self.syndrome_retriever.search(patient_text, top_k=top_k_syndromes)
        if not syndrome_results:
            raise ValueError("患者症状输入为空，无法检索。")
        selected_syndrome = syndrome_results[0].item
        syndrome_id = selected_syndrome["syndrome_id"]
        candidate_formulas = filter_formulas_by_syndrome(self.kb["formulas"], syndrome_id)
        formula_retriever = TfidfRetriever(candidate_formulas)
        formula_results = formula_retriever.search(patient_text, top_k=top_k_formulas)
        ranked_formulas = [result.item for result in formula_results]
        selected_formula = choose_consistent_formula(syndrome_id=syndrome_id, ranked_formulas=ranked_formulas, syndrome_formula_map=self.kb["syndrome_formula_map"])
        if not selected_formula:
            raise ValueError(f"证型 {syndrome_id} 下未找到候选方剂。")
        reflection = check_formula_consistency(syndrome_id=syndrome_id, formula_id=selected_formula["formula_id"], syndrome_formula_map=self.kb["syndrome_formula_map"])
        herb_records = self._get_herb_records(selected_formula)
        context = {"patient_text": patient_text, "syndrome_candidates": [self._format_scored_item(result.item, result.score) for result in syndrome_results], "formula_candidates": [self._format_scored_item(result.item, result.score) for result in formula_results], "selected_syndrome": selected_syndrome, "selected_formula": selected_formula, "reflection": reflection, "herb_records": herb_records}
        report = self.llm_client.generate(context) if use_llm else QwenClient(api_key="").generate(context)
        return {"context": context, "report": report}

    def _get_herb_records(self, formula: dict) -> list[dict]:
        records = []
        formula_id = formula.get("formula_id") or ""
        for item in formula.get("composition", []):
            herb_name = item.get("herb")
            lookup_name = HERB_ALIAS_MAP.get(herb_name, herb_name)
            herb_record = self.herb_by_name.get(lookup_name)
            trace_herb_lookup(herb_name, herb_record, formula_id=formula_id)
            if herb_record:
                merged = dict(herb_record)
                merged["formula_role"] = item.get("role")
                merged["dose"] = item.get("dose_range") or item.get("dose")
                merged["formula_herb_name"] = herb_name
                records.append(merged)
        return records

    @staticmethod
    def _format_scored_item(item: dict, score: float) -> dict:
        summary = {"score": score, "name": item.get("name") or item.get("herb_name")}
        for key in ("syndrome_id", "formula_id", "indication_syndrome"):
            if key in item:
                summary[key] = item[key]
        return summary
