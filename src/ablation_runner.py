from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DEFAULT_TOP_K_FORMULAS, DEFAULT_TOP_K_SYNDROMES, FORMULA_ALIAS_MAP, HERB_ALIAS_MAP
from .llm_client import QwenClient, build_local_report
from .rag_pipeline import OPRagPipeline
from .reflection import check_formula_consistency
from .retriever import TfidfRetriever


@dataclass
class AblationCaseInput:
    case_id: str
    patient_text: str
    primary_syndrome_id: str | None = None
    primary_syndrome_name_raw: str | None = None
    secondary_syndrome_ids: list[str] | None = None
    secondary_syndrome_names_raw: list[str] | None = None
    accepted_syndrome_ids: list[str] | None = None
    reference_main_formula_name: str | None = None
    reference_formula_name_raw: str | None = None
    reference_formula_id: str | None = None
    accepted_formula_ids: list[str] | None = None
    reference_herb_set: list[str] | None = None
    reference_herb_names_raw: list[str] | None = None
    core_herbs: list[str] | None = None
    core_herbs_raw: list[str] | None = None
    reference_syndrome_id: str | None = None
    use_llm: bool = True
    top_k_syndromes: int = DEFAULT_TOP_K_SYNDROMES
    top_k_formulas: int = DEFAULT_TOP_K_FORMULAS


@dataclass
class AblationCaseOutput:
    case_id: str
    mode: str
    context: dict[str, Any]
    report: str
    metrics: dict[str, Any] = field(default_factory=dict)
    support_assessment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(self.context.get("case_results", {}))
        return data


class AblationRunner:
    MODES = ("g0", "g1", "g2", "g3", "g4")

    def __init__(
        self,
        kb: dict[str, Any],
        pipeline: OPRagPipeline | None = None,
        *,
        core60_threshold: float = 0.60,
        strict_core_threshold: float = 0.80,
        strict_formula_threshold: float = 0.80,
    ) -> None:
        self.kb = kb
        self.pipeline = pipeline or OPRagPipeline(kb)
        self.core60_threshold = float(core60_threshold)
        self.strict_core_threshold = float(strict_core_threshold)
        self.strict_formula_threshold = float(strict_formula_threshold)
        self.syndrome_by_id = {item.get("syndrome_id"): item for item in kb.get("syndromes", [])}
        self.formula_by_id = {item.get("formula_id"): item for item in kb.get("formulas", [])}
        self.herb_by_name = {item.get("herb_name"): item for item in kb.get("herbs", [])}
        self.formula_retriever = TfidfRetriever(kb.get("formulas", []))
        self.herb_retriever = TfidfRetriever(kb.get("herbs", []), text_field="text_description")

    def run_all(self, case: AblationCaseInput) -> dict[str, dict[str, Any]]:
        return {mode: self.run_case(case, mode).to_dict() for mode in self.MODES}

    def run_case(self, case: AblationCaseInput, mode: str) -> AblationCaseOutput:
        mode = (mode or "g4").strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"不支持的消融模式：{mode}")
        if not case.patient_text.strip():
            raise ValueError("患者症状输入为空，无法评估。")

        physician_plan = self._build_physician_plan(case)
        rag_evidence = self._build_rag_evidence(case, mode, physician_plan)
        self._annotate_evidence(rag_evidence)
        support_assessment = self._build_support_assessment(case, mode, physician_plan, rag_evidence)
        metrics = self._compute_metrics(case, physician_plan, rag_evidence, support_assessment)
        corrected = self._corrected_metrics(case, mode, physician_plan, rag_evidence)
        metrics.update(corrected)
        support_assessment.update({key: value for key, value in corrected.items() if key.startswith("chain_") or key.startswith("deterministic_") or key.startswith("formula_composition_")})
        annotated_herbs = self._normalize_herb_set({record.get("herb_name") for record in rag_evidence.get("herb_mechanism_evidence", []) if record.get("has_mechanism_evidence")})
        support_assessment["core_herbs_with_mechanism"] = sorted(self._normalize_herb_set(set(case.core_herbs or [])) & annotated_herbs)
        support_assessment["reference_herbs_with_mechanism"] = sorted(self._normalize_herb_set(set(case.reference_herb_set or [])) & annotated_herbs)
        rag_evidence["deterministic_assessment"] = {
            "level": corrected.get("deterministic_assessment_level"),
            "label": corrected.get("deterministic_assessment_label"),
            "must_not_be_overridden_by_llm": True,
        }
        context = self._build_context(case, mode, physician_plan, rag_evidence, support_assessment)
        model_context = self._build_model_context(case, mode, rag_evidence)
        report = self._generate_report(model_context, case.use_llm, f"{mode.upper()} physician-plan support evaluation")
        case_results = self._build_case_results(case, mode, physician_plan, rag_evidence, metrics)
        case_results.update(self._corrected_case_fields(case, mode, physician_plan, rag_evidence, corrected))
        context["model_input_contract"] = {
            "uses_raw_physician_plan": True,
            "patient_text_visible_to_model": False,
            "retrieval_evidence_enabled": mode != "g0",
            "deterministic_assessment_visible_to_model": False,
            "rule_assessment_visible_to_model": False,
            "kb_resolution_status_visible_to_model": False,
        }
        context["case_results"] = case_results
        return AblationCaseOutput(case.case_id, mode, context, report, metrics, support_assessment)

    def _annotate_evidence(self, evidence: dict[str, Any]) -> None:
        """Attach explicit provenance/status fields without changing KB data."""
        for item in evidence.get("syndrome_evidence", []):
            item.setdefault("lookup_method", "kb_key_lookup_from_physician_label")
            item.setdefault("source_record", item.get("syndrome_id"))
            item.setdefault("evidence_status", "resolved" if item.get("syndrome_id") in self.syndrome_by_id else "missing")
        for item in evidence.get("formula_evidence", []):
            item.setdefault("lookup_method", "formula_name_normalization")
            item.setdefault("source_record", item.get("formula_id"))
            item.setdefault("evidence_status", "resolved" if item.get("formula_id") in self.formula_by_id else "missing")
            if "composition_herbs" not in item:
                item["composition_herbs"] = [
                    entry.get("herb") for entry in item.get("composition", [])
                    if isinstance(entry, dict) and entry.get("herb")
                ]
        for item in evidence.get("herb_mechanism_evidence", []):
            item.setdefault("lookup_method", "herb_alias_lookup_from_reference_prescription")
            item.setdefault("source_record", item.get("herb_name"))
            item["has_mechanism_evidence"] = bool(item.get("targets_op_related") or item.get("pathways"))

    def _build_physician_plan(self, case: AblationCaseInput) -> dict[str, Any]:
        primary = self._resolve_primary_syndrome(case)
        secondary = self._resolve_secondary_syndromes(case)
        formula = self._resolve_doctor_formula(case)
        return {
            "tcm_diagnosis_raw": case.primary_syndrome_name_raw,
            "western_diagnosis_raw": None,
            "treatment_principle_raw": None,
            "primary_syndrome": primary,
            "secondary_syndromes": secondary,
            "all_syndrome_ids": [sid for sid in [primary.get("id"), *[s.get("id") for s in secondary]] if sid],
            "formula": formula,
            "herbs": list(case.reference_herb_set or []),
            "core_herbs": list(case.core_herbs or []),
        }

    @staticmethod
    def _build_model_context(case: AblationCaseInput, mode: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Construct the LLM-visible context without leaking deterministic rules."""
        raw_secondary = [
            {"raw_name": name}
            for name in (case.secondary_syndrome_names_raw or [])
            if name
        ]
        raw_plan = {
            "tcm_diagnosis_raw": case.primary_syndrome_name_raw,
            "primary_syndrome": {"raw_name": case.primary_syndrome_name_raw},
            "secondary_syndromes": raw_secondary,
            "formula": {
                "raw_name": case.reference_main_formula_name or case.reference_formula_name_raw,
            },
            "herbs": list(case.reference_herb_set or []),
        }
        if mode == "g0":
            model_evidence = {
                "syndrome_evidence": [],
                "formula_evidence": [],
                "herb_mechanism_evidence": [],
                "retrieval": {},
            }
        else:
            def externalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [
                    {key: value for key, value in record.items() if key != "kb_status"}
                    for record in records
                ]

            model_evidence = {
                "syndrome_evidence": externalize(evidence.get("syndrome_evidence", [])),
                "formula_evidence": externalize(evidence.get("formula_evidence", [])),
                "herb_mechanism_evidence": externalize(evidence.get("herb_mechanism_evidence", [])),
                "retrieval": evidence.get("retrieval", {}),
            }
        if mode in {"g3", "g4"}:
            plan_herbs = list(case.reference_herb_set or [])
            retrieved_herbs = {
                str(record.get("herb_name") or "").strip()
                for record in evidence.get("herb_mechanism_evidence", [])
                if str(record.get("herb_name") or "").strip()
            }
            model_evidence["herb_mechanism_inventory"] = {
                "physician_plan_herbs": plan_herbs,
                "retrieved_mechanism_evidence_herbs": sorted(retrieved_herbs),
                "herbs_without_retrieved_mechanism_evidence": [
                    herb for herb in plan_herbs if HERB_ALIAS_MAP.get(herb, herb) not in retrieved_herbs
                ],
            }
        if mode in {"g2", "g3", "g4"}:
            physician_syndrome_ids = {
                syndrome_id
                for syndrome_id in [case.primary_syndrome_id or case.reference_syndrome_id, *(case.secondary_syndrome_ids or [])]
                if syndrome_id
            }
            formula_indication_syndrome_ids = {
                syndrome_id
                for record in evidence.get("formula_evidence", [])
                for syndrome_id in record.get("indication_syndrome", [])
                if syndrome_id
            }
            has_any_indication_overlap = (
                bool(physician_syndrome_ids & formula_indication_syndrome_ids)
                if physician_syndrome_ids and formula_indication_syndrome_ids
                else None
            )
            model_evidence["formula_syndrome_relation"] = {
                "physician_syndrome_ids": sorted(physician_syndrome_ids),
                "formula_indication_syndrome_ids": sorted(formula_indication_syndrome_ids),
                "has_any_indication_overlap": has_any_indication_overlap,
                "scope": "formula-level relation; additional herbs do not create a separate formula-syndrome contradiction",
            }
        return {
            "task": "evaluate_physician_plan",
            "mode": mode,
            "physician_plan": raw_plan,
            "rag_evidence": model_evidence,
            "output_requirements": {
                "same_structure_across_modes": True,
                "must_include": [
                    "patient_and_physician_plan_summary",
                    "syndrome_evidence_summary",
                    "formula_evidence_summary",
                    "herb_evidence_summary",
                    "target_pathway_evidence_summary",
                    "cross_layer_evidence_summary",
                    "evidence_boundary_statement",
                    "evidence_source_summary",
                ],
            },
        }

    def _resolve_primary_syndrome(self, case: AblationCaseInput) -> dict[str, Any]:
        sid = case.primary_syndrome_id or case.reference_syndrome_id
        item = self.syndrome_by_id.get(sid, {}) if sid else {}
        return {"id": sid, "raw_name": case.primary_syndrome_name_raw or item.get("name"), "std_name": item.get("name") or case.primary_syndrome_name_raw}

    def _resolve_secondary_syndromes(self, case: AblationCaseInput) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        ids = case.secondary_syndrome_ids or []
        raws = case.secondary_syndrome_names_raw or []
        for i, sid in enumerate(ids):
            item = self.syndrome_by_id.get(sid, {})
            out.append({"id": sid, "raw_name": raws[i] if i < len(raws) else item.get("name"), "std_name": item.get("name") or (raws[i] if i < len(raws) else None)})
        return out

    def _resolve_doctor_formula(self, case: AblationCaseInput) -> dict[str, Any]:
        raw = case.reference_main_formula_name or case.reference_formula_name_raw
        if case.reference_formula_id and case.reference_formula_id in self.formula_by_id:
            item = self.formula_by_id[case.reference_formula_id]
            return {"id": case.reference_formula_id, "raw_name": raw or item.get("name"), "std_name": item.get("name"), "kb_status": "mapped"}
        if raw:
            target = "".join(raw.split())
            normalized_target = "".join((self._normalize_formula_name(target) or target).split())
            exact_matches: list[tuple[int, dict[str, Any], str]] = []
            target_keys = self._formula_match_keys(target)
            for item in self.kb.get("formulas", []):
                name = str(item.get("name") or "").strip()
                compact_name = "".join(name.split())
                normalized_name = "".join((self._normalize_formula_name(compact_name) or compact_name).split())
                if target == compact_name:
                    exact_matches.append((3, item, name))
                elif normalized_target == normalized_name:
                    exact_matches.append((2, item, name))
                elif target_keys & self._formula_match_keys(compact_name):
                    exact_matches.append((1, item, name))
            if exact_matches:
                best_score = max(row[0] for row in exact_matches)
                best = [row for row in exact_matches if row[0] == best_score]
                if len(best) == 1:
                    _, item, name = best[0]
                    return {"id": item.get("formula_id"), "raw_name": raw, "std_name": name, "kb_status": "mapped"}
                return {"id": None, "raw_name": raw, "std_name": raw, "kb_status": "ambiguous"}
        return {"id": None, "raw_name": raw, "std_name": raw, "kb_status": "unmapped"}

    @classmethod
    def _formula_match_keys(cls, name: str | None) -> set[str]:
        if not name:
            return set()
        compact = "".join(str(name).split())
        normalized = "".join((cls._normalize_formula_name(compact) or compact).split())
        keys: set[str] = set()
        for value in (compact, normalized):
            for part in value.replace("／", "/").split("/"):
                part = part.strip()
                if not part:
                    continue
                keys.add(part)
                for suffix in ("加减方", "加减", "加味", "化裁"):
                    if part.endswith(suffix) and len(part) > len(suffix):
                        keys.add(part[: -len(suffix)])
        return keys

    def _build_rag_evidence(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any]) -> dict[str, Any]:
        if mode == "g0":
            return {"syndrome_evidence": [], "formula_evidence": [], "herb_mechanism_evidence": [], "reflection": None, "chain_assessment": None, "retrieval": {}}
        syndrome_evidence, syndrome_retrieval = self._retrieve_syndrome_evidence(case, physician_plan)
        evidence = {"syndrome_evidence": syndrome_evidence, "formula_evidence": [], "herb_mechanism_evidence": [], "reflection": None, "chain_assessment": None, "retrieval": {"syndrome": syndrome_retrieval}}
        if mode in {"g2", "g3", "g4"}:
            formula_evidence, formula_retrieval = self._retrieve_formula_evidence(case, physician_plan)
            evidence["formula_evidence"] = formula_evidence
            evidence["retrieval"]["formula"] = formula_retrieval
        if mode in {"g3", "g4"}:
            evidence["herb_mechanism_evidence"] = self._herb_mechanism_evidence(case)
        if mode == "g4":
            evidence["reflection"] = self._reflection_result(case, physician_plan, evidence)
            evidence["chain_assessment"] = self._chain_assessment(case, physician_plan, evidence)
        if mode == "g1":
            evidence["formula_evidence"] = []
            evidence["herb_mechanism_evidence"] = []
        return evidence

    def _retrieve_syndrome_evidence(self, case: AblationCaseInput, physician_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        evidence: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        planned = [("primary", physician_plan.get("primary_syndrome", {}))]
        planned.extend(("secondary", item) for item in physician_plan.get("secondary_syndromes", []))
        for role, syndrome in planned:
            query = str(syndrome.get("std_name") or syndrome.get("raw_name") or "").strip()
            if not query:
                continue
            results = self.pipeline.syndrome_retriever.search(query, top_k=case.top_k_syndromes)
            retrieved_ids = {result.item.get("syndrome_id") for result in results}
            for rank, result in enumerate(results, start=1):
                sid = result.item.get("syndrome_id")
                candidates.append({"syndrome_id": sid, "score": result.score, "rank": rank, "query_role": role, "query": query, "is_reference_label": sid == syndrome.get("id")})
            if syndrome.get("id") in retrieved_ids and syndrome.get("id") not in seen:
                item = self._syndrome_evidence(syndrome, role)
                item.update({"retrieved": True, "retrieval_rank": next((row["rank"] for row in candidates if row["syndrome_id"] == syndrome.get("id") and row["query_role"] == role), None), "retrieval_query": query})
                evidence.append(item)
                seen.add(syndrome.get("id"))
        return evidence, candidates

    def _retrieve_formula_evidence(self, case: AblationCaseInput, physician_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        formula = physician_plan.get("formula") or {}
        query = str(formula.get("std_name") or formula.get("raw_name") or "").strip()
        if not query:
            return [], []
        results = self.formula_retriever.search(query, top_k=case.top_k_formulas)
        candidates = [{"formula_id": result.item.get("formula_id"), "score": result.score, "rank": rank, "query": query, "is_reference_formula": result.item.get("formula_id") == formula.get("id")} for rank, result in enumerate(results, start=1)]
        selected = next((result.item for result in results if result.item.get("formula_id") == formula.get("id")), None)
        if selected is None:
            return [{"formula_id": None, "name": formula.get("raw_name"), "kb_status": "not_retrieved", "lookup_method": "tfidf_formula_retrieval", "evidence_status": "missing", "retrieval_query": query}], candidates
        item = self._formula_evidence(case)
        item.update({"retrieved": True, "retrieval_rank": next(row["rank"] for row in candidates if row["formula_id"] == formula.get("id")), "retrieval_query": query, "lookup_method": "tfidf_formula_retrieval"})
        return [item], candidates

    def _syndrome_evidence(self, syndrome: dict[str, Any], role: str) -> dict[str, Any]:
        item = self.syndrome_by_id.get(syndrome.get("id"), {}) if syndrome.get("id") else {}
        return {"syndrome_id": syndrome.get("id"), "name": syndrome.get("std_name") or syndrome.get("raw_name") or item.get("name"), "core_symptoms": item.get("core_symptoms", []), "secondary_symptoms": item.get("secondary_symptoms", []), "tongue": item.get("tongue", ""), "pulse": item.get("pulse", ""), "text_description": item.get("description", ""), "role": role}

    def _formula_evidence(self, case: AblationCaseInput) -> dict[str, Any]:
        formula = self._resolve_doctor_formula(case)
        if formula.get("kb_status") == "unmapped":
            return {"formula_id": None, "name": formula.get("raw_name"), "kb_status": "unmapped", "message": "该方暂未纳入结构化方剂知识库"}
        item = self.formula_by_id.get(formula.get("id"), {})
        return {"formula_id": formula.get("id"), "name": formula.get("std_name"), "composition": item.get("composition", []), "indication_syndrome": item.get("indication_syndrome", []), "classical_source": item.get("classical_source", {}), "modern_evidence": item.get("modern_evidence", []), "notes": item.get("notes", ""), "kb_status": "mapped"}

    def _herb_mechanism_evidence(self, case: AblationCaseInput) -> list[dict[str, Any]]:
        records = []
        for herb in case.reference_herb_set or []:
            normalized_herb = HERB_ALIAS_MAP.get(herb, herb)
            retrieval_results = self.herb_retriever.search(normalized_herb, top_k=1)
            item = retrieval_results[0].item if retrieval_results and retrieval_results[0].item.get("herb_name") == normalized_herb else None
            if not item:
                continue
            records.append({"herb_name": herb, "standardized_herb_name": item.get("herb_name"), "tcm_function": item.get("tcm_function", ""), "targets_op_related": item.get("targets_op_related", []), "pathways": item.get("pathways", []), "evidence_papers": item.get("evidence_papers", []), "text_description": item.get("description", ""), "lookup_method": "tfidf_herb_retrieval", "retrieval_rank": 1, "has_mechanism_evidence": bool(item.get("targets_op_related") or item.get("pathways"))})
        return records

    def _reflection_result(self, case: AblationCaseInput, physician_plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        primary = physician_plan.get("primary_syndrome") or {}
        formula = physician_plan.get("formula") or {}
        primary_consistency = self._formula_consistency(primary.get("id"), formula.get("id")) if primary.get("id") and formula.get("id") and primary.get("id") in self.kb.get("syndrome_formula_map", {}) and formula.get("kb_status") == "mapped" else None
        any_consistency = None
        if formula.get("kb_status") == "mapped":
            for syndrome in [primary, *physician_plan.get("secondary_syndromes", [])]:
                result = self._formula_consistency(syndrome.get("id"), formula.get("id")) if syndrome.get("id") else None
                if result is True:
                    any_consistency = True
                    break
                if result is False:
                    any_consistency = False if any_consistency is None else any_consistency
        return {"formula_consistent_with_primary": primary_consistency, "formula_consistent_with_any_syndrome": any_consistency, "message": "医生证型-方剂映射已评估" if formula.get("kb_status") == "mapped" else "医生方剂未映射到知识库，无法评估一致性"}

    def _formula_consistency(self, syndrome_id: str | None, formula_id: str | None) -> bool | None:
        if not syndrome_id or not formula_id:
            return None
        allowed = self.kb.get("syndrome_formula_map", {}).get(syndrome_id)
        if allowed is None:
            return None
        return formula_id in allowed

    def _chain_assessment(self, case: AblationCaseInput, physician_plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        formula = physician_plan.get("formula") or {}
        primary = physician_plan.get("primary_syndrome") or {}
        core = self._normalize_herb_set(set(case.core_herbs or []))
        ref = self._normalize_herb_set(set(case.reference_herb_set or []))
        mech = self._normalize_herb_set({h.get("herb_name") for h in evidence.get("herb_mechanism_evidence", [])})
        core_cov = (len(core & mech) / len(core)) if core else None
        ref_cov = (len(ref & mech) / len(ref)) if ref else None
        any_consistent = evidence.get("reflection", {}).get("formula_consistent_with_any_syndrome") is True
        primary_consistent = evidence.get("reflection", {}).get("formula_consistent_with_primary") is True
        formula_mapped = formula.get("kb_status") == "mapped"
        has_mech = bool(mech)
        chain_any = bool(any(evidence.get("syndrome_evidence", [])) and formula_mapped and any_consistent and has_mech)
        chain_core60 = True if chain_any and core_cov is not None and core_cov >= self.core60_threshold else False if chain_any and core_cov is not None else None
        chain_strict = True if primary.get("id") and formula_mapped and primary_consistent and core_cov is not None and core_cov >= self.strict_core_threshold and ref_cov is not None and ref_cov >= self.strict_formula_threshold else False if primary.get("id") and formula_mapped else None
        return {"chain_closed_any": chain_any, "chain_closed_core60": chain_core60, "chain_closed_strict": chain_strict}

    def _build_support_assessment(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        formula = physician_plan.get("formula") or {}
        core = self._normalize_herb_set(set(case.core_herbs or []))
        ref = self._normalize_herb_set(set(case.reference_herb_set or []))
        mech = self._normalize_herb_set({h.get("herb_name") for h in evidence.get("herb_mechanism_evidence", [])})
        return {"mode": mode, "doctor_syndrome_supported": bool(evidence.get("syndrome_evidence")) if mode in {"g1", "g2", "g3", "g4"} else None, "doctor_formula_supported": formula.get("kb_status") == "mapped" if mode in {"g2", "g3", "g4"} else None, "formula_in_accepted_set": None, "formula_consistent_with_syndrome": evidence.get("reflection") if mode == "g4" else None, "core_herbs": sorted(core), "core_herbs_with_mechanism": sorted(core & mech), "reference_herbs_with_mechanism": sorted(ref & mech), "mechanism_target_count": sum(len(h.get("targets_op_related", [])) for h in evidence.get("herb_mechanism_evidence", [])) if mode in {"g3", "g4"} else None, "mechanism_pathway_count": sum(len(h.get("pathways", [])) for h in evidence.get("herb_mechanism_evidence", [])) if mode in {"g3", "g4"} else None, **(evidence.get("chain_assessment") or {})}

    def _build_context(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any], evidence: dict[str, Any], support_assessment: dict[str, Any]) -> dict[str, Any]:
        return {"task": "evaluate_physician_plan", "mode": mode, "patient": {"text": case.patient_text}, "physician_plan": physician_plan, "rag_evidence": evidence, "support_assessment": support_assessment, "output_requirements": {"same_structure_across_modes": True, "must_include": ["患者与医生方案摘要", "证型评判", "方剂评判", "药味配伍评判", "靶点与通路机制解释", "方证一致性与证据链闭合", "医生方案总体评价", "本层证据来源说明"]}}

    def _build_case_results(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any], evidence: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
        formula = physician_plan.get("formula") or {}
        primary = physician_plan.get("primary_syndrome") or {}
        secondary = physician_plan.get("secondary_syndromes") or []
        matched = {self._normalize_name(h.get("herb_name")) for h in evidence.get("herb_mechanism_evidence", [])}
        herbs = list(case.reference_herb_set or [])
        return {"case_id": case.case_id, "mode": mode, "primary_syndrome_id": primary.get("id"), "primary_syndrome_name": primary.get("std_name"), "secondary_syndrome_ids": [s.get("id") for s in secondary if s.get("id")], "secondary_syndrome_names": [s.get("std_name") for s in secondary if s.get("std_name")], "all_syndrome_ids": physician_plan.get("all_syndrome_ids", []), "reference_formula_id": case.reference_formula_id, "reference_formula_name": formula.get("raw_name"), "formula_kb_status": formula.get("kb_status"), "reference_herb_count": len(herbs), "herbs_with_mechanism_count": len(evidence.get("herb_mechanism_evidence", [])), "herbs_without_mechanism": [h for h in herbs if self._normalize_name(h) not in matched], "formula_consistent_with_primary": evidence.get("reflection", {}).get("formula_consistent_with_primary") if evidence.get("reflection") else None, "formula_consistent_with_any_syndrome": evidence.get("reflection", {}).get("formula_consistent_with_any_syndrome") if evidence.get("reflection") else None, "herb_jaccard": metrics.get("herb_jaccard"), "core_herb_mechanism_coverage": metrics.get("core_herb_mechanism_coverage"), "mechanism_coverage_over_reference_herbs": metrics.get("mechanism_coverage_over_reference_herbs"), "chain_closed_any": metrics.get("chain_closed_any"), "chain_closed_core60": metrics.get("chain_closed_core60"), "chain_closed_strict": metrics.get("chain_closed_strict")}

    def _compute_metrics(self, case: AblationCaseInput, physician_plan: dict[str, Any], evidence: dict[str, Any], support_assessment: dict[str, Any]) -> dict[str, Any]:
        ref = self._normalize_herb_set(set(case.reference_herb_set or []))
        mech = self._normalize_herb_set({h.get("herb_name") for h in evidence.get("herb_mechanism_evidence", [])})
        core = self._normalize_herb_set(set(case.core_herbs or []))
        herb_jaccard = None if not case.reference_herb_set else self._jaccard(mech, ref)
        core_cov = (len(core & mech) / len(core)) if core else None
        ref_cov = (len(ref & mech) / len(ref)) if ref else None
        formula = physician_plan.get("formula") or {}
        primary = physician_plan.get("primary_syndrome") or {}
        return {"primary_syndrome_mapping_rate": 1.0 if primary.get("id") else None, "formula_kb_support_rate": 1.0 if formula.get("kb_status") == "mapped" else 0.0 if formula.get("kb_status") == "unmapped" else None, "formula_syndrome_consistency_rate_primary": evidence.get("reflection", {}).get("formula_consistent_with_primary") if evidence.get("reflection") else None, "formula_syndrome_consistency_rate_any": evidence.get("reflection", {}).get("formula_consistent_with_any_syndrome") if evidence.get("reflection") else None, "mean_herb_jaccard": herb_jaccard, "mean_core_herb_mechanism_coverage": core_cov, "mean_mechanism_coverage_over_reference_herbs": ref_cov, "chain_closed_any_rate": 1.0 if support_assessment.get("chain_closed_any") else 0.0 if support_assessment.get("chain_closed_any") is False else None, "chain_closed_core60_rate": 1.0 if support_assessment.get("chain_closed_core60") else 0.0 if support_assessment.get("chain_closed_core60") is False else None, "chain_closed_strict_rate": 1.0 if support_assessment.get("chain_closed_strict") else 0.0 if support_assessment.get("chain_closed_strict") is False else None, "herb_jaccard": herb_jaccard, "core_herb_mechanism_coverage": core_cov, "mechanism_coverage_over_reference_herbs": ref_cov, "chain_closed_any": support_assessment.get("chain_closed_any"), "chain_closed_core60": support_assessment.get("chain_closed_core60"), "chain_closed_strict": support_assessment.get("chain_closed_strict")}

    def _corrected_metrics(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        """Compute metrics with explicit denominators and evidence status.

        The legacy fields are retained for compatibility, while these values
        distinguish reference-prescription coverage from standard formula
        composition coverage. Only G4 receives a four-level deterministic
        assessment; earlier modes intentionally report that conclusion as not
        applicable because the rule layer is disabled.
        """
        formula = physician_plan.get("formula") or {}
        formula_evidence = (evidence.get("formula_evidence") or [{}])[0]
        formula_herbs = self._normalize_herb_set(set(formula_evidence.get("composition_herbs") or []))
        if not formula_herbs and formula.get("id") in self.formula_by_id:
            formula_herbs = self._normalize_herb_set({
                item.get("herb")
                for item in self.formula_by_id[formula.get("id")].get("composition", [])
                if isinstance(item, dict) and item.get("herb")
            })
        reference_herbs = self._normalize_herb_set(set(case.reference_herb_set or []))
        core_herbs = self._normalize_herb_set(set(case.core_herbs or []))
        herb_records = evidence.get("herb_mechanism_evidence", [])
        annotated_herbs = self._normalize_herb_set({record.get("herb_name") for record in herb_records if record.get("has_mechanism_evidence")})
        formula_annotated_herbs = {
            herb for herb in formula_herbs
            if (record := self.herb_by_name.get(herb))
            and bool(record.get("targets_op_related") or record.get("pathways"))
        }
        core_cov = len(core_herbs & annotated_herbs) / len(core_herbs) if core_herbs else None
        reference_cov = len(reference_herbs & annotated_herbs) / len(reference_herbs) if reference_herbs else None
        formula_cov = len(formula_annotated_herbs) / len(formula_herbs) if formula_herbs else None
        herb_jaccard = self._jaccard(annotated_herbs, reference_herbs) if reference_herbs else None
        if mode not in {"g3", "g4"}:
            core_cov = reference_cov = formula_cov = herb_jaccard = None
        reflection = evidence.get("reflection") or {}
        primary_consistent = reflection.get("formula_consistent_with_primary")
        any_consistent = reflection.get("formula_consistent_with_any_syndrome")
        syndrome_evidence = evidence.get("syndrome_evidence", [])
        has_syndrome = any(item.get("evidence_status") == "resolved" or item.get("syndrome_id") in self.syndrome_by_id for item in syndrome_evidence)
        formula_retrieved = any(
            item.get("formula_id") == formula.get("id") and item.get("retrieved")
            for item in evidence.get("formula_evidence", [])
        )
        formula_mapped = formula.get("kb_status") == "mapped" and (mode not in {"g2", "g3", "g4"} or formula_retrieved)
        has_mechanism = bool(annotated_herbs)

        if mode != "g4":
            level = None
            label = "not_applicable_rule_layer_disabled"
            chain_any = chain_core60 = chain_strict = None
        else:
            contradiction = formula_mapped and primary_consistent is False
            chain_any = bool(has_syndrome and formula_mapped and any_consistent is True and has_mechanism)
            chain_core60 = bool(chain_any and core_cov is not None and core_cov >= self.core60_threshold) if chain_any else None
            strict_inputs_present = bool(
                physician_plan.get("primary_syndrome", {}).get("id")
                and formula_mapped
                and primary_consistent is not None
                and core_cov is not None
                and formula_cov is not None
            )
            chain_strict = bool(primary_consistent is True and core_cov >= self.strict_core_threshold and formula_cov >= self.strict_formula_threshold) if strict_inputs_present else None
            if contradiction:
                level, label = 4, "explicit_formula_or_cross_layer_contradiction"
            elif chain_strict is True:
                level, label = 1, "evidence_supported_and_chain_complete"
            elif formula_mapped and has_syndrome and has_mechanism:
                level, label = 2, "partially_supported_but_evidence_missing"
            else:
                level, label = 3, "current_knowledge_base_evidence_insufficient"

        return {
            "herb_jaccard": herb_jaccard,
            "core_herb_mechanism_coverage": core_cov,
            "mechanism_coverage_over_reference_herbs": reference_cov,
            "formula_composition_mechanism_coverage": formula_cov,
            "formula_composition_herb_count": len(formula_herbs) if formula_herbs and mode in {"g3", "g4"} else None,
            "chain_closed_any": chain_any,
            "chain_closed_core60": chain_core60,
            "chain_closed_strict": chain_strict,
            "deterministic_assessment_level": level,
            "deterministic_assessment_label": label,
        }

    def _corrected_case_fields(self, case: AblationCaseInput, mode: str, physician_plan: dict[str, Any], evidence: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
        annotated = self._normalize_herb_set({record.get("herb_name") for record in evidence.get("herb_mechanism_evidence", []) if record.get("has_mechanism_evidence")})
        reference = list(case.reference_herb_set or [])
        core = self._normalize_herb_set(set(case.core_herbs or []))
        formula = physician_plan.get("formula") or {}
        primary = physician_plan.get("primary_syndrome") or {}
        retrieval = evidence.get("retrieval") or {}
        syndrome_rows = [
            row for row in retrieval.get("syndrome", [])
            if row.get("query_role") == "primary" and row.get("syndrome_id") == primary.get("id")
        ]
        formula_rows = [
            row for row in retrieval.get("formula", [])
            if row.get("formula_id") == formula.get("id")
        ]
        source_items = [
            *evidence.get("syndrome_evidence", []),
            *evidence.get("formula_evidence", []),
            *evidence.get("herb_mechanism_evidence", []),
        ]
        source_traceability_complete = (
            all(item.get("source_record") for item in source_items)
            if source_items else None
        )
        return {
            "resolved_formula_id": formula.get("id"),
            "resolved_formula_name": formula.get("std_name"),
            "syndrome_retrieval_hit": bool(syndrome_rows) if mode in {"g1", "g2", "g3", "g4"} and primary.get("id") else None,
            "syndrome_retrieval_rank": syndrome_rows[0].get("rank") if syndrome_rows else None,
            "formula_retrieval_hit": bool(formula_rows) if mode in {"g2", "g3", "g4"} and formula.get("id") else None,
            "formula_retrieval_rank": formula_rows[0].get("rank") if formula_rows else None,
            "source_traceability_complete": source_traceability_complete,
            "herbs_with_mechanism_count": len(annotated),
            "core_herbs_with_mechanism": sorted(core & annotated),
            "herbs_without_mechanism": [herb for herb in reference if self._normalize_name(herb) not in annotated],
            "formula_composition_mechanism_coverage": metrics.get("formula_composition_mechanism_coverage"),
            "deterministic_assessment_level": metrics.get("deterministic_assessment_level"),
            "deterministic_assessment_label": metrics.get("deterministic_assessment_label"),
        }

    @staticmethod
    def _generate_report(context: dict[str, Any], use_llm: bool, reason: str) -> str:
        return QwenClient().generate(context) if use_llm else build_local_report(context, reason=reason)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _normalize_herb_set(herbs: set[str]) -> set[str]:
        return {HERB_ALIAS_MAP.get(h, h) for h in herbs if h}

    @staticmethod
    def _normalize_name(name: str | None) -> str | None:
        return HERB_ALIAS_MAP.get(name, name) if name else None

    @staticmethod
    def _normalize_formula_name(name: str | None) -> str | None:
        if not name:
            return None
        target = name.strip()
        if target in FORMULA_ALIAS_MAP:
            return FORMULA_ALIAS_MAP[target]
        for alias, standard in FORMULA_ALIAS_MAP.items():
            if alias in target:
                return standard
        return target
