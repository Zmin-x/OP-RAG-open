from __future__ import annotations

import json
from typing import Any


PILOT_SECTION_KEYS = (
    "patient_and_physician_plan_summary",
    "syndrome_evidence_summary",
    "formula_evidence_summary",
    "herb_evidence_summary",
    "target_pathway_evidence_summary",
    "cross_layer_evidence_summary",
    "evidence_boundary_statement",
    "evidence_source_summary",
)

ASSESSMENT_SCALE = {
    1: "evidence_supported_and_chain_complete",
    2: "evidence_partially_supported_with_missing_layers",
    3: "knowledge_base_evidence_insufficient",
    4: "explicit_cross_layer_contradiction",
}

SYSTEM_PROMPT = """You are an evidence-traceability reporting component for a research prototype.

You summarize a physician-provided plan and retrieved evidence records. You do not diagnose, assess treatment effectiveness, make a prescription recommendation, or introduce facts, source identifiers, targets, pathways, formulas, herbs, or syndromes not present in the input.

Return exactly one JSON object, with no Markdown fences and no prose outside the JSON. Use this exact schema:
{
  "sections": {
    "patient_and_physician_plan_summary": "string",
    "syndrome_evidence_summary": "string",
    "formula_evidence_summary": "string",
    "herb_evidence_summary": "string",
    "target_pathway_evidence_summary": "string",
    "cross_layer_evidence_summary": "string",
    "evidence_boundary_statement": "string",
    "evidence_source_summary": "string"
  },
  "assessment_level": 1,
  "cited_source_ids": ["source ID from allowed_source_ids"],
  "missing_evidence": ["string"],
  "boundary_statement": "Research evidence-traceability summary only."
}

The sections object must contain exactly the eight named keys. cited_source_ids must be a subset of allowed_source_ids. State missing evidence plainly rather than filling gaps. Do not use clinical efficacy, diagnosis, treatment recommendation, or prescription-appropriateness language.

Only mode g4 produces assessment_level. For modes g0, g1, g2, and g3, assessment_level must be null. For g4, independently derive the level from visible input according to these operational rules, without access to a precomputed result: level 4 only when the retrieved records show an explicit formula-level or cross-layer contradiction, such as a resolved formula with no indication_syndrome overlap with any physician-provided syndrome label; level 3 when the primary syndrome or formula record is unresolved; level 2 when the primary syndrome and formula are resolved and concordant but either core-herb mechanism coverage is below 0.60 or formula-composition mechanism coverage is below 0.80; and level 1 when the primary syndrome and formula are resolved and concordant, core-herb mechanism coverage is at least 0.60, and formula-composition mechanism coverage is at least 0.80. For G2--G4, formula_syndrome_relation is the authoritative retrieved fact for the formula-level relation. When has_any_indication_overlap is true, level 4 is prohibited: missing documentation for individual added herbs is an evidence-coverage boundary, not a formula-syndrome contradiction. A physician plan may include additions to a canonical formula. Additional herbs alone are not an explicit contradiction and must not cause level 4; describe any untraced additions as an evidence-coverage boundary instead. For modes g3 and g4, herb_mechanism_inventory is the authoritative inventory of plan herbs with and without retrieved mechanism records. Do not state that every plan herb has mechanism evidence when herbs_without_retrieved_mechanism_evidence is non-empty. Do not invent a source or a coverage count beyond this inventory. A herb record counts as a mechanism-evidence record when has_mechanism_evidence is true and source_record is present. Missing evidence_papers metadata must be disclosed in missing_evidence but does not itself change the operational level. Base the level on physician-plan fields and retrieved records, not on patient-text interpretation.

To preserve the research boundary, the value of patient_and_physician_plan_summary must be exactly this sentence: "The system evaluated physician-provided structured labels against retrieved evidence records; it did not generate a clinical diagnosis, treatment recommendation, or effectiveness conclusion." Do not use the word "patient" anywhere in any output value. Do not restate symptoms, demographics, examination findings, treatments, or study outcomes. Empty strings are invalid. If an evidence layer is not enabled or returns no records in a mode, write exactly: "No retrieved evidence was available in this ablation mode." in that section."""


NON_G4_SYSTEM_PROMPT = """You are an evidence-traceability reporting component for a research prototype.

You summarize a physician-provided plan and retrieved evidence records. You do not diagnose, assess treatment effectiveness, make a prescription recommendation, or introduce facts, source identifiers, targets, pathways, formulas, herbs, or syndromes not present in the input.

Return exactly one JSON object, with no Markdown fences and no prose outside the JSON. Use this exact schema:
{
  "sections": {
    "patient_and_physician_plan_summary": "string",
    "syndrome_evidence_summary": "string",
    "formula_evidence_summary": "string",
    "herb_evidence_summary": "string",
    "target_pathway_evidence_summary": "string",
    "cross_layer_evidence_summary": "string",
    "evidence_boundary_statement": "string",
    "evidence_source_summary": "string"
  },
  "cited_source_ids": ["source ID from allowed_source_ids"],
  "missing_evidence": ["string"],
  "boundary_statement": "Research evidence-traceability summary only."
}

The sections object must contain exactly the eight named keys. cited_source_ids must be a subset of allowed_source_ids. State missing evidence plainly rather than filling gaps. Do not use clinical efficacy, diagnosis, treatment recommendation, or prescription-appropriateness language.

This is a G0--G3 report. Do not emit an assessment_level field, numerical level, or four-level conclusion. Four-level assessment is available only in G4.

To preserve the research boundary, the value of patient_and_physician_plan_summary must be exactly this sentence: "The system evaluated physician-provided structured labels against retrieved evidence records; it did not generate a clinical diagnosis, treatment recommendation, or effectiveness conclusion." Do not use the word "patient" anywhere in any output value. Do not restate symptoms, demographics, examination findings, treatments, or study outcomes. Empty strings are invalid. If an evidence layer is not enabled or returns no records in a mode, write exactly: "No retrieved evidence was available in this ablation mode." in that section."""


def build_system_prompt(mode: str | None) -> str:
    return SYSTEM_PROMPT if mode == "g4" else NON_G4_SYSTEM_PROMPT


def build_user_prompt(context: dict[str, Any]) -> str:
    payload = {
        "task": "summarize_retrieved_evidence_for_a_physician_provided_plan",
        "mode": context.get("mode"),
        "assessment_scale": ASSESSMENT_SCALE,
        "operational_thresholds": {
            "core_evidence_coverage": 0.60,
            "strict_core_evidence_coverage": 0.80,
            "strict_formula_composition_coverage": 0.80,
        },
        "allowed_source_ids": context.get("allowed_source_ids", []),
        "input": {
            "physician_plan": context.get("physician_plan", {}),
            "rag_evidence": context.get("rag_evidence", {}),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
