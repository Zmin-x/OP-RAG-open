from __future__ import annotations

from typing import Any

from protocol import (
    ASSESSMENT_LABELS,
    build_assessment_inputs,
    classify_formula_syndrome_relation,
    determine_assessment_level,
    normalize_herb,
)


def _source_supported(
    item_id: str | None,
    records: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> bool:
    if not item_id or item_id not in records or item_id not in expected:
        return False
    visible_sources = set(str(value) for value in records[item_id].get("source_ids") or [])
    expected_sources = set(str(value) for value in expected[item_id].get("source_ids") or [])
    return bool(visible_sources & expected_sources)


def _coverage_record(supported: set[str], denominator: set[str]) -> dict[str, Any]:
    present = sorted(supported & denominator)
    missing = sorted(denominator - supported)
    return {
        "numerator": len(present),
        "denominator": len(denominator),
        "value": len(present) / len(denominator) if denominator else None,
        "supported_items": present,
        "missing_items": missing,
    }


def build_evaluation_only_assessment(
    reference: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate evidence sufficiency without exposing the result to the model."""
    records = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    expected = {
        str(claim.get("item_id")): claim
        for claim in reference.get("expected_claims", [])
    }

    primary_id = reference.get("primary_syndrome_id")
    syndrome_available = _source_supported(
        f"syndrome:{primary_id}" if primary_id else None, records, expected
    )
    syndrome_claim_ids = [
        item_id
        for item_id, claim in expected.items()
        if claim.get("layer") == "syndrome"
    ]
    all_syndromes_available = bool(
        reference.get("all_case_syndromes_resolved")
        and syndrome_claim_ids
        and all(_source_supported(item_id, records, expected) for item_id in syndrome_claim_ids)
    )

    formula_id = reference.get("resolved_formula_id")
    formula_item_id = f"formula:{formula_id}" if formula_id else None
    formula_available = _source_supported(formula_item_id, records, expected)
    formula_record = records.get(formula_item_id or "") or {}
    indication_ids = {
        str(value)
        for value in ((formula_record.get("attributes") or {}).get("indication_syndrome_ids") or [])
        if value
    }
    case_ids = {
        str(value) for value in reference.get("case_syndrome_ids") or [] if value
    }
    relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids=case_ids,
        formula_indication_ids=indication_ids,
        evidence_complete=bool(formula_available and all_syndromes_available),
    )
    primary_relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids={str(primary_id)} if primary_id else set(),
        formula_indication_ids=indication_ids,
        evidence_complete=bool(formula_available and syndrome_available),
    )

    plan_herbs = {
        normalize_herb(value)
        for value in reference.get("physician_plan", {}).get("herbs", [])
        if value
    }
    visible_mechanism_herbs = {
        normalize_herb(record.get("name"))
        for record in records.values()
        if record.get("layer") == "herb"
        and record.get("name")
        and record.get("source_ids")
    }
    core_herbs = {
        normalize_herb(value) for value in reference.get("core_herbs", []) if value
    }
    formula_herbs = {
        normalize_herb(value)
        for value in reference.get("formula_composition_herbs", [])
        if value
    }
    coverage = {
        "physician_plan_herbs": _coverage_record(visible_mechanism_herbs, plan_herbs),
        "core_herbs": _coverage_record(visible_mechanism_herbs, core_herbs),
        "formula_composition_herbs": _coverage_record(
            visible_mechanism_herbs, formula_herbs
        ),
    }
    mechanism_available = coverage["physician_plan_herbs"]["numerator"] > 0
    inputs = build_assessment_inputs(
        relation_status=relation_status,
        primary_relation_status=primary_relation_status,
        syndrome_evidence_available=syndrome_available,
        formula_evidence_available=formula_available,
        mechanism_evidence_available=mechanism_available,
        core_herb_coverage=coverage["core_herbs"]["value"],
        formula_composition_coverage=coverage["formula_composition_herbs"]["value"],
    )
    level = determine_assessment_level(**inputs)
    return {
        "case_id": reference["case_id"],
        "configuration": context["configuration"],
        "scope": "experiment_only_uniform_evidence_support_level_not_system_output",
        "sent_to_qwen": False,
        "input_definition": "same source-linked evidence fields and thresholds for all configurations",
        "coverage_metrics": coverage,
        "assessment_level": level,
        "assessment_label": ASSESSMENT_LABELS[level],
        "assessment_rule_trace": {
            "inputs": inputs,
            "triggered_rule": f"level_{level}",
        },
        "formula_syndrome_relation": {
            "status": relation_status,
            "primary_status": primary_relation_status,
            "case_syndrome_ids": sorted(case_ids),
            "formula_indication_ids": sorted(indication_ids),
            "overlap_ids": sorted(case_ids & indication_ids),
            "evidence_complete": bool(formula_available and all_syndromes_available),
        },
    }
