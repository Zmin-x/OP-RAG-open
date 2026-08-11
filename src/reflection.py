from __future__ import annotations


def check_formula_consistency(
    syndrome_id: str,
    formula_id: str,
    syndrome_formula_map: dict[str, list[str]],
) -> dict:
    allowed_formula_ids = syndrome_formula_map.get(syndrome_id, [])
    is_consistent = formula_id in allowed_formula_ids

    return {
        "syndrome_id": syndrome_id,
        "formula_id": formula_id,
        "is_consistent": is_consistent,
        "allowed_formula_ids": allowed_formula_ids,
        "message": "证型-方剂一致" if is_consistent else "证型-方剂不一致，建议重选方剂",
    }


def choose_consistent_formula(
    syndrome_id: str,
    ranked_formulas: list[dict],
    syndrome_formula_map: dict[str, list[str]],
) -> dict | None:
    allowed_formula_ids = set(syndrome_formula_map.get(syndrome_id, []))
    for formula in ranked_formulas:
        if formula.get("formula_id") in allowed_formula_ids:
            return formula
    return ranked_formulas[0] if ranked_formulas else None
