from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


# These aliases are deliberately kept in the experiment package so that the
# visible-name standardization can be inspected and hashed with the run.
SYNDROME_ALIASES: list[tuple[str, str]] = [
    ("肝肾阴虚", "S2"),
    ("肝肾亏虚", "S2"),
    ("肝肾阴亏", "S2"),
    ("肝郁肾虚", "S2"),
    ("肝肾不足", "S2"),
    ("肾虚髓亏", "S2"),
    ("肾精虚", "S2"),
    ("肾阴虚", "S2"),
    ("脾肾阳虚", "S3"),
    ("脾肾亏虚", "S3"),
    ("脾肾亏损", "S3"),
    ("脾肾同虚", "S3"),
    ("肾虚髓减", "S3"),
    ("脾弱精衰", "S3"),
    ("脾阳不足", "S3"),
    ("肾阴阳两虚", "S3"),
    ("肾虚血瘀", "S4"),
    ("肾虚夹瘀", "S4"),
    ("瘀血阻络", "S4"),
    ("气滞血瘀", "S6"),
    ("血瘀气滞", "S6"),
    ("脾胃虚弱", "S5"),
    ("脾气虚", "S5"),
    ("脾虚", "S5"),
    ("肾阳亏虚", "S1"),
    ("肾阳虚损", "S1"),
    ("肾阳虚", "S1"),
]

CANONICAL_NAMES = {
    "S1": "肾阳虚证",
    "S2": "肝肾阴虚证",
    "S3": "脾肾阳虚证",
    "S4": "肾虚血瘀证",
    "S5": "脾胃虚弱证",
    "S6": "血瘀气滞证",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def clean(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()【】\[\]《》,.;:]+", "", text or "")


def line_priority(line: str) -> int:
    return 0 if re.search(r"证型|辨证|辨证分型|中医诊断|诊断调整|证候", line) else 1


def extract_candidates(text: str) -> list[dict[str, Any]]:
    """Extract explicit syndrome phrases from the case text only."""
    candidates: list[dict[str, Any]] = []
    aliases = sorted(SYNDROME_ALIASES, key=lambda pair: len(pair[0]), reverse=True)
    for position, line in enumerate((text or "").splitlines()):
        compact = clean(line)
        if not compact:
            continue
        for alias, syndrome_id in aliases:
            if alias in compact:
                candidates.append(
                    {
                        "position": position,
                        "line": line.strip(),
                        "alias": alias,
                        "syndrome_id": syndrome_id,
                        "priority": line_priority(line),
                    }
                )
    # Keep the first occurrence of each category. Summary and diagnosis lines
    # are preferred over later explanatory prose.
    unique: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda row: (row["priority"], row["position"], -len(row["alias"]))):
        unique.setdefault(candidate["syndrome_id"], candidate)
    return sorted(unique.values(), key=lambda row: (row["position"], -len(row["alias"])))


def visible_names(case: dict[str, Any]) -> tuple[str | None, list[str], list[dict[str, Any]]]:
    text = str(case.get("patient_text") or "")
    candidates = extract_candidates(text)
    if case.get("primary_syndrome_name_raw"):
        primary_raw = str(case["primary_syndrome_name_raw"])
        primary_name = primary_raw
        secondary_names = [
            str(value) for value in (case.get("secondary_syndrome_names_raw") or []) if str(value).strip()
        ]
        return primary_name, secondary_names, candidates
    diagnostic_candidates = [row for row in candidates if row["priority"] == 0]
    ordered = diagnostic_candidates or candidates
    primary = ordered[0] if ordered else None
    primary_name = primary["alias"] if primary else None
    primary_id = None
    if primary_name:
        normalized = clean(primary_name)
        for alias, syndrome_id in SYNDROME_ALIASES:
            if alias in normalized:
                primary_id = syndrome_id
                break
    secondary: list[str] = []
    for candidate in ordered:
        if primary is None or candidate["position"] != primary["position"]:
            continue
        if candidate.get("syndrome_id") != primary_id:
            name = candidate["alias"]
            if name not in secondary:
                secondary.append(name)
    return primary_name, secondary, candidates


def correct_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    corrected = deepcopy(case)
    original = {
        "primary_syndrome_id": case.get("primary_syndrome_id"),
        "primary_syndrome_name_raw": case.get("primary_syndrome_name_raw"),
        "secondary_syndrome_ids": case.get("secondary_syndrome_ids") or [],
        "secondary_syndrome_names_raw": case.get("secondary_syndrome_names_raw") or [],
        "reference_formula_id": case.get("reference_formula_id"),
    }
    primary_name, secondary_names, candidates = visible_names(case)
    corrected["original_structured_syndrome_id"] = original["primary_syndrome_id"]
    corrected["original_structured_secondary_syndrome_ids"] = original["secondary_syndrome_ids"]
    corrected["primary_syndrome_name_raw"] = primary_name
    corrected["primary_syndrome_name_std"] = CANONICAL_NAMES.get(
        next((sid for alias, sid in SYNDROME_ALIASES if alias in clean(primary_name or "")), "")
    )
    corrected["primary_syndrome_id"] = next(
        (sid for alias, sid in SYNDROME_ALIASES if alias in clean(primary_name or "")), None
    )
    corrected["secondary_syndrome_names_raw"] = secondary_names
    corrected["secondary_syndrome_ids"] = [
        next((sid for alias, sid in SYNDROME_ALIASES if alias in clean(name)), None)
        for name in secondary_names
    ]
    corrected["secondary_syndrome_ids"] = [sid for sid in corrected["secondary_syndrome_ids"] if sid]
    corrected["accepted_syndrome_ids"] = [
        sid for sid in [corrected.get("primary_syndrome_id"), *corrected["secondary_syndrome_ids"]] if sid
    ]
    corrected["reference_syndrome_id"] = corrected.get("primary_syndrome_id")
    # Formula names, rather than pre-filled formula IDs, are the visible input.
    corrected["original_reference_formula_id"] = original["reference_formula_id"]
    corrected["reference_formula_id"] = None
    audit = {
        "case_id": case.get("case_id"),
        "source_group": case.get("source_group"),
        "source_title": case.get("source_title"),
        "original": original,
        "visible_primary_name": primary_name,
        "visible_primary_id": corrected.get("primary_syndrome_id"),
        "visible_secondary_names": secondary_names,
        "visible_secondary_ids": corrected.get("secondary_syndrome_ids"),
        "source_candidates": candidates,
        "primary_id_changed": original["primary_syndrome_id"] != corrected.get("primary_syndrome_id"),
        "primary_label_recovered_from_case_text": not bool(original["primary_syndrome_name_raw"]) and bool(primary_name),
        "visible_selection_rule": "first_distinct_standardized_category_on_earliest_explicit_diagnostic_line",
        "hidden_structured_id_used_for_selection": False,
        "status": "resolved" if primary_name else "unresolved_visible_syndrome",
    }
    return corrected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    cases = read_jsonl(args.input)
    corrected: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for case in cases:
        item, audit = correct_case(case)
        corrected.append(item)
        audits.append(audit)
    if len({row.get("case_id") for row in corrected}) != len(corrected):
        raise SystemExit("case IDs are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, corrected)
    args.audit.write_text(json.dumps({"n_cases": len(audits), "cases": audits}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_cases": len(corrected),
        "resolved_primary": sum(row["status"] == "resolved" for row in audits),
        "unresolved_primary": sum(row["status"] != "resolved" for row in audits),
        "primary_id_changed": sum(row["primary_id_changed"] for row in audits),
        "output": str(args.output),
        "audit": str(args.audit),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
