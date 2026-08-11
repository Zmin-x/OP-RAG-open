from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation_runner import AblationCaseInput, AblationRunner  # noqa: E402
from src.config import FORMULA_ALIAS_MAP, HERB_ALIAS_MAP  # noqa: E402
from src.loader import load_kb  # noqa: E402
from prepare_corrected_cases import SYNDROME_ALIASES  # noqa: E402


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
# These inputs belong to the private paper audit and are intentionally not the
# default open-source dataset. The public entry point is scripts/run_ablation.py.
PAPER_DATA_DIR = Path(
    os.environ.get("OP_RAG_PAPER_DATA_DIR", str(ROOT / "data" / "paper_internal"))
).resolve()
CASES_PATH = PAPER_DATA_DIR / "eval_cases_unified_001_050.jsonl"
HERB_INTERSECTION_PATH = PAPER_DATA_DIR / "herb_targets_op_intersection_long.csv"
HERB_KEGG_PATH = PAPER_DATA_DIR / "herb_kegg_enrichment.csv"
CONFIGS = ("qwen_only", "flat_rag", "layered_rag", "op_rag")
ASSESSMENT_LABELS = {
    1: "complete_evidence_support",
    2: "partial_evidence_support",
    3: "insufficient_current_kb_evidence",
    4: "explicit_cross_layer_contradiction",
}

RELATION_SUPPORTED = "supported"
RELATION_INSUFFICIENT = "insufficient_evidence"
RELATION_INCONSISTENT = "cross_layer_inconsistency"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(value: Any, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def normalize_herb(name: str) -> str:
    value = re.sub(r"[（(].*?[）)]", "", str(name or "")).strip()
    return HERB_ALIAS_MAP.get(value, value)


def normalize_formula(name: str) -> str:
    value = re.sub(r"\s+", "", str(name or "")).strip()
    if value in FORMULA_ALIAS_MAP:
        return FORMULA_ALIAS_MAP[value]
    for alias, canonical in FORMULA_ALIAS_MAP.items():
        if alias and alias in value:
            return canonical
    return value


def runner_input(case: dict[str, Any], *, use_llm: bool = False) -> AblationCaseInput:
    allowed = set(AblationCaseInput.__dataclass_fields__)
    values = {key: value for key, value in case.items() if key in allowed}
    values["use_llm"] = use_llm
    return AblationCaseInput(**values)


def physician_plan(case: dict[str, Any]) -> dict[str, Any]:
    primary_name = case.get("primary_syndrome_name_std") or case.get("primary_syndrome_name_raw")
    secondary_names = case.get("secondary_syndrome_names_std") or case.get("secondary_syndrome_names_raw") or []
    return {
        "primary_syndrome_name": primary_name,
        "secondary_syndrome_names": unique_strings(secondary_names),
        "formula_name": case.get("reference_main_formula_name") or case.get("reference_formula_name_raw"),
        "herbs": unique_strings(case.get("reference_herb_set") or []),
    }


def normalize_syndrome_label(name: Any) -> str:
    return re.sub(r"[\s，。；：、（）()【】\[\]《》,.;:]+", "", str(name or ""))


def syndrome_id_from_visible_name(name: Any, kb: dict[str, Any]) -> str | None:
    normalized = normalize_syndrome_label(name)
    if not normalized:
        return None
    for record in kb.get("syndromes", []):
        canonical = normalize_syndrome_label(record.get("name"))
        if normalized == canonical:
            return str(record.get("syndrome_id"))
    for alias, syndrome_id in SYNDROME_ALIASES:
        if alias in normalized:
            return syndrome_id
    return None


def visible_syndrome_ids(case: dict[str, Any], kb: dict[str, Any]) -> list[str]:
    plan = physician_plan(case)
    names = [plan.get("primary_syndrome_name"), *(plan.get("secondary_syndrome_names") or [])]
    result: list[str] = []
    for name in names:
        syndrome_id = syndrome_id_from_visible_name(name, kb)
        if syndrome_id and syndrome_id not in result:
            result.append(syndrome_id)
    return result


def visible_primary_syndrome_id(case: dict[str, Any], kb: dict[str, Any]) -> str | None:
    plan = physician_plan(case)
    return syndrome_id_from_visible_name(plan.get("primary_syndrome_name"), kb)


def visible_secondary_syndrome_ids(case: dict[str, Any], kb: dict[str, Any]) -> list[str]:
    plan = physician_plan(case)
    return unique_strings(
        syndrome_id
        for name in (plan.get("secondary_syndrome_names") or [])
        if (syndrome_id := syndrome_id_from_visible_name(name, kb))
    )


def visible_case_for_runner(case: dict[str, Any], kb: dict[str, Any]) -> dict[str, Any]:
    """Remove pre-filled labels before deriving deterministic reference metrics."""
    prepared = deepcopy(case)
    primary_syndrome_id = visible_primary_syndrome_id(case, kb)
    secondary_syndrome_ids = visible_secondary_syndrome_ids(case, kb)
    syndrome_ids = unique_strings([primary_syndrome_id, *secondary_syndrome_ids])
    prepared["primary_syndrome_id"] = primary_syndrome_id
    prepared["reference_syndrome_id"] = None
    prepared["secondary_syndrome_ids"] = secondary_syndrome_ids
    prepared["accepted_syndrome_ids"] = syndrome_ids
    prepared["reference_formula_id"] = None
    return prepared


def source_document_id(case: dict[str, Any]) -> str:
    if case.get("source_group") == "hospital_real_case":
        return "HOSPITAL_COLLECTION"
    source = str(case.get("source_title") or case.get("source_ref") or case.get("case_id") or "unknown")
    digest = hashlib.sha256(source.strip().encode("utf-8")).hexdigest()[:16]
    return f"LITDOC:{digest}"


def source_ids_for_record(record: dict[str, Any], layer: str) -> list[str]:
    if layer == "syndrome":
        return unique_strings(record.get("references") or [])
    if layer == "formula":
        return unique_strings([*(record.get("references") or []), *(record.get("literature_source_ids") or [])])
    if layer == "herb":
        return unique_strings(record.get("mechanism_source_ids") or [])
    return []


def sources_after_exclusion(
    record: dict[str, Any], layer: str, excluded_source_ids: set[str] | None = None
) -> list[str]:
    excluded = excluded_source_ids or set()
    return [source for source in source_ids_for_record(record, layer) if source not in excluded]


def _csv_value_set(path: Path, column: str) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get(column) or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get(column) or "").strip()
        }


def qualify_mechanism_kb(kb: dict[str, Any]) -> dict[str, Any]:
    """Qualify mechanism records from the actual target-intersection pipeline."""
    qualified = deepcopy(kb)
    intersection_herbs = _csv_value_set(HERB_INTERSECTION_PATH, "herb_name")
    kegg_herbs = _csv_value_set(HERB_KEGG_PATH, "herb_name")
    for herb in qualified.get("herbs", []):
        herb_name = str(herb.get("herb_name") or "").strip()
        is_qualified = bool(
            herb_name in intersection_herbs
            and unique_strings(herb.get("targets_op_related") or [])
        )
        herb["mechanism_evidence_qualified"] = is_qualified
        herb["mechanism_source_ids"] = (
            [
                f"TCMSP_TARGETS:{herb_name}",
                f"OP_TARGET_INTERSECTION:{herb_name}",
                *([f"GPROFILER_KEGG:{herb_name}"] if herb_name in kegg_herbs else []),
            ]
            if is_qualified
            else []
        )
        if not is_qualified:
            herb["targets_op_related"] = []
            herb["pathways"] = []
    return qualified


def build_case_occurrence_provenance(
    cases: list[dict[str, Any]], kb: dict[str, Any]
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    runner = AblationRunner(kb)
    formula_literature_sources: dict[str, list[str]] = {}
    formula_occurrence_sources: dict[str, list[str]] = {}
    occurrence_rows: list[dict[str, Any]] = []
    document_rows: dict[str, dict[str, Any]] = {}
    valid_formula_ids = {str(item.get("formula_id")) for item in kb.get("formulas", [])}
    for case in cases:
        visible_case = visible_case_for_runner(case, kb)
        output = runner.run_case(runner_input(visible_case, use_llm=False), "g4")
        formula_id = output.context.get("case_results", {}).get("resolved_formula_id")
        if not formula_id:
            continue
        source_group = str(case.get("source_group") or "unknown")
        prefix = "HOSPITAL_RECORD" if source_group == "hospital_real_case" else "LITERATURE_CASE"
        occurrence_id = f"{prefix}:{case['case_id']}"
        formula_occurrence_sources.setdefault(str(formula_id), []).append(occurrence_id)
        document_id = source_document_id(case)
        if source_group != "hospital_real_case":
            formula_literature_sources.setdefault(str(formula_id), []).append(document_id)
            document_rows.setdefault(
                document_id,
                {
                    "source_id": document_id,
                    "source_group": source_group,
                    "source_title": compact_text(case.get("source_title") or case.get("source_ref"), 500),
                    "source_reference": compact_text(case.get("source_ref") or case.get("source_title"), 500),
                    "provenance_scope": "literature document used during formula-record curation",
                },
            )
        occurrence_rows.append(
            {
                "source_id": occurrence_id,
                "case_id": case["case_id"],
                "formula_id": formula_id,
                "source_group": source_group,
                "source_document_id": document_id,
                "source_title": (
                    "restricted de-identified hospital record"
                    if source_group == "hospital_real_case"
                    else compact_text(case.get("source_title") or case.get("source_ref"), 240)
                ),
                "provenance_scope": "formula occurrence in the internal case package; not treatment-efficacy evidence",
            }
        )
    return (
        {key: sorted(set(value)) for key, value in formula_literature_sources.items()},
        {key: sorted(set(value)) for key, value in formula_occurrence_sources.items()},
        occurrence_rows,
        [document_rows[key] for key in sorted(document_rows)],
    )


def add_formula_occurrence_provenance(
    kb: dict[str, Any], literature_sources: dict[str, list[str]], occurrence_sources: dict[str, list[str]]
) -> dict[str, Any]:
    augmented = deepcopy(kb)
    for formula in augmented.get("formulas", []):
        formula_id = str(formula.get("formula_id"))
        formula["literature_source_ids"] = list(literature_sources.get(formula_id, []))
        formula["case_occurrence_source_ids"] = list(occurrence_sources.get(formula_id, []))
    return augmented


def make_compact_records(
    kb: dict[str, Any], *, excluded_source_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    excluded_source_ids = excluded_source_ids or set()

    def usable_sources(item: dict[str, Any], layer: str) -> list[str]:
        return sources_after_exclusion(item, layer, excluded_source_ids)

    syndrome_names = {item.get("syndrome_id"): item.get("name") for item in kb.get("syndromes", [])}
    records: list[dict[str, Any]] = []
    for item in kb.get("syndromes", []):
        item_id = f"syndrome:{item.get('syndrome_id')}"
        records.append(
            {
                "item_id": item_id,
                "layer": "syndrome",
                "name": item.get("name"),
                "source_ids": usable_sources(item, "syndrome"),
                "attributes": {
                    "core_manifestations": list(item.get("core_symptoms") or [])[:8],
                    "tongue": item.get("tongue"),
                    "pulse": item.get("pulse"),
                },
                "summary": compact_text(item.get("text_description")),
            }
        )
    for item in kb.get("formulas", []):
        formula_sources = usable_sources(item, "formula")
        # A source-free formula record would expose content derived only from the
        # case's excluded document. Remove the whole record, not only its source ID.
        if not formula_sources:
            continue
        item_id = f"formula:{item.get('formula_id')}"
        indication_ids = unique_strings(item.get("indication_syndrome") or [])
        composition = unique_strings(
            entry.get("herb") for entry in (item.get("composition") or []) if isinstance(entry, dict)
        )
        records.append(
            {
                "item_id": item_id,
                "layer": "formula",
                "name": item.get("name"),
                "source_ids": formula_sources,
                "attributes": {
                    "indication_syndrome_ids": indication_ids,
                    "indication_syndrome_names": [syndrome_names.get(sid) for sid in indication_ids if syndrome_names.get(sid)],
                    "composition_herbs": composition[:24],
                },
                "summary": compact_text(item.get("text_description")),
            }
        )
    for item in kb.get("herbs", []):
        if not item.get("mechanism_evidence_qualified", bool(item.get("evidence_papers"))):
            continue
        item_id = f"herb:{item.get('herb_name')}"
        records.append(
            {
                "item_id": item_id,
                "layer": "herb",
                "name": item.get("herb_name"),
                "source_ids": usable_sources(item, "herb"),
                "attributes": {
                    "tcm_function": compact_text(item.get("tcm_function"), 120),
                    "target_count": len(item.get("targets_op_related") or []),
                    "target_examples": list(item.get("targets_op_related") or [])[:12],
                    "pathway_examples": list(item.get("pathways") or [])[:8],
                },
                "summary": compact_text(item.get("text_description")),
            }
        )
    for record in records:
        record["search_text"] = " ".join(
            [
                str(record.get("item_id") or ""),
                str(record.get("name") or ""),
                json.dumps(record.get("attributes") or {}, ensure_ascii=False),
                str(record.get("summary") or ""),
            ]
        )
    return records


class RecordRetriever:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False)
        self.matrix = self.vectorizer.fit_transform([record["search_text"] for record in records])

    def search(self, query: str, top_k: int, *, min_score: float = 0.0) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.records[index], "retrieval_score": round(float(scores[index]), 6)}
            for index in ranked
            if float(scores[index]) > min_score
        ]


def dedupe_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = str(record.get("item_id") or "")
        if not item_id:
            continue
        previous = by_id.get(item_id)
        if previous is None or float(record.get("retrieval_score") or 0.0) > float(previous.get("retrieval_score") or 0.0):
            by_id[item_id] = record
    return sorted(by_id.values(), key=lambda row: (-float(row.get("retrieval_score") or 0.0), row["item_id"]))


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"search_text"}}


def layered_retrieval(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    syndrome_top_k: int = 2,
    formula_top_k: int = 2,
    herb_top_k: int = 1,
) -> list[dict[str, Any]]:
    layers = {
        layer: RecordRetriever([record for record in records if record["layer"] == layer])
        for layer in ("syndrome", "formula", "herb")
    }
    selected: list[dict[str, Any]] = []
    syndrome_queries = unique_strings(
        [plan.get("primary_syndrome_name"), *(plan.get("secondary_syndrome_names") or [])]
    )
    for query in syndrome_queries:
        selected.extend(layers["syndrome"].search(query, top_k=syndrome_top_k))
    formula_query = str(plan.get("formula_name") or "").strip()
    if formula_query:
        selected.extend(layers["formula"].search(formula_query, top_k=formula_top_k))
    herb_records = {
        normalize_herb(str(record.get("name") or "")): record
        for record in records
        if record.get("layer") == "herb" and record.get("name")
    }
    for herb in unique_strings(plan.get("herbs") or []):
        if herb_top_k <= 0:
            continue
        normalized_herb = normalize_herb(herb)
        exact_record = herb_records.get(normalized_herb)
        if exact_record is not None:
            selected.append({**exact_record, "retrieval_score": 1.0})
    return [clean_record(record) for record in dedupe_records(selected)]


def flat_retrieval(plan: dict[str, Any], records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    query = " ".join(
        unique_strings(
            [
                plan.get("primary_syndrome_name"),
                *(plan.get("secondary_syndrome_names") or []),
                plan.get("formula_name"),
                *(plan.get("herbs") or []),
            ]
        )
    )
    return [clean_record(record) for record in RecordRetriever(records).search(query, top_k=top_k)]


def serialized_bytes(records: list[dict[str, Any]]) -> int:
    return len(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def classify_formula_syndrome_relation(
    *,
    case_syndrome_ids: Iterable[Any],
    formula_indication_ids: Iterable[Any],
    evidence_complete: bool,
) -> str:
    """Classify one relation without treating missing evidence as inconsistency."""
    case_ids = set(unique_strings(case_syndrome_ids))
    indication_ids = set(unique_strings(formula_indication_ids))
    if not evidence_complete or not case_ids or not indication_ids:
        return RELATION_INSUFFICIENT
    if case_ids & indication_ids:
        return RELATION_SUPPORTED
    return RELATION_INCONSISTENT


def build_assessment_inputs(
    *,
    relation_status: str,
    primary_relation_status: str,
    syndrome_evidence_available: bool,
    formula_evidence_available: bool,
    mechanism_evidence_available: bool,
    core_herb_coverage: float | None,
    formula_composition_coverage: float | None,
) -> dict[str, bool]:
    """Apply one evidence definition to every experimental configuration."""
    strict_coverage = bool(
        core_herb_coverage is not None
        and formula_composition_coverage is not None
        and core_herb_coverage >= 0.80
        and formula_composition_coverage >= 0.80
    )
    return {
        "contradiction": relation_status == RELATION_INCONSISTENT,
        "strict_support": bool(
            syndrome_evidence_available
            and formula_evidence_available
            and mechanism_evidence_available
            and relation_status == RELATION_SUPPORTED
            and primary_relation_status == RELATION_SUPPORTED
            and strict_coverage
        ),
        "syndrome_evidence_available": bool(syndrome_evidence_available),
        "formula_evidence_available": bool(formula_evidence_available),
        "mechanism_evidence_available": bool(mechanism_evidence_available),
    }


def determine_assessment_level(
    *,
    contradiction: bool,
    strict_support: bool,
    syndrome_evidence_available: bool,
    formula_evidence_available: bool,
    mechanism_evidence_available: bool,
) -> int:
    if contradiction:
        return 4
    if strict_support:
        return 1
    if (
        syndrome_evidence_available
        and formula_evidence_available
        and mechanism_evidence_available
    ):
        return 2
    return 3


def build_internal_reference(
    case: dict[str, Any],
    runner: AblationRunner,
    kb: dict[str, Any],
    *,
    excluded_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_source_ids = excluded_source_ids or set()
    visible_case = visible_case_for_runner(case, kb)
    output = runner.run_case(runner_input(visible_case, use_llm=False), "g4")
    results = output.context.get("case_results", {})
    syndrome_by_id = {item.get("syndrome_id"): item for item in kb.get("syndromes", [])}
    formula_by_id = {item.get("formula_id"): item for item in kb.get("formulas", [])}
    herb_by_name = {item.get("herb_name"): item for item in kb.get("herbs", [])}
    expected_claims: list[dict[str, Any]] = []
    expected_missing: list[str] = []
    expected_retrieval_item_ids: list[str] = []

    plan = physician_plan(case)
    syndrome_names = unique_strings(
        [
            plan.get("primary_syndrome_name"),
            *(plan.get("secondary_syndrome_names") or []),
        ]
    )
    all_syndromes_resolved = bool(syndrome_names) and all(
        syndrome_id_from_visible_name(name, kb) for name in syndrome_names
    )
    primary_syndrome_id = visible_primary_syndrome_id(case, kb)
    secondary_syndrome_ids = visible_secondary_syndrome_ids(case, kb)
    supplied_syndrome_ids = unique_strings(
        [primary_syndrome_id, *secondary_syndrome_ids]
    )
    for syndrome_id in supplied_syndrome_ids:
        record = syndrome_by_id.get(syndrome_id)
        syndrome_sources = source_ids_for_record(record or {}, "syndrome")
        if record and syndrome_sources:
            expected_retrieval_item_ids.append(f"syndrome:{syndrome_id}")
            expected_claims.append(
                {
                    "item_id": f"syndrome:{syndrome_id}",
                    "layer": "syndrome",
                    "expected_status": "supported",
                    "source_ids": syndrome_sources,
                }
            )
        else:
            expected_missing.append(f"syndrome:{syndrome_id}")
    if not supplied_syndrome_ids:
        expected_missing.append("syndrome:unresolved")

    formula_id = results.get("resolved_formula_id")
    formula_record = formula_by_id.get(formula_id)
    formula_composition_herbs = sorted(
        {
            normalize_herb(item.get("herb"))
            for item in ((formula_record or {}).get("composition") or [])
            if isinstance(item, dict) and item.get("herb")
        }
    )
    formula_sources = sources_after_exclusion(formula_record or {}, "formula", excluded_source_ids)
    if formula_record:
        if formula_sources:
            expected_retrieval_item_ids.append(f"formula:{formula_id}")
            expected_claims.append(
                {
                    "item_id": f"formula:{formula_id}",
                    "layer": "formula",
                    "expected_status": "supported",
                    "source_ids": formula_sources,
                }
            )
        else:
            expected_missing.append(f"formula:{formula_id}")
    else:
        expected_missing.append(f"formula:{normalize_formula(str(case.get('reference_main_formula_name') or case.get('reference_formula_name_raw') or 'unresolved'))}")

    for herb in unique_strings(normalize_herb(name) for name in (case.get("reference_herb_set") or [])):
        record = herb_by_name.get(herb)
        herb_sources = source_ids_for_record(record or {}, "herb")
        if (
            record
            and herb_sources
            and record.get("mechanism_evidence_qualified")
            and (record.get("targets_op_related") or record.get("pathways"))
        ):
            expected_retrieval_item_ids.append(f"herb:{herb}")
            expected_claims.append(
                {
                    "item_id": f"herb:{herb}",
                    "layer": "herb",
                    "expected_status": "supported",
                    "source_ids": herb_sources,
                }
            )
        else:
            expected_missing.append(f"herb:{herb}")

    indication_ids = unique_strings((formula_record or {}).get("indication_syndrome") or [])
    all_syndrome_evidence_available = bool(
        all_syndromes_resolved
        and supplied_syndrome_ids
        and all(
            (record := syndrome_by_id.get(syndrome_id))
            and source_ids_for_record(record, "syndrome")
            for syndrome_id in supplied_syndrome_ids
        )
    )
    primary_syndrome_record = syndrome_by_id.get(primary_syndrome_id)
    syndrome_evidence_available = bool(
        primary_syndrome_record
        and source_ids_for_record(primary_syndrome_record, "syndrome")
    )
    relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids=supplied_syndrome_ids,
        formula_indication_ids=indication_ids,
        evidence_complete=bool(
            formula_record and formula_sources and all_syndrome_evidence_available
        ),
    )
    primary_relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids=[primary_syndrome_id],
        formula_indication_ids=indication_ids,
        evidence_complete=bool(
            formula_record and formula_sources and syndrome_evidence_available
        ),
    )
    relation_formula_id = formula_id or normalize_formula(
        str(
            case.get("reference_main_formula_name")
            or case.get("reference_formula_name_raw")
            or "unresolved"
        )
    )
    relation_item_id = f"relation:{relation_formula_id}:syndrome"
    if relation_status in {RELATION_SUPPORTED, RELATION_INCONSISTENT}:
        expected_claims.append(
            {
                "item_id": relation_item_id,
                "layer": "cross_layer",
                "expected_status": (
                    "supported"
                    if relation_status == RELATION_SUPPORTED
                    else "contradiction"
                ),
                "source_ids": formula_sources,
            }
        )
    else:
        expected_missing.append(relation_item_id)

    formula_mapped = formula_record is not None
    formula_evidence_available = bool(formula_record and formula_sources)
    primary_consistent = results.get("formula_consistent_with_primary")
    any_consistent = results.get("formula_consistent_with_any_syndrome")
    mechanism_supported_plan_herbs = {
        claim["item_id"].split(":", 1)[1]
        for claim in expected_claims
        if claim.get("layer") == "herb"
    }
    mechanism_supported_formula_herbs = {
        herb
        for herb in formula_composition_herbs
        if (record := herb_by_name.get(herb))
        and source_ids_for_record(record, "herb")
        and record.get("mechanism_evidence_qualified")
        and bool(record.get("targets_op_related") or record.get("pathways"))
    }
    core_herbs = {
        normalize_herb(name) for name in (case.get("core_herbs") or []) if name
    }
    plan_herbs = {
        normalize_herb(name)
        for name in (case.get("reference_herb_set") or [])
        if name
    }
    core_cov = ratio_record(mechanism_supported_plan_herbs, core_herbs)["value"]
    formula_cov = ratio_record(
        mechanism_supported_formula_herbs, set(formula_composition_herbs)
    )["value"]
    has_mechanism = bool(mechanism_supported_plan_herbs & plan_herbs)
    chain_any = bool(
        syndrome_evidence_available
        and formula_evidence_available
        and relation_status == RELATION_SUPPORTED
        and has_mechanism
    )
    chain_core60 = (
        bool(core_cov is not None and core_cov >= 0.60)
        if chain_any
        else None
    )
    strict_inputs_present = bool(
        chain_any
        and primary_relation_status != RELATION_INSUFFICIENT
        and core_cov is not None
        and formula_cov is not None
    )
    chain_strict = (
        bool(
            primary_relation_status == RELATION_SUPPORTED
            and core_cov >= 0.80
            and formula_cov >= 0.80
        )
        if strict_inputs_present
        else None
    )
    assessment_inputs = build_assessment_inputs(
        relation_status=relation_status,
        primary_relation_status=primary_relation_status,
        syndrome_evidence_available=syndrome_evidence_available,
        formula_evidence_available=formula_evidence_available,
        mechanism_evidence_available=has_mechanism,
        core_herb_coverage=core_cov,
        formula_composition_coverage=formula_cov,
    )
    assessment_level = determine_assessment_level(**assessment_inputs)

    return {
        "case_id": case["case_id"],
        "source_group": case.get("source_group"),
        "physician_plan": plan,
        "primary_syndrome_id": primary_syndrome_id,
        "secondary_syndrome_ids": secondary_syndrome_ids,
        "case_syndrome_ids": supplied_syndrome_ids,
        "all_case_syndromes_resolved": all_syndromes_resolved,
        "all_syndrome_evidence_available": all_syndrome_evidence_available,
        "resolved_formula_id": formula_id,
        "core_herbs": sorted(
            set(normalize_herb(name) for name in (case.get("core_herbs") or []) if name)
        ),
        "formula_composition_herbs": formula_composition_herbs,
        "mechanism_supported_plan_herbs": sorted(mechanism_supported_plan_herbs),
        "mechanism_supported_formula_herbs": sorted(mechanism_supported_formula_herbs),
        "expected_claims": expected_claims,
        "expected_missing_items": sorted(set(expected_missing)),
        "expected_retrieval_item_ids": sorted(set(expected_retrieval_item_ids)),
        "expected_assessment_level": assessment_level,
        "expected_assessment_label": ASSESSMENT_LABELS[assessment_level],
        "formula_syndrome_relation_status": relation_status,
        "primary_formula_syndrome_relation_status": primary_relation_status,
        "expected_assessment_rule_inputs": assessment_inputs,
        "excluded_source_ids": sorted(excluded_source_ids),
        "source_cluster_id": source_document_id(case),
        "internal_case_metrics": {
            "primary_syndrome_resolved": bool(results.get("primary_syndrome_id")),
            "formula_mapped": results.get("formula_kb_status") == "mapped",
            "formula_source_supported_leave_one_source_out": formula_evidence_available,
            "primary_formula_concordance": results.get("formula_consistent_with_primary"),
            "any_formula_concordance": results.get("formula_consistent_with_any_syndrome"),
            "core_herb_annotation_coverage": results.get("core_herb_mechanism_coverage"),
            "reference_herb_annotation_coverage": results.get("mechanism_coverage_over_reference_herbs"),
            "formula_composition_annotation_coverage": results.get("formula_composition_mechanism_coverage"),
            "any_level_closure": chain_any,
            "core60_closure": chain_core60,
            "strict_closure": chain_strict,
            "assessment_level": assessment_level,
        },
        "reference_standard_scope": "versioned_internal_kb_and_predefined_rules_not_clinical_ground_truth",
    }


def ratio_record(supported: set[str], denominator_items: set[str]) -> dict[str, Any]:
    matched = sorted(supported & denominator_items)
    missing = sorted(denominator_items - supported)
    denominator = len(denominator_items)
    numerator = len(matched)
    return {
        "supported_items": matched,
        "missing_items": missing,
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def claim_visible(
    claim: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    expected_syndrome_item_ids: set[str],
) -> bool:
    item_id = str(claim.get("item_id") or "")
    record = records_by_id.get(item_id)
    if record is None and item_id.startswith("relation:"):
        parts = item_id.split(":")
        formula_id = parts[1] if len(parts) >= 3 else ""
        record = records_by_id.get(f"formula:{formula_id}")
    if record is None:
        return False
    visible_sources = set(str(value) for value in (record.get("source_ids") or []))
    expected_sources = set(str(value) for value in (claim.get("source_ids") or []))
    source_supported = bool(visible_sources & expected_sources)
    if not item_id.startswith("relation:"):
        return source_supported
    syndrome_evidence_complete = bool(expected_syndrome_item_ids) and all(
        syndrome_id in records_by_id
        and bool(records_by_id[syndrome_id].get("source_ids"))
        for syndrome_id in expected_syndrome_item_ids
    )
    return source_supported and syndrome_evidence_complete


def deterministic_claim(
    claim: dict[str, Any], records_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    item_id = str(claim["item_id"])
    status = str(claim["expected_status"])
    record = records_by_id.get(item_id)
    if record is None and item_id.startswith("relation:"):
        parts = item_id.split(":")
        formula_id = parts[1] if len(parts) >= 3 else ""
        record = records_by_id.get(f"formula:{formula_id}")
    visible_sources = set(str(value) for value in ((record or {}).get("source_ids") or []))
    expected_sources = set(str(value) for value in (claim.get("source_ids") or []))
    source_ids = sorted(visible_sources & expected_sources)
    wording = "supported by" if status == "supported" else "contradicted by"
    return {
        "item_id": item_id,
        "support_status": status,
        "source_ids": source_ids,
        "statement": f"{item_id} is {wording} the supplied source-linked evidence.",
    }


def build_structured_audit(
    reference: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Build a common evidence report and add consistency fields only for OP-RAG."""
    consistency_audit_applicable = context.get("configuration") == "op_rag"
    records_by_id = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    expected_claims = reference.get("expected_claims", [])
    auditable_claims = [
        claim
        for claim in expected_claims
        if consistency_audit_applicable or claim.get("layer") != "cross_layer"
    ]
    expected_syndrome_item_ids = {
        str(claim.get("item_id"))
        for claim in auditable_claims
        if claim.get("layer") == "syndrome"
    }
    visible_expected_claims = [
        claim
        for claim in auditable_claims
        if claim_visible(claim, records_by_id, expected_syndrome_item_ids)
    ]
    evidence_claims = [
        deterministic_claim(claim, records_by_id) for claim in visible_expected_claims
    ]

    missing_items = {
        str(value)
        for value in reference.get("expected_missing_items", [])
        if consistency_audit_applicable or not str(value).startswith("relation:")
    }
    missing_items.update(
        str(claim.get("item_id"))
        for claim in auditable_claims
        if not claim_visible(claim, records_by_id, expected_syndrome_item_ids)
    )

    plan_herbs = set(normalize_herb(name) for name in reference["physician_plan"].get("herbs", []) if name)
    core_herbs = set(normalize_herb(name) for name in reference.get("core_herbs", []) if name)
    formula_herbs = set(
        normalize_herb(name) for name in reference.get("formula_composition_herbs", []) if name
    )
    visible_mechanism_herbs = {
        normalize_herb(record.get("name"))
        for record in records_by_id.values()
        if record.get("layer") == "herb"
        and record.get("name")
        and record.get("source_ids")
    }
    plan_supported = visible_mechanism_herbs
    formula_supported = visible_mechanism_herbs

    coverage = {
        "physician_plan_herbs": ratio_record(plan_supported, plan_herbs),
        "core_herbs": ratio_record(plan_supported, core_herbs),
        "formula_composition_herbs": ratio_record(formula_supported, formula_herbs),
    }

    supported_layers = sorted(
        {
            str(claim.get("layer"))
            for claim in visible_expected_claims
            if claim.get("expected_status") == "supported"
        }
    )
    layer_names = {
        "syndrome": "syndrome",
        "formula": "formula",
        "herb": "herb mechanism",
        "cross_layer": "cross-layer relation",
    }
    readable_layers = [layer_names[layer] for layer in supported_layers]
    if readable_layers:
        layer_clause = "source-linked evidence is available for " + ", ".join(readable_layers)
    else:
        layer_clause = "no source-linked evidence layer is available"
    missing_clause = (
        "some evidence remains missing"
        if missing_items
        else "no evidence item remains missing"
    )
    if not consistency_audit_applicable:
        return {
            "generation_method": "deterministic_python",
            "audit_scope": "evidence_retrieval_and_coverage_only",
            "consistency_audit_applicable": False,
            "evidence_claims": evidence_claims,
            "missing_evidence_items": sorted(missing_items),
            "coverage_metrics": coverage,
            "assessment_level": None,
            "assessment_label": "not_applicable",
            "assessment_rule_trace": None,
            "formula_syndrome_relation": None,
            "narrative_facts": {
                "assessment_label": "evidence report",
                "supported_layers": supported_layers,
                "missing_evidence_present": bool(missing_items),
                "explicit_contradiction_present": None,
                "required_clauses": [layer_clause, missing_clause],
            },
        }

    expected_by_id = {
        str(claim.get("item_id")): claim for claim in expected_claims
    }
    primary_syndrome_id = reference.get("primary_syndrome_id")
    primary_syndrome_item_id = (
        f"syndrome:{primary_syndrome_id}" if primary_syndrome_id else None
    )
    primary_syndrome_claim = expected_by_id.get(primary_syndrome_item_id or "")
    syndrome_evidence_available = bool(
        primary_syndrome_claim
        and claim_visible(
            primary_syndrome_claim, records_by_id, expected_syndrome_item_ids
        )
    )
    syndrome_claims = [
        claim for claim in expected_claims if claim.get("layer") == "syndrome"
    ]
    all_syndrome_evidence_available = bool(
        reference.get("all_case_syndromes_resolved")
        and syndrome_claims
        and all(
            claim_visible(claim, records_by_id, expected_syndrome_item_ids)
            for claim in syndrome_claims
        )
    )

    resolved_formula_id = reference.get("resolved_formula_id")
    formula_item_id = f"formula:{resolved_formula_id}" if resolved_formula_id else None
    formula_claim = expected_by_id.get(formula_item_id or "")
    formula_evidence_available = bool(
        formula_claim
        and claim_visible(formula_claim, records_by_id, expected_syndrome_item_ids)
    )
    visible_formula_record = records_by_id.get(formula_item_id or "")
    indication_ids = unique_strings(
        ((visible_formula_record or {}).get("attributes") or {}).get(
            "indication_syndrome_ids"
        )
        or []
    )
    case_syndrome_ids = unique_strings(reference.get("case_syndrome_ids") or [])
    relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids=case_syndrome_ids,
        formula_indication_ids=indication_ids,
        evidence_complete=bool(
            formula_evidence_available and all_syndrome_evidence_available
        ),
    )
    primary_relation_status = classify_formula_syndrome_relation(
        case_syndrome_ids=[primary_syndrome_id],
        formula_indication_ids=indication_ids,
        evidence_complete=bool(
            formula_evidence_available and syndrome_evidence_available
        ),
    )
    has_mechanism = coverage["physician_plan_herbs"]["numerator"] > 0
    level_inputs = build_assessment_inputs(
        relation_status=relation_status,
        primary_relation_status=primary_relation_status,
        syndrome_evidence_available=syndrome_evidence_available,
        formula_evidence_available=formula_evidence_available,
        mechanism_evidence_available=has_mechanism,
        core_herb_coverage=coverage["core_herbs"]["value"],
        formula_composition_coverage=coverage["formula_composition_herbs"]["value"],
    )
    assessment_level = determine_assessment_level(**level_inputs)
    contradiction_present = relation_status == RELATION_INCONSISTENT

    assessment_label = ASSESSMENT_LABELS[assessment_level]
    contradiction_clause = (
        "an explicit contradiction is present"
        if contradiction_present
        else "no explicit contradiction is present"
    )
    return {
        "generation_method": "deterministic_python",
        "audit_scope": "evidence_retrieval_coverage_and_cross_layer_consistency",
        "consistency_audit_applicable": True,
        "evidence_claims": evidence_claims,
        "missing_evidence_items": sorted(missing_items),
        "coverage_metrics": coverage,
        "assessment_level": assessment_level,
        "assessment_label": assessment_label,
        "assessment_rule_trace": {
            "inputs": level_inputs,
            "triggered_rule": f"level_{assessment_level}",
        },
        "formula_syndrome_relation": {
            "status": relation_status,
            "primary_status": primary_relation_status,
            "case_syndrome_ids": case_syndrome_ids,
            "formula_indication_ids": indication_ids,
            "overlap_ids": sorted(set(case_syndrome_ids) & set(indication_ids)),
            "evidence_complete": bool(
                formula_evidence_available and all_syndrome_evidence_available
            ),
        },
        "narrative_facts": {
            "assessment_label": assessment_label.replace("_", " "),
            "supported_layers": supported_layers,
            "missing_evidence_present": bool(missing_items),
            "explicit_contradiction_present": contradiction_present,
            "required_clauses": [layer_clause, missing_clause, contradiction_clause],
        },
    }


def build_contexts(
    reference: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    syndrome_top_k: int = 2,
    formula_top_k: int = 2,
    herb_top_k: int = 1,
) -> dict[str, dict[str, Any]]:
    plan = reference["physician_plan"]
    layered = layered_retrieval(
        plan,
        records,
        syndrome_top_k=syndrome_top_k,
        formula_top_k=formula_top_k,
        herb_top_k=herb_top_k,
    )
    flat = flat_retrieval(plan, records, top_k=len(layered))
    base = {
        "task": "produce_a_source_traceable_evidence_audit_of_the_physician_recorded_plan",
        "physician_plan": plan,
        "assessment_scale": ASSESSMENT_LABELS,
        "reference_scope_notice": "The task evaluates support within supplied evidence, not diagnosis, efficacy, or prescription appropriateness.",
    }
    contexts = {
        "qwen_only": {
            **base,
            "configuration": "qwen_only",
            "evidence_context": [],
            "rule_context": None,
        },
        "flat_rag": {
            **base,
            "configuration": "flat_rag",
            "evidence_context": flat,
            "rule_context": None,
        },
        "layered_rag": {
            **base,
            "configuration": "layered_rag",
            "evidence_context": layered,
            "rule_context": None,
        },
        "op_rag": {
            **base,
            "configuration": "op_rag",
            "evidence_context": layered,
            "rule_context": None,
        },
    }
    for context in contexts.values():
        context["retrieval_budget"] = {
            "evidence_record_count": len(context["evidence_context"]),
            "serialized_evidence_bytes": serialized_bytes(context["evidence_context"]),
            "layered_reference_record_count": len(layered),
            "layered_reference_serialized_bytes": serialized_bytes(layered),
        }
        context["structured_audit"] = build_structured_audit(reference, context)
        if context["configuration"] == "op_rag":
            audit = context["structured_audit"]
            context["rule_context"] = {
                "generation_method": "deterministic_python",
                "rule_inputs": audit["assessment_rule_trace"]["inputs"],
                "formula_syndrome_relation": audit["formula_syndrome_relation"],
                "decision_rules": {
                    "level_4": "complete source-linked evidence shows no overlap between any physician-recorded syndrome and the formula indications",
                    "level_1": "the primary syndrome and target formula have source-linked evidence, the primary relation is supported, mechanism evidence is available, and both coverage values are at least 0.80",
                    "level_2": "the primary syndrome and target formula have source-linked evidence and at least one physician-plan herb has mechanism evidence, but level 1 is not met",
                    "level_3": "the supplied evidence does not meet level 1, 2, or 4",
                },
                "instruction": "Use the Python-computed structured audit; do not derive or modify its status, metrics, or level.",
            }
    return contexts


def evidence_item_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("item_id")) for record in records if record.get("item_id")}


def expected_retrievable_ids(reference: dict[str, Any]) -> set[str]:
    return set(reference.get("expected_retrieval_item_ids", []))


def retrieval_metrics(reference: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    expected = expected_retrievable_ids(reference)
    retrieved = evidence_item_ids(context.get("evidence_context", []))
    true_positive = expected & retrieved
    retrieval_applicable = context.get("configuration") != "qwen_only"
    precision = len(true_positive) / len(retrieved) if retrieval_applicable and retrieved else None
    recall = len(true_positive) / len(expected) if retrieval_applicable and expected else None
    return {
        "expected_item_count": len(expected),
        "retrieved_item_count": len(retrieved),
        "correct_item_count": len(true_positive),
        "evidence_retrieval_precision": precision,
        "evidence_retrieval_recall": recall,
        "missing_expected_item_ids": sorted(expected - retrieved),
        "extra_retrieved_item_ids": sorted(retrieved - expected),
        **context.get("retrieval_budget", {}),
    }


def strip_code_fence(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    value = json.loads(strip_code_fence(text))
    if not isinstance(value, dict):
        raise ValueError("Qwen response is not a JSON object")
    return value


def public_context(context: dict[str, Any]) -> dict[str, Any]:
    """Return the exact privacy-minimized model input used by the experiment."""
    return json.loads(json.dumps(context, ensure_ascii=False))
