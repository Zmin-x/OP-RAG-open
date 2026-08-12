from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENT = ROOT / "experiments" / "fair_rag_20260809_fixed_v2"
for import_root in (SRC, EXPERIMENT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from op_rag.loader import load_kb  # noqa: E402
from protocol import (  # noqa: E402
    CONFIGS,
    build_contexts,
    build_internal_reference,
    make_compact_records,
    read_jsonl,
    retrieval_metrics,
    write_json,
)


DEFAULT_CASES = ROOT / "data" / "demo" / "synthetic_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "public_demo_results.json"


def prepare_public_kb(kb: dict[str, Any]) -> dict[str, Any]:
    """Add explicit public-demo provenance without private mechanism CSV files."""
    prepared = deepcopy(kb)
    for herb in prepared.get("herbs", []):
        papers = [str(value) for value in herb.get("evidence_papers") or [] if value]
        qualified = bool(papers and herb.get("targets_op_related"))
        herb["mechanism_evidence_qualified"] = qualified
        herb["mechanism_source_ids"] = [f"PMID:{paper}" for paper in papers] if qualified else []
    return prepared


def run_demo(cases_path: Path = DEFAULT_CASES) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    kb = prepare_public_kb(load_kb())
    records = make_compact_records(kb)
    results: list[dict[str, Any]] = []

    for case in cases:
        reference = build_internal_reference(case, kb)
        contexts = build_contexts(reference, records)
        configurations: dict[str, Any] = {}
        for configuration in CONFIGS:
            context = contexts[configuration]
            configurations[configuration] = {
                "retrieval": retrieval_metrics(reference, context),
                "structured_audit": context["structured_audit"],
            }
        results.append(
            {
                "case_id": case["case_id"],
                "data_scope": case.get("data_scope"),
                "physician_plan": reference["physician_plan"],
                "configurations": configurations,
            }
        )

    return {
        "scope": "public_synthetic_method_demonstration_not_manuscript_results",
        "uses_qwen_api": False,
        "case_count": len(results),
        "configurations": list(CONFIGS),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OP-RAG locally on the four public synthetic plans without an API call."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = run_demo(args.cases)
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "scope": payload["scope"],
                "case_count": payload["case_count"],
                "configurations": payload["configurations"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
