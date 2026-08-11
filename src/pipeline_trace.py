"""Lightweight runtime tracing for RAG (data vs missing herb records)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("op_rag.pipeline")

_TRACE_CONFIGURED = False


def configure_pipeline_trace(*, enabled: bool = True, verbose: bool = False) -> None:
    """Enable stderr (+ optional file) warnings when herbs lack OP targets."""
    global _TRACE_CONFIGURED
    if not enabled:
        log.setLevel(logging.CRITICAL + 1)
        return
    if _TRACE_CONFIGURED:
        log.setLevel(logging.DEBUG if verbose else logging.WARNING)
        return
    log.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not log.handlers:
        fmt = logging.Formatter("%(levelname)s [%(name)s] %(message)s")
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        report_dir = Path(__file__).resolve().parents[1] / "data" / "pipeline_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(report_dir / "rag_runtime.log", encoding="utf-8", mode="a")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    _TRACE_CONFIGURED = True


def trace_herb_lookup(herb_name: str, herb_record: dict | None, *, formula_id: str = "") -> None:
    suffix = f" (formula_id={formula_id})" if formula_id else ""
    if herb_record is None:
        log.warning(
            "RAG_DATA: herb '%s' in formula but not in herbs.json — skipped mechanism block%s",
            herb_name,
            suffix,
        )
        return
    targets = herb_record.get("targets_op_related") or []
    if not targets:
        status = herb_record.get("data_status", "unknown")
        log.warning(
            "RAG_DATA: herb '%s' has empty targets_op_related (status=%s) — TCMSP∩disease gap, not retrieval bug%s",
            herb_name,
            status,
            suffix,
        )
