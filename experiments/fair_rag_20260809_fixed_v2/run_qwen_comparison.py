from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol import CONFIGS, OUTPUT_DIR, read_jsonl, sha256, write_json, write_jsonl
from qwen_protocol import GENERATION_ROLES, SYSTEM_PROMPT, FairQwenClient, validate_response


CONTEXTS_PATH = OUTPUT_DIR / "model_contexts.jsonl"
REFERENCE_PATH = OUTPUT_DIR / "internal_reference_set.jsonl"


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["case_id"]), str(record["configuration"])


def context_hash(context: dict[str, Any]) -> str:
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "qwen_comparison")
    parser.add_argument("--only-case")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-run-dir", type=Path)
    parser.add_argument("--reuse-contexts", type=Path)
    args = parser.parse_args()
    if bool(args.reuse_run_dir) != bool(args.reuse_contexts):
        raise SystemExit("--reuse-run-dir and --reuse-contexts must be supplied together")

    selected_configs = [value.strip() for value in args.configs.split(",") if value.strip()]
    unknown = sorted(set(selected_configs) - set(CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown configurations: {unknown}")
    contexts = read_jsonl(CONTEXTS_PATH)
    references = {row["case_id"]: row for row in read_jsonl(REFERENCE_PATH)}
    if args.only_case:
        contexts = [row for row in contexts if row["case_id"] == args.only_case]
    contexts = [row for row in contexts if row["configuration"] in selected_configs]
    contexts.sort(key=lambda row: (row["case_id"], CONFIGS.index(row["configuration"])))
    if not contexts:
        raise SystemExit("No model contexts matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "qwen_results.jsonl"
    existing = read_jsonl(results_path) if args.resume and results_path.exists() else []
    contexts_by_key = {record_key(row): row["context"] for row in contexts}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    discarded_resume_records = 0
    for row in existing:
        key = record_key(row)
        context = contexts_by_key.get(key)
        if context is None or row.get("context_sha256") != context_hash(context):
            discarded_resume_records += 1
            continue
        raw = {"assessment_summary": row.get("response", {}).get("assessment_summary")}
        validation = validate_response(row.get("response", {}), context, raw_model_response=raw)
        if not validation["valid"]:
            discarded_resume_records += 1
            continue
        row["validation"] = validation
        by_key[key] = row
    reused_keys: set[tuple[str, str]] = set()
    discarded_reuse_records = 0
    if args.reuse_run_dir and args.reuse_contexts:
        prior_results = {
            record_key(row): row
            for row in read_jsonl(args.reuse_run_dir / "qwen_results.jsonl")
        }
        prior_contexts = {
            record_key(row): row["context"]
            for row in read_jsonl(args.reuse_contexts)
        }
        for row in contexts:
            key = record_key(row)
            if key in by_key or key not in prior_results or key not in prior_contexts:
                continue
            if context_hash(row["context"]) != context_hash(prior_contexts[key]):
                discarded_reuse_records += 1
                continue
            reused = json.loads(json.dumps(prior_results[key], ensure_ascii=False))
            raw = {"assessment_summary": reused.get("response", {}).get("assessment_summary")}
            validation = validate_response(
                reused.get("response", {}), row["context"], raw_model_response=raw
            )
            if not validation["valid"]:
                discarded_reuse_records += 1
                continue
            reused["validation"] = validation
            reused["context_sha256"] = context_hash(row["context"])
            reused.setdefault("remote_metadata", {})["reused_from_identical_context"] = True
            by_key[key] = reused
            reused_keys.add(key)
    client = FairQwenClient()
    started_at = datetime.now(timezone.utc).isoformat()
    failures: list[dict[str, Any]] = []

    pending = [row for row in contexts if record_key(row) not in by_key]

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        case_id, configuration = record_key(row)
        context = row["context"]
        try:
            response, metadata, validation = client.request_validated(
                context, max_attempts=args.max_attempts
            )
            return {
                "ok": True,
                "result": {
                "case_id": case_id,
                "source_group": references[case_id].get("source_group"),
                "configuration": configuration,
                "response": response,
                "validation": validation,
                "remote_metadata": metadata,
                "context_sha256": context_hash(context),
                },
            }
        except Exception as exc:  # fail closed, with a resumable log
            return {
                "ok": False,
                "failure": {
                    "case_id": case_id,
                    "configuration": configuration,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": datetime.now(timezone.utc).isoformat(),
                },
            }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_rows = {executor.submit(run_one, row): row for row in pending}
        for future in as_completed(future_rows):
            outcome = future.result()
            if outcome["ok"]:
                result = outcome["result"]
                key = record_key(result)
                by_key[key] = result
                validation = result["validation"]
                print(
                    f"{result['case_id']} {result['configuration']}: "
                    f"schema_valid={validation.get('schema_valid', validation.get('valid'))} "
                    f"provenance_clean={validation.get('provenance_clean')}",
                    flush=True,
                )
            else:
                failure = outcome["failure"]
                failures.append(failure)
                print(f"FAILED {failure['case_id']} {failure['configuration']}: {failure['error']}", flush=True)
            write_jsonl(results_path, [by_key[item] for item in sorted(by_key)])
            write_json(args.output_dir / "failures.json", failures)

    ordered = [by_key[item] for item in sorted(by_key)]
    result_keys = {(row.get("case_id"), row.get("configuration")) for row in ordered}
    configuration_counts = Counter(str(row.get("configuration")) for row in ordered)
    summary = {
        "status": "completed" if len(ordered) == len(contexts) else "partial",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_expected": len(contexts),
        "n_completed": len(ordered),
        "n_valid": sum(row.get("validation", {}).get("valid") is True for row in ordered),
        "n_unique_case_configuration_keys": len(result_keys),
        "configuration_counts": dict(configuration_counts),
        "n_reused_identical_context": len(reused_keys),
        "n_requested_in_this_run": len(pending),
        "configurations": selected_configs,
        "model": client.model,
        "contexts_sha256": sha256(CONTEXTS_PATH),
        "reference_sha256": sha256(REFERENCE_PATH),
        "patient_narrative_sent_to_model": False,
        "structured_audit_fields_generated_by": "deterministic_python",
        "qwen_role": "verbalize_assessment_summary_without_calculation",
        "output_contract_version": "deterministic_structured_audit_qwen_narration_v1",
        "generation_roles": GENERATION_ROLES,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "results_sha256": sha256(results_path),
        "discarded_invalid_or_stale_resume_records": discarded_resume_records,
        "discarded_invalid_or_stale_reuse_records": discarded_reuse_records,
        "no_old_result_reuse": len(reused_keys) == 0 and not args.reuse_run_dir,
        "failures": failures,
    }
    write_json(args.output_dir / "run_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
