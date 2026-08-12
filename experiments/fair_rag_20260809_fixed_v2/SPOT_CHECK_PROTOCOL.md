# Pre-API spot-check protocol

The final 200-context Qwen run is prohibited until five different cases pass
consecutively after the 200-context local integration gate.

Each check uses a case selected without replacement by Python
`random.Random(20260810).shuffle` over the sorted 50 case IDs. It performs four
fresh API requests: Qwen-only, Flat RAG, Layered RAG, and OP-RAG. Old responses,
resume mode, and cross-run reuse are prohibited.

For every check, `run_api_spot_check.py` creates one numbered Markdown record
under `outputs/api_spot_check_gate_20260810/`. The record includes the visible
plan input, compact retrieved-record manifest, exact model payload, actual
model output, Python-assembled output, validation metadata, formulas, and
independent substituted calculations.

The spot check calls the shared per-record scoring functions directly. It does
not call the full-run aggregation or 200-key response-audit CLI, because those
commands intentionally summarize the full 50-case retrieval benchmark and are
not a valid summary container for one four-output case check.

A check fails if any configuration is missing or reused, any structured field
differs, the model emits a number or unsupported clause, provenance is invalid,
or an independently recalculated metric or audit level differs. A failure resets
the consecutive-pass counter to zero. The next check uses a new case.

The index file `SPOT_CHECK_INDEX.json` is the machine-readable gate. The final
run may start only when `formal_200_api_run_authorized` is `true`.
