# OP-RAG AutoResearch objective

## Objective

Build and verify a fail-closed OP-RAG reporting pipeline. Python must be the
single source of truth for evidence-claim counts, supported and missing herbs,
plan-herb coverage, core-herb coverage, formula-composition coverage,
missing-evidence items, assessment levels, and rule traces. Qwen may only turn
the structured result into a qualitative explanation. It must not calculate,
copy, infer, or overwrite any structured value.

Only after all local gates pass may a fresh 200-context API experiment begin.
The new run must not reuse any response from an earlier run. No manuscript,
LaTeX source, paper table, or paper figure may be changed without explicit user
permission.

### Mandatory pre-API spot-check gate added on 2026-08-10

Before the final 200-context API run, five different cases must pass in
consecutive order. Cases are selected without replacement under a recorded
fixed seed. Each spot check runs all four configurations and creates its own
numbered Markdown report containing the visible input fields, retrieved
records, the exact Qwen payload, the raw Qwen field, the code-assembled output,
the validation path, formulas, and substituted metric calculations.

If a spot check reveals any code, data, arithmetic, semantic, provenance, or
reporting defect, the defect must be repaired and regression-tested. The
consecutive-pass counter then returns to zero, and five new distinct cases must
pass before the final run. A case cannot be used twice within the same gate.
Any 200-context run completed before this gate was introduced is retained as a
provisional engineering record only and cannot serve as the final run.

### Non-negotiable numeric and semantic consistency outcome

The final 200-context run is also prohibited until both numeric ownership and
semantic consistency are demonstrated. Evidence-claim counts, herb counts,
all coverage numerators and denominators, missing-evidence items, and audit
levels must come only from Python. Qwen must receive no numeric audit fields and
must return only a qualitative `assessment_summary`. The validator must reject
numeric text, semantic contradiction, unsupported wording, changed structured
fields, invalid sources, and clinical claims.

After the final run, all 200 model reports must be revalidated against their
current structured audits. The scored result, publication-value registry,
generated table source files, and generated figure source files must share the
same final-run hashes and values. The existing manuscript tables and figures
must then be checked against that registry. This check does not authorize
editing the manuscript; any required manuscript change remains subject to the
user's explicit permission.

### Mandatory post-run 10-case audit added on 2026-08-10

After the fresh 200-context Qwen run, ten different cases must be selected
without replacement under a recorded fixed seed. To broaden coverage, cases
used by the pre-run spot-check gate are excluded from this draw. Each selected
case must be checked across Qwen-only, Flat RAG, Layered RAG, and OP-RAG using
the retained formal-run responses. This stage must not call the API again or
replace a formal response.

Each case must have a detailed Markdown record containing the visible input
fields, retrieved records, exact Qwen input and output fields, Python-assembled
output, validation metadata, processing steps, formulas, and substituted
calculations for every reported metric. All ten cases, all forty
case-configuration records, report hashes, the formal result hash, and the
zero-reuse declaration must pass before the experiment can be completed.

## Problems this objective must eliminate

1. Confusion between the number of evidence claims, the number of herbs with
   mechanism evidence, core-herb coverage, and formula-composition coverage.
2. Narrative numbers or assessment wording that disagree with the structured
   Python result.
3. A model response silently changing evidence items, missing items, coverage,
   or the audit level.
4. Stale, invalid, duplicated, or reused API responses entering a formal run.
5. Paper table or figure values being generated from a different result set.

## Fixed protocol

- Input package: 50 standardized plans and four configurations, for 200 unique
  case-configuration contexts.
- Structured computation: deterministic Python code only.
- Model role: one non-numeric qualitative `assessment_summary` based on the
  code-generated narration facts.
- Formal API settings: archived prompt and schema, `temperature=0`, fixed
  retrieval settings, and no reuse of prior responses.
- Experiment interpretation: internal engineering audit, not clinical
  validation, diagnostic evaluation, efficacy testing, or prescription advice.

## Acceptance gates

1. Every structured field has one deterministic Python source of truth.
2. Qwen cannot overwrite a structured field and its accepted output contains
   only `assessment_summary`.
3. Narrative validation rejects digits, percentages, fractions, number words,
   semantic contradictions, unsupported clauses, source IDs, and clinical or
   recommendation claims.
4. Provenance validation rejects invalid or out-of-context source IDs.
5. Unit, regression, and full 200-context local integration tests all pass.
6. The `case_018` OP-RAG regression remains fixed at: plan herbs 9/13, core
   herbs 5/8, formula composition 9/13, 10 evidence claims, 5 missing items,
   and Level 2.
7. The pre-API audit passes every context before any formal API call.
8. Five different case-level, four-configuration API spot checks pass
   consecutively and each has a numbered Markdown audit record.
9. The formal run contains exactly 200 unique, current-prompt, valid API
   outputs and reuses zero old responses.
10. Deterministic rescoring, response-consistency audit, experiment-integrity
    audit, publication-value registry, and publication-input audit all pass.
11. Ten different post-run cases pass a detailed four-configuration audit,
    with one numbered Markdown record per case and no additional API calls.
12. A no-API release verification, including sensitivity and error analyses,
    reproduces the retained artifacts.

## Controlled iteration plan

1. Establish the repaired local baseline and record every test result.
2. Attack one failure mode at a time with an explicit regression test.
3. Re-run the complete local gate after every accepted repair.
4. Run distinct case-level spot checks until five consecutive cases pass; reset
   the pass counter after any discovered defect.
5. Start the fresh 200-context API run only after the complete local gate and
   five-case spot-check gate pass.
6. Reject invalid responses and record retry/failure details; never substitute
   an old response.
7. Audit the 200 outputs, score them deterministically, and generate one
   synchronized value registry for downstream tables and figures.
8. Audit ten different retained formal-run cases and archive their full inputs,
   outputs, formulas, substituted calculations, and verdicts.
9. Run release verification and write a final experiment report. Do not modify
   the manuscript.

## Stop conditions

The experiment is not complete if the five-case consecutive spot-check gate has
not passed, any other gate fails, any output is reused, fewer than 200 unique
valid outputs exist, the ten-case post-run audit is incomplete, a narrative
contradicts its structured result, or publication values do not trace to the
final run registry.
