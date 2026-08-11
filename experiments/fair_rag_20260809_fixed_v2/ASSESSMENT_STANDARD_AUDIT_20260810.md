# Assessment-standard audit after the uniform-input fix

Date: 2026-08-10

Scope: experiment code and locally generated structured contexts only. This is not a clinical-validity audit and does not update the manuscript.

## Corrected rule boundary

All four configurations now use the same five level inputs. Each input is derived only from the source-linked evidence visible to that configuration:

- `syndrome_evidence_available`: the target primary-syndrome record and an allowed source are visible.
- `formula_evidence_available`: the target formula record retains an allowed source after case-source exclusion and is visible.
- `mechanism_evidence_available`: at least one physician-plan herb has a visible, source-linked mechanism record.
- `contradiction`: complete syndrome and formula evidence is available and no physician-recorded syndrome overlaps the formula indications.
- `strict_support`: the primary relation is supported and both core-herb and formula-composition coverage are at least 0.80.

Missing or incomplete evidence cannot trigger contradiction. A formula that supports a recorded secondary syndrome is not treated as contradictory merely because it does not support the primary syndrome.

## Verification

- 32/32 local unit and integration tests passed.
- 200/200 locally generated case-configuration contexts passed the pre-API audit.
- All 200 level-input records matched an independent recomputation from visible evidence.
- All 50 Layered/OP-RAG pairs used identical evidence, coverage, relation states, level inputs, and levels.
- Case 18 now yields Level 3 under both Layered and OP-RAG because F013 maps by name but has no independent source after case-source exclusion.

## Current local level distribution

| Configuration | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|
| Qwen-only | 0 | 0 | 50 | 0 |
| Flat RAG | 0 | 10 | 40 | 0 |
| Layered RAG | 0 | 10 | 40 | 0 |
| OP-RAG | 0 | 10 | 40 | 0 |

## Remaining experimental-design issue

Layered and OP-RAG currently receive identical evidence and both are passed through the deterministic structured-audit builder. Their levels are therefore identical by design. The corrected results cannot be used to claim that the OP-RAG consistency module improves the level relative to Layered RAG. The consistency module must instead be validated with controlled supported, insufficient-evidence, and cross-layer-inconsistency fixtures, or the non-OP configurations must stop exposing a consistency result and use `N/A` for that output.

The 50-case package contains no observed Level-4 case. Controlled fixtures prove rule execution, not the clinical validity of a contradiction label.

