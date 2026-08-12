# Paper-internal data boundary

The manuscript audit used a private, case-informed knowledge-base extension
and a restricted 50-plan audit package. These inputs are not the public demo
dataset and must not be uploaded without the required privacy and licensing
review.

The open-source default remains `data/demo/synthetic_cases.jsonl` with
`data/kb/`. An authorized local audit must point both the knowledge-base loader
and the experiment protocol to a separately stored private directory:

```powershell
$env:OP_RAG_PAPER_DATA_DIR = "C:\path\to\authorized\paper_internal"
$env:OP_RAG_DATA_DIR = "C:\path\to\authorized\paper_internal"
```

The private directory must contain the mechanism CSV resources expected by
`experiments/fair_rag_20260809_fixed_v2/protocol.py`. The case file may be
stored either at the private-directory root or at
`unified_cases/eval_cases_unified_001_050.jsonl`. The knowledge-base JSON
files selected by `OP_RAG_DATA_DIR` remain separate from the public default.
