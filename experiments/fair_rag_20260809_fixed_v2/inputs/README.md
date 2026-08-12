# Restricted paper inputs

The 50-plan input used for the manuscript audit is private and case-informed.
It is intentionally excluded from this open-source repository. The public
default is the synthetic demonstration under `data/demo/`.

For an authorized local audit, place the restricted file at the directory
specified by `OP_RAG_PAPER_DATA_DIR`, or provide the corresponding path before
running the protocol. Do not commit patient-level records, hidden reference
labels, restricted source documents, or raw API responses.

The tests under this experiment directory that check 50-case retrieval or
200-context reporting outputs are intentionally skipped when those private
artifacts are absent. This keeps the public repository test run green without
pretending that the manuscript audit can be reproduced from public files.
