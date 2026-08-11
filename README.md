<img
  align="right"
  src="assets/13428411883569884.png"
  alt="OP-RAG logo"
  width="240"
/>

<p align="left">
  <img src="https://img.shields.io/github/license/Zmin-x/2?style=flat-square&color=2F7D78&label=License" alt="MIT License">
  <img src="https://img.shields.io/badge/Python-3.13.6-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.13.6">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn 1.8.0">
  <img src="https://img.shields.io/github/stars/Zmin-x/2?style=flat-square&color=E3B341" alt="GitHub stars">
  <br>
  <img src="https://img.shields.io/badge/Model-Qwen--Plus-CF3A5B?style=flat-square" alt="Qwen-Plus">
  <img src="https://img.shields.io/badge/Field-TCM%20AI-9B59B6?style=flat-square" alt="TCM AI">
  <img src="https://img.shields.io/badge/Workflow-Three--layer%20RAG-168AAD?style=flat-square" alt="Three-layer RAG">
  <img src="https://img.shields.io/badge/Default%20data-Public%20demo-607D8B?style=flat-square" alt="Public demonstration data">
</p>

# OP-RAG Reproducibility Package

**A three-layer retrieval workflow for provenance and consistency auditing of traditional Chinese medicine prescriptions in primary osteoporosis.**

OP-RAG links syndrome, formula, and herb-target-pathway records. It retrieves source-linked evidence for a physician-provided regimen and applies predefined rules to summarize evidence coverage and cross-layer consistency.

> [!IMPORTANT]
> OP-RAG is a research workflow. It does not diagnose disease, generate prescriptions, assess treatment efficacy, or replace professional medical judgment.

<br clear="right">

## What is included

| Component | Included | Purpose |
| :--- | :---: | :--- |
| Three-layer public example knowledge base | Yes | Demonstrates syndrome, formula, and mechanism retrieval |
| Synthetic case set | Yes | Runs the public workflow without patient data |
| Deterministic evaluation code | Yes | Calculates evidence fields, coverage, missing evidence, relations, and audit levels |
| Qwen-Plus narration | Optional | Converts code-generated facts into a readable report |
| Aggregate manuscript results | Yes | Documents the reported results without releasing case-level records |
| Restricted 50-plan audit package | No | Withheld because it contains non-public research data |

## Data boundary

The default command uses the included synthetic cases and public example knowledge base:

```powershell
python scripts/run_ablation.py --no-llm
```

It reads `data/demo/synthetic_cases.jsonl` and `data/kb/`. These files reproduce the public software workflow, but they do not reproduce the manuscript's 50-plan internal audit.

The manuscript experiments used a separate private, case-informed knowledge-base extension and a restricted 50-plan audit package. These inputs are not distributed and are never selected by the default command. Their boundary and authorized local path configuration are documented in [`data/paper_internal/README.md`](data/paper_internal/README.md). Only aggregate, non-identifying manuscript outputs are provided in [`data/paper_results/`](data/paper_results/).

## Quick start

### 1. Create the environment

```powershell
conda env create -f environment.yml
conda activate op-rag-open
python --version
```

The recorded implementation used Python 3.13.6. When Conda is unavailable, install the minimal dependencies with:

```powershell
pip install -r requirements.txt
```

### 2. Run the public workflow

```powershell
python scripts/run_ablation.py --no-llm
```

By default, results are written under `outputs/demo_ablation/`:

```text
outputs/demo_ablation/
|-- g0/
|-- g1/
|-- g2/
|-- g3/
|-- g4/
`-- all_modes_summary.json
```

### 3. Run the optional test suite

```powershell
pip install pytest
python -m pytest -q
```

## Workflow

```mermaid
flowchart LR
    A[Physician-provided regimen] --> B[Syndrome evidence]
    A --> C[Formula evidence]
    A --> D[Herb-target-pathway evidence]
    B --> E[Deterministic consistency audit]
    C --> E
    D --> E
    E --> F[Structured evidence record]
    F -. optional .-> G[Qwen-readable narrative]

    style A fill:#EAF4F4,stroke:#2F7D78,color:#173F3C
    style B fill:#DCEEFF,stroke:#3776AB,color:#173F3C
    style C fill:#E6F4EA,stroke:#3C8C5A,color:#173F3C
    style D fill:#F1E6F7,stroke:#9B59B6,color:#173F3C
    style E fill:#FFF0DC,stroke:#F7931E,color:#173F3C
    style F fill:#E2F3F5,stroke:#168AAD,color:#173F3C
    style G fill:#F5F5F5,stroke:#607D8B,color:#263238
```

The public entry point retains the `G0-G4` workflow. The repaired manuscript experiment code also separates Flat-TFIDF, Layered-TFIDF, and Layered-Hybrid retrieval so that retrieval organization and matching strategy can be compared explicitly.

## Evaluation safeguards

The repaired implementation includes:

- exact standardized formula and herb resolution before fallback matching;
- rejection of zero-similarity herb results instead of returning arbitrary records;
- consistent retrieval limits and evidence-source exclusion rules;
- deterministic Python calculation of herb counts, coverage, missing evidence, relations, and Levels 1-4;
- Qwen-Plus restricted to qualitative narration of code-generated facts;
- numerical and semantic checks between structured outputs and generated summaries;
- separate Flat-TFIDF, Layered-TFIDF, and Layered-Hybrid retrieval protocols.

Qwen does not calculate the quantitative metrics. The fixed Python evaluator produces the structured results before any optional natural-language report is generated.

## Optional Qwen-Plus reports

Qwen-Plus is not required for the default reproducibility check. To enable optional report generation, configure the credentials only in the local environment:

```powershell
$env:QWEN_API_KEY = "your_api_key"
python scripts/run_ablation.py --use-qwen
```

Never commit API keys or local `.env` files. Provider-side model revisions may change wording, while the quantitative fields remain code-generated.

## Repository layout

```text
OP-RAG-open/
|-- assets/                 # Project logo and visual assets
|-- data/
|   |-- demo/               # Synthetic public demonstration cases
|   |-- kb/                 # Public example knowledge layers
|   |-- paper_internal/     # Boundary note for restricted inputs
|   `-- paper_results/      # Aggregate manuscript outputs
|-- experiments/            # Repaired manuscript experiment and audit code
|-- scripts/
|   `-- run_ablation.py     # Public workflow entry point
|-- src/                    # Retrieval, evaluation, prompting, and model client
|-- tests/                  # Public workflow tests
|-- CITATION.cff
`-- LICENSE
```

## Interpretation

Public-demo outputs show that the released code runs on the included example data. The manuscript results describe evidence coverage, provenance, and predefined consistency relations within a versioned internal resource. Neither result establishes diagnostic accuracy, prescription appropriateness, treatment efficacy, or clinical benefit.

## Citation

If you use this implementation, cite the associated OP-RAG manuscript. The bibliographic record in [`CITATION.cff`](CITATION.cff) should be updated when the final publication details become available.

## License

Project-owned code and documentation are released under the [MIT License](LICENSE). The project logo is distributed under the same license. Third-party data sources remain subject to their own terms and citation requirements.
