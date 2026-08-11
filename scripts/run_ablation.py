from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from op_rag.ablation_runner import AblationCaseInput, AblationRunner
from op_rag.loader import load_kb


@dataclass
class Result:
    case_id: str
    mode: str
    formula_kb_status: str | None
    primary_syndrome_id: str | None
    herb_jaccard: float | None
    core_herb_mechanism_coverage: float | None
    mechanism_coverage_over_reference_herbs: float | None
    formula_consistent_with_primary: bool | None
    formula_consistent_with_any_syndrome: bool | None
    chain_closed_any: bool | None
    chain_closed_core60: bool | None
    chain_closed_strict: bool | None


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def binary_average(values: list[bool | None]) -> float | None:
    return average([float(value) if value is not None else None for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public OP-RAG G0–G4 workflow on the synthetic demonstration dataset.")
    # Open-source default: public synthetic cases plus data/kb/. The manuscript
    # used a separately authorized, private knowledge-base-extended dataset;
    # that dataset is never selected implicitly by this public entry point.
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "demo" / "synthetic_cases.jsonl")
    parser.add_argument("--modes", default="g0,g1,g2,g3,g4")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--use-qwen", action="store_true", help="Generate reports with Qwen-Plus when QWEN_API_KEY is configured.")
    llm_group.add_argument("--no-llm", action="store_true", help="Use deterministic local summaries (the default public mode).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo_ablation")
    args = parser.parse_args()

    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    invalid = set(modes) - set(AblationRunner.MODES)
    if invalid:
        raise ValueError(f"Unsupported modes: {sorted(invalid)}")

    cases = load_cases(args.cases)
    runner = AblationRunner(load_kb())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for mode in modes:
        results: list[Result] = []
        for row in cases:
            output = runner.run_case(AblationCaseInput(**row, use_llm=args.use_qwen), mode)
            metric = output.metrics
            context = output.context["case_results"]
            results.append(Result(
                case_id=row["case_id"], mode=mode, formula_kb_status=context["formula_kb_status"],
                primary_syndrome_id=context["primary_syndrome_id"], herb_jaccard=metric["herb_jaccard"],
                core_herb_mechanism_coverage=metric["core_herb_mechanism_coverage"],
                mechanism_coverage_over_reference_herbs=metric["mechanism_coverage_over_reference_herbs"],
                formula_consistent_with_primary=metric["formula_syndrome_consistency_rate_primary"],
                formula_consistent_with_any_syndrome=metric["formula_syndrome_consistency_rate_any"],
                chain_closed_any=metric["chain_closed_any"], chain_closed_core60=metric["chain_closed_core60"],
                chain_closed_strict=metric["chain_closed_strict"],
            ))
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        with (mode_dir / "case_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=Result.__dataclass_fields__.keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
        summary = {
            "mode": mode, "n_cases": len(results),
            "formula_kb_support_rate": average([1.0 if r.formula_kb_status == "mapped" else 0.0 for r in results]),
            "primary_syndrome_mapping_rate": average([1.0 if r.primary_syndrome_id else 0.0 for r in results]),
            "mean_herb_jaccard": average([r.herb_jaccard for r in results]),
            "mean_core_herb_mechanism_coverage": average([r.core_herb_mechanism_coverage for r in results]),
            "mean_mechanism_coverage_over_reference_herbs": average([r.mechanism_coverage_over_reference_herbs for r in results]),
            "formula_syndrome_consistency_rate_primary": binary_average([r.formula_consistent_with_primary for r in results]),
            "formula_syndrome_consistency_rate_any": binary_average([r.formula_consistent_with_any_syndrome for r in results]),
            "chain_closed_any_rate": binary_average([r.chain_closed_any for r in results]),
            "chain_closed_core60_rate": binary_average([r.chain_closed_core60 for r in results]),
            "chain_closed_strict_rate": binary_average([r.chain_closed_strict for r in results]),
        }
        (mode_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)
    (args.output_dir / "all_modes_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {len(modes)} modes on {len(cases)} synthetic cases. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
