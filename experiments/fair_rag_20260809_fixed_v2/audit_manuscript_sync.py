from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
FINAL_RUN_DIR = OUTPUT_DIR / "qwen_comparison_v5_post_spotcheck_20260810"
FINAL_PUBLICATION_DIR = OUTPUT_DIR / "publication_v5_post_spotcheck"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "manuscript_sync_audit_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(manuscript_dir: Path) -> dict[str, str]:
    paths = list(manuscript_dir.rglob("*.tex"))
    paths.extend(
        path
        for path in (manuscript_dir / "figures").glob("Figure[45]_*")
        if path.is_file()
    )
    return {
        str(path.relative_to(manuscript_dir)).replace("\\", "/"): sha256(path)
        for path in sorted(paths)
    }


def occurrences(path: Path, needle: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            rows.append({"line": number, "text": line.strip()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only comparison of a manuscript against the final OP-RAG registry."
    )
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manuscript_dir = args.manuscript_dir.resolve()
    before = snapshot(manuscript_dir)

    registry = json.loads(
        (FINAL_PUBLICATION_DIR / "publication_value_registry.json").read_text(
            encoding="utf-8"
        )
    )
    run_manifest = json.loads(
        (FINAL_RUN_DIR / "run_manifest.json").read_text(encoding="utf-8")
    )

    stale_checks = [
        ("main.tex", "13.40", "Final matched mean record budget is 11.06."),
        ("main.tex", "0.615 versus 0.448", "Final layered versus flat precision is 0.708 versus 0.425."),
        ("main.tex", "0.996 versus 0.748", "Final layered versus flat recall is 1.000 versus 0.631."),
        ("main.tex", "30/50 formula", "Final formula mapping is 21/50."),
        ("main.tex", "20/50 retained", "Final independent formula support is 10/50."),
        ("sections/methods.tex", "All configurations used the same plan, JSON schema, and Qwen-Plus model", "Qwen now receives only non-numeric narrative facts and returns only assessment_summary; Python owns all structured fields."),
        ("sections/methods.tex", "Claims had to copy item and source IDs", "Claims, sources, coverage, missing items and levels are now assembled deterministically by Python, not emitted by Qwen."),
        ("sections/methods.tex", "We did not score free-text semantic entailment", "The final validator explicitly checks the qualitative summary for required clauses, contradictions, unsupported wording, numbers and clinical claims."),
        ("sections/results.tex", "13.40", "Final matched mean record budget is 11.06."),
        ("sections/results.tex", "0.448 to 0.615", "Final precision is 0.425 flat and 0.708 layered."),
        ("sections/results.tex", "0.748 to 0.996", "Final recall is 0.631 flat and 1.000 layered."),
        ("sections/results.tex", "30 visible formula names", "Final formula mapping is 21/50."),
        ("sections/results.tex", "Thirty visible formula names", "Final formula mapping is 21/50, with 20/21 mapped formulas concordant."),
        ("sections/results.tex", "20/50 records retained", "Final independent formula support is 10/50."),
        ("sections/results.tex", "0.722 for standard formula compositions", "Final mean formula-composition coverage is 0.6944 across 21 mapped records."),
        ("sections/results.tex", "16/50 records", "Final any-chain completion is 10/50."),
        ("sections/results.tex", "2/16 met strict", "Final strict completion is 0/10."),
        ("sections/results.tex", "Qwen only & NA & NA & NA & NA & 0.528", "Qwen-only retrieval, evidence and provenance fields are NA; deterministic consistency fields are 50/50 and are not model-performance scores."),
        ("sections/results.tex", "Flat RAG & 0.448 & 0.748", "Final Flat values are retrieval 0.425/0.631, evidence recall 0.637, link precision 1.000, and provenance 50/50."),
        ("sections/results.tex", "Layered RAG & 0.615 & 0.996", "Final Layered values are retrieval 0.708/1.000, evidence recall 1.000, link precision 1.000, and provenance 50/50."),
        ("sections/results.tex", "OP-RAG & 0.615 & 0.996", "Final OP-RAG values are retrieval 0.708/1.000, evidence recall 1.000, link precision 1.000, and provenance 50/50."),
        ("sections/results.tex", "2, 24, 24, and 0 records", "Final OP-RAG levels 1/2/3/4 are 0/21/29/0."),
        ("sections/results.tex", "Core($T$) completion was 16, 16, 9, 5, and 4 of 16", "Final Core(T) counts are 10, 10, 3, 2, and 2 of 10."),
        ("sections/results.tex", "Strict($T$) completion was 16, 14, 7, 2, and 0 of 16", "Final Strict(T) counts are 10, 10, 3, 0, and 0 of 10."),
        ("sections/results.tex", "Formula unmapped & 17", "Final mutually exclusive primary category count is 26; all-reason formula-unmapped count is 29."),
        ("sections/discussion.tex", "agreement was 0.620 for OP-RAG and 0.500 for Layered RAG", "Missing-evidence F1 and level agreement are deterministic QA fields and are 1.0, not model performance comparisons."),
        ("sections/discussion.tex", "source errors affected 10 Flat-RAG, 11 Layered-RAG, and 10 OP-RAG reports", "Final provenance is clean for 50/50 reports in each RAG configuration."),
        ("sections/discussion.tex", "mapped 30/50 formula names", "Final formula mapping is 21/50."),
        ("sections/conclusion.tex", "higher agreement with the versioned audit reference", "The final pipeline computes the level in Python; it cannot claim a Qwen agreement improvement."),
        ("supplementary.tex", "0.448 & 0.748", "Final flat retrieval values are 0.425 and 0.631."),
        ("supplementary.tex", "0.615 & 0.996", "Final layered retrieval values are 0.708 and 1.000."),
        ("supplementary.tex", "Flat RAG & 50/50 & 40/50", "Final flat provenance is 50/50."),
        ("supplementary.tex", "Layered RAG & 50/50 & 39/50", "Final layered provenance is 50/50."),
        ("supplementary.tex", "OP-RAG & 50/50 & 40/50", "Final OP-RAG provenance is 50/50."),
        ("supplementary.tex", "0.168 (0.098, 0.279)", "Final layered-minus-flat retrieval precision difference is 0.2830 (95% CI 0.2020--0.3997)."),
        ("supplementary.tex", "0.248 (0.147, 0.406)", "Final layered-minus-flat retrieval recall difference is 0.3686 (95% CI 0.2641--0.5171)."),
        ("supplementary.tex", "0.242 (0.134, 0.404)", "Final layered-minus-flat evidence recall difference is 0.3629 (95% CI 0.2568--0.5124)."),
        ("supplementary.tex", "reported OP-RAG level distribution differs", "Final levels are deterministic Python fields; there is no model-versus-reference level disagreement."),
        ("supplementary.tex", "d2eb502f35bc14e9c25114278dfbc194599c1505f9c8d41766fbf67ec1c15b4a", "Final context SHA-256 is 872972b21f3412b3fbad0a67e740bcde2718407895915c12e5502737ae5ca414."),
    ]
    findings: list[dict[str, Any]] = []
    for relative, needle, expected in stale_checks:
        path = manuscript_dir / relative
        if not path.is_file():
            findings.append(
                {
                    "severity": "major",
                    "file": relative,
                    "line": None,
                    "observed": "file missing",
                    "expected": expected,
                }
            )
            continue
        for match in occurrences(path, needle):
            findings.append(
                {
                    "severity": "major",
                    "file": relative,
                    "line": match["line"],
                    "observed": match["text"],
                    "expected": expected,
                }
            )

    required_checks = [
        ("main.tex", "11.06", "The abstract must report the final matched mean record budget."),
        ("main.tex", "21/50", "The abstract must use the final formula-mapping count."),
        ("main.tex", "10/50", "The abstract must use the final independent-source count when that outcome is reported."),
        ("sections/methods.tex", "assessment_summary", "Methods must state that assessment_summary is Qwen's only accepted output field."),
        ("sections/methods.tex", "deterministic Python", "Methods must state that Python owns the structured audit fields."),
        ("sections/methods.tex", "non-numeric", "Methods must describe the number-free Qwen input/output contract."),
        ("sections/results.tex", "11.06", "Results must use the final matched retrieval budget."),
        ("sections/results.tex", "0.425", "Results must report the final Flat retrieval precision."),
        ("sections/results.tex", "0.631", "Results must report the final Flat retrieval recall."),
        ("sections/results.tex", "0.708", "Results must report the final Layered retrieval precision."),
        ("sections/results.tex", "21/50", "Results must report the final formula-mapping count."),
        ("sections/results.tex", "10/50", "Results must report the final independent-source and any-chain numerators with explicit labels."),
        ("sections/results.tex", "0/10", "Results must report the final strict-closure count."),
        ("sections/results.tex", "0/21/29/0", "Results or its table must report final OP-RAG levels 1/2/3/4."),
        ("supplementary.tex", "0.424978", "Supplementary source data must include final Flat precision at adequate precision."),
        ("supplementary.tex", "0.631359", "Supplementary source data must include final Flat recall at adequate precision."),
        ("supplementary.tex", "0.707962", "Supplementary source data must include final Layered precision at adequate precision."),
        ("supplementary.tex", "872972b21f3412b3fbad0a67e740bcde2718407895915c12e5502737ae5ca414", "Supplementary reproducibility text must use the final context hash."),
        ("supplementary.tex", "ad8531e878b99c321e2c2ce779840401cbfaaaba7c7d73164d332e708729c139", "Supplementary reproducibility text must use the final audit-reference hash."),
    ]
    for relative, needle, requirement in required_checks:
        path = manuscript_dir / relative
        if not path.is_file() or not occurrences(path, needle):
            findings.append(
                {
                    "severity": "major",
                    "file": relative,
                    "line": None,
                    "observed": f"required final token is absent: {needle}",
                    "expected": requirement,
                }
            )

    figure_checks: dict[str, Any] = {}
    for stem in ("Figure4_fair_rag_comparison", "Figure5_internal_application"):
        for suffix in (".pdf", ".png", ".svg", ".tiff"):
            name = f"{stem}{suffix}"
            manuscript_path = manuscript_dir / "figures" / name
            final_path = FINAL_PUBLICATION_DIR / name
            match = (
                manuscript_path.is_file()
                and final_path.is_file()
                and sha256(manuscript_path) == sha256(final_path)
            )
            figure_checks[name] = {
                "match": match,
                "manuscript_sha256": sha256(manuscript_path) if manuscript_path.is_file() else None,
                "final_sha256": sha256(final_path) if final_path.is_file() else None,
            }
            if not match:
                findings.append(
                    {
                        "severity": "major",
                        "file": f"figures/{name}",
                        "line": None,
                        "observed": "manuscript figure does not match the final generated figure",
                        "expected": "replace only after explicit author permission",
                    }
                )

    after = snapshot(manuscript_dir)
    manuscript_modified = before != after
    if manuscript_modified:
        findings.append(
            {
                "severity": "critical",
                "file": str(manuscript_dir),
                "line": None,
                "observed": "manuscript changed during a read-only audit",
                "expected": "no manuscript modification",
            }
        )

    report = {
        "status": "pass" if not findings else "fail",
        "scope": "read-only manuscript-to-final-registry synchronization audit",
        "manuscript_dir": str(manuscript_dir),
        "manuscript_modified": manuscript_modified,
        "final_run_results_sha256": run_manifest.get("results_sha256"),
        "final_registry_sha256": sha256(
            FINAL_PUBLICATION_DIR / "publication_value_registry.json"
        ),
        "final_values": {
            "retrieval": registry["retrieval_benchmark"]["configurations"],
            "internal": registry["deterministic_internal_case_analysis"],
        },
        "figure_checks": figure_checks,
        "n_findings": len(findings),
        "findings": findings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANUSCRIPT_SYNC_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Manuscript-to-final-registry synchronization audit",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Manuscript modified by this audit: `{manuscript_modified}`",
        f"- Final result SHA-256: `{report['final_run_results_sha256']}`",
        f"- Final registry SHA-256: `{report['final_registry_sha256']}`",
        f"- Findings: `{len(findings)}`",
        "",
        "The audit is read-only. A FAIL means the manuscript still describes an older experiment; it does not invalidate the retained final run.",
        "",
        "| Severity | File | Line | Observed | Required correction |",
        "|---|---|---:|---|---|",
    ]
    for finding in findings:
        observed = str(finding["observed"]).replace("|", "\\|")
        expected = str(finding["expected"]).replace("|", "\\|")
        lines.append(
            f"| {finding['severity']} | `{finding['file']}` | "
            f"{finding['line'] or ''} | {observed} | {expected} |"
        )
    lines.extend(
        [
            "",
            "## Final values that must govern any later manuscript update",
            "",
            "- Mean matched record budget: 11.06.",
            "- Flat precision/recall: 0.424978/0.631359.",
            "- Layered and OP-RAG precision/recall: 0.707962/1.000000.",
            "- Formula mapping: 21/50; independent formula source: 10/50.",
            "- Any-chain: 10/50; Core60: 10/10; strict: 0/10.",
            "- OP-RAG levels 1/2/3/4: 0/21/29/0.",
            "- Qwen output is only a non-numeric qualitative assessment_summary.",
            "",
            "No manuscript correction was made because explicit author permission is required.",
            "",
        ]
    )
    (args.output_dir / "MANUSCRIPT_SYNC_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "n_findings": len(findings), "manuscript_modified": manuscript_modified}, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
