from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
THEME_PATHWAYS = (
    "Osteoclast differentiation",
    "Wnt signaling pathway",
    "Estrogen signaling pathway",
    "Calcium signaling pathway",
    "TGF-beta signaling pathway",
    "ECM-receptor interaction",
    "Protein digestion and absorption",
    "TNF signaling pathway",
    "NF-kappa B signaling pathway",
    "PI3K-Akt signaling pathway",
)


def request_kegg(identifiers: list[str]) -> dict[str, Any]:
    payload = {
        "organism": "hsapiens",
        "sources": ["KEGG"],
        "user_threshold": 0.05,
        "all_results": False,
        "ordered": False,
        "significance_threshold_method": "g_SCS",
        "query": identifiers,
        "no_evidences": False,
    }
    response = requests.post(GPROFILER_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def map_uniprot_to_entrez(accessions: list[str], *, batch_size: int = 80) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, str]] = []
    returned_accessions: set[str] = set()
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start : start + batch_size]
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        response = requests.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": f"({query})",
                "format": "tsv",
                "fields": "accession,gene_primary,xref_geneid",
                "size": 500,
            },
            timeout=90,
        )
        response.raise_for_status()
        for record in csv.DictReader(io.StringIO(response.text), delimiter="\t"):
            accession = str(record.get("Entry") or "").strip()
            if not accession:
                continue
            returned_accessions.add(accession)
            primary_symbol = str(record.get("Gene Names (primary)") or "").strip()
            gene_ids = [value.strip() for value in str(record.get("GeneID") or "").split(";") if value.strip()]
            for gene_id in gene_ids:
                rows.append(
                    {
                        "UniProt_ID": accession,
                        "Gene_Symbol_primary": primary_symbol,
                        "NCBI_Gene_ID": gene_id,
                    }
                )
    mapping = pd.DataFrame(rows).drop_duplicates()
    missing = sorted(set(accessions) - returned_accessions)
    return mapping, missing


def result_table(payload: dict[str, Any]) -> pd.DataFrame:
    query_identifiers = payload["meta"]["query_metadata"]["queries"]["query_1"]
    rows: list[dict[str, Any]] = []
    for item in payload.get("result", []):
        evidence = item.get("intersections") or []
        hit_identifiers = [identifier for identifier, flags in zip(query_identifiers, evidence) if flags]
        count = int(item["intersection_size"])
        if len(hit_identifiers) != count:
            raise RuntimeError(
                f"Intersection decoding mismatch for {item.get('native')}: "
                f"decoded {len(hit_identifiers)}, API count {count}"
            )
        query_size = int(item["query_size"])
        adjusted_p = float(item["p_value"])
        rows.append(
            {
                "ID": item["native"],
                "Description": item["name"],
                "Source": item["source"],
                "Count": count,
                "Term_size": int(item["term_size"]),
                "Query_size": query_size,
                "Bg_size": int(item["effective_domain_size"]),
                "GeneRatio": count / query_size,
                "p_adjust_gSCS": adjusted_p,
                "neglog10_adjusted_p": -math.log10(max(adjusted_p, 1e-300)),
                "hit_query_identifiers": "/".join(hit_identifiers),
            }
        )
    return pd.DataFrame(rows).sort_values("p_adjust_gSCS", kind="stable")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_theme_pathways(theme: pd.DataFrame, base: Path) -> None:
    configure_matplotlib()
    order = {name: index for index, name in enumerate(THEME_PATHWAYS)}
    data = theme.assign(_order=theme["Description"].map(order)).sort_values("_order", ascending=False)
    fig, ax = plt.subplots(figsize=(174 / 25.4, 102 / 25.4))
    sizes = 22 + data["Count"].to_numpy(float) * 2.0
    scatter = ax.scatter(
        data["GeneRatio"],
        np.arange(len(data)),
        s=sizes,
        c=data["neglog10_adjusted_p"],
        cmap="Greys",
        vmin=0,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_yticks(np.arange(len(data)), data["Description"])
    ax.set_xlabel("Gene ratio")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label(r"$-\log_{10}$(g:SCS-adjusted $P$)")
    legend_counts = sorted({int(data["Count"].min()), int(data["Count"].median()), int(data["Count"].max())})
    handles = [
        ax.scatter([], [], s=22 + count * 2.0, facecolor="#bdbdbd", edgecolor="#333333", linewidth=0.6)
        for count in legend_counts
    ]
    ax.legend(handles, [str(count) for count in legend_counts], title="Gene count", loc="upper right")
    fig.subplots_adjust(left=0.35, right=0.90, top=0.97, bottom=0.14)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--id-column", default="UniProt_ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes_frame = pd.read_csv(args.genes, encoding="utf-8-sig")
    source_identifiers = list(dict.fromkeys(genes_frame[args.id_column].dropna().astype(str).str.strip()))
    if args.id_column == "UniProt_ID":
        mapping, unmapped_source_identifiers = map_uniprot_to_entrez(source_identifiers)
        if mapping.empty:
            raise RuntimeError("UniProt-to-NCBI Gene mapping returned no records")
        mapping.to_csv(args.output_dir / "uniprot_to_ncbi_gene_mapping.csv", index=False, encoding="utf-8-sig")
        identifiers = list(dict.fromkeys(mapping["NCBI_Gene_ID"].astype(str)))
        submitted_identifier_type = "NCBI_Gene_ID mapped from UniProt_ID"
    else:
        mapping = pd.DataFrame()
        unmapped_source_identifiers = []
        identifiers = source_identifiers
        submitted_identifier_type = args.id_column
    payload = request_kegg(identifiers)
    table = result_table(payload)
    theme = table[table["Description"].isin(THEME_PATHWAYS)].copy()
    if theme.empty:
        raise RuntimeError("No prespecified osteoporosis-related pathway themes were returned")

    table.to_csv(args.output_dir / "kegg_full_single_query.csv", index=False, encoding="utf-8-sig")
    theme.to_csv(args.output_dir / "kegg_prespecified_theme_pathways.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "gprofiler_response_metadata.json").write_text(
        json.dumps(payload["meta"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    query_size = int(table["Query_size"].max())
    failed_identifiers = payload["meta"].get("genes_metadata", {}).get("failed", [])
    manifest = {
        "source_identifier_type": args.id_column,
        "source_unique_identifiers": len(source_identifiers),
        "unmapped_source_identifiers": unmapped_source_identifiers,
        "submitted_identifier_type": submitted_identifier_type,
        "submitted_unique_identifiers": len(identifiers),
        "recognized_query_identifiers": len(identifiers) - len(failed_identifiers),
        "kegg_annotated_query_size": query_size,
        "background_size": int(table["Bg_size"].max()),
        "significant_kegg_terms": len(table),
        "displayed_prespecified_theme_terms": len(theme),
        "organism": "hsapiens",
        "source": "KEGG",
        "multiple_testing_method": "g:SCS",
        "significance_threshold": 0.05,
        "query_execution": "one complete, unbatched query",
        "gprofiler_version": payload["meta"].get("version"),
        "gprofiler_timestamp": payload["meta"].get("timestamp"),
        "failed_identifiers": failed_identifiers,
        "ambiguous_identifiers": payload["meta"].get("genes_metadata", {}).get("ambiguous", {}),
    }
    (args.output_dir / "kegg_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure_base = args.output_dir / "Figure3_KEGG_enrichment"
    plot_theme_pathways(theme, figure_base)

    if args.package_dir:
        figures = args.package_dir / "figures"
        supplementary = args.package_dir / "supplementary_data"
        scripts = args.package_dir / "scripts"
        for directory in (figures, supplementary, scripts):
            directory.mkdir(parents=True, exist_ok=True)
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            shutil.copy2(figure_base.with_suffix(suffix), figures / f"Figure3_KEGG_enrichment{suffix}")
        for filename in (
            "kegg_full_single_query.csv",
            "kegg_prespecified_theme_pathways.csv",
            "gprofiler_response_metadata.json",
            "kegg_analysis_manifest.json",
            "uniprot_to_ncbi_gene_mapping.csv",
        ):
            source = args.output_dir / filename
            if source.exists():
                shutil.copy2(source, supplementary / filename)
        shutil.copy2(Path(__file__), scripts / Path(__file__).name)

    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key not in {"failed_identifiers", "ambiguous_identifiers"}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
