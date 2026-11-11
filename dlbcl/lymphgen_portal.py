"""Build auditable LymphGen portal inputs without changing case assignments.

Portal inputs use real Entrez gene IDs, explicit missing FISH values, and an
assay-wide mutation gene list. Internal LymphGen pseudo-gene IDs are forbidden.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def resolve_gene(symbol: str, hgnc: pd.DataFrame) -> tuple[str, int]:
    """Prefer an approved symbol over another gene's colliding alias."""
    symbol = str(symbol).strip()
    for col in ("symbol", "prev_symbol", "alias_symbol"):
        hits = hgnc.loc[hgnc[col].fillna("").map(lambda x: symbol in x.split("|"))]
        hits = hits.loc[hits.entrez_id.notna() & hits.entrez_id.ne("")]
        if not hits.empty:
            ids = hits.entrez_id.astype(int).unique()
            if len(ids) != 1:
                raise ValueError(f"Ambiguous {col} mapping for {symbol}: {ids}")
            return str(hits.iloc[0].symbol), int(ids[0])
    raise ValueError(f"No gene ID for {symbol}")


def fish_value(value: object) -> str:
    key = "" if pd.isna(value) else str(value).strip().lower()
    if key in {"positive", "pos", "1", "1.0"}:
        return "1"
    if key in {"negative", "neg", "0", "0.0"}:
        return "0"
    if key in {"", "unknown", "unk", "not_done", "not done", "na", "nan"}:
        return "NA"
    raise ValueError(f"Unrecognized FISH value: {value!r}")


def mutation_type(effect: object, annotation: object, *, gene_id: int,
                  position: int | None = None) -> str | None:
    effect = "" if pd.isna(effect) else str(effect).strip().lower()
    aa = "" if pd.isna(annotation) else str(annotation)
    if gene_id == 4615 and (re.search(r"L265P|LEU265PRO", aa, re.I) or position == 38182641):
        return "L265P"
    if effect in {"stopgain", "startloss", "splicing", "frameshift deletion",
                  "frameshift insertion", "frameshift substitution"}:
        return "TRUNC"
    if effect in {"nonsynonymous snv", "nonframeshift deletion", "nonframeshift insertion",
                  "nonframeshift substitution", "stoploss"}:
        return "MUTATION"
    if effect == "synonymous snv":
        return "Synon"
    # Recover explicit effects only; never infer a coding consequence from
    # the label Pathogenic alone. c.+1/+2 and -1/-2 affect canonical splice sites.
    if re.search(r"fs|p\.[^,: ]*\*|c\.\d+[+-][12](?:\D|$)", aa, re.I):
        return "TRUNC"
    if re.search(r"(?:p\.|:)\s*[A-Z]\d+[A-Z](?:$|[,; ])", aa):
        return "MUTATION"
    return None


def validate_bundle(samples: pd.DataFrame, mutations: pd.DataFrame, genes: list[int]) -> None:
    if samples["Sample"].duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if not samples["Sample"].str.fullmatch(r"[TV]\d+").all():
        raise ValueError("Only study aliases may be submitted")
    if not set(mutations.Sample).issubset(samples.Sample):
        raise ValueError("Mutation has no sample annotation")
    if not set(mutations["ENTREZ.ID"]).issubset(genes):
        raise ValueError("Mutation outside declared panel coverage")
    if any(x >= 1_000_000_000 for x in genes):
        raise ValueError("Internal pseudo-gene ID in portal input")
    if not mutations.loc[mutations.Type.eq("L265P"), "ENTREZ.ID"].eq(4615).all():
        raise ValueError("L265P must be encoded as MYD88/4615")
    if not set(mutations.Type).issubset({"TRUNC", "MUTATION", "Synon", "L265P"}):
        raise ValueError("Unknown mutation type")
    if not samples["Copy.Number"].eq(0).all():
        raise ValueError("This builder is for mutation/FISH runs without usable CN data")
    for col in ["BCL2.Translocation", "BCL6.Translocation"]:
        if not set(samples[col]).issubset({"0", "1", "NA"}):
            raise ValueError("Invalid or missing FISH encoding")


def write_bundle(directory: Path, samples: pd.DataFrame, mutations: pd.DataFrame,
                 genes: list[int]) -> None:
    validate_bundle(samples, mutations, genes)
    directory.mkdir(parents=True, exist_ok=True)
    samples.rename(columns={"BCL2.Translocation": "BCL2.transloc",
                            "BCL6.Translocation": "BCL6.transloc"}).to_csv(
        directory / "sample_annotation.txt", sep="\t", index=False)
    mutations.to_csv(directory / "mutation_flat.txt", sep="\t", index=False)
    pd.DataFrame({"ENTREZ.ID": sorted(set(genes))}).to_csv(
        directory / "mutation_gene_list.txt", sep="\t", index=False)
