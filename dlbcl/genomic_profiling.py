"""Build oncoprint-ready genomic alteration tables for adata.uns['genomic_profiling']."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LOCATION_ONCOPRINT_ORDER = ["PCNS", "Bone", "Nodal", "Testis"]

LOCATION_RECODE = {
    "pBONE": "bone",
    "polyOST": "bone",
    "disseminated": "bone",
}

HOTSPOT_RULES: list[tuple[int, set[str]]] = [
    (38182641, {"C"}),  # MYD88
    (45003745, {"T", "G", "C"}),  # B2M
    (45003746, {"C", "A", "G"}),
    (45003747, {"A", "T", "C"}),
    (62006799, {"C", "G", "T"}),  # CD79B
    (62006798, {"A", "G", "C"}),
    (148508727, {"A", "G", "C"}),  # EZH2
    (148508728, {"T", "G", "C"}),
]

ALTERATION_PALETTE = {
    "STAR": "#6a78ff",
    "NON": "#63d16b",
    "SPLI": "#d1a252",
    "STOP": "#c34600",
    "HOT": "#7d6c3c",
    "FRAM": "#7ca9b6",
    "REAR": "#fff2b8",
    "GAIN": "#001b21",
    "LOSS": "#ffa19f",
}

LOCATION_PALETTE = {
    "PCNS": "#7F3C8D",
    "Bone": "#11A579",
    "Nodal": "#3969AC",
    "Testis": "#F2B701",
}

# Patients withheld from genomic profiling / oncoprint outputs.
GENOMIC_PROFILING_EXCLUDE_PATIENTS = frozenset({
    "T5", "T11", "T22", "T35", "T47", "T57",
})


def _is_hotspot(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for start, alts in HOTSPOT_RULES:
        mask |= (df["Start"] == start) & df["Alt"].isin(alts)
    return mask


def _map_sequence_alteration(exonic_func: str) -> str | None:
    if exonic_func == "startloss":
        return "STAR"
    if exonic_func in ("stopgain", "stoploss"):
        return "STOP"
    if exonic_func == "nonsynonymous SNV":
        return "NON"
    if exonic_func == "splicing":
        return "SPLI"
    if exonic_func in {
        "frameshift deletion",
        "nonframeshift deletion",
        "frameshift insertion",
        "frameshift substitution",
        "nonframeshift insertion",
        "nonframeshift substitution",
    }:
        return "FRAM"
    if exonic_func == "rearrangement":
        return "REAR"
    return None


def _filter_patient_table(
    df: pd.DataFrame,
    exclude: set[str] | frozenset[str],
) -> pd.DataFrame:
    if df.empty or "patient_id" not in df.columns:
        return df
    return df.loc[~df["patient_id"].astype(str).isin(exclude)].reset_index(drop=True)


def apply_genomic_profiling_exclusions(
    gp: dict,
    exclude: set[str] | frozenset[str] | None = None,
) -> dict:
    """Remove withheld patients from genomic profiling tables."""
    exclude = set(exclude or GENOMIC_PROFILING_EXCLUDE_PATIENTS)
    out = dict(gp)
    for key in ("events", "location_class", "sample_map"):
        if key in out:
            out[key] = _filter_patient_table(pd.DataFrame(out[key]), exclude)
    flat = out.get("flat")
    if isinstance(flat, dict):
        out["flat"] = {pid: row for pid, row in flat.items() if pid not in exclude}
    return out


def patients_without_source_variants(
    gp: dict,
    *,
    patient_ids: set[str] | frozenset[str] | list[str] | None = None,
) -> frozenset[str]:
    """Patients with no alteration rows in ``genomic_profiling['events']``.

    Uses the oncoprint event table embedded in the h5ad (built from the variant
    xlsx). Patients like T15 with registered SNVs/CNVs are retained; patients with
    an empty event table (e.g. T29, T41, T48) are excluded from genomic analyses.
    """
    gp = apply_genomic_profiling_exclusions(gp)
    events = pd.DataFrame(gp.get("events", []))
    if events.empty or "patient_id" not in events.columns:
        pool = set(patient_ids) if patient_ids is not None else set()
        pool -= set(GENOMIC_PROFILING_EXCLUDE_PATIENTS)
        if not is_nested_genomic_profiling(gp) and isinstance(gp, dict):
            in_gp = {str(k) for k in gp.keys()}
            return frozenset(p for p in pool if p not in in_gp)
        return frozenset(pool)

    counts = events.groupby(events["patient_id"].astype(str)).size()
    if patient_ids is None:
        pool = {str(p) for p in counts.index}
    else:
        pool = {str(p) for p in patient_ids}
    pool -= set(GENOMIC_PROFILING_EXCLUDE_PATIENTS)
    return frozenset(p for p in pool if int(counts.get(p, 0)) == 0)


def read_sample_map(flat_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(flat_csv, usecols=["filename", "patient_id"])
    return df.assign(
        filename=lambda d: d["filename"].astype(str),
        patient_id=lambda d: d["patient_id"].astype(str),
    ).drop_duplicates()


def assign_patient_id_from_tnr(tnr: pd.Series) -> pd.Series:
    """Map sequencing sample id (Tnr) to patient id (strip trailing .N replicate suffix)."""
    return tnr.astype(str).str.replace(r"\.[0-9]+$", "", regex=True)


def read_pathogenic_variants_from_xlsx(variant_xlsx: Path) -> pd.DataFrame:
    """Pathogenic, non-synonymous variants from the Variants sheet (full coordinates)."""
    variants_raw = pd.read_excel(variant_xlsx, sheet_name="Variants")
    pathogenic = variants_raw.loc[
        (variants_raw["Path"] == "Pathogenic")
        & (variants_raw["ExonicFunc.refGene"] != "synonymous SNV")
    ].copy()
    pathogenic["patient_id"] = assign_patient_id_from_tnr(pathogenic["Tnr"])
    return pathogenic.reset_index(drop=True)


def read_cnv_gene_events_from_xlsx(
    variant_xlsx: Path,
    sample_map: pd.DataFrame,
) -> pd.DataFrame:
    """Gene-level GAIN/LOSS from the CNV sheet (patient_id, gene, alteration)."""
    cnv_raw = pd.read_excel(variant_xlsx, sheet_name="CNV")
    return (
        cnv_raw.assign(filename=cnv_raw["Sample"].astype(str))
        .merge(sample_map, on="filename", how="left")
        .rename(columns={"Gene": "gene"})
        .assign(alteration=lambda d: d["CNV"].astype(str).str.upper())
        .loc[lambda d: d["alteration"].isin(["GAIN", "LOSS"]), ["patient_id", "gene", "alteration"]]
        .dropna(subset=["patient_id", "gene"])
        .astype({"patient_id": str, "gene": str})
        .reset_index(drop=True)
    )


def _map_sample_to_patient_alias(
    sample_ids: pd.Series,
    id_map: pd.DataFrame | None,
) -> pd.Series:
    """Map sequencing sample ids (``filename`` / ``NGS_ID``) to ``patient_alias``."""
    if id_map is None or id_map.empty:
        return sample_ids.astype(str)
    mapping: dict[str, str] = {}
    if "NGS_ID" in id_map.columns:
        for ngs_id, alias in (
            id_map.dropna(subset=["NGS_ID"])
            .drop_duplicates(subset=["NGS_ID"])
            .set_index("NGS_ID")["patient_alias"]
            .astype(str)
            .items()
        ):
            mapping[str(ngs_id).strip()] = str(alias)
    if "filename" in id_map.columns:
        for fname, alias in (
            id_map.drop_duplicates(subset=["filename"])
            .set_index("filename")["patient_alias"]
            .astype(str)
            .items()
        ):
            mapping[str(fname).strip()] = str(alias)
    return sample_ids.astype(str).str.strip().map(lambda s: mapping.get(s, s))


_CNV_CALL_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".GAIN", "GAIN"),
    ("_GAIN", "GAIN"),
    (".AMP", "GAIN"),
    ("_AMP", "GAIN"),
    (".LOSS", "LOSS"),
    ("_LOSS", "LOSS"),
    (".DEL", "LOSS"),
    ("_DEL", "LOSS"),
    (".HETLOSS", "LOSS"),
    ("_HETLOSS", "LOSS"),
    (".HOMDEL", "LOSS"),
    ("_HOMDEL", "LOSS"),
)


def _truthy_cnv_call(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().upper()
    if text in {"", "0", "FALSE", "F", "NO", "N", "WT", "NONE", "NAN"}:
        return False
    return text in {"GAIN", "LOSS", "HETLOSS", "HOMDEL", "AMP", "DEL", "TRUE", "T", "YES", "Y", "1"}


def _gene_from_cnv_column(column: str) -> tuple[str, str] | None:
    col = str(column).strip()
    upper = col.upper()
    for suffix, alteration in _CNV_CALL_SUFFIXES:
        if upper.endswith(suffix):
            gene = col[: -len(suffix)].strip("._-")
            if gene:
                return gene, alteration
    return None


def read_cnv_gene_events_from_call_table(
    call_table: pd.DataFrame,
    *,
    id_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract gene-level GAIN/LOSS rows from a long or wide genomics call list."""
    df = pd.DataFrame(call_table).copy()
    if df.empty:
        return pd.DataFrame(columns=["patient_id", "gene", "alteration"])

    patient_col = next(
        (c for c in ("patient_alias", "patient_id", "Sample", "filename", "sample_id") if c in df.columns),
        None,
    )
    gene_col = next((c for c in ("Gene", "gene", "Gene.refGene") if c in df.columns), None)
    cnv_col = next((c for c in ("CNV", "cn_state", "alteration", "call", "Call") if c in df.columns), None)

    if patient_col and gene_col and cnv_col:
        out = df[[patient_col, gene_col, cnv_col]].copy()
        out.columns = ["patient_id", "gene", "alteration"]
        if patient_col in {"Sample", "filename"}:
            out["patient_id"] = _map_sample_to_patient_alias(out["patient_id"], id_map)
        out["alteration"] = out["alteration"].astype(str).str.strip().str.upper()
        out.loc[out["alteration"] == "HETLOSS", "alteration"] = "LOSS"
        out.loc[out["alteration"] == "HOMDEL", "alteration"] = "LOSS"
        out = out.loc[out["alteration"].isin(["GAIN", "LOSS"])]
        return (
            out.dropna(subset=["patient_id", "gene"])
            .astype({"patient_id": str, "gene": str})
            .drop_duplicates(["patient_id", "gene", "alteration"])
            .reset_index(drop=True)
        )

    if patient_col and "ExonicFunc.refGene" in df.columns:
        out = df[[patient_col, gene_col or "Gene.refGene", "ExonicFunc.refGene"]].copy()
        out.columns = ["patient_id", "gene", "alteration"]
        if patient_col in {"Sample", "filename"}:
            out["patient_id"] = _map_sample_to_patient_alias(out["patient_id"], id_map)
        out["alteration"] = out["alteration"].astype(str).str.strip().str.upper()
        out.loc[out["alteration"] == "HETLOSS", "alteration"] = "LOSS"
        out.loc[out["alteration"] == "HOMDEL", "alteration"] = "LOSS"
        out = out.loc[out["alteration"].isin(["GAIN", "LOSS"])]
        return (
            out.dropna(subset=["patient_id", "gene"])
            .astype({"patient_id": str, "gene": str})
            .drop_duplicates(["patient_id", "gene", "alteration"])
            .reset_index(drop=True)
        )

    if patient_col is None:
        return pd.DataFrame(columns=["patient_id", "gene", "alteration"])

    meta_cols = {
        patient_col,
        "patient_alias",
        "patient_id",
        "Sample",
        "filename",
        "sample_id",
        "Run",
    }
    value_cols = [c for c in df.columns if c not in meta_cols]
    if not value_cols:
        return pd.DataFrame(columns=["patient_id", "gene", "alteration"])

    pid = df[patient_col].astype(str)
    if patient_col in {"Sample", "filename"}:
        pid = _map_sample_to_patient_alias(pid, id_map)
    rows: list[dict[str, str]] = []
    suffix_cols = [c for c in value_cols if _gene_from_cnv_column(c) is not None]
    if suffix_cols:
        for i, patient_id in enumerate(pid):
            for col in suffix_cols:
                if not _truthy_cnv_call(df.iloc[i][col]):
                    continue
                gene, alteration = _gene_from_cnv_column(col)
                if gene is None:
                    continue
                rows.append({"patient_id": patient_id, "gene": gene, "alteration": alteration})
        if rows:
            return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    for i, patient_id in enumerate(pid):
        for col in value_cols:
            val = str(df.iloc[i][col]).strip().upper()
            if val in {"GAIN", "LOSS"}:
                rows.append({"patient_id": patient_id, "gene": str(col), "alteration": val})
    if not rows:
        return pd.DataFrame(columns=["patient_id", "gene", "alteration"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def read_cnv_gene_events_from_validation_workbook(
    xlsx_path: Path | str,
    id_map: pd.DataFrame,
    *,
    sheet_name: str = "cnv_data",
) -> pd.DataFrame:
    """Gene-level GAIN/LOSS from optional validation workbook ``cnv_data`` sheet."""
    xlsx_path = Path(xlsx_path)
    sheet_names = set(pd.ExcelFile(xlsx_path).sheet_names)
    if sheet_name not in sheet_names and "CNV" in sheet_names:
        sheet_name = "CNV"
    if sheet_name not in sheet_names:
        return pd.DataFrame(columns=["patient_id", "gene", "alteration"])
    return read_cnv_gene_events_from_call_table(
        pd.read_excel(xlsx_path, sheet_name=sheet_name),
        id_map=id_map,
    )


def read_flat_matrix(flat_csv: Path) -> dict[str, dict[str, str]]:
    """Legacy per-patient boolean matrix (patient_id -> column -> 'True'/'False')."""
    df = pd.read_csv(flat_csv, low_memory=False)
    df["patient_id"] = df["patient_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    variant_cols = [c for c in df.columns if c not in ("filename", "patient_id")]
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        patient = row["patient_id"]
        out[patient] = {col: str(row[col]) for col in variant_cols}
        out[patient]["filename"] = row["filename"]
    return out


def build_events_from_variants_xlsx(
    variant_xlsx: Path,
    sample_map: pd.DataFrame,
    *,
    patient_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Replicate oncoprintplot_unified.Rmd variant + CNV processing."""
    pathogenic = read_pathogenic_variants_from_xlsx(variant_xlsx)

    hotspot = pathogenic.loc[_is_hotspot(pathogenic)].assign(alteration="HOT")
    sequence = pathogenic.loc[~_is_hotspot(pathogenic)].copy()
    sequence["alteration"] = sequence["ExonicFunc.refGene"].map(_map_sequence_alteration)

    sequence_events = pd.concat(
        [
            sequence[["patient_id", "Gene.refGene", "alteration"]].rename(
                columns={"Gene.refGene": "gene"}
            ),
            hotspot[["patient_id", "Gene.refGene", "alteration"]].rename(
                columns={"Gene.refGene": "gene"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["patient_id", "gene", "alteration"])

    cnv_events = read_cnv_gene_events_from_xlsx(variant_xlsx, sample_map)

    events = (
        pd.concat([sequence_events, cnv_events], ignore_index=True)
        .dropna(subset=["patient_id", "gene", "alteration"])
        .drop_duplicates(["patient_id", "gene", "alteration"])
    )
    allowed = set(patient_ids) if patient_ids is not None else None
    if allowed is not None:
        allowed -= set(GENOMIC_PROFILING_EXCLUDE_PATIENTS)
        events = events.loc[events["patient_id"].isin(allowed)]
    else:
        events = events.loc[~events["patient_id"].isin(GENOMIC_PROFILING_EXCLUDE_PATIENTS)]
    return events.reset_index(drop=True)


def location_class_from_locations(location: pd.Series) -> pd.Series:
    loc = location.astype(str).replace(LOCATION_RECODE)
    mapped = loc.map(
        {
            "PCNS": "PCNS",
            "bone": "Bone",
            "nodal": "Nodal",
            "testis": "Testis",
        }
    )
    return mapped


def build_location_class(case_classifications: pd.DataFrame) -> pd.DataFrame:
    cc = pd.DataFrame(case_classifications).copy()
    if "patient_id" in cc.columns:
        cc = cc.set_index("patient_id", drop=True)
    elif cc.index.name != "patient_id":
        cc = cc.rename_axis("patient_id")
    out = (
        pd.DataFrame({"patient_id": cc.index.astype(str), "Location": cc["Location"].values})
        .loc[lambda d: ~d["patient_id"].isin(GENOMIC_PROFILING_EXCLUDE_PATIENTS)]
        .assign(Location_class=lambda d: location_class_from_locations(d["Location"]))
        .dropna(subset=["Location_class"])
        .drop_duplicates("patient_id")
    )
    out["Location_class"] = pd.Categorical(
        out["Location_class"], categories=LOCATION_ONCOPRINT_ORDER, ordered=True
    )
    return out.sort_values(["Location_class", "patient_id"]).reset_index(drop=True)


def build_genomic_profiling_uns(
    *,
    variant_xlsx: Path,
    flat_csv: Path,
    case_classifications: pd.DataFrame,
) -> dict:
    sample_map = read_sample_map(flat_csv)
    location_class = build_location_class(case_classifications)
    patient_ids = set(location_class["patient_id"])
    events = build_events_from_variants_xlsx(
        variant_xlsx, sample_map, patient_ids=patient_ids
    )
    flat = read_flat_matrix(flat_csv)
    return apply_genomic_profiling_exclusions({
        "events": events,
        "sample_map": sample_map,
        "location_class": location_class,
        "flat": flat,
        "alteration_palette": ALTERATION_PALETTE,
        "location_palette": LOCATION_PALETTE,
    })


def is_nested_genomic_profiling(gp: object) -> bool:
    return isinstance(gp, dict) and "events" in gp


# Keys safe for public release (no sequencing filenames / local paths).
RELEASE_KEYS = ("events", "location_class", "alteration_palette", "location_palette", "ref")


def _load_panel_symbol_list(panel_csv: Path) -> list[str]:
    symbols = pd.read_csv(panel_csv, header=None)[0].astype(str).str.strip()
    return [s for s in symbols if s and s.lower() != "nan"]


def _hgnc_mapping_from_table(table: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, row in table.iterrows():
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym:
            continue
        entrez = row.get("entrez_id")
        if pd.isna(entrez) or str(entrez).strip() == "":
            continue
        try:
            out[sym] = int(float(entrez))
        except (TypeError, ValueError):
            continue
    return out


def _nonempty_seq(value: object) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def is_genomic_profiling_ref_complete(ref: object) -> bool:
    """Return True when ``ref`` contains embedded classifier reference tables."""
    if not isinstance(ref, dict):
        return False
    panel = ref.get("blymv2_panel", {})
    hgnc = ref.get("hgnc_symbol_to_entrez", {})
    gsm = ref.get("dlbclass_template_gsm", {})
    return (
        _nonempty_seq(panel.get("genes"))
        and _nonempty_seq(hgnc.get("symbols"))
        and _nonempty_seq(hgnc.get("entrez_ids"))
        and _nonempty_seq(gsm.get("features"))
    )


def panel_symbols_from_ref(ref: dict[str, object]) -> list[str]:
    genes = ref.get("blymv2_panel", {}).get("genes", [])
    return [str(g).strip() for g in list(genes) if str(g).strip()]


def hgnc_mapping_from_ref(ref: dict[str, object]) -> dict[str, int]:
    entry = ref.get("hgnc_symbol_to_entrez", {})
    raw = entry.get("mapping")
    if isinstance(raw, dict):
        out: dict[str, int] = {}
        for sym, entrez in raw.items():
            try:
                out[str(sym).strip().upper()] = int(entrez)
            except (TypeError, ValueError):
                continue
        return out
    symbols = list(entry.get("symbols", []))
    entrez_ids = list(entry.get("entrez_ids", []))
    if not symbols or not entrez_ids:
        return {}
    out = {}
    for sym, entrez in zip(symbols, entrez_ids, strict=False):
        try:
            out[str(sym).strip().upper()] = int(entrez)
        except (TypeError, ValueError):
            continue
    return out


def gsm_features_from_ref(ref: dict[str, object]) -> list[str]:
    feats = ref.get("dlbclass_template_gsm", {}).get("features", [])
    return [str(f).strip() for f in list(feats) if str(f).strip()]


def build_genomic_profiling_ref(repo_root: Path | str) -> dict[str, object]:
    """Build embedded classifier reference tables for ``uns['genomic_profiling']['ref']``."""
    root = Path(repo_root)
    ref_dir = root / "data" / "reference"
    panel_csv = ref_dir / "blymv2_panel_symbols.csv"
    hgnc_tsv = ref_dir / "hgnc_symbol_to_entrez.tsv"
    gsm_csv = root / "data" / "DLBCL_template_gsm.csv"

    panel_genes = _load_panel_symbol_list(panel_csv) if panel_csv.exists() else []
    gsm_features = (
        pd.read_csv(gsm_csv)["classifier_name"].astype(str).tolist()
        if gsm_csv.exists()
        else []
    )
    hgnc_mapping: dict[str, int] = {}
    if hgnc_tsv.exists():
        hgnc_mapping = _hgnc_mapping_from_table(pd.read_csv(hgnc_tsv, sep="\t"))

    hgnc_symbols = sorted(hgnc_mapping)
    hgnc_entrez_ids = [hgnc_mapping[sym] for sym in hgnc_symbols]

    return {
        "blymv2_panel": {
            "description": "LymphGen BLYMv2 gene panel manifest (not inferred from per-patient events)",
            "n_genes": len(panel_genes),
            "genes": panel_genes,
        },
        "hgnc_symbol_to_entrez": {
            "description": "HGNC symbol/alias/prev_symbol to Entrez ID map for LymphGen exports",
            "n_symbols": len(hgnc_symbols),
            "symbols": hgnc_symbols,
            "entrez_ids": hgnc_entrez_ids,
        },
        "dlbclass_template_gsm": {
            "description": "DLBclass preset GSM feature order",
            "n_features": len(gsm_features),
            "features": gsm_features,
        },
    }


def attach_genomic_profiling_ref(adata, repo_root: Path | str) -> dict[str, object]:
    """Embed classifier reference tables in ``adata.uns['genomic_profiling']['ref']``."""
    ref = build_genomic_profiling_ref(repo_root)
    gp = adata.uns.get("genomic_profiling")
    if not isinstance(gp, dict):
        gp = {}
    gp = dict(gp)
    gp["ref"] = ref
    adata.uns["genomic_profiling"] = gp
    return ref


def get_genomic_profiling_ref(adata) -> dict[str, object]:
    """Return embedded classifier reference tables from ``adata.uns``."""
    gp = adata.uns.get("genomic_profiling")
    if not isinstance(gp, dict):
        raise KeyError("Expected adata.uns['genomic_profiling']")
    ref = gp.get("ref")
    if not is_genomic_profiling_ref_complete(ref):
        raise KeyError(
            "Expected complete adata.uns['genomic_profiling']['ref'] with "
            "blymv2_panel.genes, hgnc_symbol_to_entrez.symbols/entrez_ids, and dlbclass_template_gsm.features."
        )
    return ref


def sanitize_genomic_profiling(gp: dict) -> dict:
    """Drop lab sample filenames, legacy flat matrix, and local source paths."""
    return apply_genomic_profiling_exclusions(
        {key: gp[key] for key in RELEASE_KEYS if key in gp}
    )


def audit_genomic_profiling_phi(gp: dict) -> list[dict]:
    """Return rows describing potential linkable identifiers in genomic_profiling."""
    import re

    patterns = {
        "hospital_sample_id": re.compile(r"^(H\d{2}[-_]|R\d{2}[-_]|MD\d|T\d+-\d|\d{6,})", re.I),
    }
    findings: list[dict] = []

    def _check(value: str, field: str) -> None:
        for label, pat in patterns.items():
            if pat.search(str(value)):
                findings.append({"field": field, "pattern": label, "example": str(value)[:80]})

    if "sample_map" in gp:
        for fn in pd.DataFrame(gp["sample_map"])["filename"].astype(str):
            _check(fn, "sample_map.filename")

    flat = gp.get("flat")
    if isinstance(flat, dict):
        for patient_id, row in flat.items():
            if isinstance(row, dict) and "filename" in row:
                _check(row["filename"], f"flat.{patient_id}.filename")

    for path in (gp.get("source_files") or {}).values():
        _check(path, "source_files")

    return findings


def export_for_r(gp: dict, out_dir: Path) -> dict[str, Path]:
    """Write CSV inputs expected by r/oncoprint_unified.R."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = apply_genomic_profiling_exclusions(gp)
    paths = {
        "events": out_dir / "oncoprint_events.csv",
        "location": out_dir / "oncoprint_location_class.csv",
    }
    pd.DataFrame(gp["events"]).to_csv(paths["events"], index=False)
    pd.DataFrame(gp["location_class"]).to_csv(paths["location"], index=False)
    return paths
