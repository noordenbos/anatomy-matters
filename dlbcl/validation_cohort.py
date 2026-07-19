"""Validation cohort NanoString GEP export, normalization, and archetype scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANNOT_COLS = ("Probe", "Accession", "Code.Class")
VALIDATION_COHORT_UNS_KEY = "validation_cohort"
VALIDATION_COHORT_UNS_VERSION = 2

# PHI identifiers stripped from de-identified clinical meta (kept in local ``validation_id_map.csv``).
CLINICAL_PHI_COLUMNS = (
    "gep_id",
    "iPLUS_ID",
    "ID",
    "castor_id",
    "pa_id",
    "Goldfile_ID",
    "NGS_ID",
    "Isolation",
    "IMC_ID",
)
NGS_SAMPLE_ID_COLUMN = "filename"

ARCHETYPE_LABELS = {
    1: "low immune",
    2: "cytotoxic predominant",
    3: "complex immune",
}


def patient_aliases(n: int) -> list[str]:
    return [f"V{i}" for i in range(1, n + 1)]


def _sample_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ANNOT_COLS]


def geomean_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Positive-control geometric-mean scaling (``NS_RawData.Rmd`` GeoMean)."""
    out = df.copy()
    sample_cols = _sample_columns(out)
    out[sample_cols] = out[sample_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    pos = out.loc[out["Code.Class"] == "Positive", sample_cols].apply(pd.to_numeric, errors="coerce")
    geo = np.exp(np.log(pos.replace(0, np.nan)).mean(axis=0))
    scale = geo.mean() / geo
    hk_end = out["Code.Class"].isin(["Endogenous", "Housekeeping"])
    hk_vals = out.loc[hk_end, sample_cols]
    out.loc[hk_end, sample_cols] = hk_vals.mul(scale, axis=1)
    return out


def nanonorm(df: pd.DataFrame, *, is_logged: bool = False, corr: float = 1e-4) -> pd.DataFrame:
    """Housekeeping-normalized log2 expression (``NS_RawData.Rmd`` NanoNorm)."""
    sample_cols = _sample_columns(df)
    endo = df.loc[df["Code.Class"] == "Endogenous", sample_cols].apply(pd.to_numeric, errors="coerce")
    hk = df.loc[df["Code.Class"] == "Housekeeping", sample_cols].apply(pd.to_numeric, errors="coerce")
    if is_logged:
        hk_means = hk.mean(axis=0)
        gx_norm = endo.sub(hk_means, axis=1)
        hk_norm = hk.sub(hk_means, axis=1)
    else:
        gx_log = np.log2(endo + corr)
        hk_log = np.log2(hk + corr)
        hk_means = hk_log.mean(axis=0)
        gx_norm = gx_log.sub(hk_means, axis=1)
        hk_norm = hk_log.sub(hk_means, axis=1)
    annot = df.loc[df["Code.Class"].isin(["Endogenous", "Housekeeping"]), list(ANNOT_COLS)]
    norm_vals = pd.concat([gx_norm, hk_norm], axis=0)
    return pd.concat([annot.reset_index(drop=True), norm_vals.reset_index(drop=True)], axis=1)


def normalize_nanos_string_gep(df: pd.DataFrame) -> pd.DataFrame:
    return nanonorm(geomean_scale(df))


def gene_expression_matrix(
    df: pd.DataFrame,
    *,
    discovery_genes: list[str] | None = None,
    normalized: bool = True,
) -> pd.DataFrame:
    """Return genes × patients matrix (Endogenous + Housekeeping probes)."""
    if normalized:
        sample_cols = _sample_columns(df)
        mat = df.set_index("Probe")[sample_cols].apply(pd.to_numeric, errors="coerce")
    else:
        hk_end = df["Code.Class"].isin(["Endogenous", "Housekeeping"])
        sample_cols = _sample_columns(df)
        mat = df.loc[hk_end].set_index("Probe")[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat.index = mat.index.astype(str)
    if discovery_genes is not None:
        genes = [g for g in discovery_genes if g in mat.index]
        mat = mat.loc[genes]
    return mat


def read_validation_workbook(
    xlsx_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Return ``(clinical, gep_raw, ngs_raw, cnv_raw)`` from the validation workbook."""
    xlsx_path = Path(xlsx_path)
    with pd.ExcelFile(xlsx_path) as xls:
        sheet_names = set(xls.sheet_names)
        clinical = pd.read_excel(xls, sheet_name="clinical_data")
        gep_raw = pd.read_excel(xls, sheet_name="gep_data")
        ngs_raw = pd.read_excel(xls, sheet_name="ngs_data") if "ngs_data" in sheet_names else None
        if "cnv_data" in sheet_names:
            cnv_raw = pd.read_excel(xls, sheet_name="cnv_data")
        elif "CNV" in sheet_names:
            cnv_raw = pd.read_excel(xls, sheet_name="CNV")
        else:
            cnv_raw = None
    clinical = clinical.copy()
    clinical["gep_id"] = clinical["gep_id"].astype(str)
    return clinical, gep_raw, ngs_raw, cnv_raw


def build_patient_id_map(clinical: pd.DataFrame) -> pd.DataFrame:
    """PHI bridge table: ``gep_id``, optional ``NGS_ID``, and ``patient_alias`` (V1…)."""
    aliases = patient_aliases(len(clinical))
    id_map = pd.DataFrame(
        {
            "patient_alias": aliases,
            "gep_id": clinical["gep_id"].astype(str).tolist(),
        }
    )
    if "NGS_ID" in clinical.columns:
        ngs = clinical["NGS_ID"].copy()
        id_map["NGS_ID"] = ngs.map(lambda v: pd.NA if pd.isna(v) else str(v).strip())
    return id_map


def deidentify_clinical_meta(clinical: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Clinical workbook row → PHI-stripped meta indexed by ``patient_alias``."""
    meta = clinical.merge(id_map[["gep_id", "patient_alias"]], on="gep_id", how="left")
    drop_cols = [c for c in CLINICAL_PHI_COLUMNS if c in meta.columns]
    meta = meta.drop(columns=drop_cols)
    meta = meta.set_index("patient_alias")
    meta.index = meta.index.astype(str)
    return meta


def deidentify_ngs_data(ngs_raw: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Map ``ngs_data.filename`` (NGS_ID) → ``patient_alias``; drop raw sample identifiers."""
    if NGS_SAMPLE_ID_COLUMN not in ngs_raw.columns:
        raise ValueError(f"ngs_data missing required column {NGS_SAMPLE_ID_COLUMN!r}")
    if "NGS_ID" not in id_map.columns:
        raise ValueError("id_map missing NGS_ID; cannot de-identify ngs_data")

    ngs = ngs_raw.copy()
    ngs_to_alias = (
        id_map.dropna(subset=["NGS_ID"])
        .drop_duplicates(subset=["NGS_ID"])
        .set_index("NGS_ID")["patient_alias"]
        .astype(str)
    )
    sample_ids = ngs[NGS_SAMPLE_ID_COLUMN].astype(str).str.strip()
    ngs["patient_alias"] = sample_ids.map(ngs_to_alias)
    ngs = ngs.drop(columns=[NGS_SAMPLE_ID_COLUMN])
    front = ["patient_alias"]
    rest = [c for c in ngs.columns if c not in front]
    return ngs[front + rest]


def deidentify_cnv_data(cnv_raw: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Map CNV sheet sample ids to ``patient_alias`` (same bridge as ``ngs_data``)."""
    cnv = cnv_raw.copy()
    sample_col = next((c for c in ("Sample", "filename", "sample_id") if c in cnv.columns), None)
    if sample_col is None:
        raise ValueError("cnv_data missing Sample/filename column")
    if "NGS_ID" not in id_map.columns:
        raise ValueError("id_map missing NGS_ID; cannot de-identify cnv_data")

    ngs_to_alias = (
        id_map.dropna(subset=["NGS_ID"])
        .drop_duplicates(subset=["NGS_ID"])
        .set_index("NGS_ID")["patient_alias"]
        .astype(str)
    )
    sample_ids = cnv[sample_col].astype(str).str.strip()
    cnv["patient_alias"] = sample_ids.map(ngs_to_alias)
    gene_col = next((c for c in ("Gene", "gene", "Gene.refGene") if c in cnv.columns), None)
    cnv_col = next((c for c in ("CNV", "cn_state", "alteration") if c in cnv.columns), None)
    keep = ["patient_alias"]
    if gene_col:
        keep.append(gene_col)
    if cnv_col:
        keep.append(cnv_col)
    for col in cnv.columns:
        if col not in keep and col != sample_col:
            keep.append(col)
    return cnv[[c for c in keep if c in cnv.columns]]


def build_validation_tables(
    clinical: pd.DataFrame,
    gep_raw: pd.DataFrame,
    *,
    ngs_raw: pd.DataFrame | None = None,
    cnv_raw: pd.DataFrame | None = None,
    discovery_genes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """De-identify workbook sheets → meta, GEP matrices, optional NGS (no PHI id map)."""
    id_map = build_patient_id_map(clinical)
    rename = dict(zip(clinical["gep_id"], id_map["patient_alias"], strict=True))
    gep_renamed = gep_raw.rename(columns=rename)

    meta = deidentify_clinical_meta(clinical, id_map)

    gep_norm = normalize_nanos_string_gep(gep_renamed)
    gene_expression_raw = gene_expression_matrix(gep_renamed, normalized=False)
    gene_expression = gene_expression_matrix(gep_norm, discovery_genes=discovery_genes)
    annot_cols = [c for c in ANNOT_COLS if c in gep_renamed.columns]
    sample_cols = [c for c in gep_renamed.columns if c not in annot_cols]
    gep_raw_probes = gep_renamed[annot_cols + sample_cols].copy()

    out: dict[str, pd.DataFrame] = {
        "id_map": id_map,
        "meta": meta,
        "gene_expression_raw": gene_expression_raw,
        "gene_expression": gene_expression,
        "gep_raw_probes": gep_raw_probes,
        "gep_normalized_probes": gep_norm,
    }
    if ngs_raw is not None:
        out["ngs_data"] = deidentify_ngs_data(ngs_raw, id_map)
    if cnv_raw is not None:
        out["cnv_data"] = deidentify_cnv_data(cnv_raw, id_map)
    return out


def _as_dataframe(obj, *, index_col: str | None = None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = pd.DataFrame(obj).copy()
    if index_col:
        if index_col in df.columns:
            df = df.set_index(index_col)
        elif df.index.name != index_col:
            df.index.name = index_col
    return df


def _sanitize_h5ad_columns(df: pd.DataFrame) -> pd.DataFrame:
    """h5py interprets ``/`` in dataset names as group paths; normalize for ``uns``."""
    out = df.copy()
    rename = {
        col: str(col).replace("/", "_").replace("\\", "_")
        for col in out.columns
        if "/" in str(col) or "\\" in str(col)
    }
    if rename:
        out = out.rename(columns=rename)
    return out


def _h5ad_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type columns so AnnData can serialize the frame in ``uns``."""
    out = _sanitize_h5ad_columns(df)
    for col in out.columns:
        ser = out[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            out[col] = ser.astype(str).replace("NaT", "")
        elif pd.api.types.is_numeric_dtype(ser):
            out[col] = pd.to_numeric(ser, errors="coerce")
        else:
            out[col] = ser.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def build_validation_cohort_uns(
    xlsx_path: Path | str,
    *,
    discovery_genes: list[str] | None = None,
    model_bundle: dict | None = None,
    classifications_path: Path | str | None = None,
) -> dict[str, object]:
    """Build the full ``adata.uns['validation_cohort']`` payload (PHI-free, Zenodo-ready)."""
    clinical, gep_raw, ngs_raw, cnv_raw = read_validation_workbook(xlsx_path)
    tables = build_validation_tables(
        clinical, gep_raw, ngs_raw=ngs_raw, cnv_raw=cnv_raw, discovery_genes=discovery_genes
    )
    gep = tables["gene_expression"]
    gep_raw_mat = tables["gene_expression_raw"]

    uns: dict[str, object] = {
        "version": VALIDATION_COHORT_UNS_VERSION,
        "patient_aliases": gep.columns.astype(str).tolist(),
        "n_patients": int(gep.shape[1]),
        "n_genes": int(gep.shape[0]),
        "n_genes_raw": int(gep_raw_mat.shape[0]),
        "meta": _h5ad_safe_dataframe(tables["meta"]),
        "gene_expression_raw": gep_raw_mat.apply(pd.to_numeric, errors="coerce"),
        "gene_expression": gep.apply(pd.to_numeric, errors="coerce"),
        "gep_raw_probes": tables["gep_raw_probes"],
        "gep_normalized_probes": tables["gep_normalized_probes"],
    }
    if "ngs_data" in tables:
        uns["ngs_data"] = _h5ad_safe_dataframe(tables["ngs_data"])
        uns["n_ngs_patients"] = int(tables["ngs_data"]["patient_alias"].nunique())
        uns["n_ngs_variants"] = int(len(tables["ngs_data"]))
    if model_bundle is not None:
        pred = predict_validation_archetypes(model_bundle, uns)
        pred = pred.set_index("patient_alias")
        pred.index = pred.index.astype(str)
        uns["predictions"] = _h5ad_safe_dataframe(pred)

    if classifications_path is not None and Path(classifications_path).exists():
        from .validation_classifications import (
            attach_validation_classifications,
            load_validation_classifier_tsv,
        )

        clf = load_validation_classifier_tsv(classifications_path)
        pred_df = _as_dataframe(uns.get("predictions", pd.DataFrame()))
        if "patient_alias" in pred_df.columns:
            pred_df = pred_df.set_index("patient_alias")
        pred_df.index = pred_df.index.astype(str)
        uns = attach_validation_classifications(
            uns,
            pred_df,
            clf,
        )
        if "case_classification_validation" in uns:
            uns["case_classification_validation"] = _h5ad_safe_dataframe(
                uns["case_classification_validation"]
            )
        if "ecotyper_b_state" in uns:
            uns["ecotyper_b_state"] = _h5ad_safe_dataframe(uns["ecotyper_b_state"])
    return uns


def validation_cohort_present(adata) -> bool:
    vc = getattr(adata, "uns", {}).get(VALIDATION_COHORT_UNS_KEY)
    return isinstance(vc, dict) and "gene_expression" in vc


def require_validation_cohort(adata) -> dict[str, object]:
    """Return ``adata.uns['validation_cohort']`` or raise with maintainer instructions."""
    if not validation_cohort_present(adata):
        raise KeyError(
            f"adata.uns['{VALIDATION_COHORT_UNS_KEY}'] not found in the AnnData bundle.\n"
            "Maintainers: run scripts/inject_validation_cohort.py to embed the validation cohort "
            "before publishing the Zenodo h5ad."
        )
    return adata.uns[VALIDATION_COHORT_UNS_KEY]


def cohort_notebook_inputs(adata) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Notebook helper: ``(predictions, gene_expression, archetype_labels)``."""
    vc = require_validation_cohort(adata)
    gep = _as_dataframe(vc["gene_expression"])
    gep.index = gep.index.astype(str)
    gep.columns = gep.columns.astype(str)

    if "predictions" not in vc:
        raise KeyError(
            f"adata.uns['{VALIDATION_COHORT_UNS_KEY}'] is missing 'predictions'. "
            "Re-run inject_validation_cohort.py with a trained nb5 model bundle."
        )
    pred = _as_dataframe(vc["predictions"], index_col="patient_alias")
    pred.index = pred.index.astype(str)
    labels = pred["pred_abundance_cluster_30_label"].astype(str)
    return pred, gep, labels


def export_validation_csvs(
    xlsx_path: Path | str,
    out_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Path]:
    """Split workbook into PHI map, de-identified meta/GEP/NGS (local maintainer export)."""
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    repo_root = Path(repo_root or out_dir.parent.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical, gep_raw, ngs_raw, cnv_raw = read_validation_workbook(xlsx_path)
    tables = build_validation_tables(clinical, gep_raw, ngs_raw=ngs_raw, cnv_raw=cnv_raw)

    paths = {
        "id_map": out_dir / "validation_id_map.csv",
        "meta": out_dir / "validation_meta.csv",
        "gep": out_dir / "validation_gep.csv",
        "gep_raw": out_dir / "validation_gep_raw.csv",
    }
    tables["id_map"].to_csv(paths["id_map"], index=False)
    tables["meta"].to_csv(paths["meta"])
    tables["gene_expression"].to_csv(paths["gep"])
    tables["gep_raw_probes"].to_csv(paths["gep_raw"], index=False)
    if "ngs_data" in tables:
        paths["ngs"] = out_dir / "validation_ngs.csv"
        tables["ngs_data"].to_csv(paths["ngs"], index=False)
    if "cnv_data" in tables:
        from .genomic_profiling import read_cnv_gene_events_from_call_table

        paths["cnv"] = out_dir / "validation_cnv_gene_events.csv"
        cnv_events = read_cnv_gene_events_from_call_table(
            tables["cnv_data"],
            id_map=tables["id_map"],
        )
        cnv_events.to_csv(paths["cnv"], index=False)

    from .validation_classifications import (
        build_case_classification_validation,
        load_validation_classifier_tsv,
        resolve_classifier_tsv,
    )

    clf_path = resolve_classifier_tsv(out_dir)
    if clf_path is not None:
        clf = load_validation_classifier_tsv(clf_path)
        pred = tables["meta"].copy()
        pred.index = pred.index.astype(str)
        for pred_csv in (
            repo_root / "figures" / "notebook5" / "validation_archetype_predictions.csv",
            out_dir / "validation_predictions.csv",
        ):
            if pred_csv.exists():
                scored = pd.read_csv(pred_csv)
                scored["patient_alias"] = scored["patient_alias"].astype(str)
                scored = scored.set_index("patient_alias")
                pred = scored.reindex(pred.index).combine_first(pred)
                break
        cc = build_case_classification_validation(pred, clf)
        paths["case_classifications"] = out_dir / "validation_case_classifications.csv"
        cc.to_csv(paths["case_classifications"])
    return paths


def load_validation_cohort_uns(
    validation_dir: Path | str,
    *,
    discovery_genes: list[str] | None = None,
) -> dict[str, object]:
    """Load prepared CSVs into an ``adata.uns``-compatible dict (legacy local path)."""
    validation_dir = Path(validation_dir)
    id_map = pd.read_csv(validation_dir / "validation_id_map.csv", dtype=str)
    meta = pd.read_csv(validation_dir / "validation_meta.csv", index_col=0)
    gep = pd.read_csv(validation_dir / "validation_gep.csv", index_col=0)
    gep.index = gep.index.astype(str)
    gep.columns = gep.columns.astype(str)
    if discovery_genes is not None:
        genes = [g for g in discovery_genes if g in gep.index]
        gep = gep.loc[genes]
    uns: dict[str, object] = {
        "version": VALIDATION_COHORT_UNS_VERSION,
        "id_map": id_map,
        "meta": meta,
        "gene_expression": gep,
        "patient_aliases": gep.columns.tolist(),
        "n_patients": int(gep.shape[1]),
        "n_genes": int(gep.shape[0]),
    }
    ngs_path = validation_dir / "validation_ngs.csv"
    if ngs_path.exists():
        ngs = pd.read_csv(ngs_path, dtype=str)
        uns["ngs_data"] = ngs
        if "patient_alias" in ngs.columns:
            uns["n_ngs_patients"] = int(ngs["patient_alias"].nunique())
        uns["n_ngs_variants"] = int(len(ngs))
    return uns


def predict_validation_archetypes(
    model_bundle: dict,
    validation_uns: dict,
) -> pd.DataFrame:
    """Apply trained elastic-net bundle to validation ``gene_expression``."""
    gep = _as_dataframe(validation_uns["gene_expression"])
    gene_order = list(model_bundle["gene_order"])
    class_names = [int(c) for c in model_bundle["class_names"]]

    missing = [g for g in gene_order if g not in gep.index]
    if missing:
        raise ValueError(f"{len(missing)} model genes missing from validation GEP, e.g. {missing[:5]}")

    x = gep.loc[gene_order, :].T.apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        raise ValueError(f"validation matrix has {int(x.isna().sum().sum())} NA after alignment")

    pipeline = model_bundle["pipeline"]
    proba = pipeline.predict_proba(x)
    pred_idx = proba.argmax(axis=1)
    pred_class = [class_names[i] for i in pred_idx]

    out = pd.DataFrame(index=x.index)
    out.index.name = "patient_alias"
    out["pred_abundance_cluster_30"] = pred_class
    out["pred_abundance_cluster_30_label"] = out["pred_abundance_cluster_30"].map(ARCHETYPE_LABELS)
    for i, cls in enumerate(class_names):
        out[f"prob_{cls}"] = proba[:, i]
    out["max_prob"] = proba.max(axis=1)

    meta = validation_uns.get("meta")
    if isinstance(meta, pd.DataFrame):
        out = out.join(meta, how="left")
    elif meta is not None:
        out = out.join(_as_dataframe(meta), how="left")
    return out.reset_index()


def prepare_validation_cohort(
    xlsx_path: Path | str,
    out_dir: Path | str,
    *,
    discovery_genes: list[str] | None = None,
) -> dict[str, object]:
    """Export CSVs from the workbook and return a local ``validation_cohort`` uns payload."""
    paths = export_validation_csvs(xlsx_path, out_dir)
    uns = load_validation_cohort_uns(out_dir, discovery_genes=discovery_genes)
    uns["source_files"] = {k: str(v) for k, v in paths.items()}
    return uns


def inject_validation_cohort_into_adata(
    adata,
    xlsx_path: Path | str,
    *,
    model_bundle: dict | None = None,
    discovery_genes: list[str] | None = None,
    classifications_path: Path | str | None = None,
) -> dict[str, object]:
    """Attach validation cohort tables to ``adata.uns`` (in memory)."""
    if discovery_genes is None and "gene_expression" in adata.uns:
        discovery_genes = adata.uns["gene_expression"].index.astype(str).tolist()
    if classifications_path is None:
        from .validation_classifications import resolve_classifier_tsv

        classifications_path = resolve_classifier_tsv(Path(xlsx_path).parent)
    uns = build_validation_cohort_uns(
        xlsx_path,
        discovery_genes=discovery_genes,
        model_bundle=model_bundle,
        classifications_path=classifications_path,
    )
    adata.uns[VALIDATION_COHORT_UNS_KEY] = uns
    return uns
