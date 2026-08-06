"""Validation cohort classifier labels aligned to discovery ``case_classifications``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation_figures import ARCHETYPE_NAME_MAP, DISEASE_TYPE_TO_LOCATION

CLASSIFIER_TSV_CANDIDATES = (
    "patient_classifications_with_trained_classifier.tsv",
    "patient_classifications.tsv",
    "validation_ecotyper_lymphomap_classifications.tsv",
)

DISCOVERY_CLASSIFICATION_COLUMNS = (
    "Location",
    "lymphomap",
    "Lymphoma_Ecotype",
    "New_Ecotype",
    "Ciav_Cluster",
    "KotlovSig",
    "COO_NanoString",
    "tumorimmune_archetype_id",
    "tumorimmune_archetype",
)

# Workbook column names mapped to discovery-aligned names in case_classification_validation.
WORKBOOK_CLASSIFIER_COLUMNS: dict[str, str] = {
    "Ciav_Cluster": "Ciav_Cluster",
    "Lymphoma_Ecotype": "Lymphoma_Ecotype",
    "New_Ecotype": "New_Ecotype",
    "KotlovssGSEA": "KotlovSig",
    "COO_NanoString": "COO_NanoString",
}


def resolve_classifier_tsv(validation_dir: Path | str) -> Path | None:
    """Return the first available validation classifier export."""
    validation_dir = Path(validation_dir)
    for name in CLASSIFIER_TSV_CANDIDATES:
        path = validation_dir / name
        if path.exists():
            return path
    return None


def load_validation_classifier_tsv(path: Path | str) -> pd.DataFrame:
    """Load EcoTyper / LymphoMAP export (tab-separated)."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "patient_alias" not in df.columns:
        raise ValueError(f"{path} must contain patient_alias")
    df["patient_alias"] = df["patient_alias"].astype(str)
    return df


def _normalize_lymphomap(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def _normalize_classifier_label(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _normalize_ecotype(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    out = out.mask(out.str.lower().eq("unassigned"), "unknown")
    return out


def _ecotype_missing(series: pd.Series) -> bool:
    if series.isna().all():
        return True
    known = series.dropna().astype(str).str.strip()
    if known.empty:
        return True
    return known.str.lower().isin({"unknown", "unassigned", "nan", "none", ""}).all()


def _ecotype_label_missing(series: pd.Series) -> pd.Series:
    """Per-patient mask: True when Steen ecotype is missing / unknown / unassigned."""
    labels = series.astype(str).str.strip()
    return series.isna() | labels.str.lower().isin({"unknown", "unassigned", "nan", "none", ""})


def finalize_lymphoma_ecotype_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Split Steen ecotype into confident assignment vs best-match (New_Ecotype fill)."""
    if "Lymphoma_Ecotype" not in out.columns:
        return out
    if "Lymphoma_Ecotype_confident" in out.columns:
        return out.copy()
    out = out.copy()
    missing = _ecotype_label_missing(out["Lymphoma_Ecotype"])
    out["Lymphoma_Ecotype_confident"] = out["Lymphoma_Ecotype"].mask(missing)

    imputed = pd.Series(False, index=out.index)
    if "New_Ecotype" in out.columns:
        fill_mask = missing & ~_ecotype_label_missing(out["New_Ecotype"])
        if fill_mask.any():
            out.loc[fill_mask, "Lymphoma_Ecotype"] = _normalize_ecotype(
                out.loc[fill_mask, "New_Ecotype"]
            )
            imputed.loc[fill_mask] = True

    out["Lymphoma_Ecotype_imputed"] = imputed
    return out


_BCELL_STATE_MAP = {
    "S1": "S01",
    "S2": "S02",
    "S3": "S03",
    "S4": "S04",
    "S5": "S05",
    "S01": "S01",
    "S02": "S02",
    "S03": "S03",
    "S04": "S04",
    "S05": "S05",
}


def _normalize_bcell_state(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.upper()
    out = out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return out.map(lambda v: _BCELL_STATE_MAP.get(v, v) if pd.notna(v) else v)


def merge_validation_metadata_with_classifications(
    meta: pd.DataFrame,
    case_cc: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join discovery-aligned classifier columns onto validation metadata."""
    out = meta.copy()
    out.index = out.index.astype(str)
    case_cc = case_cc.copy()
    case_cc.index = case_cc.index.astype(str)
    for col in case_cc.columns:
        out[col] = case_cc[col].reindex(out.index)
    return out


def apply_trained_classifier_archetypes(
    pred: pd.DataFrame,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Override elastic-net archetype calls with trained_classifier labels when present."""
    clf = classifications.copy()
    if "patient_alias" in clf.columns:
        clf = clf.set_index("patient_alias")
    clf.index = clf.index.astype(str)

    out = pred.copy()
    if out.index.name != "patient_alias" and "patient_alias" in out.columns:
        out = out.set_index("patient_alias")
    out.index = out.index.astype(str)

    if "trained_classifier" in clf.columns:
        trained = pd.to_numeric(clf["trained_classifier"], errors="coerce").reindex(out.index)
        if trained.notna().any():
            out["pred_tumorimmune_archetype_id"] = trained.combine_first(
                pd.to_numeric(out.get("pred_tumorimmune_archetype_id"), errors="coerce")
            )
    if "trained_classifier_label" in clf.columns:
        labels = clf["trained_classifier_label"].reindex(out.index)
        if labels.notna().any():
            out["pred_tumorimmune_archetype"] = labels.combine_first(
                out.get("pred_tumorimmune_archetype")
            )
    if "trained_classifier_max_probability" in clf.columns:
        max_prob = pd.to_numeric(clf["trained_classifier_max_probability"], errors="coerce").reindex(
            out.index
        )
        if max_prob.notna().any():
            out["max_prob"] = max_prob.combine_first(pd.to_numeric(out.get("max_prob"), errors="coerce"))
    return out


def build_case_classification_validation(
    pred: pd.DataFrame,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Build discovery-aligned ``case_classification_validation`` for V-aliases."""
    clf = classifications.copy()
    if "patient_alias" in clf.columns:
        clf = clf.set_index("patient_alias")
    clf.index = clf.index.astype(str)

    pred_idx = pred.index.astype(str)
    out = pd.DataFrame(index=pred_idx)
    out.index.name = "patient_id"

    out["Location"] = pred["disease_type"].map(DISEASE_TYPE_TO_LOCATION)

    # Archetype: nb5 elastic-net on validation GEP (not external trained_classifier TSV).
    out["tumorimmune_archetype_id"] = pd.to_numeric(
        pred["pred_tumorimmune_archetype_id"], errors="coerce"
    ).astype("Int64")
    if "pred_tumorimmune_archetype" in pred.columns:
        out["tumorimmune_archetype"] = pred["pred_tumorimmune_archetype"].reindex(pred_idx)
    else:
        out["tumorimmune_archetype"] = out["tumorimmune_archetype_id"].map(ARCHETYPE_NAME_MAP)

    for dst_col in ("Ciav_Cluster", "COO_NanoString", "KotlovSig", "Lymphoma_Ecotype", "New_Ecotype"):
        out[dst_col] = pd.NA

    for src_col, dst_col in WORKBOOK_CLASSIFIER_COLUMNS.items():
        if src_col not in pred.columns:
            continue
        normalizer = (
            _normalize_ecotype
            if dst_col in {"Lymphoma_Ecotype", "New_Ecotype"}
            else _normalize_classifier_label
        )
        out[dst_col] = normalizer(pred[src_col]).reindex(pred_idx)

    if out["COO_NanoString"].isna().all() and "coo_hans" in pred.columns:
        coo_map = {"GCB": "GCB", "Non-GCB": "ABC", "ABC": "ABC", "Unclassified": "Intermediate"}
        out["COO_NanoString"] = pred["coo_hans"].map(coo_map).reindex(pred_idx)

    if "lymphomap_label" in clf.columns:
        out["lymphomap"] = _normalize_lymphomap(clf["lymphomap_label"]).reindex(pred_idx)
    elif "lymphomap" in clf.columns:
        out["lymphomap"] = _normalize_lymphomap(clf["lymphomap"]).reindex(pred_idx)

    if _ecotype_missing(out["Lymphoma_Ecotype"]):
        ecotype_col = None
        for candidate in ("ecotyper_ecotype", "ecotyper_label", "Lymphoma_Ecotype"):
            if candidate in clf.columns:
                ecotype_col = candidate
                break
        if ecotype_col is not None:
            out["Lymphoma_Ecotype"] = _normalize_ecotype(clf[ecotype_col]).reindex(pred_idx)

    out = finalize_lymphoma_ecotype_columns(out)
    return out[out["Location"].notna()].copy()


def add_validation_genomic_tested(
    case_cc: pd.DataFrame,
    validation_uns: dict,
) -> pd.DataFrame:
    """Mark patients with embedded validation NGS variants (grey genomic rings when False)."""
    out = case_cc.copy()
    tested: set[str] = set()
    ngs = validation_uns.get("ngs_data")
    if isinstance(ngs, pd.DataFrame) and "patient_alias" in ngs.columns:
        tested = set(ngs["patient_alias"].astype(str).unique())
    out["genomic_tested"] = out.index.astype(str).isin(tested)
    return out


def build_ecotyper_b_state_validation(classifications: pd.DataFrame) -> pd.DataFrame:
    """Discovery-compatible ``ecotyper_b_state`` table for validation patients."""
    clf = classifications.copy()
    if "patient_alias" in clf.columns:
        clf = clf.set_index("patient_alias")
    clf.index = clf.index.astype(str)
    if "ecotyper_bcell_state_max" not in clf.columns:
        raise ValueError("classifications missing ecotyper_bcell_state_max")

    out = pd.DataFrame(index=clf.index)
    out.index.name = "patient_id"
    out["Dominant_B_cell_state"] = _normalize_bcell_state(clf["ecotyper_bcell_state_max"])
    if "ecotyper_bcell_state_maxprob" in clf.columns:
        out["Dominant_score"] = pd.to_numeric(clf["ecotyper_bcell_state_maxprob"], errors="coerce")
    else:
        out["Dominant_score"] = pd.NA
    out["Second_score"] = pd.NA
    out["Margin"] = pd.NA
    return out.dropna(subset=["Dominant_B_cell_state"])


def attach_validation_classifications(
    validation_uns: dict,
    pred: pd.DataFrame,
    classifications: pd.DataFrame,
) -> dict:
    """Add ``case_classification_validation`` and ``ecotyper_b_state`` to uns payload."""
    validation_uns = dict(validation_uns)
    case_cc = build_case_classification_validation(pred, classifications)
    case_cc = add_validation_genomic_tested(case_cc, validation_uns)
    validation_uns["case_classification_validation"] = case_cc
    try:
        validation_uns["ecotyper_b_state"] = build_ecotyper_b_state_validation(classifications)
    except ValueError:
        pass
    return validation_uns


def load_case_classification_validation(
    validation_uns: dict,
    pred: pd.DataFrame,
) -> pd.DataFrame:
    """Read embedded ``case_classification_validation`` from the validation cohort."""
    del pred  # retained for notebook call compatibility
    case_cc = validation_uns.get("case_classification_validation")
    if not isinstance(case_cc, pd.DataFrame):
        raise KeyError(
            "case_classification_validation not found in adata.uns['validation_cohort']"
        )
    case_cc = case_cc.copy()
    case_cc.index = case_cc.index.astype(str)
    if "genomic_tested" not in case_cc.columns:
        case_cc = add_validation_genomic_tested(case_cc, validation_uns)
    return case_cc


def load_ecotyper_b_state_validation(validation_uns: dict) -> pd.DataFrame:
    """EcoTyper B-state from embedded validation cohort table."""
    bstate = validation_uns.get("ecotyper_b_state")
    if not isinstance(bstate, pd.DataFrame):
        raise KeyError("ecotyper_b_state not found in adata.uns['validation_cohort']")
    out = bstate.copy()
    out.index = out.index.astype(str)
    return out
