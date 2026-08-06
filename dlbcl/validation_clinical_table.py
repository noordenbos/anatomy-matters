"""Validation-cohort clinical characteristics table (aligned to discovery gt layout)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

# disease_type -> discovery table column labels
DISEASE_TYPE_TO_LOCATION_GROUP = {
    "Bone": "Bone",
    "Brain": "PCNSL",
    "Nodal": "Nodal",
    "Testis": "Testis",
}

LOCATION_ORDER = ["Bone", "PCNSL", "Testis", "Nodal"]

EXPECTED_VALIDATION_N = 303
EXPECTED_LOCATION_COUNTS: dict[str, int] = {
    "Bone": 47,
    "PCNSL": 130,
    "Testis": 62,
    "Nodal": 64,
}

NA_PLACEHOLDER = "n.a."

# Discovery-aligned first-line treatment buckets (validation workbook labels mapped below).
TREATMENT_CATEGORY_ORDER = [
    "HD-MTX-based polychemotherapy",
    "RCHOP-like",
]

# Exact strings from validation_cohort.xlsx ``clinical_data.treatment``.
TREATMENT_EXACT_MAP: dict[str, str] = {
    "HD-MTX-based polychemotherapy with intent to consolidate": "HD-MTX-based polychemotherapy",
    "HD-MTX-based polychemoterapy without intent to consolidate": "HD-MTX-based polychemotherapy",
    "RCHOP-like (w/wo radiotherapy/MTX/orchidectomy)": "RCHOP-like",
    "Chemotherapy (w/wo radiotherapy/orchidectomy)": "RCHOP-like",
}

HD_MTX_INTENT_LABEL = "HD-MTX-based polychemotherapy with intent to consolidate"
HD_MTX_NO_INTENT_LABEL = "HD-MTX-based polychemoterapy without intent to consolidate"

HD_MTX_PATTERNS = (
    r"mbvp",
    r"matrix",
    r"\bmp\b",
    r"\brmp\b",
    r"mcpm",
    r"hd-mtx",
    r"r-mbvp",
    r"hd-arac",
    r"r \+ hd-mtx",
    r"rdhap\+mtx",
    r"r-mcpm",
)

SECTION_HEADER_ROWS = {
    "Ann Arbor stage:",
    "IPI/MSKCC-score:",
    "First-line treatment:",
    "Cell-of-origin (Lymph2CX):",
    "In situ hybridization:",
}

# Discovery anatomy_matters table uses legacy row labels; map to validation canonical labels.
CHARACTERISTIC_LABEL_ALIASES: dict[str, str] = {
    "IPI/IELSG-score:": "IPI/MSKCC-score:",
    "  RCHOP like": "  RCHOP-like",
    "  R-CHOP-like": "  RCHOP-like",
    "  R-CHOP-like (w/o RT)": "  RCHOP-like",
}

COO_LEVELS = ["ABC", "GCB", "Intermediate"]

DISCOVERY_COLUMN_PREFIX = "d"
VALIDATION_COLUMN_PREFIX = "v"


@dataclass
class ClinicalTableConfig:
    """Paths and source selection for the validation clinical table."""

    meta_csv: Path | None = None
    adata_path: Path | None = None
    classifications_csv: Path | None = None
    source: Literal["csv", "adata"] = "adata"
    output_dir: Path | None = None
    table_stem: str = "validation_clinical_characteristics_table"


SectionFootnotes = dict[str, list[str]]


@dataclass
class GroupStats:
    location: str
    total: str = ""
    age: str = NA_PLACEHOLDER
    sex_female: str = "NA"
    stage_I: str = "0"
    stage_II: str = "0"
    stage_III: str = "0"
    stage_IV: str = "0"
    stage_n: int = 0
    ipi_0_2: str = NA_PLACEHOLDER
    ipi_ge3: str = NA_PLACEHOLDER
    ipi_missing: int = 0
    treatment_by_level: dict[str, int] = field(default_factory=dict)
    treatment_n: int = 0
    treatment_unknown: int = 0
    coo_ABC: str = "0"
    coo_GCB: str = "0"
    coo_Intermediate: str = "0"
    myc_bcl2: str = "0"
    ish_MYC: str = "0 (n = 0)"
    ish_BCL2: str = "0 (n = 0)"
    ish_BCL6: str = "0 (n = 0)"
    ish_EBER: str = NA_PLACEHOLDER


def _is_missing(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "na", "unk", "unknown"}


def _is_fish_tested(value: object) -> bool:
    if _is_missing(value):
        return False
    text = str(value).strip()
    return not re.match(r"^(unk|unknown|not performed)$", text, flags=re.IGNORECASE)


def _is_fish_positive(value: object) -> bool:
    if not _is_fish_tested(value):
        return False
    text = str(value).strip().lower()
    if text in {"pos", "positive", "1", "yes", "y", "true"}:
        return True
    if text in {"neg", "negative", "0", "no", "n", "false"}:
        return False
    return bool(re.search(r"positive|^pos$", text, flags=re.IGNORECASE))


def _fish_positive_count(series: pd.Series) -> tuple[int, int]:
    tested_mask = series.map(_is_fish_tested)
    tested = series[tested_mask]
    n_tested = len(tested)
    n_pos = int(tested.map(_is_fish_positive).sum()) if n_tested else 0
    return n_pos, n_tested


def _parse_ann_arbor_stage(value: object, *, location_group: str) -> str | None:
    if location_group == "PCNSL":
        return "IV"
    if _is_missing(value):
        return None
    text = str(value).strip()
    if re.match(r"^pcnsl$", text, flags=re.IGNORECASE):
        return "IV"
    # Arabic stage used in discovery case_clinical (optional E/A/B/X suffix).
    arabic = re.match(r"^([1-4])([EABX/\-]*)?$", text, flags=re.IGNORECASE)
    if arabic:
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}[arabic.group(1)]
    # Match roman stage token at start (III before II before I); allow trailing A/B/E/X.
    if re.match(r"^iv(?![a-z])", text, flags=re.IGNORECASE) or re.match(r"^iv\(", text, flags=re.IGNORECASE):
        return "IV"
    if re.match(r"^iii(?![a-z])", text, flags=re.IGNORECASE) or re.match(r"^iii\(", text, flags=re.IGNORECASE):
        return "III"
    if re.match(r"^ii(?![a-z])", text, flags=re.IGNORECASE) or re.match(r"^ii\(", text, flags=re.IGNORECASE):
        return "II"
    if re.match(r"^i(?![iv])", text, flags=re.IGNORECASE):
        return "I"
    return None


def classify_treatment(value: object) -> str | None:
    """Map free-text treatment to discovery-aligned buckets; None = unknown."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text in TREATMENT_EXACT_MAP:
        return TREATMENT_EXACT_MAP[text]
    lower = text.lower()
    if re.search(r"rchop", lower):
        return "RCHOP-like"
    for pattern in HD_MTX_PATTERNS:
        if re.search(pattern, lower):
            return "HD-MTX-based polychemotherapy"
    return None


def _parse_ipi_bucket(value: object) -> Literal["low", "high"] | None:
    """Legacy helper: map a raw score/string to primary low/high when unambiguous.

    Prefer ``assign_ipi_ielsg_buckets`` / ``attach_primary_ipi_buckets`` for table builds.
    """
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"0-2", "0–2"} or "low-risk" in text or text in {"0", "1", "0-1", "2"}:
        return "low"
    if text in {">=3", "≥3", ">3"} or text in {"3", "4", "5"}:
        return "high"
    if "high-risk" in text or text in {"2-5"}:
        # Spans both primary buckets — ambiguous without exact score.
        return None
    num = pd.to_numeric(value, errors="coerce")
    if pd.notna(num):
        return "low" if float(num) <= 2 else "high"
    return None


def attach_primary_ipi_buckets(data: pd.DataFrame) -> pd.DataFrame:
    """Attach ``ipi_ielsg_primary`` (``0-2`` / ``>=3``) via component-aware scoring."""
    from .validation_ipi import assign_ipi_ielsg_buckets

    out = data.copy()
    buckets = assign_ipi_ielsg_buckets(out)
    out["ipi_ielsg_primary"] = buckets["bucket_primary"].reindex(out.index.astype(str)).values
    out["ipi_ielsg_system"] = buckets["system"].reindex(out.index.astype(str)).values
    return out


def collect_unresolved_treatments(data: pd.DataFrame) -> pd.DataFrame:
    """Return patients whose ``treatment`` string does not map to a table bucket."""
    work = data.copy()
    if "treatment" not in work.columns:
        return pd.DataFrame(columns=["patient_alias", "disease_type", "Location_group", "treatment"])
    if "Location_group" not in work.columns:
        work = add_location_group(work)
    cats = work["treatment"].map(classify_treatment)
    unresolved = work.loc[cats.isna()].copy()
    unresolved = unresolved.reset_index()
    if "patient_alias" not in unresolved.columns:
        # index name may be patient_alias / patient_id / None
        idx_name = work.index.name or "index"
        if idx_name in unresolved.columns:
            unresolved = unresolved.rename(columns={idx_name: "patient_alias"})
        else:
            unresolved.insert(0, "patient_alias", unresolved.index.astype(str))
    unresolved["patient_alias"] = unresolved["patient_alias"].astype(str)
    keep = [c for c in ("patient_alias", "disease_type", "Location_group", "treatment") if c in unresolved.columns]
    out = unresolved.loc[:, keep].copy()
    return out.sort_values([c for c in ("Location_group", "treatment", "patient_alias") if c in out.columns])


def _format_median_age_range(ages: pd.Series) -> str:
    known = pd.to_numeric(ages, errors="coerce").dropna()
    if known.empty:
        return NA_PLACEHOLDER
    med = float(known.median())
    return f"{med:.0f} ({known.min():.0f}–{known.max():.0f})"


def hd_mtx_consolidation_footnote(data: pd.DataFrame) -> str | None:
    """Share of HD-MTX patients with stated intent to consolidate."""
    if "treatment" not in data.columns:
        return None
    hd = data["treatment"].astype(str).str.strip()
    with_intent = int((hd == HD_MTX_INTENT_LABEL).sum())
    without_intent = int((hd == HD_MTX_NO_INTENT_LABEL).sum())
    total = with_intent + without_intent
    if total == 0:
        return None
    pct = round(100 * with_intent / total, 1)
    return (
        f"HD-MTX-based polychemotherapy: {pct}% ({with_intent}/{total}) "
        "with intent to consolidate"
    )


def coo_source_footnote(data: pd.DataFrame) -> str:
    """Describe how COO_NanoString was assigned for the validation table."""
    if "COO_NanoString" not in data.columns:
        return "Validation cell-of-origin: COO_NanoString not available"
    coo = data["COO_NanoString"]
    n_known = int((~coo.map(_is_missing)).sum())
    if "coo_hans" in data.columns:
        hans_proxy = coo.map(_is_missing) & data["coo_hans"].map(lambda v: not _is_missing(v))
        n_proxy = int(hans_proxy.sum())
        if n_proxy:
            return (
                f"Validation cell-of-origin: Lymph2CX (COO_NanoString) from validation workbook "
                f"({n_known - n_proxy}/{len(data)}); Hans IHC proxy for {n_proxy} missing NanoString"
            )
    return (
        f"Validation cell-of-origin: Lymph2CX (COO_NanoString) labels from validation workbook "
        f"({n_known}/{len(data)})"
    )


def enrich_with_coo_nanostring(
    df: pd.DataFrame,
    classifications_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Attach COO_NanoString from case classifications or workbook / Hans proxy."""
    out = df.copy()
    out.index = out.index.astype(str)
    if classifications_csv is not None and Path(classifications_csv).exists():
        cc = pd.read_csv(classifications_csv)
        id_col = "patient_id" if "patient_id" in cc.columns else "patient_alias"
        cc = cc.set_index(id_col)
        cc.index = cc.index.astype(str)
        if "COO_NanoString" in cc.columns:
            out["COO_NanoString"] = cc["COO_NanoString"].reindex(out.index)
            return out

    if "COO_NanoString" in out.columns and out["COO_NanoString"].notna().any():
        return out

    if "coo_hans" in out.columns:
        coo_map = {"GCB": "GCB", "Non-GCB": "ABC", "ABC": "ABC", "Unclassified": "Intermediate"}
        out["COO_NanoString"] = out["coo_hans"].map(coo_map)
    return out


def _count_coo_label(series: pd.Series, label: str) -> int:
    known = series[~series.map(_is_missing)]
    if known.empty:
        return 0
    return int(known.astype(str).str.strip().str.fullmatch(label, case=False).sum())


def normalize_characteristic_label(label: object) -> str:
    """Map legacy discovery row labels onto the validation canonical layout."""
    text = str(label)
    return CHARACTERISTIC_LABEL_ALIASES.get(text, text)


def _simplify_total_cell(value: object) -> object:
    """Render Total cells as bare n (``47``), accepting legacy ``Total (n = 47)``."""
    if _is_missing(value):
        return value
    text = str(value).strip()
    if text.isdigit():
        return text
    match = re.search(r"n\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return value


def normalize_clinical_table(table: pd.DataFrame) -> pd.DataFrame:
    """Rename alias characteristic rows and collapse duplicate labels."""
    out = table.copy()
    out["Characteristic"] = out["Characteristic"].astype(str).map(normalize_characteristic_label)
    value_cols = [c for c in out.columns if c != "Characteristic"]
    if not value_cols:
        return out

    def _coalesce_cell(series: pd.Series) -> object:
        for value in series:
            text = str(value).strip()
            if text and text.lower() != "nan":
                return value
        return series.iloc[0] if len(series) else ""

    out = out.groupby("Characteristic", as_index=False)[value_cols].agg(_coalesce_cell)
    total_mask = out["Characteristic"].astype(str) == "Total"
    if total_mask.any():
        for col in value_cols:
            out.loc[total_mask, col] = out.loc[total_mask, col].map(_simplify_total_cell)
    return out


def _parse_total_n(total_cell: object) -> int | None:
    if _is_missing(total_cell):
        return None
    text = str(total_cell).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"n\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _sum_table_row(table: pd.DataFrame, label: str, location: str) -> int | None:
    indexed = table.set_index("Characteristic")
    if label not in indexed.index:
        return None
    value = indexed.at[label, location]
    if _is_missing(value):
        return 0
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _table_category_sum_footnotes(
    table: pd.DataFrame,
    *,
    cohort_label: str,
    treatment_rows: list[str] | None = None,
) -> list[str]:
    """Explain when rendered category counts do not sum to the location total."""
    table = normalize_clinical_table(table)
    indexed = table.set_index("Characteristic")
    if "Total" not in indexed.index:
        return []

    treatment_rows = treatment_rows or [
        "  HD-MTX-based polychemotherapy",
        "  RCHOP-like",
    ]
    stage_rows = ["  I(X)B/E", "  II(X)A/E", "  III", "  IV"]
    ipi_rows = ["  0–2", "  ≥3"]
    coo_rows = ["  ABC", "  GCB", "  Intermediate"]
    footnotes: list[str] = []

    for loc in LOCATION_ORDER:
        if loc not in table.columns:
            continue
        n_total = _parse_total_n(indexed.at["Total", loc])
        if n_total is None:
            continue

        stage_sum = sum(_sum_table_row(table, row, loc) or 0 for row in stage_rows)
        if stage_sum != n_total:
            footnotes.append(
                f"{cohort_label} Ann Arbor stage: category counts sum to {stage_sum}/{n_total} "
                f"for {loc} (missing or non-applicable stage excluded)"
            )

        ipi_sum = sum(_sum_table_row(table, row, loc) or 0 for row in ipi_rows)
        if ipi_sum != n_total:
            footnotes.append(
                f"{cohort_label} IPI/IELSG: category counts sum to {ipi_sum}/{n_total} for {loc} "
                "(unassignable after partial primary bucketing 0–2 vs ≥3 excluded)"
            )

        treat_sum = sum(_sum_table_row(table, row, loc) or 0 for row in treatment_rows)
        if treat_sum != n_total:
            footnotes.append(
                f"{cohort_label} first-line treatment: category counts sum to {treat_sum}/{n_total} "
                f"for {loc} (unknown treatment or no curative intent excluded)"
            )

        coo_sum = sum(_sum_table_row(table, row, loc) or 0 for row in coo_rows)
        if coo_sum != n_total:
            footnotes.append(
                f"{cohort_label} cell-of-origin: category counts sum to {coo_sum}/{n_total} "
                f"for {loc} (missing COO excluded)"
            )

    return footnotes


def default_discovery_metadata_path(repo_root: Path | str) -> Path:
    """Best-effort path to discovery de-identified metadata (optional local CSV)."""
    repo_root = Path(repo_root).resolve()
    candidates = [
        repo_root / "data" / "discovery_meta" / "deidentified_patient_metadata_2026-07-03.csv",
        repo_root / "data" / "discovery_deidentified_patient_metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_discovery_elements_path(repo_root: Path | str) -> Path:
    """Path to discovery IPI/IELSG component table (partial scores for incomplete cases)."""
    repo_root = Path(repo_root).resolve()
    candidates = [
        repo_root / "data" / "discovery_meta" / "discovery_clinical_elements.csv",
        repo_root / "data" / "discovery_clinical_elements.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_discovery_clinical_elements(path: Path | str | None) -> pd.DataFrame:
    """Load per-patient IPI/IELSG components keyed by ``patient_id`` (``IMC_ID``)."""
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path).dropna(how="all")
    id_col = next((c for c in ("IMC_ID", "patient_id") if c in df.columns), None)
    if id_col is None:
        return pd.DataFrame()
    df = df.dropna(subset=[id_col]).copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df[df[id_col].ne("") & ~df[id_col].str.lower().isin({"nan", "none"})]
    df = df.drop_duplicates(subset=[id_col], keep="last").set_index(id_col)
    df.index.name = "patient_id"
    return df


def discovery_primary_bucket_series(
    metadata_csv: Path | str,
    *,
    elements_csv: Path | str | None = None,
) -> pd.Series:
    """Primary IPI/IELSG bucket per discovery patient (``0-2`` / ``>=3`` / NA).

    Concrete metadata scores win when present; otherwise component partial bucketing
    (same strategy as validation) is used when elements are available.
    """
    meta = pd.read_csv(metadata_csv)
    if "patient_id" not in meta.columns:
        raise ValueError(f"{metadata_csv} must contain patient_id")
    return discovery_primary_bucket_series_from_frame(meta, elements_csv=elements_csv)


def discovery_primary_bucket_series_from_frame(
    meta: pd.DataFrame,
    *,
    elements_csv: Path | str | None = None,
) -> pd.Series:
    """Primary IPI/IELSG bucket from a discovery clinical DataFrame."""
    from .validation_ipi import assign_ipi_ielsg_buckets

    work = meta.copy()
    if "patient_id" not in work.columns:
        work = work.reset_index()
        if work.columns[0] != "patient_id":
            work = work.rename(columns={work.columns[0]: "patient_id"})
    work["patient_id"] = work["patient_id"].astype(str)
    ipi_col = next((c for c in ("IPI/IELSG", "ipi_score", "IPI") if c in work.columns), None)
    patient_ids = work["patient_id"]
    buckets = pd.Series(pd.NA, index=patient_ids, dtype=object)
    buckets.index.name = "patient_id"

    stored_col = next((c for c in ("ipi_mskcc", "ipi_ielsg") if c in work.columns), None)
    if stored_col is not None:
        for pid, raw in zip(patient_ids, work[stored_col], strict=True):
            text_v = None if _is_missing(raw) else str(raw).strip()
            if text_v in {"0-2", "0–2"}:
                buckets.loc[pid] = "0-2"
            elif text_v in {">=3", "≥3"}:
                buckets.loc[pid] = ">=3"

    if ipi_col is not None:
        for pid, raw in zip(patient_ids, work[ipi_col], strict=True):
            if not pd.isna(buckets.loc[pid]):
                continue
            parsed = _parse_ipi_bucket(raw)
            if parsed == "low":
                buckets.loc[pid] = "0-2"
            elif parsed == "high":
                buckets.loc[pid] = ">=3"

    need = buckets.isna()
    if not need.any():
        return buckets

    if elements_csv is None:
        elements_csv = default_discovery_elements_path(Path.cwd())

    scored = assign_ipi_ielsg_buckets(
        discovery_ipi_score_frame_from_meta(work, elements_csv=elements_csv)
    )
    for pid in buckets.index[need]:
        if pid not in scored.index:
            continue
        primary = scored.at[pid, "bucket_primary"]
        if primary is None or (isinstance(primary, float) and pd.isna(primary)):
            continue
        text = str(primary).strip()
        if text in {"0-2", ">=3"}:
            buckets.loc[pid] = text
    return buckets


def discovery_ipi_score_frame_from_meta(
    meta: pd.DataFrame,
    *,
    elements_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Build a scoring frame for ``assign_ipi_ielsg_buckets`` from a discovery clinical frame."""
    work = meta.copy()
    if "patient_id" not in work.columns:
        work = work.reset_index()
        if work.columns[0] != "patient_id":
            work = work.rename(columns={work.columns[0]: "patient_id"})
    work["patient_id"] = work["patient_id"].astype(str)
    loc_col = "Location_group" if "Location_group" in work.columns else "Location"
    ipi_col = next((c for c in ("IPI/IELSG", "ipi_score", "IPI") if c in work.columns), None)

    out = pd.DataFrame(index=work["patient_id"].astype(str))
    out.index.name = "patient_id"
    age_col = "Age" if "Age" in work.columns else ("age" if "age" in work.columns else None)
    if age_col is not None:
        out["age"] = pd.to_numeric(work[age_col], errors="coerce").values
    if "Ann_Arbor_at_Dx" in work.columns:
        out["Ann_Arbor_at_Dx"] = work["Ann_Arbor_at_Dx"].values
    out["Location_group"] = work[loc_col].astype(str).values
    out["Location"] = out["Location_group"]
    out["location"] = out["Location_group"]
    out["disease_type"] = out["Location_group"].map(
        {"PCNSL": "Brain", "Bone": "Bone", "Testis": "Testis", "Nodal": "Nodal"}
    )
    if ipi_col is not None:
        out["ipi_score"] = pd.to_numeric(work[ipi_col], errors="coerce").values

    elements = load_discovery_clinical_elements(elements_csv)
    if elements.empty:
        return out

    for col in (
        "Ann_Arbor_at_Dx",
        "ipi_score",
        "ipi_extranodal",
        "ipi_ldh",
        "ipi_who",
        "ielsg_ecog",
        "ielsg_ldh",
        "ielsg_csf",
        "ielsg_deepbrain",
    ):
        if col not in elements.columns:
            continue
        if col not in out.columns:
            out[col] = pd.NA
        el = elements[col].reindex(out.index)
        use = ~el.map(_is_missing)
        out.loc[use, col] = el.loc[use].values
    return out


def discovery_ipi_score_frame(
    metadata_csv: Path | str,
    *,
    elements_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Build a scoring frame for ``assign_ipi_ielsg_buckets`` from discovery metadata + elements."""
    meta = pd.read_csv(metadata_csv)
    if "patient_id" not in meta.columns:
        raise ValueError(f"{metadata_csv} must contain patient_id")
    return discovery_ipi_score_frame_from_meta(meta, elements_csv=elements_csv)


def _format_site_missing_counts(counts: dict[str, int]) -> str:
    """Format ``10 (3 Bone, 4 PCNSL, 1 Testis, 2 Nodal)`` preserving location order."""
    total = int(sum(int(v) for v in counts.values()))
    parts = [f"{int(counts[loc])} {loc}" for loc in LOCATION_ORDER if int(counts.get(loc, 0))]
    if parts:
        return f"{total} ({', '.join(parts)})"
    return str(total)


def _missing_counts_from_table(
    table: pd.DataFrame,
    category_rows: list[str],
) -> dict[str, int]:
    """Per-location missing = Total − sum(category rows) for a clinical table."""
    table = normalize_clinical_table(table)
    indexed = table.set_index("Characteristic")
    if "Total" not in indexed.index:
        return {}
    missing: dict[str, int] = {}
    for loc in LOCATION_ORDER:
        if loc not in table.columns:
            continue
        n_total = _parse_total_n(indexed.at["Total", loc])
        if n_total is None:
            continue
        cat_sum = sum(_sum_table_row(table, row, loc) or 0 for row in category_rows)
        miss = int(n_total) - int(cat_sum)
        if miss > 0:
            missing[loc] = miss
    return missing


def _validation_stage_missing_counts(results: list[GroupStats]) -> dict[str, int]:
    missing: dict[str, int] = {}
    for stats in results:
        try:
            n_total = int(str(stats.total).strip())
        except ValueError:
            continue
        miss = n_total - int(stats.stage_n)
        if miss > 0:
            missing[stats.location] = miss
    return missing


def _validation_coo_missing_counts(results: list[GroupStats]) -> dict[str, int]:
    missing: dict[str, int] = {}
    for stats in results:
        try:
            n_total = int(str(stats.total).strip())
        except ValueError:
            continue
        known = int(stats.coo_ABC) + int(stats.coo_GCB) + int(stats.coo_Intermediate)
        miss = n_total - known
        if miss > 0:
            missing[stats.location] = miss
    return missing


def _validation_ipi_missing_counts(results: list[GroupStats]) -> dict[str, int]:
    return {
        stats.location: int(stats.ipi_missing)
        for stats in results
        if stats.ipi_missing
    }


def assignable_missing_footnote(
    feature: str,
    *,
    discovery: dict[str, int] | None = None,
    validation: dict[str, int] | None = None,
) -> str | None:
    """Combined missingness note, e.g. ``patients without assignable IPI/IELSG: Discovery: …; Validation: …``."""
    parts: list[str] = []
    if discovery and sum(discovery.values()) > 0:
        parts.append(f"Discovery: {_format_site_missing_counts(discovery)}")
    if validation and sum(validation.values()) > 0:
        parts.append(f"Validation: {_format_site_missing_counts(validation)}")
    if not parts:
        return None
    return f"patients without assignable {feature}: " + "; ".join(parts)


STAGE_CATEGORY_ROWS = ["  I(X)B/E", "  II(X)A/E", "  III", "  IV"]
COO_CATEGORY_ROWS = ["  ABC", "  GCB", "  Intermediate"]


def discovery_ipi_missing_counts(
    metadata: Path | str | pd.DataFrame | None,
    *,
    elements_csv: Path | str | None = None,
) -> dict[str, int]:
    """Discovery patients without an assignable primary IPI/IELSG bucket."""
    if metadata is None:
        return {}
    if isinstance(metadata, pd.DataFrame):
        meta = metadata.copy()
        if "patient_id" not in meta.columns:
            meta = meta.reset_index()
            if meta.columns[0] != "patient_id":
                meta = meta.rename(columns={meta.columns[0]: "patient_id"})
        buckets = discovery_primary_bucket_series_from_frame(meta, elements_csv=elements_csv)
    else:
        if not Path(metadata).exists():
            return {}
        meta = pd.read_csv(metadata)
        buckets = discovery_primary_bucket_series(metadata, elements_csv=elements_csv)
    loc_col = "Location_group" if "Location_group" in meta.columns else "Location"
    loc = meta.set_index(meta["patient_id"].astype(str))[loc_col].astype(str)
    missing = buckets.isna()
    if not missing.any():
        return {}
    return {
        site: int((missing & (loc == site)).sum())
        for site in LOCATION_ORDER
        if int((missing & (loc == site)).sum())
    }


def discovery_ipi_missing_footnote(
    metadata_csv: Path | str | None,
    *,
    elements_csv: Path | str | None = None,
) -> str | None:
    """Discovery-only IPI/IELSG missingness footnote."""
    return assignable_missing_footnote(
        "IPI/MSKCC",
        discovery=discovery_ipi_missing_counts(metadata_csv, elements_csv=elements_csv),
    )


def validation_ipi_missing_footnote(results: list[GroupStats]) -> str | None:
    """Validation-only IPI/MSKCC missingness footnote."""
    return assignable_missing_footnote(
        "IPI/MSKCC",
        validation=_validation_ipi_missing_counts(results),
    )


def combined_ipi_missing_footnote(
    *,
    group_stats: list[GroupStats],
    discovery_metadata_csv: Path | str | pd.DataFrame | None = None,
    discovery_elements_csv: Path | str | None = None,
) -> str | None:
    return assignable_missing_footnote(
        "IPI/MSKCC",
        discovery=discovery_ipi_missing_counts(
            discovery_metadata_csv, elements_csv=discovery_elements_csv
        ),
        validation=_validation_ipi_missing_counts(group_stats),
    )


def combined_ann_arbor_missing_footnote(
    *,
    discovery_table: pd.DataFrame | None,
    group_stats: list[GroupStats],
) -> str | None:
    discovery = (
        _missing_counts_from_table(discovery_table, STAGE_CATEGORY_ROWS)
        if discovery_table is not None
        else {}
    )
    return assignable_missing_footnote(
        "Ann Arbor stage",
        discovery=discovery,
        validation=_validation_stage_missing_counts(group_stats),
    )


def combined_coo_missing_footnote(
    *,
    discovery_table: pd.DataFrame | None,
    group_stats: list[GroupStats],
) -> str | None:
    discovery = (
        _missing_counts_from_table(discovery_table, COO_CATEGORY_ROWS)
        if discovery_table is not None
        else {}
    )
    return assignable_missing_footnote(
        "cell-of-origin",
        discovery=discovery,
        validation=_validation_coo_missing_counts(group_stats),
    )


def validation_ann_arbor_missing_footnote(results: list[GroupStats]) -> str | None:
    return assignable_missing_footnote(
        "Ann Arbor stage",
        validation=_validation_stage_missing_counts(results),
    )


def validation_coo_missing_footnote(results: list[GroupStats]) -> str | None:
    return assignable_missing_footnote(
        "cell-of-origin",
        validation=_validation_coo_missing_counts(results),
    )


def discovery_no_curative_intent_footnote(
    metadata: Path | str | pd.DataFrame | None,
) -> str | None:
    """Discovery patients excluded from first-line treatment counts."""
    if metadata is None:
        return None
    if isinstance(metadata, pd.DataFrame):
        meta = metadata.copy()
    else:
        if not Path(metadata).exists():
            return None
        meta = pd.read_csv(metadata)

    loc_col = "Location_group" if "Location_group" in meta.columns else "Location"
    treat_col = next(
        (c for c in ("Treatment category", "treatment") if c in meta.columns),
        None,
    )
    if treat_col is None or loc_col not in meta.columns:
        return None

    tc = meta[loc_col].astype(str)
    no_curative = meta[treat_col].astype(str).str.match(
        r"^(No curative intent|Geen curatieve behandeling)$", case=False, na=False
    )
    if not no_curative.any():
        return None

    parts = []
    for loc in LOCATION_ORDER:
        n = int((no_curative & (tc == loc)).sum())
        if n:
            parts.append(f"{n} {loc}")
    return (
        f"Discovery: {int(no_curative.sum())} patients with no curative intent excluded "
        f"from treatment rows ({', '.join(parts)})"
    )


def rchop_like_without_rituximab_footnote(data: pd.DataFrame) -> str | None:
    """Patients bucketed as RCHOP-like despite non-R-CHOP workbook wording."""
    if "treatment" not in data.columns:
        return None
    categories = data["treatment"].map(classify_treatment)
    mask = categories == "RCHOP-like"
    non_r_label = mask & ~data["treatment"].astype(str).str.lower().str.contains("rchop", na=False)
    n = int(non_r_label.sum())
    if n == 0:
        return None
    parts = []
    if "Location_group" in data.columns:
        for loc in LOCATION_ORDER:
            count = int((non_r_label & (data["Location_group"] == loc)).sum())
            if count:
                parts.append(f"{count} {loc}")
    detail = f" ({', '.join(parts)})" if parts else ""
    return (
        f"Validation: RCHOP-like includes {n} patient{'s' if n != 1 else ''}{detail} "
        "with anthracycline-based chemotherapy without rituximab"
    )


def discovery_primary_ipi_counts(
    metadata: Path | str | pd.DataFrame,
    *,
    elements_csv: Path | str | None = None,
) -> dict[str, dict[str, int]]:
    """Count discovery patients per location in primary IPI buckets.

    Uses concrete ``IPI/IELSG`` when present; otherwise component partial bucketing.
    """
    if isinstance(metadata, pd.DataFrame):
        meta = metadata.copy()
        if "patient_id" not in meta.columns:
            meta = meta.reset_index()
            if meta.columns[0] != "patient_id":
                meta = meta.rename(columns={meta.columns[0]: "patient_id"})
        buckets = discovery_primary_bucket_series_from_frame(meta, elements_csv=elements_csv)
    else:
        meta = pd.read_csv(metadata)
        buckets = discovery_primary_bucket_series(metadata, elements_csv=elements_csv)
    loc_col = "Location_group" if "Location_group" in meta.columns else "Location"
    loc = meta.set_index(meta["patient_id"].astype(str))[loc_col].astype(str)

    counts = {site: {"0-2": 0, ">=3": 0, "missing": 0} for site in LOCATION_ORDER}
    for pid, bucket in buckets.items():
        site = str(loc.get(pid, ""))
        if site not in counts:
            continue
        if pd.isna(bucket):
            counts[site]["missing"] += 1
        elif str(bucket) == "0-2":
            counts[site]["0-2"] += 1
        elif str(bucket) == ">=3":
            counts[site][">=3"] += 1
        else:
            counts[site]["missing"] += 1
    return counts


def rebucket_discovery_ipi_table(
    discovery_table: pd.DataFrame,
    metadata: Path | str | pd.DataFrame | None,
    *,
    elements_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Replace discovery IPI rows with primary ``0–2`` / ``≥3`` (concrete + components)."""
    out = normalize_clinical_table(discovery_table)
    if metadata is None:
        return out
    if not isinstance(metadata, pd.DataFrame) and not Path(metadata).exists():
        return out

    if elements_csv is None and not isinstance(metadata, pd.DataFrame):
        meta_path = Path(metadata).resolve()
        sidecar = meta_path.parent / "discovery_clinical_elements.csv"
        elements_csv = sidecar if sidecar.exists() else default_discovery_elements_path(meta_path.parents[1])

    counts = discovery_primary_ipi_counts(metadata, elements_csv=elements_csv)
    legacy = {"  0–1", "  0-1", "  2–3", "  2-3", "  >3", "  ≥3", "  0–2", "  0-2", "  >=3"}
    out = out.loc[~out["Characteristic"].astype(str).isin(legacy)].copy()

    header_idx = out.index[
        out["Characteristic"].astype(str).isin({"IPI/MSKCC-score:", "IPI/IELSG-score:"})
    ].tolist()
    insert_at = int(header_idx[0]) + 1 if header_idx else len(out)

    low_row = {"Characteristic": "  0–2"}
    high_row = {"Characteristic": "  ≥3"}
    for loc in LOCATION_ORDER:
        if loc in out.columns:
            low_row[loc] = str(counts[loc]["0-2"])
            high_row[loc] = str(counts[loc][">=3"])

    top = out.iloc[:insert_at]
    bottom = out.iloc[insert_at:]
    out = pd.concat(
        [top, pd.DataFrame([low_row, high_row]), bottom],
        ignore_index=True,
    )
    return out


# Footnote placement: (row Characteristic label, column name or None for row header, note)
FootnotePlacement = tuple[str, str | None, str]

# Journal SVG layout (matches manuscript figure font)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_MARGIN_MM = 12.0
TABLE_FONT_FAMILY = "DejaVu Sans"
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _to_superscript(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT)


def _footnote_marker_map(
    placements: list[FootnotePlacement],
) -> tuple[dict[tuple[str, str | None], list[int]], list[str]]:
    """Map (row_label, column) -> footnote numbers (1-based) and ordered note texts."""
    markers: dict[tuple[str, str | None], list[int]] = {}
    notes: list[str] = []
    for row_label, column, note in placements:
        notes.append(note)
        markers.setdefault((row_label, column), []).append(len(notes))
    return markers, notes


def _cell_display(
    value: object,
    *,
    row_label: str,
    column: str | None,
    markers: dict[tuple[str, str | None], list[int]],
) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    if text.lower() == "nan":
        text = ""
    marks = markers.get((row_label, column), [])
    if not marks:
        return text
    return text + "".join(_to_superscript(n) for n in marks)


def _measure_text_width_in(
    text: str,
    *,
    fontsize: float,
    weight: str = "normal",
    style: str = "normal",
) -> float:
    """Approximate text width in inches for DejaVu Sans (no renderer needed)."""
    # DejaVu Sans average glyph width ≈ 0.52 em for mixed alphanumeric; pad bold/italic lightly.
    scale = 0.56 if weight == "bold" else 0.52
    if style == "italic":
        scale *= 1.02
    return max(len(text), 1) * fontsize * scale / 72.0


def render_clinical_table_svg(
    table_data: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str,
    value_columns: list[str],
    column_labels: dict[str, str] | None = None,
    spanners: list[tuple[str, list[str]]] | None = None,
    footnote_placements: list[FootnotePlacement] | None = None,
    font_size: float = 7.5,
    title_size: float = 9.0,
    page: Literal["a4", "tight"] = "a4",
) -> Path:
    """Draw a journal-ready SVG table (DejaVu Sans, editable text via svg.fonttype=none).

    ``page='a4'`` places the table on an A4 portrait canvas at content-minimal width
    (columns sized to text + padding so cells do not collide). ``page='tight'`` crops
    to the table bbox.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    markers, notes = _footnote_marker_map(footnote_placements or [])
    labels = column_labels or {c: c for c in value_columns}
    display_cols = ["Characteristic", *value_columns]
    rows = table_data[display_cols].copy()
    rows["Characteristic"] = rows["Characteristic"].astype(str)

    # Cell strings (with footnote markers)
    cell_text: list[list[str]] = []
    cell_weight: list[list[str]] = []
    cell_style: list[list[str]] = []
    for _, row in rows.iterrows():
        row_label = str(row["Characteristic"])
        is_section = row_label in SECTION_HEADER_ROWS
        line: list[str] = []
        weights: list[str] = []
        styles: list[str] = []
        for col in display_cols:
            mark_col: str | None = None if col == "Characteristic" else col
            line.append(
                _cell_display(
                    row[col],
                    row_label=row_label,
                    column=mark_col,
                    markers=markers,
                )
            )
            if is_section and col == "Characteristic":
                weights.append("bold")
                styles.append("italic")
            else:
                weights.append("normal")
                styles.append("normal")
        cell_text.append(line)
        cell_weight.append(weights)
        cell_style.append(styles)

    header_labels = ["Characteristic", *[labels.get(c, c) for c in value_columns]]

    pad_in = 0.08  # horizontal padding inside each cell
    min_col_in = 0.55
    char_min_in = 2.05

    col_widths = []
    for j, col in enumerate(display_cols):
        candidates = [header_labels[j]]
        for i in range(len(cell_text)):
            candidates.append(cell_text[i][j])
        if spanners:
            for spanner_label, span_cols in spanners:
                if j > 0 and display_cols[j] in span_cols:
                    # Attribute a share of spanner width
                    candidates.append(spanner_label)
        widest = max(
            (
                _measure_text_width_in(
                    t,
                    fontsize=font_size,
                    weight="bold" if j == 0 and t in SECTION_HEADER_ROWS else ("bold" if j > 0 else "normal"),
                )
                for t in candidates
            ),
            default=min_col_in,
        )
        width = widest + 2 * pad_in
        if j == 0:
            width = max(width, char_min_in)
        else:
            width = max(width, min_col_in)
        col_widths.append(width)

    # Spanner row may need wider grouped columns
    if spanners:
        for spanner_label, span_cols in spanners:
            idxs = [display_cols.index(c) for c in span_cols if c in display_cols]
            if not idxs:
                continue
            span_width = sum(col_widths[i] for i in idxs)
            need = _measure_text_width_in(spanner_label, fontsize=font_size, weight="bold") + 2 * pad_in
            if need > span_width:
                extra = (need - span_width) / len(idxs)
                for i in idxs:
                    col_widths[i] += extra

    table_width = sum(col_widths)
    row_h = font_size / 72.0 * 1.55
    header_h = row_h * 1.15
    spanner_h = row_h * 1.05 if spanners else 0.0
    title_h = title_size / 72.0 * 1.8
    n_body = len(cell_text)
    table_body_h = n_body * row_h
    footnote_line_h = font_size / 72.0 * 1.35
    # wrap footnotes roughly to table width
    footnote_lines = 0
    chars_per_line = max(int(table_width / (_measure_text_width_in("M", fontsize=font_size - 0.5) or 0.05)), 40)
    for i, note in enumerate(notes, start=1):
        wrapped = (len(f"{i} {note}") // chars_per_line) + 1
        footnote_lines += wrapped
    footnotes_h = footnote_lines * footnote_line_h + (0.12 if notes else 0.0)

    content_h = title_h + spanner_h + header_h + table_body_h + footnotes_h + 0.15

    margin_in = _mm_to_in(A4_MARGIN_MM)
    a4_w = _mm_to_in(A4_WIDTH_MM)
    a4_h = _mm_to_in(A4_HEIGHT_MM)
    usable_w = a4_w - 2 * margin_in

    # If table wider than usable A4 width, scale font-derived widths down
    scale = 1.0
    if table_width > usable_w:
        scale = usable_w / table_width
        col_widths = [w * scale for w in col_widths]
        table_width = sum(col_widths)
        row_h *= scale
        header_h *= scale
        spanner_h *= scale
        title_h *= scale
        table_body_h = n_body * row_h
        footnotes_h *= scale
        content_h = title_h + spanner_h + header_h + table_body_h + footnotes_h + 0.15
        font_size *= scale
        title_size *= scale

    if page == "a4":
        fig_w, fig_h = a4_w, a4_h
        # Left-align within margins; vertically from top margin
        origin_x = margin_in
        origin_y = a4_h - margin_in - content_h
        if origin_y < margin_in:
            # overflow: shrink to fit remaining height
            fit = (a4_h - 2 * margin_in) / content_h
            col_widths = [w * fit for w in col_widths]
            table_width = sum(col_widths)
            row_h *= fit
            header_h *= fit
            spanner_h *= fit
            title_h *= fit
            table_body_h = n_body * row_h
            footnotes_h *= fit
            content_h = title_h + spanner_h + header_h + table_body_h + footnotes_h + 0.15
            font_size *= fit
            title_size *= fit
            origin_y = margin_in
    else:
        fig_w = table_width + 2 * margin_in * 0.4
        fig_h = content_h + 2 * margin_in * 0.4
        origin_x = margin_in * 0.4
        origin_y = margin_in * 0.4

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = TABLE_FONT_FAMILY

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    y = origin_y + content_h

    def _draw_text(x, y_center, text, *, fontsize, weight="normal", style="normal", ha="left", color="black"):
        ax.text(
            x,
            y_center,
            text,
            fontsize=fontsize,
            fontfamily=TABLE_FONT_FAMILY,
            fontweight=weight,
            fontstyle=style,
            ha=ha,
            va="center",
            color=color,
            clip_on=False,
        )

    # Title
    y -= title_h / 2
    _draw_text(origin_x, y, title, fontsize=title_size, weight="bold")
    y -= title_h / 2

    # Spanner row
    if spanners and spanner_h > 0:
        y_top = y
        y_bot = y - spanner_h
        x = origin_x + col_widths[0]
        # left characteristic blank under spanner
        ax.plot([origin_x, origin_x + table_width], [y_top, y_top], color="black", lw=0.8, solid_capstyle="butt")
        for spanner_label, span_cols in spanners:
            idxs = [display_cols.index(c) for c in span_cols if c in display_cols]
            if not idxs:
                continue
            x0 = origin_x + sum(col_widths[: idxs[0]])
            x1 = origin_x + sum(col_widths[: idxs[-1] + 1])
            mid = (x0 + x1) / 2
            _draw_text(mid, (y_top + y_bot) / 2, spanner_label, fontsize=font_size, weight="bold", ha="center")
            ax.plot([x0, x1], [y_bot + 0.02, y_bot + 0.02], color="black", lw=0.6)
        y = y_bot

    # Column header row
    y_top = y
    y_bot = y - header_h
    ax.plot([origin_x, origin_x + table_width], [y_top, y_top], color="black", lw=0.8)
    x = origin_x
    for j, lab in enumerate(header_labels):
        cx = x + col_widths[j] / 2 if j > 0 else x + pad_in
        ha = "center" if j > 0 else "left"
        _draw_text(cx, (y_top + y_bot) / 2, lab, fontsize=font_size, weight="bold", ha=ha)
        x += col_widths[j]
    ax.plot([origin_x, origin_x + table_width], [y_bot, y_bot], color="black", lw=0.8)
    y = y_bot

    # Body rows
    for i, line in enumerate(cell_text):
        y_top = y
        y_bot = y - row_h
        x = origin_x
        for j, text in enumerate(line):
            cx = x + col_widths[j] / 2 if j > 0 else x + pad_in
            ha = "center" if j > 0 else "left"
            _draw_text(
                cx,
                (y_top + y_bot) / 2,
                text,
                fontsize=font_size,
                weight=cell_weight[i][j],
                style=cell_style[i][j],
                ha=ha,
            )
            x += col_widths[j]
        # light rule under section headers and after last row
        row_label = str(rows.iloc[i]["Characteristic"])
        if row_label in SECTION_HEADER_ROWS or i == len(cell_text) - 1:
            ax.plot([origin_x, origin_x + table_width], [y_bot, y_bot], color="black", lw=0.5)
        y = y_bot

    ax.plot([origin_x, origin_x + table_width], [y, y], color="black", lw=0.8)

    # Footnotes
    if notes:
        y -= 0.1
        for n, note in enumerate(notes, start=1):
            prefix = f"{_to_superscript(n)} "
            full = prefix + note
            # simple wrap
            max_chars = max(int(chars_per_line * (table_width / max(sum(col_widths), 1e-6))), 50)
            words = full.split()
            cur = ""
            for word in words:
                trial = f"{cur} {word}".strip()
                if len(trial) > max_chars and cur:
                    y -= footnote_line_h / 2
                    _draw_text(origin_x, y, cur, fontsize=max(font_size - 0.5, 5.5))
                    y -= footnote_line_h / 2
                    cur = word
                else:
                    cur = trial
            if cur:
                y -= footnote_line_h / 2
                _draw_text(origin_x, y, cur, fontsize=max(font_size - 0.5, 5.5))
                y -= footnote_line_h / 2

    fig.savefig(out_path, format="svg", facecolor="white")
    plt.close(fig)
    return out_path


def save_clinical_table_svg(
    table_data: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str,
    footnote_placements: list[FootnotePlacement] | None = None,
    combined: bool = False,
    page: Literal["a4", "tight"] = "a4",
) -> Path:
    """Convenience wrapper for validation-only or combined SVG export."""
    if combined:
        discovery_cols = _combined_column_names(DISCOVERY_COLUMN_PREFIX)
        validation_cols = _combined_column_names(VALIDATION_COLUMN_PREFIX)
        value_columns = [*discovery_cols, *validation_cols]
        column_labels = {
            **{f"{DISCOVERY_COLUMN_PREFIX}_{loc}": loc for loc in LOCATION_ORDER},
            **{f"{VALIDATION_COLUMN_PREFIX}_{loc}": loc for loc in LOCATION_ORDER},
        }
        spanners = [
            ("Discovery", discovery_cols),
            ("Validation", validation_cols),
        ]
    else:
        value_columns = list(LOCATION_ORDER)
        column_labels = {loc: loc for loc in LOCATION_ORDER}
        spanners = None
    return render_clinical_table_svg(
        table_data,
        out_path,
        title=title,
        value_columns=value_columns,
        column_labels=column_labels,
        spanners=spanners,
        footnote_placements=footnote_placements,
        page=page,
    )


def _attach_footnote_placements(
    gt_table: object,
    display_df: pd.DataFrame,
    placements: list[FootnotePlacement],
) -> object:
    from great_tables import loc

    row_by_label = {
        label: idx for idx, label in enumerate(display_df["Characteristic"].astype(str))
    }
    for row_label, column, note in placements:
        row_idx = row_by_label.get(row_label)
        if row_idx is None:
            continue
        col = column if column is not None else "Characteristic"
        if col not in display_df.columns:
            continue
        gt_table = gt_table.tab_footnote(
            footnote=note,
            locations=loc.body(columns=col, rows=row_idx),
        )
    return gt_table


def _attach_section_footnotes(gt_table: object, display_df: pd.DataFrame, footnotes: SectionFootnotes) -> object:
    placements: list[FootnotePlacement] = []
    for section, notes in footnotes.items():
        for note in notes:
            placements.append((section, None, note))
    return _attach_footnote_placements(gt_table, display_df, placements)


def validation_section_footnotes(
    results: list[GroupStats],
    data: pd.DataFrame,
) -> SectionFootnotes:
    """Section-anchored footnotes for the validation-only clinical table (legacy dict)."""
    notes: SectionFootnotes = {}
    ann_notes = [
        "PCNSL reported here as stage IV (formal staging not applicable)."
    ]
    ann_missing = validation_ann_arbor_missing_footnote(results)
    if ann_missing:
        ann_notes.append(ann_missing)
    notes["Ann Arbor stage:"] = ann_notes

    ipi_notes = [
        "MSKCC prognostic class is used exclusively for PCNSL; Bone, Testis, and Nodal use IPI.",
    ]
    ipi_note = validation_ipi_missing_footnote(results)
    if ipi_note:
        ipi_notes.append(ipi_note)
    notes["IPI/MSKCC-score:"] = ipi_notes

    coo_missing = validation_coo_missing_footnote(results)
    if coo_missing:
        notes["Cell-of-origin (Lymph2CX):"] = [coo_missing]

    notes["In situ hybridization:"] = ["ISH/FISH/EBER: positive cases (n = tested)."]
    return notes


def validation_footnote_placements(
    results: list[GroupStats],
    data: pd.DataFrame,
) -> list[FootnotePlacement]:
    """Validation-only footnote targets (section headers + treatment cells/rows)."""
    placements: list[FootnotePlacement] = []
    for section, notes in validation_section_footnotes(results, data).items():
        for note in notes:
            placements.append((section, None, note))

    anthracycline_note = rchop_like_without_rituximab_footnote(data)
    if anthracycline_note:
        placements.append(("  RCHOP-like", None, anthracycline_note))
    return placements


def combined_section_footnotes(
    *,
    validation_data: pd.DataFrame,
    group_stats: list[GroupStats],
    discovery_metadata_csv: Path | str | pd.DataFrame | None = None,
    discovery_table: pd.DataFrame | None = None,
    discovery_elements_csv: Path | str | None = None,
) -> SectionFootnotes:
    """Section-header footnotes for the side-by-side discovery/validation table."""
    notes: SectionFootnotes = {}
    ann_notes = [
        "PCNSL reported here as stage IV (formal staging not applicable)."
    ]
    ann_missing = combined_ann_arbor_missing_footnote(
        discovery_table=discovery_table,
        group_stats=group_stats,
    )
    if ann_missing:
        ann_notes.append(ann_missing)
    notes["Ann Arbor stage:"] = ann_notes

    elements = discovery_elements_csv
    if (
        elements is None
        and discovery_metadata_csv is not None
        and not isinstance(discovery_metadata_csv, pd.DataFrame)
    ):
        parent = Path(discovery_metadata_csv).resolve().parent
        sidecar = parent / "discovery_clinical_elements.csv"
        elements = sidecar if sidecar.exists() else None

    ipi_notes = [
        "MSKCC prognostic class is used exclusively for PCNSL; Bone, Testis, and Nodal use IPI.",
    ]
    ipi_note = combined_ipi_missing_footnote(
        group_stats=group_stats,
        discovery_metadata_csv=discovery_metadata_csv,
        discovery_elements_csv=elements,
    )
    if ipi_note:
        ipi_notes.append(ipi_note)
    notes["IPI/MSKCC-score:"] = ipi_notes

    coo_missing = combined_coo_missing_footnote(
        discovery_table=discovery_table,
        group_stats=group_stats,
    )
    if coo_missing:
        notes["Cell-of-origin (Lymph2CX):"] = [coo_missing]

    notes["In situ hybridization:"] = ["ISH/FISH/EBER: positive cases (n = tested)."]
    return notes


def combined_footnote_placements(
    *,
    validation_data: pd.DataFrame,
    group_stats: list[GroupStats],
    discovery_metadata_csv: Path | str | pd.DataFrame | None = None,
    discovery_table: pd.DataFrame | None = None,
    discovery_elements_csv: Path | str | None = None,
) -> list[FootnotePlacement]:
    """Combined-table footnote targets including treatment cell/row markers."""
    placements: list[FootnotePlacement] = []
    for section, notes in combined_section_footnotes(
        validation_data=validation_data,
        group_stats=group_stats,
        discovery_metadata_csv=discovery_metadata_csv,
        discovery_table=discovery_table,
        discovery_elements_csv=discovery_elements_csv,
    ).items():
        for note in notes:
            placements.append((section, None, note))

    no_curative = discovery_no_curative_intent_footnote(discovery_metadata_csv)
    if no_curative:
        placements.append(("First-line treatment:", None, no_curative))

    anthracycline_note = rchop_like_without_rituximab_footnote(validation_data)
    if anthracycline_note:
        placements.append(("  RCHOP-like", None, anthracycline_note))
    return placements


def load_case_clinical_frame(adata: object) -> pd.DataFrame:
    """Return ``adata.uns['case_clinical']`` with ``patient_id`` index."""
    if "case_clinical" not in getattr(adata, "uns", {}):
        raise KeyError("Expected adata.uns['case_clinical']")
    df = pd.DataFrame(adata.uns["case_clinical"]).copy()
    if "patient_id" not in df.columns:
        df = df.reset_index()
        if df.columns[0] != "patient_id":
            df = df.rename(columns={df.columns[0]: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str)
    return df.set_index("patient_id")


def prepare_discovery_clinical_frame(
    adata: object,
    *,
    elements_csv: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> pd.DataFrame:
    """Map discovery ``case_clinical`` onto the validation clinical-table schema."""
    raw = load_case_clinical_frame(adata).reset_index()
    if elements_csv is None and repo_root is not None:
        elements_csv = default_discovery_elements_path(repo_root)

    out = pd.DataFrame(index=raw["patient_id"].astype(str))
    out.index.name = "patient_id"
    out["Location_group"] = raw["Location_group"].astype(str).values
    out["Location"] = raw["Location"].values if "Location" in raw.columns else out["Location_group"]
    out["disease_type"] = out["Location_group"].map(
        {"PCNSL": "Brain", "Bone": "Bone", "Testis": "Testis", "Nodal": "Nodal"}
    )
    out["age"] = pd.to_numeric(raw["Age"], errors="coerce").values
    gender = raw["Gender"]
    out["sex"] = (
        gender.map({0: "male", 1: "female", "0": "male", "1": "female"})
        .fillna(gender)
        .values
    )
    out["Ann_Arbor_at_Dx"] = raw["Ann_Arbor_at_Dx"].values
    if "IPI/IELSG" in raw.columns:
        out["ipi_score"] = raw["IPI/IELSG"].values
    elif "ipi_score" in raw.columns:
        out["ipi_score"] = raw["ipi_score"].values
    if "MSKCC" in raw.columns:
        out["MSKCC"] = raw["MSKCC"].values
    if "ipi_mskcc" in raw.columns:
        out["ipi_mskcc"] = raw["ipi_mskcc"].values
    elif "ipi_ielsg" in raw.columns:
        out["ipi_mskcc"] = raw["ipi_ielsg"].values
    out["treatment"] = raw["Treatment category"].values
    out["COO_NanoString"] = raw["COO_NanoString"].values
    out["myc_fish"] = raw["tMYC"].values
    out["bcl2_fish"] = raw["tBCL2"].values
    out["bcl6_fish"] = raw["tBCL6"].values
    out["eber"] = raw["EBER"].values
    if "Curative_intent" in raw.columns:
        out["Curative_intent"] = raw["Curative_intent"].values

    # Merge component overrides used for partial IPI/IELSG bucketing.
    elements = load_discovery_clinical_elements(elements_csv)
    if not elements.empty:
        for col in (
            "Ann_Arbor_at_Dx",
            "ipi_score",
            "ipi_extranodal",
            "ipi_ldh",
            "ipi_who",
            "ielsg_ecog",
            "ielsg_ldh",
            "ielsg_csf",
            "ielsg_deepbrain",
        ):
            if col not in elements.columns:
                continue
            if col not in out.columns:
                out[col] = pd.NA
            el = elements[col].reindex(out.index)
            use = ~el.map(_is_missing)
            out.loc[use, col] = el.loc[use].values

    stored = next((c for c in ("ipi_mskcc", "ipi_ielsg") if c in raw.columns), None)
    if stored is not None:
        mapped = raw.set_index(raw["patient_id"].astype(str))[stored]
        primary = mapped.reindex(out.index).map(
            lambda v: None
            if _is_missing(v)
            else (
                "0-2"
                if str(v).strip() in {"0-2", "0–2"}
                else (">=3" if str(v).strip() in {">=3", "≥3"} else None)
            )
        )
        need = primary.isna()
        if need.any():
            buckets = discovery_primary_bucket_series_from_frame(raw, elements_csv=elements_csv)
            primary = primary.where(~need, buckets.reindex(out.index))
        out["ipi_ielsg_primary"] = primary.values
    else:
        buckets = discovery_primary_bucket_series_from_frame(raw, elements_csv=elements_csv)
        out["ipi_ielsg_primary"] = buckets.reindex(out.index).values
    return out


def build_discovery_clinical_table_from_adata(
    adata: object,
    *,
    elements_csv: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> pd.DataFrame:
    """Aggregate discovery ``case_clinical`` into the characteristics table layout."""
    data = prepare_discovery_clinical_frame(
        adata, elements_csv=elements_csv, repo_root=repo_root
    )
    treatment_levels = list(TREATMENT_CATEGORY_ORDER)
    results = [
        calculate_group_stats(data, loc, treatment_levels=treatment_levels)
        for loc in LOCATION_ORDER
    ]
    labels = build_characteristic_labels(treatment_levels)
    table = pd.DataFrame({"Characteristic": labels})
    for stats in results:
        table[stats.location] = build_site_column(stats, treatment_levels)
    table.attrs["group_stats"] = results
    table.attrs["n_patients"] = len(data)
    table.attrs["source_df"] = data
    table.attrs["case_clinical"] = load_case_clinical_frame(adata)
    table.attrs["unresolved_treatments"] = collect_unresolved_treatments(data)
    return normalize_clinical_table(table)


def _merge_extra_columns(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    on: str,
) -> pd.DataFrame:
    """Left-join ``extra`` onto ``base``, keeping only non-overlapping extra columns."""
    if extra is None or extra.empty:
        return base
    work = extra.copy()
    if on not in work.columns:
        work = work.reset_index()
        if work.columns[0] != on:
            work = work.rename(columns={work.columns[0]: on})
    work[on] = work[on].astype(str)
    base = base.copy()
    base[on] = base[on].astype(str)
    keep = [on] + [c for c in work.columns if c != on and c not in base.columns]
    if len(keep) == 1:
        return base
    return base.merge(work.loc[:, keep], on=on, how="left", sort=False)


def export_patient_clinical_metadata_xlsx(
    adata: object,
    path: Path | str,
    *,
    elements_csv: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    """Write one Excel workbook with Discovery and Validation patient-level clinical sheets.

    Each sheet has a single row per patient and includes all patient-level clinical
    columns present in the AnnData bundle (discovery ``case_clinical``; validation
    ``validation_cohort['meta']``), plus ``case_classifications`` /
    ``case_classification_validation`` as extra columns, and a derived primary
    IPI/IELSG bucket column when missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    discovery = load_case_clinical_frame(adata).reset_index()
    if elements_csv is None and repo_root is not None:
        elements_csv = default_discovery_elements_path(repo_root)
    if "ipi_mskcc" not in discovery.columns and "ipi_ielsg" not in discovery.columns:
        buckets = discovery_primary_bucket_series_from_frame(discovery, elements_csv=elements_csv)
        discovery["ipi_mskcc"] = buckets.reindex(discovery["patient_id"].astype(str)).values

    if "case_classifications" in getattr(adata, "uns", {}):
        discovery = _merge_extra_columns(
            discovery,
            pd.DataFrame(adata.uns["case_classifications"]),
            on="patient_id",
        )

    from .validation_cohort import require_validation_cohort

    vc = require_validation_cohort(adata)
    validation = pd.DataFrame(vc["meta"]).copy()
    if "patient_alias" not in validation.columns:
        validation = validation.reset_index()
        id_col = validation.columns[0]
        if id_col != "patient_alias":
            validation = validation.rename(columns={id_col: "patient_alias"})
    validation["patient_alias"] = validation["patient_alias"].astype(str)
    scored = attach_primary_ipi_buckets(validation.set_index("patient_alias"))
    validation = validation.set_index("patient_alias")
    validation["ipi_ielsg"] = scored["ipi_ielsg_primary"].reindex(validation.index).values
    validation = validation.reset_index()

    if "case_classification_validation" in vc:
        vclass = pd.DataFrame(vc["case_classification_validation"]).copy()
        # Align join key: classifications are indexed by patient_alias / patient_id.
        if "patient_alias" not in vclass.columns:
            vclass = vclass.reset_index()
            id_col = vclass.columns[0]
            if id_col != "patient_alias":
                vclass = vclass.rename(columns={id_col: "patient_alias"})
        validation = _merge_extra_columns(validation, vclass, on="patient_alias")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        discovery.to_excel(writer, sheet_name="Discovery", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)
    return path


def load_discovery_clinical_table(path: Path | str) -> pd.DataFrame:
    """Load pre-rendered discovery cohort clinical table CSV."""
    table = pd.read_csv(path)
    if "Characteristic" not in table.columns:
        raise ValueError(f"{path} must contain a Characteristic column")
    return normalize_clinical_table(table)


def default_discovery_table_path(repo_root: Path | str) -> Path:
    """Best-effort path to the discovery clinical table in anatomy_matters."""
    repo_root = Path(repo_root).resolve()
    candidates = [
        repo_root / "data" / "discovery_clinical_characteristics_table.csv",
        repo_root.parent / "gitnoordenbos" / "anatomy_matters" / "output" / "tables" / "clinical_characteristics_table.csv",
        repo_root.parent / "anatomy_matters" / "output" / "tables" / "clinical_characteristics_table.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[1]


def load_validation_metadata(
    *,
    source: Literal["csv", "adata"] = "adata",
    meta_csv: Path | str | None = None,
    adata_path: Path | str | None = None,
    classifications_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Load validation clinical metadata from CSV or adata.uns['validation_cohort']."""
    from .validation_classifications import (
        load_case_classification_validation,
        merge_validation_metadata_with_classifications,
    )

    if source == "csv":
        if meta_csv is None:
            raise ValueError("meta_csv is required when source='csv'")
        df = pd.read_csv(meta_csv)
        if "patient_alias" in df.columns:
            df = df.set_index("patient_alias")
        df.index = df.index.astype(str)
        if classifications_csv is not None and Path(classifications_csv).exists():
            cc = pd.read_csv(classifications_csv)
            id_col = "patient_id" if "patient_id" in cc.columns else "patient_alias"
            if id_col in cc.columns:
                cc = cc.set_index(id_col)
            cc.index = cc.index.astype(str)
            df = merge_validation_metadata_with_classifications(df, cc)
        else:
            df = enrich_with_coo_nanostring(df, classifications_csv)
        return df

    if adata_path is None:
        raise ValueError("adata_path is required when source='adata'")
    import scanpy as sc
    from .validation_cohort import cohort_notebook_inputs, require_validation_cohort

    adata = sc.read_h5ad(adata_path)
    pred, _, _ = cohort_notebook_inputs(adata)
    vc = require_validation_cohort(adata)
    case_cc = load_case_classification_validation(vc, pred)
    return merge_validation_metadata_with_classifications(pred, case_cc)


@dataclass
class ClinicalInputAudit:
    """Sanity-check report for validation clinical-table inputs."""

    n_patients: int
    location_counts: dict[str, int]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def audit_validation_clinical_inputs(
    df: pd.DataFrame,
    *,
    expected_n: int = EXPECTED_VALIDATION_N,
    expected_location_counts: dict[str, int] | None = None,
) -> ClinicalInputAudit:
    """Assert validation metadata is complete and aligned with the embedded cohort."""
    expected_location_counts = expected_location_counts or EXPECTED_LOCATION_COUNTS
    data = add_location_group(df)
    counts = (
        data["Location_group"]
        .value_counts()
        .reindex(LOCATION_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    audit = ClinicalInputAudit(n_patients=len(data), location_counts=counts)

    if len(data) != expected_n:
        audit.issues.append(f"Expected n={expected_n}, got n={len(data)}")
    for loc, expected in expected_location_counts.items():
        observed = counts.get(loc, 0)
        if observed != expected:
            audit.issues.append(f"{loc}: expected n={expected}, got n={observed}")

    if "treatment" in data.columns:
        unknown_treat = int(data["treatment"].map(classify_treatment).isna().sum())
        if unknown_treat:
            unresolved = collect_unresolved_treatments(data)
            examples = (
                unresolved["treatment"].astype(str).value_counts().head(8).to_dict()
                if len(unresolved)
                else {}
            )
            audit.issues.append(
                f"Treatment unclassified for {unknown_treat} patients; "
                f"unique labels needing review: {examples}"
            )

    if "COO_NanoString" in data.columns:
        missing_coo = int(data["COO_NanoString"].map(_is_missing).sum())
        if missing_coo:
            audit.warnings.append(f"COO_NanoString missing for {missing_coo} patients")

    scored = attach_primary_ipi_buckets(data)
    ipi_missing = int(scored["ipi_ielsg_primary"].isna().sum())
    if ipi_missing:
        by_site = (
            scored.loc[scored["ipi_ielsg_primary"].isna(), "Location_group"]
            .value_counts()
            .to_dict()
        )
        audit.warnings.append(
            f"IPI/IELSG primary bucket unassignable for {ipi_missing} patients "
            f"(excluded from 0–2 / ≥3 rows): {by_site}"
        )

    if "age" in data.columns:
        missing_age = int(pd.to_numeric(data["age"], errors="coerce").isna().sum())
        if missing_age:
            audit.warnings.append(f"Age missing for {missing_age} patients")

    for loc in LOCATION_ORDER:
        subset = data[data["Location_group"] == loc]
        if "Ann_Arbor_at_Dx" not in subset.columns:
            continue
        stage_n = sum(
            1
            for value in subset["Ann_Arbor_at_Dx"]
            if _parse_ann_arbor_stage(value, location_group=loc) is not None
        )
        if stage_n != len(subset):
            audit.warnings.append(
                f"Ann Arbor stage missing/unparsed for {len(subset) - stage_n} {loc} patients"
            )

    return audit


def add_location_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "disease_type" not in out.columns:
        raise KeyError("Expected column 'disease_type' in validation metadata")
    out["Location_group"] = out["disease_type"].map(DISEASE_TYPE_TO_LOCATION_GROUP)
    out = out[out["Location_group"].notna()].copy()
    return out


def calculate_group_stats(
    data: pd.DataFrame,
    location: str,
    *,
    treatment_levels: list[str] | None = None,
) -> GroupStats:
    subset = data[data["Location_group"] == location]
    n_total = len(subset)
    stats = GroupStats(location=location, total=str(n_total))

    if "age" in subset.columns:
        stats.age = _format_median_age_range(subset["age"])

    # Sex female
    if "sex" in subset.columns:
        sex = subset["sex"].astype(str).str.strip()
        known = sex[~sex.map(_is_missing)]
        if len(known):
            n_female = int(known.str.lower().isin({"female", "f", "1"}).sum())
            pct = round(100 * n_female / len(known), 1)
            stats.sex_female = f"{n_female} ({pct}%)"

    # Ann Arbor stage
    if "Ann_Arbor_at_Dx" in subset.columns:
        stage_counts = {"I": 0, "II": 0, "III": 0, "IV": 0}
        stage_n = 0
        for value in subset["Ann_Arbor_at_Dx"]:
            stage = _parse_ann_arbor_stage(value, location_group=location)
            if stage is not None:
                stage_counts[stage] += 1
                stage_n += 1
        stats.stage_I = str(stage_counts["I"])
        stats.stage_II = str(stage_counts["II"])
        stats.stage_III = str(stage_counts["III"])
        stats.stage_IV = str(stage_counts["IV"])
        stats.stage_n = stage_n

    if "ipi_ielsg_primary" in subset.columns:
        primary = subset["ipi_ielsg_primary"]
        low = int((primary == "0-2").sum())
        high = int((primary == ">=3").sum())
        ipi_missing = int(primary.isna().sum())
        stats.ipi_0_2 = str(low)
        stats.ipi_ge3 = str(high)
        stats.ipi_missing = ipi_missing
    elif "ipi_score" in subset.columns:
        # Fallback if caller did not attach primary buckets.
        low = high = 0
        ipi_missing = 0
        for value in subset["ipi_score"]:
            bucket = _parse_ipi_bucket(value)
            if bucket == "low":
                low += 1
            elif bucket == "high":
                high += 1
            else:
                ipi_missing += 1
        stats.ipi_0_2 = str(low)
        stats.ipi_ge3 = str(high)
        stats.ipi_missing = ipi_missing

    # Treatment
    levels = treatment_levels or TREATMENT_CATEGORY_ORDER
    if "treatment" in subset.columns:
        categories = subset["treatment"].map(classify_treatment)
        stats.treatment_unknown = int(categories.isna().sum())
        known = categories.dropna()
        stats.treatment_n = len(known)
        stats.treatment_by_level = {lv: int((known == lv).sum()) for lv in levels}

    if "COO_NanoString" in subset.columns:
        coo = subset["COO_NanoString"]
        stats.coo_ABC = str(_count_coo_label(coo, "ABC"))
        stats.coo_GCB = str(_count_coo_label(coo, "GCB"))
        stats.coo_Intermediate = str(_count_coo_label(coo, "Intermediate"))

    # FISH / ISH
    if {"myc_fish", "bcl2_fish"}.issubset(subset.columns):
        both_tested = subset["myc_fish"].map(_is_fish_tested) & subset["bcl2_fish"].map(
            _is_fish_tested
        )
        double_hit = subset.loc[both_tested].apply(
            lambda row: _is_fish_positive(row["myc_fish"]) and _is_fish_positive(row["bcl2_fish"]),
            axis=1,
        )
        stats.myc_bcl2 = str(int(double_hit.sum()))

    for col, attr in (
        ("myc_fish", "ish_MYC"),
        ("bcl2_fish", "ish_BCL2"),
        ("bcl6_fish", "ish_BCL6"),
    ):
        if col in subset.columns:
            n_pos, n_tested = _fish_positive_count(subset[col])
            stats.__dict__[attr] = f"{n_pos} (n = {n_tested})"

    if "eber" in subset.columns:
        n_pos, n_tested = _fish_positive_count(subset["eber"])
        stats.ish_EBER = f"{n_pos} (n = {n_tested})" if n_tested else NA_PLACEHOLDER

    return stats


def build_site_column(stats: GroupStats, treatment_levels: list[str]) -> list[str]:
    treat_vals = [str(stats.treatment_by_level.get(lv, 0)) for lv in treatment_levels]
    return [
        stats.total,
        stats.age,
        stats.sex_female,
        "",
        stats.stage_I,
        stats.stage_II,
        stats.stage_III,
        stats.stage_IV,
        "",
        stats.ipi_0_2,
        stats.ipi_ge3,
        "",
        *treat_vals,
        "",
        stats.coo_ABC,
        stats.coo_GCB,
        stats.coo_Intermediate,
        "",
        stats.myc_bcl2,
        stats.ish_MYC,
        stats.ish_BCL2,
        stats.ish_BCL6,
        stats.ish_EBER,
    ]


def build_characteristic_labels(treatment_levels: list[str]) -> list[str]:
    treatment_rows = [f"  {lv}" for lv in treatment_levels]
    return [
        "Total",
        "Median age (min-max, years)",
        "Sex female (%)",
        "Ann Arbor stage:",
        "  I(X)B/E",
        "  II(X)A/E",
        "  III",
        "  IV",
        "IPI/MSKCC-score:",
        "  0–2",
        "  ≥3",
        "First-line treatment:",
        *treatment_rows,
        "Cell-of-origin (Lymph2CX):",
        "  ABC",
        "  GCB",
        "  Intermediate",
        "In situ hybridization:",
        "  MYC/BCL2",
        "  MYC",
        "  BCL2",
        "  BCL6",
        "  EBER",
    ]


def build_clinical_table_dataframe(
    df: pd.DataFrame,
    *,
    classifications_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Aggregate validation metadata into a discovery-aligned wide table."""
    data = add_location_group(enrich_with_coo_nanostring(df, classifications_csv))
    data = attach_primary_ipi_buckets(data)
    treatment_levels = list(TREATMENT_CATEGORY_ORDER)
    results = [
        calculate_group_stats(data, loc, treatment_levels=treatment_levels)
        for loc in LOCATION_ORDER
    ]
    labels = build_characteristic_labels(treatment_levels)
    table = pd.DataFrame({"Characteristic": labels})
    for stats in results:
        table[stats.location] = build_site_column(stats, treatment_levels)
    table.attrs["group_stats"] = results
    table.attrs["n_patients"] = len(data)
    table.attrs["source_df"] = data
    table.attrs["unresolved_treatments"] = collect_unresolved_treatments(data)
    return table


def _ipi_missing_footnote(results: list[GroupStats]) -> str | None:
    return validation_ipi_missing_footnote(results)


def validation_table_footnotes(
    results: list[GroupStats],
    data: pd.DataFrame,
    **kwargs: object,
) -> SectionFootnotes:
    """Footnotes for the validation-only clinical table (legacy flat-list wrapper)."""
    del kwargs
    return validation_section_footnotes(results, data)


def _footnote_rows(results: list[GroupStats], data: pd.DataFrame) -> SectionFootnotes:
    return validation_section_footnotes(results, data)


def _combined_column_names(prefix: str) -> list[str]:
    return [f"{prefix}_{loc}" for loc in LOCATION_ORDER]


def align_characteristic_rows(*tables: pd.DataFrame) -> list[str]:
    """Align rows to the canonical clinical-table layout."""
    master = build_characteristic_labels(TREATMENT_CATEGORY_ORDER)
    master_set = set(master)
    extras: list[str] = []
    for table in tables:
        normalized = normalize_clinical_table(table)
        for label in normalized["Characteristic"].astype(str):
            if label not in master_set and label not in extras:
                extras.append(label)
    return master + extras


def build_combined_clinical_table(
    discovery_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    *,
    discovery_metadata_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Side-by-side discovery (left) and validation (right) clinical tables."""
    discovery_table = rebucket_discovery_ipi_table(discovery_table, discovery_metadata_csv)
    validation_table = normalize_clinical_table(validation_table)
    labels = align_characteristic_rows(discovery_table, validation_table)
    disc = discovery_table.set_index("Characteristic").reindex(labels)
    val = validation_table.set_index("Characteristic").reindex(labels)

    combined = pd.DataFrame({"Characteristic": labels})
    for loc in LOCATION_ORDER:
        combined[f"{DISCOVERY_COLUMN_PREFIX}_{loc}"] = disc[loc].fillna("").astype(str).values
        combined[f"{VALIDATION_COLUMN_PREFIX}_{loc}"] = val[loc].fillna("").astype(str).values
    return combined


def render_combined_great_table(
    combined_table: pd.DataFrame,
    *,
    title: str = "Clinical Characteristics — Discovery and Validation Cohorts",
    footnotes: SectionFootnotes | None = None,
    footnote_placements: list[FootnotePlacement] | None = None,
) -> object:
    from great_tables import GT, loc, style

    discovery_cols = _combined_column_names(DISCOVERY_COLUMN_PREFIX)
    validation_cols = _combined_column_names(VALIDATION_COLUMN_PREFIX)
    display_df = pd.DataFrame(combined_table[["Characteristic", *discovery_cols, *validation_cols]].to_dict(orient="list"))
    section_row_idx = display_df.index[
        display_df["Characteristic"].isin(SECTION_HEADER_ROWS)
    ].tolist()

    gt_table = (
        GT(display_df)
        .tab_header(title=title)
        .tab_spanner(label="Discovery", columns=discovery_cols)
        .tab_spanner(label="Validation", columns=validation_cols)
        .cols_label(
            **{f"{DISCOVERY_COLUMN_PREFIX}_{loc}": loc for loc in LOCATION_ORDER},
            **{f"{VALIDATION_COLUMN_PREFIX}_{loc}": loc for loc in LOCATION_ORDER},
        )
        .tab_style(style=style.text(weight="bold"), locations=loc.column_labels())
        .tab_style(
            style=style.text(weight="bold", style="italic"),
            locations=loc.body(columns="Characteristic", rows=section_row_idx),
        )
        .cols_align(align="left", columns=["Characteristic"])
        .cols_align(align="center", columns=[*discovery_cols, *validation_cols])
        .cols_width(**_gt_column_widths([*discovery_cols, *validation_cols], value_px=82))
        .tab_options(
            table_width="auto",
            table_font_names=[TABLE_FONT_FAMILY, "sans-serif"],
            table_font_size="11px",
            heading_title_font_size="14px",
            data_row_padding_horizontal="10px",
            column_labels_padding_horizontal="10px",
        )
    )

    if footnote_placements is not None:
        return _attach_footnote_placements(gt_table, display_df, footnote_placements)
    return _attach_section_footnotes(gt_table, display_df, footnotes or {})


def _dejavu_font_face_css() -> str:
    """CSS @font-face so Chrome/nokap PDF+PNG can use manuscript DejaVu Sans."""
    try:
        from matplotlib.font_manager import FontProperties, findfont

        regular = Path(findfont(FontProperties(family=TABLE_FONT_FAMILY)))
        bold = Path(findfont(FontProperties(family=TABLE_FONT_FAMILY, weight="bold")))
    except Exception:
        return ""
    faces = [
        (regular, "normal"),
        (bold, "bold"),
    ]
    chunks: list[str] = []
    seen: set[str] = set()
    for path, weight in faces:
        key = f"{path.resolve()}:{weight}"
        if not path.exists() or key in seen:
            continue
        seen.add(key)
        uri = path.resolve().as_uri()
        chunks.append(
            "@font-face {"
            f"font-family:'{TABLE_FONT_FAMILY}';"
            f"src:url('{uri}') format('truetype');"
            f"font-weight:{weight};font-style:normal;"
            "}"
        )
    return "\n".join(chunks)


def _gt_column_widths(value_columns: list[str], *, characteristic_px: int = 210, value_px: int = 78) -> dict[str, str]:
    """Minimum column widths so values like ``14 (29.8%)`` stay on one line."""
    widths = {"Characteristic": f"{characteristic_px}px"}
    for col in value_columns:
        widths[col] = f"{value_px}px"
    return widths


def _gt_html_with_dejavu(gt_table: object, *, make_page: bool = True) -> str:
    """Raw GT HTML with DejaVu Sans embedded for print exports."""
    html = gt_table.as_raw_html(make_page=make_page)  # type: ignore[union-attr]
    css = _dejavu_font_face_css()
    layout = (
        "table { table-layout: auto; width: max-content; max-width: none; }\n"
        "th, td { white-space: nowrap; }\n"
        "td.gt_row, th.gt_col_heading, th.gt_column_spanner { white-space: nowrap; }\n"
    )
    inject = (
        "<style>\n"
        f"{css}\n"
        f"body, table {{ font-family: '{TABLE_FONT_FAMILY}', sans-serif; }}\n"
        f"{layout}"
        "</style>"
    )
    if "</head>" in html:
        return html.replace("</head>", inject + "\n</head>", 1)
    return inject + html


def _gtsave_pdf_png(
    gt_table: object,
    *,
    pdf_path: Path,
    png_path: Path,
    vwidth: int = 1100,
) -> tuple[Path | None, Path | None]:
    """Save GT table as PDF (primary journal format) and PNG preview via nokap."""
    import nokap

    html = _gt_html_with_dejavu(gt_table, make_page=False)
    pdf_out: Path | None = pdf_path
    png_out: Path | None = png_path
    try:
        nokap.from_html(
            html=html,
            file=pdf_path,
            selector="table",
            expand=8,
            zoom=2.0,
            delay=0.25,
            vwidth=vwidth,
            vheight=1400,
        )
    except Exception:
        pdf_out = None
    try:
        nokap.from_html(
            html=html,
            file=png_path,
            selector="table",
            expand=8,
            zoom=2.0,
            delay=0.25,
            vwidth=vwidth,
            vheight=1400,
        )
    except Exception:
        png_out = None
    return pdf_out, png_out


def save_combined_clinical_table_outputs(
    combined_table: pd.DataFrame,
    output_dir: Path | str,
    *,
    stem: str = "combined_clinical_characteristics_table",
    title: str = "Clinical Characteristics — Discovery and Validation Cohorts",
    footnotes: SectionFootnotes | None = None,
    footnote_placements: list[FootnotePlacement] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "csv": output_dir / f"{stem}.csv",
        "html": output_dir / f"{stem}.html",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
    }
    combined_table.to_csv(paths["csv"], index=False)

    gt_table = render_combined_great_table(
        combined_table,
        title=title,
        footnotes=footnotes,
        footnote_placements=footnote_placements,
    )
    paths["html"].write_text(_gt_html_with_dejavu(gt_table, make_page=True), encoding="utf-8")

    pdf_out, png_out = _gtsave_pdf_png(gt_table, pdf_path=paths["pdf"], png_path=paths["png"])
    if pdf_out is None:
        paths.pop("pdf", None)
    if png_out is None:
        paths.pop("png", None)

    placements = footnote_placements or [
        (section, None, note)
        for section, notes in (footnotes or {}).items()
        for note in notes
    ]
    save_clinical_table_svg(
        combined_table,
        paths["svg"],
        title=title,
        footnote_placements=placements,
        combined=True,
        page="a4",
    )
    return paths


def render_great_table(
    table_data: pd.DataFrame,
    *,
    title: str,
    group_stats: list[GroupStats] | None = None,
    source_df: pd.DataFrame | None = None,
) -> object:
    from great_tables import GT, loc, style

    results: list[GroupStats] = group_stats or table_data.attrs.get("group_stats", [])
    if source_df is None:
        source_df = table_data.attrs.get("source_df")
    loc_df = (
        add_location_group(source_df)
        if source_df is not None
        else pd.DataFrame()
    )
    footnotes = validation_footnote_placements(results, loc_df)

    display_df = pd.DataFrame(
        table_data[["Characteristic", *LOCATION_ORDER]].to_dict(orient="list")
    )
    section_row_idx = display_df.index[
        display_df["Characteristic"].isin(SECTION_HEADER_ROWS)
    ].tolist()

    gt_table = (
        GT(display_df)
        .tab_header(title=title)
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.text(weight="bold", style="italic"),
            locations=loc.body(columns="Characteristic", rows=section_row_idx),
        )
        .cols_align(align="left", columns=["Characteristic"])
        .cols_align(align="center", columns=LOCATION_ORDER)
        .cols_width(**_gt_column_widths(list(LOCATION_ORDER), value_px=90))
        .tab_options(
            table_width="auto",
            table_font_names=[TABLE_FONT_FAMILY, "sans-serif"],
            table_font_size="11px",
            heading_title_font_size="14px",
            data_row_padding_horizontal="10px",
            column_labels_padding_horizontal="10px",
        )
    )

    return _attach_footnote_placements(gt_table, display_df, footnotes)


def save_clinical_table_outputs(
    table_data: pd.DataFrame,
    output_dir: Path | str,
    *,
    stem: str = "validation_clinical_characteristics_table",
    title: str = "Clinical Characteristics — Validation Cohort",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "csv": output_dir / f"{stem}.csv",
        "html": output_dir / f"{stem}.html",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
    }
    table_data[["Characteristic", *LOCATION_ORDER]].to_csv(paths["csv"], index=False)

    gt_table = render_great_table(
        table_data,
        title=title,
        group_stats=table_data.attrs.get("group_stats"),
        source_df=table_data.attrs.get("source_df"),
    )
    paths["html"].write_text(_gt_html_with_dejavu(gt_table, make_page=True), encoding="utf-8")

    pdf_out, png_out = _gtsave_pdf_png(gt_table, pdf_path=paths["pdf"], png_path=paths["png"])
    if pdf_out is None:
        paths.pop("pdf", None)
    if png_out is None:
        paths.pop("png", None)

    source_df = table_data.attrs.get("source_df")
    loc_df = (
        add_location_group(source_df)
        if source_df is not None and "Location_group" not in getattr(source_df, "columns", [])
        else source_df
    )
    placements = validation_footnote_placements(
        table_data.attrs.get("group_stats") or [],
        loc_df if loc_df is not None else pd.DataFrame(),
    )
    save_clinical_table_svg(
        table_data,
        paths["svg"],
        title=title,
        footnote_placements=placements,
        combined=False,
        page="a4",
    )

    return paths


def run_validation_clinical_table(config: ClinicalTableConfig) -> tuple[pd.DataFrame, dict[str, Path]]:
    """End-to-end: load metadata, build table, export outputs."""
    df = load_validation_metadata(
        source=config.source,
        meta_csv=config.meta_csv,
        adata_path=config.adata_path,
        classifications_csv=config.classifications_csv,
    )
    table = build_clinical_table_dataframe(
        df,
        classifications_csv=config.classifications_csv,
    )
    if config.output_dir is None:
        raise ValueError("output_dir is required")
    paths = save_clinical_table_outputs(
        table,
        config.output_dir,
        stem=config.table_stem,
    )
    unresolved = table.attrs.get("unresolved_treatments")
    if isinstance(unresolved, pd.DataFrame):
        unresolved_path = Path(config.output_dir) / f"{config.table_stem}_unresolved_treatments.csv"
        unresolved.to_csv(unresolved_path, index=False)
        paths["unresolved_treatments"] = unresolved_path
    return table, paths


def combined_clinical_table_footnotes(
    discovery_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    *,
    validation_data: pd.DataFrame | None = None,
    discovery_metadata_csv: Path | str | pd.DataFrame | None = None,
    discovery_elements_csv: Path | str | None = None,
) -> list[FootnotePlacement]:
    """Footnote placements for the side-by-side discovery/validation clinical table."""
    if validation_data is None or validation_data.empty:
        return []

    loc_df = add_location_group(validation_data) if "Location_group" not in validation_data.columns else validation_data
    group_stats = validation_table.attrs.get("group_stats")
    if not group_stats:
        rebuilt = build_clinical_table_dataframe(loc_df)
        group_stats = rebuilt.attrs.get("group_stats", [])
    return combined_footnote_placements(
        validation_data=loc_df,
        group_stats=group_stats,
        discovery_metadata_csv=discovery_metadata_csv,
        discovery_table=discovery_table,
        discovery_elements_csv=discovery_elements_csv,
    )


def run_combined_clinical_table(
    config: ClinicalTableConfig,
    discovery_table_csv: Path | str | None = None,
    *,
    discovery_metadata_csv: Path | str | pd.DataFrame | None = None,
    discovery_elements_csv: Path | str | None = None,
    adata: object | None = None,
    repo_root: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Build discovery + validation clinical tables and export combined outputs.

    Preferred public-release path: pass ``adata`` (builds discovery from
    ``uns['case_clinical']``). Legacy path: pass ``discovery_table_csv``.
    """
    validation_table, _ = run_validation_clinical_table(config)

    root = Path(repo_root) if repo_root is not None else None
    if root is None and config.adata_path is not None:
        root = Path(config.adata_path).resolve().parent.parent
    if discovery_elements_csv is None and root is not None:
        discovery_elements_csv = default_discovery_elements_path(root)

    if adata is not None:
        discovery_table = build_discovery_clinical_table_from_adata(
            adata,
            elements_csv=discovery_elements_csv,
            repo_root=root,
        )
        discovery_meta: Path | str | pd.DataFrame | None = discovery_table.attrs.get(
            "case_clinical"
        )
        if isinstance(discovery_meta, pd.DataFrame):
            discovery_meta = discovery_meta.reset_index()
        elif discovery_metadata_csv is not None:
            discovery_meta = discovery_metadata_csv
    else:
        if discovery_table_csv is None:
            raise ValueError("Pass adata=... or discovery_table_csv=...")
        if discovery_metadata_csv is None and root is not None:
            discovery_metadata_csv = default_discovery_metadata_path(root)
        discovery_table = load_discovery_clinical_table(discovery_table_csv)
        discovery_meta = discovery_metadata_csv

    discovery_table = rebucket_discovery_ipi_table(
        discovery_table,
        discovery_meta,
        elements_csv=discovery_elements_csv,
    )
    combined = build_combined_clinical_table(
        discovery_table,
        validation_table,
        discovery_metadata_csv=None,
    )

    if config.output_dir is None:
        raise ValueError("output_dir is required")

    source_df = validation_table.attrs.get("source_df")
    loc_df = (
        add_location_group(source_df)
        if source_df is not None and "Location_group" not in getattr(source_df, "columns", [])
        else source_df
    )
    if loc_df is None:
        loc_df = pd.DataFrame()
    combined_placements = combined_clinical_table_footnotes(
        discovery_table,
        validation_table,
        validation_data=loc_df,
        discovery_metadata_csv=discovery_meta,
        discovery_elements_csv=discovery_elements_csv,
    )
    paths = save_combined_clinical_table_outputs(
        combined,
        config.output_dir,
        stem="combined_clinical_characteristics_table",
        footnote_placements=combined_placements,
    )
    return discovery_table, validation_table, combined, paths