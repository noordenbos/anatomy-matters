"""Kaplan–Meier sensitivity analyses (cohort exclusions for KM panels)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import scanpy as sc

from .dlbcl_io import DISCOVERY_PATIENTS, NotebookPaths, load_archetype_assignments
from .validation_figures import ARCHETYPE_NAME_MAP, LOCATION_ORDER

DISCOVERY_TIME_COL = "OS_time_JV"
DISCOVERY_ENDPOINTS = {
    "OS": {"time": "OS_time_JV", "event": "OS_status_JV"},
    "DSS": {"time": "OS_time_JV", "event": "OS_specific_status_JV"},
    "PFS": {"time": "pfs_time", "event": "pfs_status"},
}
REQUIRED_DISCOVERY_ENDPOINTS = ("OS", "DSS")


def _discovery_endpoint_columns() -> tuple[list[str], list[str]]:
    time_cols = list(dict.fromkeys(ep["time"] for ep in DISCOVERY_ENDPOINTS.values()))
    event_cols = [ep["event"] for ep in DISCOVERY_ENDPOINTS.values()]
    return time_cols, event_cols

DISCOVERY_LOCATION_RECODE = {
    "pBONE": "bone",
    "polyOST": "bone",
    "disseminated": "bone",
}

# Original-cluster outliers excluded from discovery archetype KM (Figs 2F/2G).
DISCOVERY_ARCHETYPE_OUTLIERS: tuple[str, ...] = ("T7", "T55", "T36", "T42")

# PCNS / Brain patients excluded from validation Fig 5A archetype KM sensitivity.
VALIDATION_EXCLUDE_LOCATIONS: tuple[str, ...] = ("PCNS",)


def _patient_frame(adata, uns_key: str) -> pd.DataFrame:
    df = pd.DataFrame(adata.uns[uns_key]).copy()
    if "patient_id" not in df.columns:
        df = df.reset_index(names="patient_id")
    df["patient_id"] = df["patient_id"].astype(str)
    return df.set_index("patient_id")


def build_discovery_km_survival_table(adata, paths: NotebookPaths) -> pd.DataFrame:
    """Curative-intent discovery KM table (same filters as ``nb11_km_survival``)."""
    mask = (
        adata.obs["patient_id"].isin(DISCOVERY_PATIENTS)
        & (adata.obs["filtering_status"] == "Unfiltered")
    )
    adata_sub = adata[mask].copy()

    arch_df = load_archetype_assignments(paths)
    cluster_dict = dict(
        zip(arch_df["patient_id"].astype(str), arch_df["abundance_cluster_30"].astype(int))
    )

    clinical = _patient_frame(adata_sub, "case_clinical")
    classif = _patient_frame(adata_sub, "case_classifications")

    clinical["Curative_intent"] = pd.to_numeric(clinical["Curative_intent"], errors="coerce")
    time_cols, event_cols = _discovery_endpoint_columns()
    surv = clinical.loc[
        clinical["Curative_intent"] == 1,
        list(dict.fromkeys([*time_cols, *event_cols])),
    ].copy()

    surv["Location"] = classif["Location"].replace(DISCOVERY_LOCATION_RECODE)

    arch = pd.Series(cluster_dict, name="archetype_id")
    surv["archetype_id"] = arch
    surv["Archetype"] = surv["archetype_id"].map(ARCHETYPE_NAME_MAP)

    required_cols = [
        DISCOVERY_ENDPOINTS[ep]["time"] for ep in REQUIRED_DISCOVERY_ENDPOINTS
    ] + [DISCOVERY_ENDPOINTS[ep]["event"] for ep in REQUIRED_DISCOVERY_ENDPOINTS]
    numeric_cols = list(dict.fromkeys([*time_cols, *event_cols, "archetype_id"]))
    for col in numeric_cols:
        surv[col] = pd.to_numeric(surv[col], errors="coerce")

    return surv.dropna(subset=[*required_cols, "Location", "Archetype"]).copy()


def discovery_to_km_frame(
    surv: pd.DataFrame,
    *,
    event_col: str,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Map discovery survival table to validation KM plotter columns."""
    if time_col is None:
        time_col = next(
            (ep["time"] for ep in DISCOVERY_ENDPOINTS.values() if ep["event"] == event_col),
            DISCOVERY_TIME_COL,
        )
    out = surv.copy()
    out["time"] = pd.to_numeric(out[time_col], errors="coerce")
    out["event"] = pd.to_numeric(out[event_col], errors="coerce")
    return out.dropna(subset=["time", "event", "Location", "archetype_id", "Archetype"])


def exclude_patients(surv: pd.DataFrame, patient_ids: tuple[str, ...] | list[str]) -> pd.DataFrame:
    drop = {str(p) for p in patient_ids}
    return surv.loc[~surv.index.astype(str).isin(drop)].copy()


def exclude_locations(surv: pd.DataFrame, locations: tuple[str, ...] | list[str]) -> pd.DataFrame:
    drop = set(locations)
    return surv.loc[~surv["Location"].isin(drop)].copy()


def validation_location_order_excluding(excluded: tuple[str, ...] | list[str] = VALIDATION_EXCLUDE_LOCATIONS) -> list[str]:
    return [loc for loc in LOCATION_ORDER if loc not in set(excluded)]


def load_validation_survival_table(adata) -> pd.DataFrame:
    from .validation_figures import build_survival_table, load_from_adata

    pred, _gep, _raw = load_from_adata(adata)
    return build_survival_table(pred)


def load_discovery_km_from_h5ad(adata_path: Path | str, paths: NotebookPaths) -> pd.DataFrame:
    adata = sc.read_h5ad(adata_path)
    return build_discovery_km_survival_table(adata, paths)
