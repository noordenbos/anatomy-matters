"""Canonical 30-class phenotype labels (ordered old -> new rename for manuscript)."""

from __future__ import annotations

import json
from typing import Any

import anndata as ad
import pandas as pd

PHENOTYPE_30_LABELS_VERSION = 3

PHENOTYPE_30_COLUMNS = ("phenotype_30_clean", "phenotype_30_bm")
COLORMAP_30_UNS_KEY = "celltype_colormap_30"
COLORMAP_30_JSON_KEY = "celltype_colormap_30_json"

# Consistent-order old labels (37 classes).
PHENOTYPE_30_OLD_ORDER: list[str] = [
    "neuronal",
    "CD4 T cell, memory",
    "Regulatory T cell",
    "unclassified",
    "CD4 T cell, follicular helper-like PD1+",
    "CD4 T cell, naive",
    "CD4 T cell, follicular helper-like PD1-",
    "CD4 T cell, activated ICOS KI67",
    "Th1 ILC1",
    "CD8 T cell, non-cytotoxic",
    "CD8 T cell, cytotoxic",
    "CD8 T cell, naive",
    "Double negative T cell",
    "Other lymphocytes (unspecified)",
    "Monocyte CD163+",
    "Monocyte CD163-",
    "Macrophage CD163+",
    "Myeloid cell CD11b+",
    "Macrophage CD163-",
    "Dendritic cell CD163-",
    "Myeloid cell, IDO+",
    "Activated stromal cell",
    "Resting stromal cell",
    "Activated endothelial cell",
    "Endothelial cell",
    "cd20_cxcr5+ki67+",
    "cd20_cxcr5+ki67-",
    "cd20_cxcr5-ki67+",
    "cd20_cxcr5-ki67-",
    "cd20_cxcr5+caspase+",
    "Neutrophils",
    "Gamma-delta T cell",
    "Double negative T cell, cytotoxic",
    "nk cell, cytotoxic",
    "Dendritic cell CD163+",
    "cd20_cxcr5-caspase+",
    "Activated regulatory T cell ICOS KI67",
]

# Consistent-order new labels (same positions as ``PHENOTYPE_30_OLD_ORDER``).
PHENOTYPE_30_ORDER: list[str] = [
    "neuronal",
    "CD4 T cell, memory",
    "Regulatory T cell",
    "unclassified",
    "CD4 T cell, follicular helper-like PD1+",
    "CD4 T cell, CD45RA+",
    "CD4 T cell, follicular helper-like PD1-",
    "CD4 T cell, activated ICOS KI67",
    "Tbet expressing lymphocytes (Th1 ILC1)",
    "CD8 T cell, other",
    "CD8 T cell, cytotoxic (GZMB)",
    "CD8 T cell, CD45RA",
    "Double negative T cell",
    "Other lymphocytes (unspecified)",
    "Myeloid cell CD163+",
    "Monocyte",
    "Macrophage CD163+",
    "Myeloid cell CD11b+",
    "Macrophage CD163-",
    "Dendritic cell CD163-",
    "Myeloid cell, IDO+",
    "Activated stromal cell",
    "Resting stromal cell",
    "Activated endothelial cell",
    "Endothelial cell",
    "cd20_cxcr5+ki67+",
    "cd20_cxcr5+ki67-",
    "cd20_cxcr5-ki67+",
    "cd20_cxcr5-ki67-",
    "cd20_cxcr5+caspase+",
    "Neutrophils",
    "Gamma-delta T cell",
    "Double negative T cell, cytotoxic (GZMB)",
    "nk cell, cytotoxic (GZMB)",
    "Dendritic cell CD163+",
    "cd20_cxcr5-caspase+",
    "Activated regulatory T cell ICOS KI67",
]

if len(PHENOTYPE_30_OLD_ORDER) != len(PHENOTYPE_30_ORDER):
    raise ValueError("PHENOTYPE_30_OLD_ORDER and PHENOTYPE_30_ORDER must have equal length")

PHENOTYPE_30_RENAME: dict[str, str] = {
    old: new for old, new in zip(PHENOTYPE_30_OLD_ORDER, PHENOTYPE_30_ORDER, strict=True) if old != new
}

PHENOTYPE_30_GROUPS: dict[str, list[str]] = {
    "Unknown": ["unclassified"],
    "Tissue / stromal": [
        "neuronal",
        "Activated stromal cell",
        "Resting stromal cell",
        "Activated endothelial cell",
        "Endothelial cell",
    ],
    "CD4 / Treg": [
        "CD4 T cell, memory",
        "Regulatory T cell",
        "CD4 T cell, follicular helper-like PD1+",
        "CD4 T cell, CD45RA+",
        "CD4 T cell, follicular helper-like PD1-",
        "CD4 T cell, activated ICOS KI67",
        "Activated regulatory T cell ICOS KI67",
    ],
    "CD8 / other T": [
        "Tbet expressing lymphocytes (Th1 ILC1)",
        "CD8 T cell, other",
        "CD8 T cell, cytotoxic (GZMB)",
        "CD8 T cell, CD45RA",
        "Double negative T cell",
        "Other lymphocytes (unspecified)",
        "Gamma-delta T cell",
        "Double negative T cell, cytotoxic (GZMB)",
        "nk cell, cytotoxic (GZMB)",
    ],
    "Myeloid": [
        "Myeloid cell CD163+",
        "Monocyte",
        "Macrophage CD163+",
        "Myeloid cell CD11b+",
        "Macrophage CD163-",
        "Dendritic cell CD163-",
        "Myeloid cell, IDO+",
        "Neutrophils",
        "Dendritic cell CD163+",
    ],
    "B cells": [
        "cd20_cxcr5+ki67+",
        "cd20_cxcr5+ki67-",
        "cd20_cxcr5-ki67+",
        "cd20_cxcr5-ki67-",
        "cd20_cxcr5+caspase+",
        "cd20_cxcr5-caspase+",
    ],
}

# Phenotypes excluded from nb4 abundance heatmap (substring match, case-insensitive).
PHENOTYPE_30_HEATMAP_EXCLUDE_TERMS = ("unclassified", "cd20", "neuronal", "neutrophil")

# Fig S1K (nb2): phenotype × protein heatmap row order (grouped for display; uses renamed labels).
PHENOTYPE_30_HEATMAP_ORDER: list[str] = [
    "unclassified",
    "cd20_cxcr5+ki67+", "cd20_cxcr5+ki67-", "cd20_cxcr5-ki67-", "cd20_cxcr5-ki67+",
    "cd20_cxcr5+caspase+", "cd20_cxcr5-caspase+",
    "Neutrophils",
    "Resting stromal cell", "Activated stromal cell", "Endothelial cell", "Activated endothelial cell",
    "Monocyte", "Myeloid cell CD11b+", "Dendritic cell CD163-", "Dendritic cell CD163+",
    "Macrophage CD163+", "Macrophage CD163-", "Myeloid cell CD163+", "Myeloid cell, IDO+",
    "Double negative T cell",
    "CD4 T cell, memory", "CD4 T cell, CD45RA+", "CD4 T cell, follicular helper-like PD1+",
    "CD4 T cell, follicular helper-like PD1-", "CD4 T cell, activated ICOS KI67",
    "Activated regulatory T cell ICOS KI67", "Regulatory T cell",
    "Tbet expressing lymphocytes (Th1 ILC1)",
    "CD8 T cell, CD45RA", "CD8 T cell, other", "CD8 T cell, cytotoxic (GZMB)",
    "Double negative T cell, cytotoxic (GZMB)",
    "nk cell, cytotoxic (GZMB)", "Gamma-delta T cell", "Other lymphocytes (unspecified)",
    "neuronal",
]

if set(PHENOTYPE_30_HEATMAP_ORDER) - set(PHENOTYPE_30_ORDER):
    raise ValueError("PHENOTYPE_30_HEATMAP_ORDER contains labels absent from PHENOTYPE_30_ORDER")


def rename_phenotype_30_value(label: str) -> str:
    return PHENOTYPE_30_RENAME.get(str(label), str(label))


def flatten_colormap(cmap: dict[str, Any]) -> dict[str, Any]:
    """Undo h5ad ``/`` nesting in ``uns`` colormap dicts."""
    flat: dict[str, Any] = {}
    for key, value in cmap.items():
        if isinstance(value, dict):
            parent = str(key).rstrip()
            for subkey, subvalue in value.items():
                flat[f"{parent} / {str(subkey).lstrip()}"] = subvalue
        else:
            flat[str(key)] = value
    return flat


def read_celltype_colormap_30(adata: ad.AnnData) -> dict[str, Any]:
    """Return a flat phenotype -> color map safe for plotting."""
    if COLORMAP_30_JSON_KEY in adata.uns:
        payload = adata.uns[COLORMAP_30_JSON_KEY]
        if isinstance(payload, str):
            return {str(k): v for k, v in json.loads(payload).items()}
        return {str(k): v for k, v in dict(payload).items()}
    if COLORMAP_30_UNS_KEY in adata.uns:
        return flatten_colormap(dict(adata.uns[COLORMAP_30_UNS_KEY]))
    return {}


def write_celltype_colormap_30(adata: ad.AnnData, cmap: dict[str, Any]) -> None:
    """Persist colormap; JSON avoids h5ad splitting keys that contain ``/``."""
    flat = {str(k): v for k, v in cmap.items() if not isinstance(v, dict)}
    adata.uns[COLORMAP_30_JSON_KEY] = json.dumps(flat)
    adata.uns[COLORMAP_30_UNS_KEY] = {k: v for k, v in flat.items() if "/" not in k}


def rename_colormap(cmap: dict[str, Any]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in flatten_colormap(cmap).items():
        if isinstance(value, dict):
            continue
        renamed[rename_phenotype_30_value(key)] = value
    return renamed


def sync_celltype_colormap_30(adata: ad.AnnData) -> None:
    """Flatten, rename, and persist the 30-class colormap."""
    if COLORMAP_30_UNS_KEY in adata.uns or COLORMAP_30_JSON_KEY in adata.uns:
        write_celltype_colormap_30(adata, rename_colormap(read_celltype_colormap_30(adata)))


def apply_phenotype_30_labels(adata: ad.AnnData, *, inplace: bool = False) -> ad.AnnData:
    """Rename ``phenotype_30_clean`` / ``phenotype_30_bm`` and sync the 30-class colormap."""
    out = adata if inplace else adata.copy()
    for col in PHENOTYPE_30_COLUMNS:
        if col in out.obs.columns:
            out.obs[col] = out.obs[col].astype(str).map(rename_phenotype_30_value)
    sync_celltype_colormap_30(out)
    out.uns["phenotype_30_label_map"] = dict(PHENOTYPE_30_RENAME)
    out.uns["phenotype_30_labels_version"] = PHENOTYPE_30_LABELS_VERSION
    return out


def restore_phenotype_30_from_source(
    adata: ad.AnnData,
    source: ad.AnnData,
    *,
    inplace: bool = False,
) -> ad.AnnData:
    """Re-copy phenotype columns from a donor h5ad (old labels), then apply rename."""
    out = adata if inplace else adata.copy()
    common = out.obs.index.intersection(source.obs.index)
    if len(common) != out.n_obs:
        missing = out.n_obs - len(common)
        raise ValueError(f"{missing:,} cells in target are absent from source obs.index")
    for col in PHENOTYPE_30_COLUMNS:
        if col not in source.obs.columns:
            raise KeyError(f"source missing obs[{col!r}]")
        out.obs[col] = out.obs[col].astype(str)
        out.obs.loc[common, col] = source.obs.loc[common, col].astype(str)
    if COLORMAP_30_UNS_KEY in source.uns:
        out.uns[COLORMAP_30_UNS_KEY] = dict(source.uns[COLORMAP_30_UNS_KEY])
    out.uns.pop(COLORMAP_30_JSON_KEY, None)
    out.uns.pop("phenotype_30_labels_version", None)
    return apply_phenotype_30_labels(out, inplace=True)


def ensure_phenotype_30_labels(adata: ad.AnnData) -> ad.AnnData:
    """Apply label map once (idempotent via ``uns['phenotype_30_labels_version']``)."""
    version = adata.uns.get("phenotype_30_labels_version")
    if version == PHENOTYPE_30_LABELS_VERSION and COLORMAP_30_JSON_KEY in adata.uns:
        return adata
    if version == PHENOTYPE_30_LABELS_VERSION:
        sync_celltype_colormap_30(adata)
        return adata
    return apply_phenotype_30_labels(adata, inplace=True)


def filter_phenotype_index(df: pd.DataFrame, terms: tuple[str, ...] = PHENOTYPE_30_HEATMAP_EXCLUDE_TERMS) -> pd.DataFrame:
    """Drop phenotype rows whose index matches any exclude term (nb4 heatmap helper)."""
    if not terms:
        return df
    pattern = "|".join(terms)
    return df.loc[~df.index.astype(str).str.contains(pattern, case=False, na=False)]
