"""Fig 1C cohort modality matrix — aggregated (V1) and per-patient strips (V2).

Row definitions live in ``DISCOVERY_MODALITY_ROWS`` / ``VALIDATION_MODALITY_ROWS`` so
validation genomics (or new modalities) can be enabled by editing one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .validation_figures import DISEASE_TYPE_TO_LOCATION

LOCATION_ORDER = ["PCNS", "bone", "nodal", "testis"]
LOCATION_LABELS = {"PCNS": "PCNS", "bone": "bone", "nodal": "nodal", "testis": "testis"}
LOCATION_COLORS = {
    "bone": "#ff7f0e",
    "nodal": "#2ca02c",
    "PCNS": "#1f77b4",
    "testis": "#d62728",
}

ROW_COLORS = ("#dbe9f6", "#e4ddf4")
ABSENT_COLOR = "#ffffff"
GRID_COLOR = "#c7c7c7"
PATIENT_CELL_EDGE_COLOR = "#b8b8b8"
PATIENT_CELL_EDGE_LW = 0.12
ROW_GAP = 0.35
SITE_GAP_FRAC = 0.75  # gap between location blocks, as a fraction of cell_w


@dataclass(frozen=True)
class ModalityRow:
    """One horizontal strip in the modality figure."""

    key: str
    label: str
    discovery: Callable[[pd.DataFrame], pd.Series] | None = None
    validation: Callable[[pd.DataFrame], pd.Series] | None = None
    v1_aggregate: str = "all"  # "all" | "any" — per anatomical site in V1 blocks


def _bool_series(index: pd.Index, value: bool) -> pd.Series:
    return pd.Series(value, index=index, dtype=bool)



def _any_ihc(row: pd.Series, cols: list[str]) -> bool:
    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        text = str(val).strip()
        if text and text.lower() not in {"nan", "unk", "na", "none"}:
            return True
    return False


DISCOVERY_IHC_COLS = [
    "CD10_IHC",
    "MUM1_IHC",
    "BCL6_IHC",
    "BCL6_IHC ",
    "BCL2_IHC",
    "MYC_IHC",
    "MYC_IHC_TRUE",
    "EBER",
]

VALIDATION_IHC_COLS = ["myc_ihc", "bcl2_ihc", "tp53_ihc", "coo_hans"]


def discovery_survival_eligible(df: pd.DataFrame) -> pd.Series:
    """OS available and curative intent (survival-analysis cohort)."""
    curative = pd.to_numeric(df.get("Curative_intent"), errors="coerce") == 1
    has_os = df.get("OS_time_JV").notna() & df.get("OS_status_JV").notna()
    return curative & has_os


def discovery_genomic_tested(df: pd.DataFrame) -> pd.Series:
    if "genomic_tested" in df.columns:
        return df["genomic_tested"].fillna(False).astype(bool)
    return _bool_series(df.index, True)


def discovery_imc(df: pd.DataFrame) -> pd.Series:
    if "imc_multiplex" in df.columns:
        return df["imc_multiplex"].fillna(False).astype(bool)
    return _bool_series(df.index, True)


def validation_survival(df: pd.DataFrame) -> pd.Series:
    time_ok = pd.to_numeric(df.get("follow_up_time"), errors="coerce").notna()
    event_ok = pd.to_numeric(df.get("vital_status"), errors="coerce").notna()
    return time_ok & event_ok


def validation_genomic_tested(df: pd.DataFrame) -> pd.Series:
    if "genomic_tested" in df.columns:
        return df["genomic_tested"].fillna(False).astype(bool)
    return _bool_series(df.index, False)


def validation_ihc(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in VALIDATION_IHC_COLS if c in df.columns]
    if not cols:
        return _bool_series(df.index, False)
    return df.apply(lambda row: _any_ihc(row, cols), axis=1)


DISCOVERY_MODALITY_ROWS: tuple[ModalityRow, ...] = (
    ModalityRow(
        "clinical",
        "Clinical data",
        discovery=lambda df: _bool_series(df.index, True),
        validation=lambda df: _bool_series(df.index, True),
    ),
    ModalityRow(
        "survival",
        "Survival",
        discovery=discovery_survival_eligible,
        validation=validation_survival,
    ),
    ModalityRow(
        "histology",
        "Histology",
        discovery=lambda df: df.get("COO", df.get("Location")).notna(),
        validation=lambda df: df.get("disease_type").notna(),
    ),
    ModalityRow(
        "genomic",
        "Genomic profiling",
        discovery=discovery_genomic_tested,
        validation=validation_genomic_tested,
        v1_aggregate="any",
    ),
    ModalityRow(
        "transcriptomic",
        "Transcriptomic profiling",
        discovery=lambda df: df.get("has_transcriptomic", _bool_series(df.index, True)),
        validation=lambda df: df.get("has_transcriptomic", _bool_series(df.index, True)),
    ),
    ModalityRow(
        "ihc",
        "IHC tumor profiling",
        discovery=lambda df: df.apply(lambda r: _any_ihc(r, DISCOVERY_IHC_COLS), axis=1),
        validation=validation_ihc,
    ),
    ModalityRow(
        "imc",
        "IMC multiplex",
        discovery=discovery_imc,
        validation=lambda df: _bool_series(df.index, False),
        v1_aggregate="any",
    ),
)


def _attach_discovery_flags(meta: pd.DataFrame, adata) -> pd.DataFrame:
    from .genomic_profiling import is_nested_genomic_profiling, patients_without_source_variants

    out = meta.copy()
    if "patient_id" in out.columns:
        out = out.set_index("patient_id")
    out.index = out.index.astype(str)

    gp = adata.uns.get("genomic_profiling")
    if gp is not None:
        no_var = patients_without_source_variants(gp, patient_ids=set(out.index))
        out["genomic_tested"] = ~out.index.isin(no_var)
    else:
        out["genomic_tested"] = True

    expr_cols = set(map(str, adata.uns["gene_expression"].columns))
    out["has_transcriptomic"] = out.index.isin(expr_cols)

    obs = adata.obs
    if "fov" in obs.columns and "patient_id" in obs.columns:
        has_imc = (
            obs.groupby("patient_id", observed=True)
            .apply(lambda g: g["fov"].notna().any(), include_groups=False)
            .astype(bool)
        )
        out["imc_multiplex"] = out.index.map(has_imc).fillna(False)
    else:
        out["imc_multiplex"] = True

    if "Location" in out.columns:
        out["Location"] = out["Location"].astype(str)
    return out


def build_discovery_patient_matrix(adata, patient_ids: list[str] | None = None) -> pd.DataFrame:
    """Patients × modality booleans for discovery (V2)."""
    from .dlbcl_io import DISCOVERY_PATIENTS, load_discovery_metadata

    subset = list(patient_ids or DISCOVERY_PATIENTS)
    meta = load_discovery_metadata(adata, subset)
    work = _attach_discovery_flags(meta, adata)

    data = {}
    for spec in DISCOVERY_MODALITY_ROWS:
        fn = spec.discovery
        if fn is None:
            data[spec.key] = _bool_series(work.index, False)
        else:
            data[spec.key] = fn(work).reindex(work.index).fillna(False).astype(bool)
    return pd.DataFrame(data, index=work.index)


def build_validation_patient_matrix(adata) -> pd.DataFrame:
    """Patients × modality booleans for validation (V2)."""
    meta = _validation_meta_frame(adata)
    data = {}
    for spec in DISCOVERY_MODALITY_ROWS:
        fn = spec.validation
        if fn is None:
            data[spec.key] = _bool_series(meta.index, False)
        else:
            data[spec.key] = fn(meta).reindex(meta.index).fillna(False).astype(bool)
    return pd.DataFrame(data, index=meta.index)


def _validation_meta_frame(adata) -> pd.DataFrame:
    from .validation_cohort import require_validation_cohort

    meta = pd.DataFrame(require_validation_cohort(adata)["meta"]).copy()
    meta.index = meta.index.astype(str)
    meta["has_transcriptomic"] = True
    meta["Location"] = meta["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    return _attach_validation_flags(meta, adata)


def _attach_validation_flags(meta: pd.DataFrame, adata) -> pd.DataFrame:
    """Attach per-patient modality flags for validation (e.g. NGS → genomic_tested)."""
    from .validation_cohort import require_validation_cohort

    out = meta.copy()
    if "patient_id" in out.columns:
        out = out.set_index("patient_id")
    out.index = out.index.astype(str)

    vc = require_validation_cohort(adata)
    ngs_patients: set[str] = set()
    if "ngs_data" in vc:
        ngs = pd.DataFrame(vc["ngs_data"])
        if "patient_alias" in ngs.columns:
            ngs_patients = set(ngs["patient_alias"].astype(str).str.strip().unique())
    out["genomic_tested"] = out.index.isin(ngs_patients)
    return out


def discovery_patient_locations(adata, patient_ids: list[str] | None = None) -> pd.Series:
    from .dlbcl_io import DISCOVERY_PATIENTS, load_discovery_metadata

    meta = load_discovery_metadata(adata, patient_ids or DISCOVERY_PATIENTS)
    work = _attach_discovery_flags(meta, adata)
    return work["Location"].astype(str)


def validation_patient_locations(adata) -> pd.Series:
    return _validation_meta_frame(adata)["Location"].astype(str)


def location_counts_from_matrix(
    matrix: pd.DataFrame,
    locations: pd.Series,
) -> dict[str, int]:
    loc = locations.reindex(matrix.index).astype(str)
    counts = {site: int((loc == site).sum()) for site in LOCATION_ORDER}
    return counts


def discovery_location_counts(adata, patient_ids: list[str] | None = None) -> dict[str, int]:
    from .dlbcl_io import DISCOVERY_PATIENTS, load_discovery_metadata

    meta = load_discovery_metadata(adata, patient_ids or DISCOVERY_PATIENTS)
    vc = meta["Location"].astype(str).value_counts()
    return {site: int(vc.get(site, 0)) for site in LOCATION_ORDER}


def validation_location_counts(adata) -> dict[str, int]:
    matrix = build_validation_patient_matrix(adata)
    from .validation_cohort import require_validation_cohort

    meta = pd.DataFrame(require_validation_cohort(adata)["meta"])
    meta.index = meta.index.astype(str)
    loc = meta["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    return location_counts_from_matrix(matrix, loc)


def build_v1_presence_matrix(
    matrix: pd.DataFrame,
    locations: pd.Series,
    *,
    rows: tuple[ModalityRow, ...] = DISCOVERY_MODALITY_ROWS,
) -> np.ndarray:
    """Aggregate patient matrix to location × modality (V1 blocks)."""
    loc = locations.reindex(matrix.index).astype(str)
    presence = np.zeros((len(rows), len(LOCATION_ORDER)), dtype=int)
    for i, spec in enumerate(rows):
        flags = matrix[spec.key]
        for j, site in enumerate(LOCATION_ORDER):
            site_mask = loc == site
            if not site_mask.any():
                presence[i, j] = 0
                continue
            site_vals = flags.loc[site_mask]
            if spec.v1_aggregate == "any":
                presence[i, j] = int(site_vals.any())
            else:
                presence[i, j] = int(site_vals.all())
    return presence


def _row_sorted_flags(flags: pd.Series) -> np.ndarray:
    vals = flags.fillna(False).astype(bool).to_numpy()
    order = np.argsort(~vals, kind="stable")  # True left
    return vals[order]


def _location_block_layout(
    location_counts: dict[str, int],
    *,
    cell_w: float,
) -> tuple[list[tuple[str, float, int]], float, float]:
    """Return ``[(site, x_start, n_patients), ...]``, total width, site gap."""
    site_gap = cell_w * SITE_GAP_FRAC
    blocks: list[tuple[str, float, int]] = []
    x = 0.0
    for site in LOCATION_ORDER:
        n = int(location_counts.get(site, 0))
        if n <= 0:
            continue
        blocks.append((site, x, n))
        x += n * cell_w + site_gap
    total_w = x - site_gap if blocks else 0.0
    return blocks, total_w, site_gap


def patient_cell_width(location_counts: dict[str, int], *, panel_width: float = 6.5) -> float:
    """Per-patient column width when a cohort fills ``panel_width``."""
    ncols = sum(int(location_counts.get(site, 0)) for site in LOCATION_ORDER)
    n_blocks = sum(1 for site in LOCATION_ORDER if location_counts.get(site, 0) > 0)
    return panel_width / max(ncols + n_blocks * SITE_GAP_FRAC, 1)


def uniform_patient_cell_width(*location_counts: dict[str, int], panel_width: float = 6.5) -> float:
    """Shared patient column width across cohorts (sized to the largest cohort)."""
    return min(patient_cell_width(c, panel_width=panel_width) for c in location_counts)


def _panel_x_extent(location_counts: dict[str, int], cell_w: float) -> float:
    """Horizontal data extent for subplot width ratios."""
    _blocks, total_w, site_gap = _location_block_layout(location_counts, cell_w=cell_w)
    label_pad = max(1.8, cell_w * 4)
    return label_pad + total_w + site_gap * 1.8


def plot_modality_matrix_v1(
    ax,
    presence: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    *,
    cell_w: float = 1.45,
    cell_h: float = 0.55,
) -> None:
    nrows, ncols = presence.shape
    for i in range(nrows):
        for j in range(ncols):
            base = ROW_COLORS[i % 2]
            facecolor = base if presence[i, j] else ABSENT_COLOR
            ax.add_patch(Rectangle((j * cell_w, i * cell_h), cell_w, cell_h, facecolor=facecolor, edgecolor="none"))
    for j in range(ncols + 1):
        ax.plot([j * cell_w, j * cell_w], [0, nrows * cell_h], color=GRID_COLOR, linewidth=1.1)
    ax.set_xlim(0, ncols * cell_w)
    ax.set_ylim(nrows * cell_h, 0)
    ax.set_xticks(np.arange(ncols) * cell_w + cell_w / 2)
    ax.set_xticklabels(col_labels, fontsize=10.5)
    ax.set_yticks(np.arange(nrows) * cell_h + cell_h / 2)
    ax.set_yticklabels(row_labels, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13.5, pad=30, weight="bold")


def plot_modality_matrix_v2(
    ax,
    matrix: pd.DataFrame,
    locations: pd.Series,
    row_labels: list[str],
    title: str,
    *,
    location_counts: dict[str, int] | None = None,
    cell_w: float | None = None,
    cell_h: float = 0.42,
    row_gap: float = ROW_GAP,
    panel_width: float = 6.5,
    patient_cell_edges: bool = False,
) -> None:
    """Per-patient strips grouped by anatomical site (V1 layout).

    Within each site block, every row sorts patients with the modality present to the left.
    Present cells use ``LOCATION_COLORS``; absent cells stay white.
    """
    if matrix.empty:
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, weight="bold")
        return

    loc = locations.reindex(matrix.index).astype(str)
    if location_counts is None:
        location_counts = {site: int((loc == site).sum()) for site in LOCATION_ORDER}

    ncols = len(matrix)
    if cell_w is None:
        n_blocks = sum(1 for site in LOCATION_ORDER if location_counts.get(site, 0) > 0)
        site_gap_est = 0.08
        cell_w = panel_width / max(ncols + n_blocks * SITE_GAP_FRAC, 1)

    blocks, total_w, site_gap = _location_block_layout(location_counts, cell_w=cell_w)
    keys = list(matrix.columns)
    nrows = len(row_labels)
    total_h = nrows * cell_h + (nrows - 1) * row_gap
    edgecolor = PATIENT_CELL_EDGE_COLOR if patient_cell_edges else "none"
    edgewidth = PATIENT_CELL_EDGE_LW if patient_cell_edges else 0.0

    for i, (key, label) in enumerate(zip(keys, row_labels, strict=True)):
        y0 = i * (cell_h + row_gap)
        row_true = 0
        for site, x_start, n_pts in blocks:
            site_pts = matrix.index[loc == site]
            flags = matrix.loc[site_pts, key]
            sorted_flags = _row_sorted_flags(flags)
            row_true += int(sorted_flags.sum())
            site_color = LOCATION_COLORS.get(site, ROW_COLORS[i % 2])
            for j, present in enumerate(sorted_flags):
                facecolor = site_color if present else ABSENT_COLOR
                ax.add_patch(
                    Rectangle(
                        (x_start + j * cell_w, y0),
                        cell_w,
                        cell_h,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=edgewidth,
                    )
                )
        ax.text(-0.35, y0 + cell_h / 2, label, ha="right", va="center", fontsize=9.5)
        ax.text(
            total_w + site_gap * 0.35,
            y0 + cell_h / 2,
            f"{row_true}/{ncols}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#555555",
        )

    for bi, (site, x_start, n_pts) in enumerate(blocks):
        x_end = x_start + n_pts * cell_w
        ax.plot([x_start, x_start], [0, total_h], color=GRID_COLOR, linewidth=1.1)
        ax.plot([x_end, x_end], [0, total_h], color=GRID_COLOR, linewidth=1.1)
        cx = x_start + (n_pts * cell_w) / 2
        ax.text(
            cx,
            total_h + row_gap * 0.55,
            f"{LOCATION_LABELS[site]}\n({n_pts})",
            ha="center",
            va="bottom",
            fontsize=10,
            color=LOCATION_COLORS.get(site, "#333333"),
        )
        if bi < len(blocks) - 1:
            divider = x_end + site_gap / 2
            ax.plot([divider, divider], [0, total_h], color=GRID_COLOR, linewidth=0.8, linestyle=":")

    label_pad = max(1.8, cell_w * 4)
    ax.set_xlim(-label_pad, total_w + site_gap * 1.8)
    ax.set_ylim(total_h + row_gap * 1.35, -row_gap * 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, weight="bold", pad=28)


def save_fig1c_v1(
    adata,
    fig_dir,
    *,
    patient_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write ``fig1C_cohort_modality_matrix`` V1 SVG + PNG."""
    from .dlbcl_io import DISCOVERY_PATIENTS, load_discovery_metadata

    fig_dir = Path(fig_dir)
    rows = DISCOVERY_MODALITY_ROWS
    row_labels = [r.label for r in rows]

    d_counts = discovery_location_counts(adata, patient_ids)
    v_counts = validation_location_counts(adata)
    d_total = sum(d_counts.values())
    v_total = sum(v_counts.values())

    d_meta = load_discovery_metadata(adata, patient_ids or DISCOVERY_PATIENTS)
    d_work = _attach_discovery_flags(d_meta, adata)
    d_matrix = build_discovery_patient_matrix(adata, patient_ids)
    d_presence = build_v1_presence_matrix(d_matrix, d_work["Location"])

    v_matrix = build_validation_patient_matrix(adata)
    from .validation_cohort import require_validation_cohort

    v_meta = pd.DataFrame(require_validation_cohort(adata)["meta"])
    v_meta.index = v_meta.index.astype(str)
    v_loc = v_meta["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    v_presence = build_v1_presence_matrix(v_matrix, v_loc)

    d_cols = [f"{LOCATION_LABELS[k]}\n({v})" for k, v in d_counts.items()]
    v_cols = [f"{LOCATION_LABELS[k]}\n({v})" for k, v in v_counts.items()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.8), sharey=True)
    plot_modality_matrix_v1(axes[0], d_presence, row_labels, d_cols, f"Discovery cohort (n={d_total})")
    plot_modality_matrix_v1(axes[1], v_presence, row_labels, v_cols, f"Validation cohort (n={v_total})")
    axes[1].tick_params(axis="y", labelleft=False)
    plt.subplots_adjust(left=0.28, right=0.98, top=0.78, bottom=0.08, wspace=0.08)

    out_base = fig_dir / "fig1C_cohort_modality_matrix"
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".svg"), out_base.with_suffix(".png")


def save_fig1c_v2(
    adata,
    fig_dir,
    *,
    patient_ids: list[str] | None = None,
    patient_cell_w: float | None = None,
    panel_width: float = 6.5,
) -> tuple[Path, Path]:
    """Write ``fig1C_cohort_modality_matrix_v2`` SVG + PNG."""
    return _save_fig1c_patient_strips(
        adata,
        fig_dir,
        stem="fig1C_cohort_modality_matrix_v2",
        uniform_cell_width=False,
        patient_cell_edges=False,
        patient_ids=patient_ids,
        patient_cell_w=patient_cell_w,
        panel_width=panel_width,
    )


def save_fig1c_v3(
    adata,
    fig_dir,
    *,
    patient_ids: list[str] | None = None,
    patient_cell_w: float | None = None,
    panel_width: float = 6.5,
) -> tuple[Path, Path]:
    """Write ``fig1C_cohort_modality_matrix_v3`` SVG + PNG.

    Like V2 but both cohorts share the same per-patient column width; subplot widths
    scale with the number of patients per cohort.
    """
    return _save_fig1c_patient_strips(
        adata,
        fig_dir,
        stem="fig1C_cohort_modality_matrix_v3",
        uniform_cell_width=True,
        patient_cell_edges=True,
        patient_ids=patient_ids,
        patient_cell_w=patient_cell_w,
        panel_width=panel_width,
    )


def _save_fig1c_patient_strips(
    adata,
    fig_dir,
    *,
    stem: str,
    uniform_cell_width: bool,
    patient_cell_edges: bool = False,
    patient_ids: list[str] | None = None,
    patient_cell_w: float | None = None,
    panel_width: float = 6.5,
) -> tuple[Path, Path]:
    fig_dir = Path(fig_dir)
    row_labels = [r.label for r in DISCOVERY_MODALITY_ROWS]

    d_counts = discovery_location_counts(adata, patient_ids)
    v_counts = validation_location_counts(adata)
    d_matrix = build_discovery_patient_matrix(adata, patient_ids)
    v_matrix = build_validation_patient_matrix(adata)
    d_locs = discovery_patient_locations(adata, patient_ids)
    v_locs = validation_patient_locations(adata)
    d_total = len(d_matrix)
    v_total = len(v_matrix)

    if patient_cell_w is not None:
        cell_w = patient_cell_w
        width_ratios = [_panel_x_extent(d_counts, cell_w), _panel_x_extent(v_counts, cell_w)]
        fig_w = 2.8 + 0.42 * sum(width_ratios)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(fig_w, 3.8),
            gridspec_kw={"width_ratios": width_ratios},
        )
    elif uniform_cell_width:
        cell_w = uniform_patient_cell_width(d_counts, v_counts, panel_width=panel_width)
        width_ratios = [_panel_x_extent(d_counts, cell_w), _panel_x_extent(v_counts, cell_w)]
        fig_w = 2.8 + 0.42 * sum(width_ratios)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(fig_w, 3.8),
            gridspec_kw={"width_ratios": width_ratios},
        )
    else:
        cell_w = None
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 3.8))

    plot_modality_matrix_v2(
        axes[0],
        d_matrix,
        d_locs,
        row_labels,
        f"Discovery cohort (n={d_total})",
        location_counts=d_counts,
        cell_w=cell_w,
        panel_width=panel_width,
        patient_cell_edges=patient_cell_edges,
    )
    plot_modality_matrix_v2(
        axes[1],
        v_matrix,
        v_locs,
        row_labels,
        f"Validation cohort (n={v_total})",
        location_counts=v_counts,
        cell_w=cell_w,
        panel_width=panel_width,
        patient_cell_edges=patient_cell_edges,
    )
    axes[1].tick_params(axis="y", labelleft=False)
    plt.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.08, wspace=0.22)

    out_base = fig_dir / stem
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".svg"), out_base.with_suffix(".png")
