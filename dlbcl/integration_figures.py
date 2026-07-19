"""Integration figures (Fig 4A–4D): donut/circos, associations, enrichment dotplots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

ARCHETYPE_NAME_MAP = {
    1: "low immune",
    2: "cytotoxic predominant",
    3: "complex immune",
}

LOCATION_RECODE = {
    "pBONE": "bone",
    "polyOST": "bone",
    "disseminated": "bone",
}

COLUMN_ALIASES = {
    "KotlovSig_update": "KotlovSig",
    "Ciav_Cluster_update": "Ciav_Cluster",
    "COO_NanoString_update": "COO_NanoString",
    "LymphGen_update": "Lymphgen",
    "Lymphgen_aug2024": "Lymphgen",
    "DLBclass_update": "DLBclass",
}

ECOTYPE_CONFIDENT_COL = "Lymphoma_Ecotype_confident"

# Genomic-classifier rings in Fig 4A circos (grey when ``genomic_tested`` is False).
GENOMIC_CLASSIFIER_RING_LABELS = frozenset({
    "Wright 2020\nLymphgen",
    "Chapuy 2025\nDLBclass",
    "Lacy 2020\nHMRN",
    "Shen 2023\nLymphPlex",
})
STEEN_ECOTYPE_RING = "Steen 2021\nLymphoma_Ecotype"
GENOMIC_NOT_TESTED_COLOR = "#d9d9d9"
GENOMIC_NOT_TESTED_LABEL = "missing genomic data"
GENOMIC_NOT_TESTED_LINEWIDTH = 0.55
HLA_CLASS_STATE_COL = "HLA_class_state"
HLA_CLASS_STATE_LABEL = "Tumor HLA\nclass state"
HLA_CLASS_STATE_DISPLAY = "Tumor HLA class state"
HLA_CLASS_STATE_ORDER = ["HLA-I+/HLA-II+", "HLA-I-/HLA-II+", "HLA-I-/HLA-II-"]
HLA_CLASS_STATE_COLORS = {
    "HLA-I+/HLA-II+": "#d99227",
    "HLA-I-/HLA-II+": "#9d8ac7",
    "HLA-I-/HLA-II-": "#6f3b8f",
    "NA": "whitesmoke",
}

DONUT_COLOR_PALETTES = {
    "Location": {
        "PCNS": "#1f77b4",
        "bone": "#ff7f0e",
        "nodal": "#2ca02c",
        "testis": "#d62728",
        "other": "#9467bd",
        "NA": "whitesmoke",
    },
    "Tumor Immune Archetype\n(this work)": {
        "cytotoxic predominant": "#e41a1c",
        "low immune": "rebeccapurple",
        "complex immune": "#2ca02c",
        "unknown": "whitesmoke",
        "nan": "whitesmoke",
        "NA": "whitesmoke",
    },
    HLA_CLASS_STATE_LABEL: HLA_CLASS_STATE_COLORS,
    "Li 2025\nLymphoMAP": {
        "FMAC": "#984ea3",
        "LN": "#ff7f00",
        "TEX": "#ffff33",
        "NA": "whitesmoke",
    },
    "Kotlov 2021\nLME": {
        "Depleted": "mediumpurple",
        "GC-like": "darkorange",
        "Inflammatory": "crimson",
        "Mesenchymal": "darkgoldenrod",
        "NA": "whitesmoke",
    },
    "Ciavarella 2018\nCluster": {
        "Hot": "coral",
        "Intermediate": "orchid",
        "Cold": "darkorchid",
        "NA": "whitesmoke",
    },
    "Steen 2021\nLymphoma_Ecotype": {
        "LE1": "#a6cee3",
        "LE2": "#e78ac3",
        "LE3": "#a6d854",
        "LE4": "#ffd92f",
        "LE5": "#b2df8a",
        "LE6": "#fb8072",
        "LE7": "#80b1d3",
        "LE8": "#e5c494",
        "LE9": "#fccde5",
        "unknown": "whitesmoke",
        "NA": "whitesmoke",
    },
    "Alizadeh 2000\nCell of Origin": {
        "GCB": "gold",
        "ABC": "darkviolet",
        "Intermediate": "pink",
        "unknown": "whitesmoke",
        "NA": "whitesmoke",
    },
    "Wright 2020\nLymphgen": {
        "MCD": "blueviolet",
        "BN2": "#80b1d3",
        "EZB": "darkgoldenrod",
        "Other": "lightgray",
        "ST2": "#fdb462",
        "BN2/MCD": "purple",
        "unknown": "whitesmoke",
        "NA": "whitesmoke",
    },
    "Chapuy 2025\nDLBclass": {
        "C1": "darkgoldenrod",
        "C3": "orange",
        "C4": "#d62728",
        "C5": "indigo",
        "0": "lightgray",
        "NA": "whitesmoke",
    },
    "Lacy 2020\nHMRN": {
        "C1": "darkgoldenrod",
        "C2": "#1b9e77",
        "C3": "orange",
        "C4": "#d62728",
        "C5": "indigo",
        "C6": "#7570b3",
        "NA": "whitesmoke",
    },
    "Shen 2023\nLymphPlex": {
        "MCD": "blueviolet",
        "BN2": "#80b1d3",
        "EZB": "darkgoldenrod",
        "ST2": "#fdb462",
        "N1": "#b3de69",
        "TP53": "#d62728",
        "Others": "lightgray",
        "Other": "lightgray",
        "NA": "whitesmoke",
    },
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "DejaVu Sans Display", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_show_close(fig, out_svg: Path | None = None, *, show: bool = True) -> None:
    """Write SVG to disk and render inline when running inside Jupyter."""
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        # Keep text as font elements (not path outlines) for Illustrator/Inkscape editability.
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)


def _circos_ring_label(ax, radius: float, label: str) -> None:
    """White bold ring label with a black underlay (font-backed SVG text, no path_effects)."""
    common = dict(ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0, radius, label, color="black", zorder=3, **common)
    ax.text(0, radius, label, color="white", zorder=4, **common)


def fdr_within_groups(p_values: pd.Series, groups: pd.Series) -> np.ndarray:
    """Benjamini–Hochberg FDR applied independently within each group."""

    def _bh(series: pd.Series) -> pd.Series:
        return pd.Series(benjamini_hochberg(series.to_numpy()), index=series.index)

    return (
        p_values.groupby(groups, group_keys=False)
        .apply(_bh)
        .reindex(p_values.index)
        .to_numpy()
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    q = np.full(n, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    p_fin = p[finite]
    m = len(p_fin)
    order = np.argsort(p_fin)
    ranked = p_fin[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_fin = np.empty(m, dtype=float)
    q_fin[order] = q_ranked
    q[finite] = q_fin
    return q


def build_integration_metadata(
    adata,
    arch_df: pd.DataFrame,
    *,
    patient_subset: list[str],
    require_source_variants: bool = False,
) -> pd.DataFrame:
    """Patient-level table for integration figures from ``adata.uns`` + archetype CSV.

    Adds ``genomic_tested`` when ``adata.uns['genomic_profiling']`` is present.
    Fig 4A keeps all patients and greys genomic rings when not tested; set
    ``require_source_variants=True`` to drop untested patients (legacy pie-grid behaviour).
    """
    classif = pd.DataFrame(adata.uns["case_classifications"]).copy()
    if "patient_id" in classif.columns:
        classif = classif.set_index("patient_id")
    classif.index = classif.index.astype(str)

    keep = set(map(str, patient_subset))
    meta = classif.loc[classif.index.isin(keep)].copy()
    meta["Location"] = meta["Location"].replace(LOCATION_RECODE)

    from .genomic_profiling import patients_without_source_variants

    gp = adata.uns.get("genomic_profiling")
    if gp is not None:
        no_var = patients_without_source_variants(gp, patient_ids=keep)
        meta["genomic_tested"] = ~meta.index.isin(no_var)
    else:
        meta["genomic_tested"] = True

    if require_source_variants:
        meta = meta.loc[meta["genomic_tested"]].copy()

    cluster_dict = dict(
        zip(arch_df["patient_id"].astype(str), arch_df["abundance_cluster_30"].astype(int))
    )
    meta["abundance_cluster_30"] = meta.index.map(cluster_dict)
    meta["abundance_cluster_30_label"] = meta["abundance_cluster_30"].map(ARCHETYPE_NAME_MAP)

    for old_col, new_col in COLUMN_ALIASES.items():
        if old_col in meta.columns:
            meta[new_col] = meta[old_col]

    from .validation_classifications import finalize_lymphoma_ecotype_columns

    if ECOTYPE_CONFIDENT_COL not in meta.columns:
        meta = finalize_lymphoma_ecotype_columns(meta)
    if ECOTYPE_CONFIDENT_COL in meta.columns:
        meta["Lymphoma_Ecotype"] = meta[ECOTYPE_CONFIDENT_COL].fillna("unknown")

    if HLA_CLASS_STATE_COL not in meta.columns:
        hla_cols = [c for c in ("HLAABC", "HLADR") if c in meta.columns]
        if len(hla_cols) < 2 and "case_path" in adata.uns:
            case_path = pd.DataFrame(adata.uns["case_path"]).copy()
            if "patient_id" in case_path.columns:
                case_path = case_path.set_index("patient_id")
            case_path.index = case_path.index.astype(str)
            for col in ("HLAABC", "HLADR"):
                if col not in meta.columns and col in case_path.columns:
                    meta[col] = case_path[col].reindex(meta.index)
        if {"HLAABC", "HLADR"}.issubset(meta.columns):
            hla_i = _clean_hla_call(meta["HLAABC"]).map({"retained": "HLA-I+", "loss": "HLA-I-"})
            hla_ii = _clean_hla_call(meta["HLADR"]).map({"retained": "HLA-II+", "loss": "HLA-II-"})
            meta[HLA_CLASS_STATE_COL] = np.where(
                hla_i.notna() & hla_ii.notna(),
                hla_i.astype(str) + "/" + hla_ii.astype(str),
                pd.NA,
            )

    meta = meta[meta["Location"].notna()].copy()
    return meta


def _clean_hla_call(values: pd.Series) -> pd.Series:
    """Normalize binary HLA calls to retained/loss for integration fallbacks."""
    return (
        values.astype("object")
        .replace(
            {
                0: "loss",
                1: "retained",
                0.0: "loss",
                1.0: "retained",
                "0": "loss",
                "1": "retained",
                "0.0": "loss",
                "1.0": "retained",
                "loss": "loss",
                "retained": "retained",
                "Loss": "loss",
                "Retained": "retained",
                "UNK": pd.NA,
                "unknown": pd.NA,
                "Unknown": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "": pd.NA,
            }
        )
    )


def circos_ring_sort_columns(df: pd.DataFrame) -> list[str]:
    """Metadata columns used for hierarchical patient order in Fig 4A.

    Must match the ring order in ``plot_donut_circos`` (center → outermost) so
    each ring shows contiguous segments within its parent blocks.
    """
    spatial_col = "abundance_cluster_30_label"
    ringplan_cols = [
        "Location",
        spatial_col,
        HLA_CLASS_STATE_COL,
        "lymphomap",
        "Ciav_Cluster",
        "KotlovSig",
        "Lymphoma_Ecotype",
        "COO_NanoString",
        "Lymphgen",
        "LymphPlex",
        "HMRN",
        "DLBclass",
    ]
    return [c for c in ringplan_cols if c in df.columns]


def _within_group_frequency_rank(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Rank ``value_col`` categories by descending frequency within each group."""
    if not group_cols:
        freq = df[value_col].value_counts()
        cat_order = freq.index.tolist()
        return df[value_col].map({k: j for j, k in enumerate(cat_order)})

    ranks: list[pd.Series] = []
    for _, subdf in df.groupby(group_cols, sort=False):
        freq = subdf[value_col].value_counts()
        cat_order = freq.index.tolist()
        ranks.append(subdf[value_col].map({k: j for j, k in enumerate(cat_order)}))
    return pd.concat(ranks).sort_index()


def order_patients_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """Hierarchical frequency ordering aligned with Fig 4A circos ring order."""
    order_cols = circos_ring_sort_columns(df)

    work = df.copy()
    sort_keys: list[str] = []
    for i, col in enumerate(order_cols):
        group_cols = order_cols[:i]
        work[f"_rank_{col}"] = _within_group_frequency_rank(
            work, group_cols=group_cols, value_col=col
        )
        sort_keys.extend([f"_rank_{col}", col])

    patient_key = "_patient_id_sort"
    work[patient_key] = work.index.astype(str)
    sort_keys.append(patient_key)

    ordered = work.sort_values(by=sort_keys, kind="mergesort")
    remaining = [
        c
        for c in ordered.columns
        if c not in order_cols and not c.startswith("_rank_") and c != patient_key
    ]
    return ordered[order_cols + remaining]


def _donut_color(col: str, val) -> str:
    palette = DONUT_COLOR_PALETTES.get(col, {})
    return palette.get(str(val), palette.get("NA", "#cccccc"))


def _circos_segment_color(col: str, val, *, genomic_tested: bool) -> str:
    if col in GENOMIC_CLASSIFIER_RING_LABELS and not genomic_tested:
        return GENOMIC_NOT_TESTED_COLOR
    return _donut_color(col, val)


def _polar_wedge_corners(
    theta_center: float,
    wedge_width: float,
    r_bottom: float,
    r_top: float,
) -> np.ndarray:
    """Four wedge vertices in polar data order: CCW inner → CW inner → CW outer → CCW outer."""
    th_lo = theta_center - wedge_width / 2
    th_hi = theta_center + wedge_width / 2
    return np.array(
        [
            [th_lo, r_bottom],
            [th_hi, r_bottom],
            [th_hi, r_top],
            [th_lo, r_top],
        ]
    )


def _stroke_sw_ne_on_polar_wedge(
    ax,
    theta_center: float,
    wedge_width: float,
    r_bottom: float,
    r_top: float,
    *,
    color: str = "0.15",
    linewidth: float = GENOMIC_NOT_TESTED_LINEWIDTH,
) -> None:
    """Thin SW→NE stroke between exact wedge vertices (inner CCW → outer CW)."""
    corners = _polar_wedge_corners(theta_center, wedge_width, r_bottom, r_top)
    # Wedge-local SW = (th_lo, r_bottom); NE = (th_hi, r_top) — always corners 0 and 2.
    ax.plot(
        corners[[0, 2], 0],
        corners[[0, 2], 1],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=4,
        clip_on=True,
    )


def _legend_missing_genomic_patch(ax, x: float, y: float, size: float) -> None:
    """Legend swatch: grey fill + SW→NE stroke (``transAxes`` coords)."""
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            size,
            size,
            facecolor=GENOMIC_NOT_TESTED_COLOR,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.plot(
        [x, x + size],
        [y, y + size],
        color="0.15",
        linewidth=GENOMIC_NOT_TESTED_LINEWIDTH * 1.4,
        solid_capstyle="butt",
        transform=ax.transAxes,
        clip_on=False,
        zorder=10,
    )


def _genomic_classifier_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to patients with source WES variants (for Fig 4B–4D statistics)."""
    if "genomic_tested" not in df.columns:
        return df
    return df.loc[df["genomic_tested"]].copy()


def _donut_segment_linewidth(n_patients: int, *, linewidth: float | None = None) -> float:
    """Edge width for circos ring segments; thinner when many patients."""
    if linewidth is not None:
        return linewidth
    if n_patients <= 100:
        return 0.5
    return max(0.05, min(0.5, 35.0 / n_patients))


def plot_donut_circos(
    df: pd.DataFrame,
    out_svg: Path,
    *,
    legend_svg: Path | None = None,
    segment_linewidth: float | None = None,
    show: bool = True,
) -> tuple[Path, Path | None]:
    """Radial integration plot (Fig 4A; ``make_donut_plot.py``)."""
    configure_matplotlib()
    work = df.copy()
    column_mapping = {
        "abundance_cluster_30_label": "Tumor Immune Archetype\n(this work)",
        HLA_CLASS_STATE_COL: HLA_CLASS_STATE_LABEL,
        "lymphomap": "Li 2025\nLymphoMAP",
        "Lymphoma_Ecotype": "Steen 2021\nLymphoma_Ecotype",
        "KotlovSig": "Kotlov 2021\nLME",
        "Ciav_Cluster": "Ciavarella 2018\nCluster",
        "COO_NanoString": "Alizadeh 2000\nCell of Origin",
        "Lymphgen": "Wright 2020\nLymphgen",
        "DLBclass": "Chapuy 2025\nDLBclass",
        "HMRN": "Lacy 2020\nHMRN",
        "LymphPlex": "Shen 2023\nLymphPlex",
    }
    for old_col, new_col in column_mapping.items():
        if old_col in work.columns:
            work[new_col] = work[old_col]

    center_col = "Location"
    rings = [
        "Tumor Immune Archetype\n(this work)",
        HLA_CLASS_STATE_LABEL,
        "Li 2025\nLymphoMAP",
        "Ciavarella 2018\nCluster",
        "Kotlov 2021\nLME",
        "Steen 2021\nLymphoma_Ecotype",
        "Alizadeh 2000\nCell of Origin",
        "Wright 2020\nLymphgen",
        "Shen 2023\nLymphPlex",
        "Lacy 2020\nHMRN",
        "Chapuy 2025\nDLBclass",
    ]
    rings = [r for r in rings if r in work.columns]
    # Blank width after a ring, in units of ring_width (half-ring gaps after HLA/COO;
    # when HLA is absent, e.g. validation Fig 6A, the same half-spacer follows archetype).
    archetype_label = "Tumor Immune Archetype\n(this work)"
    spacing_after: dict[str, float] = {}
    if HLA_CLASS_STATE_LABEL in rings:
        spacing_after[HLA_CLASS_STATE_LABEL] = 0.5
    elif archetype_label in rings:
        spacing_after[archetype_label] = 0.5
    if "Alizadeh 2000\nCell of Origin" in rings:
        spacing_after["Alizadeh 2000\nCell of Origin"] = 0.5

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    n_patients = len(work)
    edge_lw = _donut_segment_linewidth(n_patients, linewidth=segment_linewidth)
    width = 2 * np.pi / n_patients
    theta = np.linspace(0, 2 * np.pi, n_patients, endpoint=False) + width / 2

    inner_radius = 0.2
    ring_width = 0.08
    all_columns = [center_col] + rings
    outer_radius = inner_radius + len(all_columns) * ring_width
    pie_radius = outer_radius * 1.1

    location_counts = work[center_col].value_counts().sort_values(ascending=False)
    current_angle = 0.0
    for size, loc in zip(location_counts.values, location_counts.index):
        theta1 = current_angle
        theta2 = current_angle + (size / location_counts.sum()) * 360
        theta1_rad = np.deg2rad(theta1)
        theta2_rad = np.deg2rad(theta2)
        ax.bar(
            (theta1_rad + theta2_rad) / 2,
            pie_radius,
            width=abs(theta2_rad - theta1_rad),
            bottom=0,
            color=_donut_color(center_col, loc),
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )
        current_angle = theta2

    ring_idx = 0.0
    has_untested = "genomic_tested" in work.columns and not work["genomic_tested"].all()
    for col in all_columns[1:]:
        r_inner = inner_radius + ring_idx * ring_width
        colors = []
        missing_genomic = []
        for pid in work.index:
            genomic_tested = (
                bool(work.loc[pid, "genomic_tested"]) if "genomic_tested" in work.columns else True
            )
            colors.append(
                _circos_segment_color(col, work.loc[pid, col], genomic_tested=genomic_tested)
            )
            missing_genomic.append(col in GENOMIC_CLASSIFIER_RING_LABELS and not genomic_tested)
        for i, color in enumerate(colors):
            ax.bar(
                theta[i],
                ring_width,
                width=width,
                bottom=r_inner,
                color=color,
                edgecolor="white",
                linewidth=edge_lw,
                zorder=1,
            )
            if missing_genomic[i]:
                _stroke_sw_ne_on_polar_wedge(ax, theta[i], width, r_inner, r_inner + ring_width)
        label_radius = r_inner + ring_width / 2
        _circos_ring_label(ax, label_radius, col)
        gap_rings = spacing_after.get(col)
        if gap_rings is not None:
            ring_idx += 1.0
            spacing_r_inner = inner_radius + ring_idx * ring_width
            spacing_height = gap_rings * ring_width
            for i in range(n_patients):
                ax.bar(
                    theta[i],
                    spacing_height,
                    width=width,
                    bottom=spacing_r_inner,
                    color="none",
                    edgecolor="none",
                    linewidth=0,
                    zorder=1,
                )
            ring_idx += gap_rings
        else:
            ring_idx += 1.0

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    spacer_rings = sum(spacing_after.get(c, 0.0) for c in all_columns[1:])
    ax.set_ylim(0, inner_radius + (len(all_columns) + spacer_rings) * ring_width + 0.1)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)

    if legend_svg is not None:
        legend_cols = [center_col] + rings
        legend_fig, legend_ax = plt.subplots(figsize=(6, 8))
        y = 0.95
        for col in legend_cols:
            palette = DONUT_COLOR_PALETTES.get(col, {})
            legend_ax.text(0.01, y, col, fontsize=12, fontweight="bold", va="top")
            y -= 0.04
            entries = [(k, v) for k, v in palette.items() if k != "NA"]
            if col == "Tumor Immune Archetype\n(this work)":
                order = ["cytotoxic predominant", "low immune", "complex immune"]
                rank = {name: i for i, name in enumerate(order)}
                entries.sort(key=lambda x: (rank.get(str(x[0]), 999), str(x[0])))
            if col == HLA_CLASS_STATE_LABEL:
                rank = {name: i for i, name in enumerate(HLA_CLASS_STATE_ORDER)}
                entries.sort(key=lambda x: (rank.get(str(x[0]), 999), str(x[0])))
            for k, v in entries:
                legend_ax.add_patch(
                    plt.Rectangle((0.03, y - 0.015), 0.025, 0.025, color=v, transform=legend_ax.transAxes, clip_on=False)
                )
                legend_ax.text(0.06, y, str(k), fontsize=10, va="center", transform=legend_ax.transAxes)
                y -= 0.03
            if col in GENOMIC_CLASSIFIER_RING_LABELS and has_untested:
                _legend_missing_genomic_patch(legend_ax, 0.03, y - 0.015, 0.025)
                legend_ax.text(
                    0.06,
                    y,
                    GENOMIC_NOT_TESTED_LABEL,
                    fontsize=10,
                    va="center",
                    transform=legend_ax.transAxes,
                )
                y -= 0.03
            y -= 0.01
        legend_ax.axis("off")
        legend_fig.tight_layout()
        _save_show_close(legend_fig, legend_svg, show=show)

    return out_svg, legend_svg


def _association_metrics(df: pd.DataFrame, *, stratifier: str) -> tuple[list[str], dict[str, str]]:
    spatial_col = "abundance_cluster_30_label"
    if stratifier == "location":
        metrics = [
            spatial_col,
            HLA_CLASS_STATE_COL,
            "lymphomap",
            "Ciav_Cluster",
            "KotlovSig",
            "Lymphoma_Ecotype",
            "COO_NanoString",
            "Lymphgen",
            "DLBclass",
            "HMRN",
            "LymphPlex",
        ]
        display = {
            spatial_col: "This work",
            HLA_CLASS_STATE_COL: HLA_CLASS_STATE_DISPLAY,
            "lymphomap": "Li 2025 LymphoMAP",
            "Ciav_Cluster": "Ciavarella 2018 Cluster",
            "KotlovSig": "Kotlov 2021 LME",
            "Lymphoma_Ecotype": "Steen 2021 EcoTyper (confident)",
            "COO_NanoString": "Cell of Origin",
            "Lymphgen": "Wright 2020 Lymphgen",
            "DLBclass": "Chapuy 2025 DLBclass",
            "HMRN": "Lacy 2020 HMRN",
            "LymphPlex": "Shen 2023 LymphPlex",
        }
    else:
        metrics = [
            "Location",
            HLA_CLASS_STATE_COL,
            "lymphomap",
            "Ciav_Cluster",
            "KotlovSig",
            "Lymphoma_Ecotype",
            "COO_NanoString",
            "Lymphgen",
            "DLBclass",
            "HMRN",
            "LymphPlex",
        ]
        display = {
            "Location": "Tumor Location",
            HLA_CLASS_STATE_COL: HLA_CLASS_STATE_DISPLAY,
            "lymphomap": "Li 2025 LymphoMAP",
            "Ciav_Cluster": "Ciavarella 2018 Cluster",
            "KotlovSig": "Kotlov 2021 LME",
            "Lymphoma_Ecotype": "Steen 2021 EcoTyper (confident)",
            "COO_NanoString": "Cell of Origin",
            "Lymphgen": "Wright 2020 Lymphgen",
            "DLBclass": "Chapuy 2025 DLBclass",
            "HMRN": "Lacy 2020 HMRN",
            "LymphPlex": "Shen 2023 LymphPlex",
        }
    metrics = [m for m in metrics if m in df.columns]
    return metrics, display


def _global_association_test(contingency_test: pd.DataFrame) -> tuple[str, float, float | None, float | None]:
    if contingency_test.empty or contingency_test.size == 0:
        return "NA", float("nan"), None, None
    expected = chi2_contingency(contingency_test)[3]
    min_expected = expected.min()
    chi2 = None
    odds_ratio = None
    if min_expected >= 5:
        chi2, p_value, _, _ = chi2_contingency(contingency_test)
        test_type = "Chi-square"
    elif contingency_test.shape == (2, 2):
        odds_ratio, p_value = fisher_exact(contingency_test)
        test_type = "Fisher's exact"
    else:
        chi2, p_value, _, _ = chi2_contingency(contingency_test, correction=True)
        test_type = "Chi-square"
    n = contingency_test.sum().sum()
    min_dim = min(contingency_test.shape) - 1
    cramer_v = float(np.sqrt(chi2 / (n * min_dim))) if chi2 is not None and min_dim > 0 else 0.0
    if cramer_v < 0.1:
        effect = "Negligible"
    elif cramer_v < 0.3:
        effect = "Small"
    elif cramer_v < 0.5:
        effect = "Medium"
    else:
        effect = "Large"
    return test_type, p_value, chi2 if test_type == "Chi-square" else odds_ratio, cramer_v


def _neg_log10_fdr(fdr_values: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(fdr_values.astype(float), 1e-300, 1.0))


def _fdr_barchart_colors(fdr_values: np.ndarray) -> list:
    return [
        "darkred" if fdr < 0.01 else "red" if fdr < 0.05 else "0.88"
        for fdr in fdr_values
    ]


def plot_association_barchart(
    results_df: pd.DataFrame,
    title: str,
    out_svg: Path,
    *,
    y_axis: str = "cramer",
    show: bool = True,
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    metric_names = results_df["Metric"].tolist()
    fdr_values = np.clip(results_df["FDR"].values.astype(float), 1e-300, 1.0)
    cramer_v = results_df["Cramer_V"].values.astype(float)

    fig, ax = plt.subplots(figsize=(12, 7))

    if y_axis == "fdr":
        y_vals = _neg_log10_fdr(fdr_values)
        colors = _fdr_barchart_colors(fdr_values)
        bars = ax.bar(range(len(metric_names)), y_vals, color=colors, edgecolor="0.45", linewidth=0.6)
        ax.axhline(
            -np.log10(0.05),
            color="0.35",
            linestyle="--",
            alpha=0.8,
            label="FDR=0.05",
        )
        ax.set_ylabel("-log10(FDR)")
        ymax = float(np.nanmax(y_vals)) if len(y_vals) else 1.0
        ax.set_ylim(0, max(1.5, ymax * 1.12 + 0.1))
        legend_handles = [
            Patch(facecolor="darkred", edgecolor="0.45", label="FDR<0.01"),
            Patch(facecolor="red", edgecolor="0.45", label="FDR<0.05"),
            Patch(facecolor="0.88", edgecolor="0.45", label="FDR≥0.05"),
            Line2D([0], [0], color="0.35", linestyle="--", label="FDR=0.05"),
        ]
    else:
        y_vals = cramer_v
        cmap = plt.colormaps.get_cmap("Blues")
        colors = []
        for fdr in fdr_values:
            if fdr < 0.05:
                intensity = 0.35 + 0.65 * min(float(-np.log10(fdr)) / 4.0, 1.0)
                colors.append(cmap(intensity))
            else:
                colors.append("0.88")
        bars = ax.bar(range(len(metric_names)), y_vals, color=colors, edgecolor="0.45", linewidth=0.6)
        ax.axhline(
            CRAMER_V_HIGHLIGHT_MIN,
            color="0.35",
            linestyle="--",
            alpha=0.8,
            label=f"V={CRAMER_V_HIGHLIGHT_MIN:g} (small effect)",
        )
        ax.set_ylabel("Cramér's V")
        ax.set_ylim(0, max(0.55, float(np.nanmax(cramer_v)) * 1.15 + 0.05))
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=0.05))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("FDR (bar color if q<0.05)")
        legend_handles = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="0.12", markersize=10, label="FDR<0.05"),
            Line2D([0], [0], color="0.35", linestyle="--", label=f"V={CRAMER_V_HIGHLIGHT_MIN:g}"),
            Patch(facecolor="0.88", edgecolor="0.45", label="FDR≥0.05"),
        ]

    ax.set_xlabel("Metrics")
    ax.set_title(title)
    ax.set_xticks(range(len(metric_names)), metric_names, rotation=45, ha="right")

    for i, bar in enumerate(bars):
        if y_axis != "fdr" and fdr_values[i] < 0.05:
            ax.scatter(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                marker="*",
                s=70,
                c="0.12",
                zorder=5,
            )
        row = results_df.iloc[i]
        n_note = f"n={int(row['n_tested'])}" if "n_tested" in row and pd.notna(row["n_tested"]) else ""
        label = f"FDR={row['FDR']:.3g}"
        if n_note:
            label = f"{n_note}  {label}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="0.25",
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)


def plot_combined_association_barchart(
    location_results: pd.DataFrame,
    archetype_results: pd.DataFrame,
    out_svg: Path,
    *,
    y_axis: str = "cramer",
    show: bool = True,
) -> Path:
    """Fig 4B — grouped bars for location vs archetype associations (``create_combined_association_barchart``)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    loc_df = location_results.set_index("Metric")
    arch_df = archetype_results.set_index("Metric")
    metric_order = sorted(
        set(loc_df.index).union(set(arch_df.index)),
        key=lambda m: (float(arch_df["FDR"].get(m, np.inf)), m),
    )

    y_loc = loc_df["Cramer_V"].reindex(metric_order).values.astype(float)
    y_arch = arch_df["Cramer_V"].reindex(metric_order).values.astype(float)
    fdr_loc = np.clip(loc_df["FDR"].reindex(metric_order).values.astype(float), 1e-300, 1.0)
    fdr_arch = np.clip(arch_df["FDR"].reindex(metric_order).values.astype(float), 1e-300, 1.0)

    x = np.arange(len(metric_order))
    width = 0.38
    loc_color = "#4C78A8"
    arch_color = "#F58518"

    fig, ax = plt.subplots(figsize=(14, 8))

    if y_axis == "fdr":
        y_loc = _neg_log10_fdr(fdr_loc)
        y_arch = _neg_log10_fdr(fdr_arch)
        ax.bar(x - width / 2, y_loc, width=width, color=loc_color, label="Tumor Location", edgecolor="0.35", linewidth=0.5)
        ax.bar(x + width / 2, y_arch, width=width, color=arch_color, label="Archetype", edgecolor="0.35", linewidth=0.5)
        ax.axhline(-np.log10(0.05), color="0.35", linestyle="--", alpha=0.8, label="FDR=0.05")
        ax.set_ylabel("-log10(FDR)")
        ymax = float(np.nanmax(np.r_[y_loc, y_arch])) if len(metric_order) else 1.0
        ax.set_ylim(0, max(1.5, ymax * 1.12 + 0.1))
        legend_handles = [
            Patch(facecolor=loc_color, label="Tumor Location"),
            Patch(facecolor=arch_color, label="Archetype"),
            Line2D([0], [0], color="0.35", linestyle="--", label="FDR=0.05"),
        ]
    else:
        ax.bar(x - width / 2, y_loc, width=width, color=loc_color, label="Tumor Location", edgecolor="0.35", linewidth=0.5)
        ax.bar(x + width / 2, y_arch, width=width, color=arch_color, label="Archetype", edgecolor="0.35", linewidth=0.5)
        ax.axhline(
            CRAMER_V_HIGHLIGHT_MIN,
            color="0.35",
            linestyle="--",
            alpha=0.8,
            label=f"V={CRAMER_V_HIGHLIGHT_MIN:g}",
        )
        ax.set_ylabel("Cramér's V")
        ax.set_ylim(0, max(0.55, float(np.nanmax(np.r_[y_loc, y_arch])) * 1.15 + 0.05))
        for xi, (vl, va, fl, fa) in enumerate(zip(y_loc, y_arch, fdr_loc, fdr_arch)):
            if np.isfinite(vl) and fl < 0.05:
                ax.scatter(xi - width / 2, vl + 0.015, marker="*", s=70, c=loc_color, edgecolors="0.12", zorder=5)
            if np.isfinite(va) and fa < 0.05:
                ax.scatter(xi + width / 2, va + 0.015, marker="*", s=70, c=arch_color, edgecolors="0.12", zorder=5)
        legend_handles = [
            Patch(facecolor=loc_color, label="Tumor Location"),
            Patch(facecolor=arch_color, label="Archetype"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="0.2", markersize=10, label="FDR<0.05"),
            Line2D([0], [0], color="0.35", linestyle="--", label=f"V={CRAMER_V_HIGHLIGHT_MIN:g}"),
        ]

    ax.set_xlabel("Metric")
    ax.set_title(
        "Association of tumor location and archetype with known classifiers (χ² test over crosstab)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x, metric_order, rotation=45, ha="right")
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)
    return out_svg


# Reflexive rows shown as single-dot dumbbells (not paired with external classifiers).
ASSOCIATION_DUMBBELL_LOCATION_SINGLE = "This work"
ASSOCIATION_DUMBBELL_ARCHETYPE_SINGLE = "Tumor Location"
ASSOCIATION_DUMBBELL_PAIR_EXCLUDE: frozenset[str] = frozenset({
    ASSOCIATION_DUMBBELL_LOCATION_SINGLE,
    ASSOCIATION_DUMBBELL_ARCHETYPE_SINGLE,
})
# Legacy: validation dumbbell once listed confident separately from best match.
ASSOCIATION_DUMBBELL_METRIC_EXCLUDE: frozenset[str] = frozenset()

ASSOCIATION_DUMBBELL_SHORT_LABELS: dict[str, str] = {
    "This work": "This work",
    "Tumor Location": "Location",
    HLA_CLASS_STATE_DISPLAY: "HLA class",
    "Kotlov 2021 LME": "LME",
    "Cell of Origin": "COO",
    "Steen 2021 Lymphoma Ecotype": "EcoTyper",
    "Steen 2021 EcoTyper (best match)": "EcoTyper",
    "Steen 2021 EcoTyper (confident)": "EcoTyper (conf.)",
    "Chapuy 2025 DLBclass": "DLBclass",
    "Wright 2020 Lymphgen": "Lymphgen",
    "Li 2025 LymphoMAP": "LymphoMAP",
    "Ciavarella 2018 Cluster": "Ciavarella",
    "Lacy 2020 HMRN": "HMRN",
    "Shen 2023 LymphPlex": "LymphPlex",
}

CLASSIFIER_PAIRWISE_COLUMNS: list[tuple[str, str]] = [
    ("abundance_cluster_30_label", "This work"),
    ("Location", "Tumor Location"),
    (HLA_CLASS_STATE_COL, HLA_CLASS_STATE_DISPLAY),
    ("KotlovSig", "Kotlov 2021 LME"),
    ("lymphomap", "Li 2025 LymphoMAP"),
    ("Ciav_Cluster", "Ciavarella 2018 Cluster"),
    ("COO_NanoString", "Cell of Origin"),
    ("Lymphgen", "Wright 2020 Lymphgen"),
    ("DLBclass", "Chapuy 2025 DLBclass"),
    ("HMRN", "Lacy 2020 HMRN"),
    ("LymphPlex", "Shen 2023 LymphPlex"),
    ("Lymphoma_Ecotype", "Steen 2021 EcoTyper (confident)"),
]


def _association_dumbbell_rows(
    location_results: pd.DataFrame,
    archetype_results: pd.DataFrame,
    *,
    exclude_metrics: frozenset[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (archetype-higher rows, location-higher rows) for grouped dumbbell layout."""
    skip = ASSOCIATION_DUMBBELL_METRIC_EXCLUDE | (exclude_metrics or frozenset())
    loc_df = location_results.set_index("Metric").drop(skip, errors="ignore")
    arch_df = archetype_results.set_index("Metric").drop(skip, errors="ignore")

    arch_group: list[dict[str, object]] = []
    loc_group: list[dict[str, object]] = []

    arch_metric = ASSOCIATION_DUMBBELL_ARCHETYPE_SINGLE
    if arch_metric in arch_df.index and np.isfinite(arch_df.loc[arch_metric, "Cramer_V"]):
        va = float(arch_df.loc[arch_metric, "Cramer_V"])
        arch_group.append(
            {
                "metric": arch_metric,
                "loc_v": np.nan,
                "arch_v": va,
                "loc_fdr": np.nan,
                "arch_fdr": float(arch_df.loc[arch_metric, "FDR"]),
                "sort_v": va,
            }
        )

    loc_metric = ASSOCIATION_DUMBBELL_LOCATION_SINGLE
    if loc_metric in loc_df.index and np.isfinite(loc_df.loc[loc_metric, "Cramer_V"]):
        vl = float(loc_df.loc[loc_metric, "Cramer_V"])
        loc_group.append(
            {
                "metric": loc_metric,
                "loc_v": vl,
                "arch_v": np.nan,
                "loc_fdr": float(loc_df.loc[loc_metric, "FDR"]),
                "arch_fdr": np.nan,
                "sort_v": vl,
            }
        )

    shared = set(loc_df.index).intersection(arch_df.index) - ASSOCIATION_DUMBBELL_PAIR_EXCLUDE
    for metric in shared:
        vl = loc_df.loc[metric, "Cramer_V"] if metric in loc_df.index else np.nan
        va = arch_df.loc[metric, "Cramer_V"] if metric in arch_df.index else np.nan
        if not (np.isfinite(vl) and np.isfinite(va)):
            continue
        vl_f, va_f = float(vl), float(va)
        row = {
            "metric": metric,
            "loc_v": vl_f,
            "arch_v": va_f,
            "loc_fdr": float(loc_df.loc[metric, "FDR"]),
            "arch_fdr": float(arch_df.loc[metric, "FDR"]),
            "sort_v": max(vl_f, va_f),
        }
        if va_f > vl_f:
            arch_group.append(row)
        else:
            loc_group.append(row)

    arch_group.sort(key=lambda r: (-float(r["sort_v"]), str(r["metric"])))
    loc_group.sort(key=lambda r: (-float(r["sort_v"]), str(r["metric"])))
    return arch_group, loc_group


def _association_dumbbell_metric_order(
    location_results: pd.DataFrame,
    archetype_results: pd.DataFrame,
    *,
    exclude: frozenset[str] | None = None,
) -> list[str]:
    """Metric labels in dumbbell row order (archetype-higher block, then location-higher)."""
    arch_group, loc_group = _association_dumbbell_rows(location_results, archetype_results)
    rows = arch_group + loc_group
    if exclude is not None:
        rows = [r for r in rows if str(r["metric"]) not in exclude]
    return [str(r["metric"]) for r in rows]


def _draw_dumbbell_dot(
    ax,
    x: float,
    y: float,
    *,
    color: str,
    fdr: float,
) -> None:
    """Filled dot if FDR<0.05, else hollow."""
    significant = np.isfinite(fdr) and fdr < 0.05
    if significant:
        ax.scatter(
            x,
            y,
            s=62,
            facecolors=color,
            edgecolors="0.25",
            linewidths=0.6,
            zorder=3,
        )
    else:
        ax.scatter(
            x,
            y,
            s=62,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            zorder=3,
        )


def plot_combined_association_dumbbell(
    location_results: pd.DataFrame,
    archetype_results: pd.DataFrame,
    out_svg: Path,
    *,
    metric_order: list[str] | None = None,
    show: bool = True,
) -> Path:
    """Fig 4B — dumbbell plot of Cramér's V vs location (left) and archetype (right)."""
    from matplotlib.lines import Line2D

    arch_group, loc_group = _association_dumbbell_rows(location_results, archetype_results)
    group_gap = 0.9 if arch_group and loc_group else 0.0

    def _y_positions(n_arch: int, n_loc: int) -> tuple[np.ndarray, np.ndarray]:
        arch_y = np.arange(n_arch, dtype=float)
        loc_y = np.arange(n_loc, dtype=float) + n_arch + group_gap
        return arch_y, loc_y

    if metric_order is not None:
        order_index = {m: i for i, m in enumerate(metric_order)}
        arch_group = sorted(arch_group, key=lambda r: order_index.get(str(r["metric"]), 999))
        loc_group = sorted(loc_group, key=lambda r: order_index.get(str(r["metric"]), 999))
    if not arch_group and not loc_group:
        raise ValueError("No classifier metrics for dumbbell plot")

    loc_color = "#4C78A8"
    arch_color = "#F58518"

    arch_y, loc_y = _y_positions(len(arch_group), len(loc_group))
    y_max = (loc_y[-1] if len(loc_group) else arch_y[-1]) if (arch_group or loc_group) else 0.0

    fig_h = max(4.0, 0.38 * (len(arch_group) + len(loc_group) + (1 if group_gap else 0)) + 2.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))

    all_v: list[float] = []

    def _plot_rows(rows: list[dict[str, object]], y_values: np.ndarray) -> None:
        for row, y in zip(rows, y_values):
            vl = float(row["loc_v"]) if np.isfinite(row["loc_v"]) else np.nan
            va = float(row["arch_v"]) if np.isfinite(row["arch_v"]) else np.nan
            loc_fdr = float(row["loc_fdr"]) if np.isfinite(row["loc_fdr"]) else np.nan
            arch_fdr = float(row["arch_fdr"]) if np.isfinite(row["arch_fdr"]) else np.nan
            if np.isfinite(vl):
                all_v.append(vl)
            if np.isfinite(va):
                all_v.append(va)
            if np.isfinite(vl) and np.isfinite(va):
                ax.plot([vl, va], [y, y], color="0.78", linewidth=1.8, solid_capstyle="round", zorder=1)
            if np.isfinite(vl):
                _draw_dumbbell_dot(ax, vl, y, color=loc_color, fdr=loc_fdr)
            if np.isfinite(va):
                _draw_dumbbell_dot(ax, va, y, color=arch_color, fdr=arch_fdr)

    _plot_rows(arch_group, arch_y)
    _plot_rows(loc_group, loc_y)

    if arch_group:
        ax.text(
            -0.22,
            float(np.mean(arch_y)),
            "Archetype higher",
            transform=ax.get_yaxis_transform(),
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="0.25",
        )
    if loc_group:
        ax.text(
            -0.22,
            float(np.mean(loc_y)),
            "Location higher",
            transform=ax.get_yaxis_transform(),
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="0.25",
        )

    if arch_group and loc_group:
        divider = (arch_y[-1] + loc_y[0]) / 2.0
        ax.axhline(divider, color="0.82", linewidth=0.9, zorder=0)

    y_ticks = list(arch_y) + list(loc_y)
    y_labels = (
        [ASSOCIATION_DUMBBELL_SHORT_LABELS.get(str(r["metric"]), str(r["metric"])) for r in arch_group]
        + [ASSOCIATION_DUMBBELL_SHORT_LABELS.get(str(r["metric"]), str(r["metric"])) for r in loc_group]
    )

    xmax = float(np.nanmax(all_v)) if all_v else 0.5
    ax.set_xlim(0, max(0.45, xmax * 1.12 + 0.04))
    ax.set_ylim(-0.6, y_max + 0.6)
    ax.invert_yaxis()
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlabel("Cramér's V")
    ax.set_title(
        "Classifier association with tumor location vs immune archetype",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.35, linewidth=0.6)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="0.15", markeredgecolor="0.15",
            markersize=8, label="FDR < 0.05",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.15",
            markersize=8, markeredgewidth=1.4, label="FDR ≥ 0.05",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5, framealpha=0.95)
    fig.subplots_adjust(left=0.36)
    _save_show_close(fig, out_svg, show=show)
    return out_svg


def _fdr_significance_stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def compute_classifier_pairwise_associations(
    df: pd.DataFrame,
    *,
    columns: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise Cramér's V between patient-level classifiers (complete cases per pair)."""
    df = _genomic_classifier_cohort(df)
    pairs = columns or CLASSIFIER_PAIRWISE_COLUMNS
    present = [(col, label) for col, label in pairs if col in df.columns]
    labels = [label for _, label in present]

    records: list[dict] = []
    for i, (col_a, label_a) in enumerate(present):
        for col_b, label_b in present[i + 1 :]:
            series_a = _prepare_classifier_series(df[col_a], label_a)
            series_b = _prepare_classifier_series(df[col_b], label_b)
            series_a = _mask_nonspecific_classifier_levels(series_a)
            series_b = _mask_nonspecific_classifier_levels(series_b)
            valid = series_a.notna() & series_b.notna()
            if int(valid.sum()) < 4:
                continue
            table = pd.crosstab(series_a[valid], series_b[valid])
            table = _drop_nonspecific_crosstab_levels(table)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            _, p_value, _, cramer_v = _global_association_test(table)
            records.append(
                {
                    "classifier_a": label_a,
                    "classifier_b": label_b,
                    "n": int(valid.sum()),
                    "Cramer_V": cramer_v,
                    "p_value": p_value,
                }
            )

    assoc = pd.DataFrame(records)
    v_mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    q_mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    if assoc.empty:
        return assoc, v_mat, q_mat

    assoc["FDR"] = benjamini_hochberg(assoc["p_value"].values)
    for _, row in assoc.iterrows():
        a, b = row["classifier_a"], row["classifier_b"]
        v_mat.loc[a, b] = row["Cramer_V"]
        v_mat.loc[b, a] = row["Cramer_V"]
        q_mat.loc[a, b] = row["FDR"]
        q_mat.loc[b, a] = row["FDR"]
    np.fill_diagonal(v_mat.values, 1.0)
    return assoc, v_mat, q_mat


def plot_classifier_pairwise_heatmap(
    v_mat: pd.DataFrame,
    q_mat: pd.DataFrame,
    out_svg: Path,
    *,
    title: str = "Pairwise classifier association (Cramér's V)",
    show: bool = True,
) -> Path:
    """Symmetric heatmap of classifier–classifier Cramér's V; stars mark FDR significance."""
    import seaborn as sns
    from matplotlib.lines import Line2D

    short = ASSOCIATION_DUMBBELL_SHORT_LABELS
    row_labels = [short.get(str(x), str(x)) for x in v_mat.index]
    col_labels = [short.get(str(x), str(x)) for x in v_mat.columns]

    annot = pd.DataFrame("", index=v_mat.index, columns=v_mat.columns, dtype=object)
    for i in annot.index:
        for j in annot.columns:
            if i != j:
                annot.loc[i, j] = _fdr_significance_stars(q_mat.loc[i, j])

    n = len(v_mat)
    figsize = (max(6.5, 0.52 * n + 2.2), max(5.5, 0.52 * n + 1.8))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        v_mat.astype(float),
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "0.95"},
        cbar_kws={"label": "Cramér's V", "shrink": 0.82},
        ax=ax,
    )
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.95", markersize=12, linestyle="none", label="FDR < 0.05"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.95", markersize=14, linestyle="none", label="FDR < 0.01"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.12, 1), fontsize=8, framealpha=0.95)
    fig.tight_layout()
    _save_show_close(fig, out_svg, show=show)
    return out_svg


def run_association_analysis(
    df: pd.DataFrame,
    *,
    stratifier: str,
    out_dir: Path,
    pairwise_csv: str,
    barchart_svg: Path | None = None,
    barchart_title: str = "",
) -> pd.DataFrame:
    """Global + pairwise association tests vs Location or archetype (spatial protein)."""
    df = _genomic_classifier_cohort(df)
    strat_col = "Location" if stratifier == "location" else "abundance_cluster_30_label"
    metrics, display_names = _association_metrics(df, stratifier=stratifier)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for metric in metrics:
        metric_label = display_names[metric]
        class_series = _prepare_classifier_series(
            df[metric], metric_label, strat_series=df[strat_col]
        )
        class_series = _mask_nonspecific_classifier_levels(class_series)
        valid = class_series.notna()
        n_tested = int(valid.sum())
        n_levels = int(class_series[valid].nunique()) if n_tested else 0
        if n_tested == 0 or n_levels < 2:
            results.append(
                {
                    "Metric": display_names[metric],
                    "Test_Type": "NA",
                    "Chi2_or_OR": None,
                    "p_value": float("nan"),
                    "Cramer_V": None,
                    "Effect_Size": "NA",
                    "n_tested": n_tested,
                    "n_levels": n_levels,
                }
            )
            continue
        contingency_test = pd.crosstab(
            class_series[valid],
            df.loc[valid, strat_col].astype(str),
        )
        contingency_test = _drop_nonspecific_crosstab_levels(contingency_test)
        if contingency_test.shape[0] < 2 or contingency_test.shape[1] < 2:
            results.append(
                {
                    "Metric": display_names[metric],
                    "Test_Type": "NA",
                    "Chi2_or_OR": None,
                    "p_value": float("nan"),
                    "Cramer_V": None,
                    "Effect_Size": "NA",
                    "n_tested": n_tested,
                    "n_levels": n_levels,
                }
            )
            continue
        if contingency_test.empty or contingency_test.size == 0:
            results.append(
                {
                    "Metric": display_names[metric],
                    "Test_Type": "NA",
                    "Chi2_or_OR": None,
                    "p_value": float("nan"),
                    "Cramer_V": None,
                    "Effect_Size": "NA",
                    "n_tested": n_tested,
                    "n_levels": n_levels,
                }
            )
            continue
        test_type, p_value, chi2_or_or, cramer_v = _global_association_test(contingency_test)
        effect = (
            "Negligible"
            if cramer_v < 0.1
            else "Small"
            if cramer_v < 0.3
            else "Medium"
            if cramer_v < 0.5
            else "Large"
        )
        results.append(
            {
                "Metric": display_names[metric],
                "Test_Type": test_type,
                "Chi2_or_OR": chi2_or_or,
                "p_value": p_value,
                "Cramer_V": cramer_v,
                "Effect_Size": effect,
                "n_tested": n_tested,
                "n_levels": n_levels,
            }
        )

    results_df = pd.DataFrame(results)
    results_df["FDR"] = benjamini_hochberg(results_df["p_value"].values)
    results_df.to_csv(out_dir / f"{stratifier}_association_results.csv", index=False)
    if barchart_svg is not None:
        sorted_results = results_df.sort_values("FDR").reset_index(drop=True)
        plot_association_barchart(sorted_results, barchart_title, barchart_svg, y_axis="cramer")
        fdr_svg = barchart_svg.with_name(f"{barchart_svg.stem}_fdr{barchart_svg.suffix}")
        plot_association_barchart(sorted_results, barchart_title, fdr_svg, y_axis="fdr")

    pairwise_rows = []
    groups = (
        sorted(df[strat_col].dropna().astype(str).unique().tolist(), key=str)
        if metrics
        else []
    )
    for metric in metrics:
        metric_label = display_names[metric]
        class_series = _prepare_classifier_series(
            df[metric], metric_label, strat_series=df[strat_col]
        )
        class_series = _mask_nonspecific_classifier_levels(class_series)
        valid = class_series.notna()
        contingency = pd.crosstab(
            class_series[valid],
            df.loc[valid, strat_col].astype(str),
        )
        contingency = _drop_nonspecific_crosstab_levels(contingency)
        metric_values = [v for v in contingency.index.tolist() if not _is_nonspecific_classifier_level(v)]
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                for metric_val in metric_values:
                    if metric_val not in contingency.index:
                        continue
                    others = [v for v in metric_values if v != metric_val]
                    table = pd.DataFrame(
                        {
                            g1: [contingency.loc[metric_val, g1], contingency.loc[others, g1].sum()],
                            g2: [contingency.loc[metric_val, g2], contingency.loc[others, g2].sum()],
                        },
                        index=[metric_val, "Others"],
                    )
                    if table.values.min() == 0:
                        continue
                    odds_ratio, p_value = fisher_exact(table)
                    pairwise_rows.append(
                        {
                            "Metric": display_names[metric],
                            "Value": metric_val,
                            "Location1": g1,
                            "Location2": g2,
                            "Pct1": table.loc[metric_val, g1] / table[g1].sum() * 100,
                            "Pct2": table.loc[metric_val, g2] / table[g2].sum() * 100,
                            "Odds_Ratio": odds_ratio,
                            "p_value": p_value,
                        }
                    )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["FDR"] = benjamini_hochberg(pairwise_df["p_value"].values)
        pairwise_df.to_csv(out_dir / pairwise_csv, index=False)
    return results_df


LOCATION_GROUP_ORDER = ["PCNS", "bone", "nodal", "testis"]
ARCHETYPE_GROUP_ORDER = ["low immune", "cytotoxic predominant", "complex immune"]

_SKIP_FEATURE_VALUES = {"unknown", "unassigned", "nan", "na", ""}

# Catch-all classifier levels excluded from association crosstabs (global + pairwise).
CROSSTAB_EXCLUDE_LEVELS = frozenset({
    "other",
    "others",
    "unknown",
    "unassigned",
    "nan",
    "na",
    "n/a",
    "unk",
    "",
})


def _is_nonspecific_classifier_level(label: object) -> bool:
    return str(label).strip().lower() in CROSSTAB_EXCLUDE_LEVELS


def _mask_nonspecific_classifier_levels(series: pd.Series) -> pd.Series:
    """Mask Other / unknown / unassigned levels before building association crosstabs."""
    return series.mask(series.map(_is_nonspecific_classifier_level))


def _drop_nonspecific_crosstab_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Drop catch-all rows/columns from a contingency table."""
    keep_rows = [idx for idx in table.index if not _is_nonspecific_classifier_level(idx)]
    keep_cols = [col for col in table.columns if not _is_nonspecific_classifier_level(col)]
    if not keep_rows or not keep_cols:
        return table.iloc[0:0, 0:0]
    out = table.loc[keep_rows, keep_cols]
    return out.loc[out.sum(axis=1) > 0, out.sum(axis=0) > 0]

# Global association barchart: require at least this Cramér's V to highlight FDR significance.
CRAMER_V_HIGHLIGHT_MIN = 0.15

# Drop classifier levels with fewer than this many patients before global chi-square tests.
MIN_CLASSIFIER_LEVEL_N = 10

# Sparse subtype pooling (Lymphgen has many low-count classes in n=61 discovery cohort).
MIN_CLASSIFIER_LEVEL_N_BY_METRIC: dict[str, int] = {
    "Wright 2020 Lymphgen": 5,
}

# Classifier panels omitted from enrichment dotplots (Fig 4C/D only; still in Fig 4B).
METRIC_PANEL_EXCLUDE: frozenset[str] = frozenset({
    "Steen 2021 Lymphoma Ecotype",
})

# Classifier panel order (top → bottom) in enrichment dotplots.
METRIC_PANEL_ORDER: list[str] = [
    HLA_CLASS_STATE_DISPLAY,
    "Kotlov 2021 LME",
    "Li 2025 LymphoMAP",
    "Ciavarella 2018 Cluster",
    "Cell of Origin",
    "Wright 2020 Lymphgen",
    "Chapuy 2025 DLBclass",
    "Lacy 2020 HMRN",
    "Shen 2023 LymphPlex",
    "Steen 2021 EcoTyper (confident)",
    "Steen 2021 EcoTyper (best match)",
    "Spatial Protein",
    "Tumor Location",
]

# Panels shown only for one stratifier (location columns vs archetype columns).
METRIC_PANEL_STRATIFIER: dict[str, str] = {
    "Spatial Protein": "location",
    "Tumor Location": "archetype",
}

# Row order within each classifier panel (keys = ``Metric`` column in enrichment tables).
_STEEN_ECOTYPE_LEVELS = [f"LE{i}" for i in range(1, 10)]

METRIC_FEATURE_ORDER: dict[str, list[str]] = {
    HLA_CLASS_STATE_DISPLAY: HLA_CLASS_STATE_ORDER,
    "Cell of Origin": ["ABC", "Intermediate", "GCB"],
    "Chapuy 2025 DLBclass": ["C1", "C2", "C3", "C4", "C5"],
    "Ciavarella 2018 Cluster": ["Cold", "Intermediate", "Hot"],
    "Kotlov 2021 LME": ["Depleted", "Inflammatory", "Mesenchymal", "GC-like"],
    "Spatial Protein": ["low immune", "cytotoxic predominant", "complex immune"],
    "Steen 2021 Lymphoma Ecotype": _STEEN_ECOTYPE_LEVELS,
    "Steen 2021 EcoTyper (confident)": _STEEN_ECOTYPE_LEVELS,
    "Steen 2021 EcoTyper (best match)": _STEEN_ECOTYPE_LEVELS,
    "Wright 2020 Lymphgen": ["MCD", "BN2", "BN2/MCD", "ST2", "EZB", "N1", "Other"],
    "Lacy 2020 HMRN": ["C1", "C2", "C3", "C4", "C5", "C6"],
    "Shen 2023 LymphPlex": ["MCD", "BN2", "EZB", "ST2", "N1", "TP53"],
    "Li 2025 LymphoMAP": ["FMAC", "TEX", "LN"],
    "Tumor Location": ["PCNS", "bone", "nodal", "testis"],
}

METRIC_FEATURE_EXCLUDE: dict[str, tuple[str, ...]] = {
    "Chapuy 2025 DLBclass": ("0",),
    "Shen 2023 LymphPlex": ("Others", "Other"),
}


def _ordered_groups(values: list, *, stratifier: str) -> list:
    order = LOCATION_GROUP_ORDER if stratifier == "location" else ARCHETYPE_GROUP_ORDER
    rank = {g: i for i, g in enumerate(order)}
    return sorted(values, key=lambda g: (rank.get(str(g), len(order)), str(g)))


def _order_metric_panels(
    metrics: list[str],
    *,
    stratifier: str,
    exclude: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Fixed top-to-bottom classifier order (shared across location and archetype dotplots)."""
    skip = set(exclude or ()) | set(METRIC_PANEL_EXCLUDE)
    present = {str(m) for m in metrics if str(m) not in skip}
    ordered: list[str] = []
    for panel in METRIC_PANEL_ORDER:
        if panel not in present:
            continue
        only = METRIC_PANEL_STRATIFIER.get(panel)
        if only is not None and only != stratifier:
            continue
        ordered.append(panel)
    for panel in sorted(present - set(ordered)):
        ordered.append(panel)
    return ordered


def _prepare_classifier_series(
    series: pd.Series,
    metric_label: str,
    *,
    strat_series: pd.Series | None = None,
    min_level_n: int | None = None,
) -> pd.Series:
    """Mask excluded / missing classifier levels before crosstabs."""
    if min_level_n is None:
        min_level_n = MIN_CLASSIFIER_LEVEL_N_BY_METRIC.get(metric_label, MIN_CLASSIFIER_LEVEL_N)
    out = series.astype(str).str.strip()
    exclude = {x.lower() for x in METRIC_FEATURE_EXCLUDE.get(metric_label, ())}
    exclude |= _SKIP_FEATURE_VALUES
    if metric_label == "Wright 2020 Lymphgen":
        exclude.add("unknown")
    out = out.mask(out.str.lower().isin(exclude))
    if strat_series is not None:
        out = out.mask(strat_series.isna())
    if min_level_n > 1 and out.notna().any():
        counts = out.value_counts()
        rare = {str(lv) for lv, n in counts.items() if n < min_level_n}
        if rare:
            out = out.mask(out.isin(rare))
    return out


def _order_metric_features(metric: str, features: list[str]) -> list[str]:
    """Biological row order within a classifier panel (case-insensitive level matching)."""
    if not features:
        return []
    canonical = list(dict.fromkeys(str(f).strip() for f in features))
    by_lower = {f.lower(): f for f in canonical}
    excluded = {x.lower() for x in METRIC_FEATURE_EXCLUDE.get(metric, ())}
    ordered: list[str] = []

    for level in METRIC_FEATURE_ORDER.get(metric, []):
        hit = by_lower.get(level.lower())
        if hit and hit not in ordered:
            ordered.append(hit)

    for feat in canonical:
        if feat.lower() in excluded:
            continue
        if feat not in ordered:
            ordered.append(feat)
    return ordered


def compute_group_enrichment_table(
    df: pd.DataFrame,
    *,
    stratifier: str,
) -> pd.DataFrame:
    """Per-feature enrichment in each location or archetype (feature vs rest, Fisher OR).

    FDR is Benjamini–Hochberg **within each classifier** (not across the full dotplot).
    """
    df = _genomic_classifier_cohort(df)
    strat_col = "Location" if stratifier == "location" else "abundance_cluster_30_label"
    metrics, display_names = _association_metrics(df, stratifier=stratifier)

    groups = _ordered_groups(df[strat_col].dropna().astype(str).unique().tolist(), stratifier=stratifier)
    rows: list[dict] = []

    for metric in metrics:
        metric_label = display_names[metric]
        metric_series = _prepare_classifier_series(
            df[metric], metric_label, strat_series=df[strat_col]
        )
        valid = metric_series.notna()
        strat_valid = df.loc[valid, strat_col].astype(str)
        for metric_val in metric_series.loc[valid].unique():
            val = str(metric_val).strip()

            has_val = metric_series.loc[valid] == val
            for group in groups:
                in_group = strat_valid == str(group)
                table = pd.crosstab(has_val, in_group)
                if table.shape != (2, 2):
                    continue
                odds_ratio, p_value = fisher_exact(table.values)
                log2_or = float(np.log2(max(odds_ratio, 1e-12)))
                rows.append(
                    {
                        "Metric": metric_label,
                        "Value": val,
                        "Feature": val,
                        "Group": str(group),
                        "Odds_Ratio": float(odds_ratio),
                        "log2_OR": log2_or,
                        "p_value": float(p_value),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["FDR"] = fdr_within_groups(out["p_value"], out["Metric"])
    return out


def _select_enrichment_features(enrichment: pd.DataFrame, *, max_rows: int = 36) -> pd.DataFrame:
    """Keep features with FDR-significant or strong |log2(OR)| enrichment in any group."""
    if enrichment.empty:
        return enrichment
    summary = (
        enrichment.groupby(["Metric", "Feature"], as_index=False)
        .agg(max_abs_log2or=("log2_OR", lambda s: float(s.abs().max())), min_fdr=("FDR", "min"))
    )
    selected = summary.loc[(summary["min_fdr"] < 0.05) | (summary["max_abs_log2or"] >= 1.0)]
    if selected.empty:
        selected = summary.nlargest(min(max_rows, len(summary)), "max_abs_log2or")
    else:
        selected = selected.sort_values("max_abs_log2or", ascending=False).head(max_rows)
    return enrichment.merge(selected[["Metric", "Feature"]], on=["Metric", "Feature"], how="inner")


# Backwards-compatible alias (older notebook / local edits).
_select_heatmap_features = _select_enrichment_features


def plot_association_enrichment_dotplot_by_classifier(
    df: pd.DataFrame,
    *,
    stratifier: str,
    out_svg: Path,
    title: str,
    show: bool = True,
    max_rows: int = 36,
    min_dot: float = 12,
    max_dot: float = 260,
    log2_or_cap: float | None = 5.0,
    grey_nonsignificant: bool = False,
    fdr_thresh: float = 0.05,
    nonsignificant_dot: float | None = None,
    x_spacing: float = 0.55,
    fig_w: float | None = None,
    row_height: float = 0.26,
    hspace: float = 0.04,
    left_frac: float = 0.20,
    right_frac: float = 0.84,
    ylabel_pad: float = 22.0,
    show_grid: bool = False,
    exclude_panels: set[str] | frozenset[str] | None = None,
) -> Path | None:
    """
    Dotplot enrichment figure with one subplot per classifier/metric.

    Dot color: log2(OR) — red = enriched in group, blue = depleted.
    Dot size: −log10(FDR), with FDR corrected within each classifier panel.

    Color scale uses the 95th percentile of |log2 OR| (minimum ±1.0), then caps at
    ``log2_or_cap`` (default ±5) when the heuristic exceeds that bound.

    If ``grey_nonsignificant`` is True, cells with FDR ≥ ``fdr_thresh`` are drawn
    as fixed-size grey placeholders instead of coloured dots.
    """

    full_enrichment = compute_group_enrichment_table(df, stratifier=stratifier)

    if full_enrichment.empty:
        print(f"No enrichment rows for {out_svg.name}; figure not written.")
        return None

    panel_exclude = set(METRIC_PANEL_EXCLUDE) | set(exclude_panels or ())
    enrichment = full_enrichment.loc[~full_enrichment["Metric"].isin(panel_exclude)].copy()

    if enrichment.empty:
        print(f"No enrichment rows after panel exclusion for {out_svg.name}; figure not written.")
        return None

    if log2_or_cap is not None:
        enrichment["log2_OR"] = enrichment["log2_OR"].clip(-log2_or_cap, log2_or_cap)

    groups = _ordered_groups(
        enrichment["Group"].dropna().unique().tolist(),
        stratifier=stratifier,
    )
    group_x = {g: i * x_spacing for i, g in enumerate(groups)}
    x_max = max(group_x.values()) if group_x else 0.0

    # --------------------------------------------------------
    # Order metrics/classifiers and features
    # --------------------------------------------------------

    metric_order = _order_metric_panels(
        enrichment["Metric"].unique().tolist(),
        stratifier=stratifier,
        exclude=panel_exclude,
    )

    metric_feature_order = {}
    height_ratios = []

    for metric in metric_order:
        feats = _order_metric_features(
            metric,
            enrichment.loc[enrichment["Metric"] == metric, "Feature"].unique().tolist(),
        )
        metric_feature_order[metric] = feats
        height_ratios.append(max(1, len(feats)))

    # --------------------------------------------------------
    # Global color and size scaling
    # --------------------------------------------------------

    dummy_dot = min_dot if nonsignificant_dot is None else nonsignificant_dot
    scale_enrichment = enrichment
    if grey_nonsignificant:
        scale_enrichment = enrichment.loc[
            enrichment["FDR"].replace([np.inf, -np.inf], np.nan) < fdr_thresh
        ]

    finite_log2or = scale_enrichment["log2_OR"].replace([np.inf, -np.inf], np.nan).dropna()

    # Symmetric heat limits: 95th |log2 OR|, floored at 1.0; cap at ±log2_or_cap when set.
    vmax = (
        max(1.0, float(np.nanpercentile(np.abs(finite_log2or), 95)))
        if len(finite_log2or)
        else 1.0
    )
    if log2_or_cap is not None:
        vmax = min(vmax, float(log2_or_cap))

    fdr_vals = scale_enrichment["FDR"].replace([np.inf, -np.inf], np.nan).dropna()
    neglog_fdr_vals = -np.log10(np.clip(fdr_vals, 1e-300, 1.0))

    size_cap = (
        max(1.5, float(np.nanpercentile(neglog_fdr_vals, 95)))
        if len(neglog_fdr_vals)
        else 1.5
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    n_metrics = len(metric_order)

    plot_w = fig_w if fig_w is not None else max(3.6, x_max + 1.6)
    fig_h = max(4.0, row_height * sum(height_ratios) + 1.2 + 0.18 * n_metrics)

    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(plot_w, fig_h),
        sharex=True,
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": hspace,
        },
    )

    if n_metrics == 1:
        axes = [axes]

    last_sc = None

    for ax, metric in zip(axes, metric_order):

        feats = metric_feature_order[metric]
        sub = enrichment.loc[enrichment["Metric"] == metric].copy()

        # Complete grid: Feature × Group
        full_index = pd.MultiIndex.from_product(
            [feats, groups],
            names=["Feature", "Group"],
        )

        sub_full = (
            sub
            .set_index(["Feature", "Group"])
            .reindex(full_index)
            .reset_index()
        )

        sub_full["Metric"] = metric

        xs, ys, cs, ss = [], [], [], []
        grey_xs, grey_ys = [], []

        for row in sub_full.itertuples(index=False):
            i = feats.index(row.Feature)
            x_pos = group_x.get(row.Group)
            if x_pos is None:
                continue

            log2_or = getattr(row, "log2_OR", np.nan)
            fdr = getattr(row, "FDR", np.nan)

            if not np.isfinite(log2_or) or not np.isfinite(fdr):
                continue

            if grey_nonsignificant and float(fdr) >= fdr_thresh:
                grey_xs.append(x_pos)
                grey_ys.append(i)
                continue

            neglog_fdr = -np.log10(np.clip(float(fdr), 1e-300, 1.0))
            s_scaled = min(neglog_fdr, size_cap) / size_cap
            dot_size = min_dot + s_scaled * (max_dot - min_dot)

            xs.append(x_pos)
            ys.append(i)
            cs.append(float(log2_or))
            ss.append(dot_size)

        if grey_xs:
            ax.scatter(
                grey_xs,
                grey_ys,
                s=dummy_dot,
                color="0.82",
                edgecolors="0.65",
                linewidths=0.45,
                alpha=0.9,
                zorder=1,
            )

        if len(xs):
            last_sc = ax.scatter(
                xs,
                ys,
                c=cs,
                s=ss,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                edgecolors="0.35",
                linewidths=0.45,
                alpha=0.95,
                zorder=2,
            )

        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)

        ax.set_ylim(len(feats) - 0.5, -0.5)
        ax.set_xlim(-0.35 * x_spacing, x_max + 0.35 * x_spacing)

        clean_metric = (
            str(metric)
            .replace(" 2018", "")
            .replace(" 2020", "")
            .replace(" 2021", "")
            .replace(" 2025", "")
            .replace("Lymphoma Ecotype", "Ecotype")
            .replace("Cell of Origin", "COO")
        )

        ax.set_ylabel(
            clean_metric,
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            color="0.35",
            labelpad=ylabel_pad,
        )

        if show_grid:
            ax.grid(axis="x", color="0.88", linewidth=0.8)
            ax.grid(axis="y", color="0.94", linewidth=0.6)
            ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xticks([group_x[g] for g in groups])
    axes[-1].set_xticklabels(
        [_format_group_label(g) for g in groups],
        fontsize=10,
    )

    axes[-1].set_xlabel(
        "Tumor location" if stratifier == "location" else "Spatial archetype",
        fontweight="bold",
        fontsize=10,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    fig.subplots_adjust(
        left=left_frac,
        right=right_frac,
        top=0.95,
        bottom=0.09,
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    if last_sc is not None:
        cax = fig.add_axes([right_frac + 0.02, 0.36, 0.016, 0.30])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label(
            "log₂ odds ratio",
            rotation=270,
            labelpad=14,
            fontsize=8,
        )

    # --------------------------------------------------------
    # Dot-size legend
    # --------------------------------------------------------

    legend_vals = [0.05, 0.01, 0.001]
    legend_scores = [-np.log10(v) for v in legend_vals]

    handles = []

    for fdr, score in zip(legend_vals, legend_scores):
        s_scaled = min(score, size_cap) / size_cap
        s = min_dot + s_scaled * (max_dot - min_dot)

        handles.append(
            axes[0].scatter(
                [],
                [],
                s=s,
                color="0.75",
                edgecolors="0.35",
                linewidths=0.45,
                label=f"FDR {fdr:g}",
            )
        )

    if grey_nonsignificant:
        handles.append(
            axes[0].scatter(
                [],
                [],
                s=dummy_dot,
                color="0.82",
                edgecolors="0.65",
                linewidths=0.45,
                label=f"FDR ≥ {fdr_thresh:g}",
            )
        )

    axes[0].legend(
        handles=handles,
        title="Dot size",
        loc="upper left",
        bbox_to_anchor=(right_frac + 0.12, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        borderaxespad=0,
    )

    out_svg.parent.mkdir(parents=True, exist_ok=True)

    _save_show_close(fig, out_svg, show=show)

    return out_svg


def _format_group_label(group: str) -> str:
    g = str(group)
    if g == "bone":
        return "Bone"
    if g == "nodal":
        return "Nodal"
    if g == "testis":
        return "Testis"
    if g == "cytotoxic predominant":
        return "Cytotoxic"
    if g == "low immune":
        return "Low immune"
    if g == "complex immune":
        return "Complex immune"
    return g
