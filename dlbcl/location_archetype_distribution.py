"""Location × immune-archetype distribution plots (discovery S2C / validation analogue)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from .integration_figures import LOCATION_GROUP_ORDER, configure_matplotlib
from .validation_figures import (
    ARCHETYPE_NAME_MAP,
    CLASS_COLORS,
    CLUSTER_ORDER,
    DISEASE_TYPE_TO_LOCATION,
    SHORT_LABELS,
)

LOCATION_ORDER = list(LOCATION_GROUP_ORDER)


@dataclass(frozen=True)
class LocationArchetypeAssociation:
    counts: pd.DataFrame
    row_props: pd.DataFrame
    expected: pd.DataFrame
    residuals: pd.DataFrame
    annot_prop: pd.DataFrame
    chi2: float
    p_value: float
    dof: int
    cramers_v: float
    low_expected_cells: int
    enrichment_summary: list[str]


def build_location_archetype_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Location + predicted archetype labels from validation ``pred`` table."""
    df = pd.DataFrame(index=pred.index)
    df["Location"] = pred["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    archetype_id = pd.to_numeric(pred["pred_abundance_cluster_30"], errors="coerce")
    df["Archetype"] = archetype_id.map(ARCHETYPE_NAME_MAP)
    return df.dropna(subset=["Location", "Archetype"]).copy()


def compute_location_archetype_association(
    df: pd.DataFrame,
    *,
    location_col: str = "Location",
    archetype_col: str = "Archetype",
    location_order: list[str] | None = None,
    archetype_order: list[str] | None = None,
    residual_threshold: float = 1.0,
) -> LocationArchetypeAssociation:
    """Crosstab, chi-square, and standardized residuals (nb4 S2C logic)."""
    loc_order = list(location_order) if location_order is not None else list(LOCATION_ORDER)
    arch_order = list(archetype_order) if archetype_order is not None else list(CLUSTER_ORDER)

    plot_df = df[[location_col, archetype_col]].dropna().copy()
    plot_df[location_col] = pd.Categorical(plot_df[location_col], categories=loc_order, ordered=True)
    plot_df[archetype_col] = pd.Categorical(plot_df[archetype_col], categories=arch_order, ordered=True)

    ct = pd.crosstab(plot_df[location_col], plot_df[archetype_col], dropna=False)
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]

    row_props = ct.div(ct.sum(axis=1), axis=0)
    annot_prop = ct.copy().astype(str)
    for loc in ct.index:
        for arch in ct.columns:
            annot_prop.loc[loc, arch] = f"{ct.loc[loc, arch]}\n({row_props.loc[loc, arch] * 100:.0f}%)"

    chi2, p_value, dof, expected = chi2_contingency(ct)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    n = ct.values.sum()
    r, k = ct.shape
    cramers_v = float(np.sqrt(chi2 / (n * min(r - 1, k - 1)))) if n and min(r - 1, k - 1) else float("nan")
    low_expected_cells = int((expected_df < 5).sum().sum())

    residuals = (ct - expected_df) / np.sqrt(expected_df)
    hits: list[str] = []
    for loc in residuals.index:
        for arch in residuals.columns:
            val = float(residuals.loc[loc, arch])
            if val >= residual_threshold:
                hits.append(f"Enriched: {loc} × {arch} (residual={val:.2f})")
            elif val <= -residual_threshold:
                hits.append(f"Depleted: {loc} × {arch} (residual={val:.2f})")

    return LocationArchetypeAssociation(
        counts=ct,
        row_props=row_props,
        expected=expected_df,
        residuals=residuals,
        annot_prop=annot_prop,
        chi2=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        cramers_v=cramers_v,
        low_expected_cells=low_expected_cells,
        enrichment_summary=hits,
    )


def association_stats_table(assoc: LocationArchetypeAssociation) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chi2": assoc.chi2,
                "p_value": assoc.p_value,
                "dof": assoc.dof,
                "cramers_v": assoc.cramers_v,
                "low_expected_cells_lt5": assoc.low_expected_cells,
            }
        ]
    )


def plot_location_archetype_heatmaps(
    assoc: LocationArchetypeAssociation,
    *,
    out_path: Path | str | None = None,
    title: str = "Validation — Location × predicted immune archetype",
    show: bool = True,
) -> plt.Figure:
    """Three-panel figure: row-%-colored counts, row-%, standardized residuals."""
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    annot_counts = assoc.counts.astype(int).astype(str)
    hm0 = sns.heatmap(
        assoc.row_props,
        annot=annot_counts,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=True,
        ax=axes[0],
    )
    cbar0 = hm0.collections[0].colorbar
    cbar0.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    cbar0.set_label("Row %", rotation=270, labelpad=15)
    axes[0].set_title("Counts\n(color = row %; annotation = n)")
    axes[0].set_xlabel("Immune archetype")
    axes[0].set_ylabel("Location")

    hm = sns.heatmap(
        assoc.row_props,
        annot=assoc.annot_prop,
        fmt="",
        cmap="magma",
        vmin=0,
        vmax=0.8,
        cbar=True,
        ax=axes[1],
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    cbar.set_label("Row %", rotation=270, labelpad=15)
    cbar.set_ticks(np.linspace(0, 0.8, 5))
    axes[1].set_title("Row-normalized proportions\n(annotation = count + row %)")
    axes[1].set_xlabel("Immune archetype")
    axes[1].set_ylabel("Location")

    v = float(np.nanmax(np.abs(assoc.residuals.values)))
    sns.heatmap(
        assoc.residuals,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-v,
        vmax=v,
        cbar=True,
        ax=axes[2],
    )
    axes[2].set_title("Standardized residuals")
    axes[2].set_xlabel("Immune archetype")
    axes[2].set_ylabel("Location")

    fig.suptitle(
        f"{title}\n"
        f"Chi2={assoc.chi2:.2f}, p={assoc.p_value:.2e}, Cramer's V={assoc.cramers_v:.2f}",
        y=1.04,
        fontsize=14,
    )
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix.lstrip(".") or "svg", bbox_inches="tight")
        if path.suffix.lower() != ".png":
            fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_archetype_pies_by_location(
    df: pd.DataFrame,
    *,
    out_path: Path | str | None = None,
    location_order: list[str] | None = None,
    archetype_order: list[str] | None = None,
    archetype_colors: dict[str, str] | None = None,
    show: bool = True,
) -> plt.Figure:
    """Archetype composition within each anatomical location (2×2 grid)."""
    configure_matplotlib()
    loc_order = list(location_order) if location_order is not None else list(LOCATION_ORDER)
    arch_order = list(archetype_order) if archetype_order is not None else list(CLUSTER_ORDER)
    colors = dict(archetype_colors) if archetype_colors is not None else dict(CLASS_COLORS)

    active_locs = [loc for loc in loc_order if (df["Location"] == loc).any()]
    n = len(active_locs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, loc in zip(axes_flat, active_locs, strict=False):
        sub = df[df["Location"] == loc]
        counts = (
            sub["Archetype"]
            .value_counts()
            .reindex(arch_order, fill_value=0)
        )
        counts = counts[counts > 0]
        ax.pie(
            counts.values,
            labels=[f"{SHORT_LABELS.get(a, a)}\n(n={v})" for a, v in counts.items()],
            colors=[colors[a] for a in counts.index],
            autopct="%1.0f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        ax.set_title(f"{loc}\nTotal n={int(counts.sum())}", fontsize=11)
        ax.axis("equal")

    for ax in axes_flat[len(active_locs):]:
        ax.axis("off")

    fig.suptitle("Predicted immune archetype composition by location", fontsize=13, y=1.02)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix.lstrip(".") or "svg", bbox_inches="tight", dpi=300)
        if path.suffix.lower() != ".png":
            fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_location_archetype_distribution(
    pred: pd.DataFrame,
    out_dir: Path | str,
    *,
    prefix: str = "val",
    show: bool = False,
) -> dict[str, Path]:
    """Write heatmaps, pies, and tabular exports for validation cohort."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_location_archetype_frame(pred)
    assoc = compute_location_archetype_association(df)

    heatmap_path = out / f"{prefix}_location_archetype_heatmaps.svg"
    pies_path = out / f"{prefix}_location_archetype_pies.svg"
    plot_location_archetype_heatmaps(assoc, out_path=heatmap_path, show=show)
    plot_archetype_pies_by_location(df, out_path=pies_path, show=show)

    counts_path = out / f"{prefix}_location_archetype_counts.csv"
    props_path = out / f"{prefix}_location_archetype_row_props.csv"
    resid_path = out / f"{prefix}_location_archetype_residuals.csv"
    stats_path = out / f"{prefix}_location_archetype_chi2_stats.csv"
    assoc.counts.to_csv(counts_path)
    assoc.row_props.to_csv(props_path)
    assoc.residuals.to_csv(resid_path)
    association_stats_table(assoc).to_csv(stats_path, index=False)

    return {
        "heatmap": heatmap_path,
        "pies": pies_path,
        "counts": counts_path,
        "row_props": props_path,
        "residuals": resid_path,
        "chi2_stats": stats_path,
    }
