"""Table-style Cox forest plots: monochrome markers, grouped rows, structured HR / p / FDR columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

FOREST_MARKER = "0.12"
FOREST_LINE = "0.12"
REF_MARKER = "0.35"
HEADER_TEXT = "0.08"
BODY_TEXT = "0.15"
BAND_COLOR = "0.96"
GROUP_GAP = 0.35
ROW_STEP = 1.0
HEADER_STEP = 0.55

FDR_BUCKET_THRESHOLDS = (0.001, 0.01, 0.05)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


def add_fdr_column(plot_df: pd.DataFrame, *, p_col: str = "p") -> pd.DataFrame:
    df = plot_df.copy()
    mask = (
        ~df.get("is_reference", pd.Series(False, index=df.index)).fillna(False)
    ) & df[p_col].notna()
    fdr = np.full(len(df), np.nan)
    if mask.any():
        fdr[mask.to_numpy()] = benjamini_hochberg(df.loc[mask, p_col].to_numpy())
    df["FDR"] = fdr
    return df


def add_fdr_column_by_group(
    plot_df: pd.DataFrame,
    group_col: str,
    *,
    p_col: str = "p",
) -> pd.DataFrame:
    """Benjamini–Hochberg FDR applied independently within each group."""
    df = plot_df.copy()
    df["FDR"] = np.nan
    for group_idx in df.groupby(group_col, sort=False).groups.values():
        p_vals = df.loc[group_idx, p_col]
        mask = p_vals.notna()
        if mask.any():
            df.loc[p_vals.loc[mask].index, "FDR"] = benjamini_hochberg(p_vals.loc[mask].to_numpy())
    return df


def format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}".rstrip("0").rstrip(".")
    return f"{p:.2f}"


def format_fdr(fdr: float) -> str:
    if not np.isfinite(fdr):
        return ""
    for thresh in FDR_BUCKET_THRESHOLDS:
        if fdr < thresh:
            return f"<{thresh:g}"
    return f"{fdr:.2f}"


def format_hr_ci(hr: float, lo: float, hi: float, *, is_reference: bool = False) -> str:
    if is_reference:
        return "Reference"
    if not np.isfinite(hr):
        return ""
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{hr:.2f} ({lo:.2f}-{hi:.2f})"
    return f"{hr:.2f}"


def level_only_label(label: str) -> str:
    text = str(label).strip()
    for suffix in (" (reference)", " (ref)"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


@dataclass
class _TableRow:
    kind: Literal["header", "data"]
    y: float
    group: str = ""
    label: str = ""
    n_text: str = ""
    hr: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    p: float = np.nan
    fdr: float = np.nan
    is_reference: bool = False
    is_pooled: bool = False


def build_table_layout(
    plot_df: pd.DataFrame,
    *,
    group_col: str = "factor",
    n_col: str = "n_patients",
) -> list[_TableRow]:
    """Variable header row + level rows per factor, with vertical gaps between groups."""
    rows: list[_TableRow] = []
    y = 0.0
    prev_group = None

    for _, record in plot_df.iterrows():
        group = str(record.get(group_col, ""))
        if prev_group is not None and group != prev_group:
            y += GROUP_GAP
        if group != prev_group:
            rows.append(_TableRow(kind="header", y=y, group=group, label=group))
            y += HEADER_STEP

        n_val = record.get(n_col, np.nan)
        n_text = "" if pd.isna(n_val) else str(int(n_val))
        rows.append(
            _TableRow(
                kind="data",
                y=y,
                group=group,
                label=level_only_label(record.get("label", "")),
                n_text=n_text,
                hr=float(record.get("HR", np.nan)),
                ci_lower=float(record.get("CI_lower", np.nan)),
                ci_upper=float(record.get("CI_upper", np.nan)),
                p=float(record.get("p", np.nan)),
                fdr=float(record.get("FDR", np.nan)),
                is_reference=bool(record.get("is_reference", False)),
            )
        )
        y += ROW_STEP
        prev_group = group

    return rows


def _draw_forest_markers(
    ax,
    data_rows: list[_TableRow],
    *,
    log_x: bool,
    study_marker: str = "s",
    pooled_marker: str = "D",
    study_marker_size: float = 5.0,
    pooled_marker_size: float = 8.0,
) -> None:
    for row in data_rows:
        yi = row.y
        if row.is_reference:
            ax.plot(
                1.0,
                yi,
                marker="s",
                color=REF_MARKER,
                markerfacecolor=REF_MARKER,
                markeredgecolor=REF_MARKER,
                markersize=5.5,
                linestyle="none",
                zorder=4,
            )
            continue
        if not np.isfinite(row.hr):
            ax.plot(np.nan, yi, marker="x", color=FOREST_MARKER, markersize=5.5, linestyle="none", zorder=4)
            continue
        marker = pooled_marker if row.is_pooled else study_marker
        marker_size = pooled_marker_size if row.is_pooled else study_marker_size
        line_width = 2.0 if row.is_pooled else 1.4
        ax.errorbar(
            row.hr,
            yi,
            xerr=[[row.hr - row.ci_lower], [row.ci_upper - row.hr]],
            fmt=marker,
            color=FOREST_MARKER,
            ecolor=FOREST_LINE,
            elinewidth=line_width,
            capsize=3.0 if row.is_pooled else 2.5,
            capthick=line_width,
            markersize=marker_size,
            markeredgewidth=0.0,
            zorder=5 if row.is_pooled else 4,
        )

    ax.axvline(1.0, color="0.55", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax.invert_yaxis()


def _forest_xlim(data_rows: list[_TableRow], *, log_x: bool) -> tuple[float, float]:
    finite = [
        r
        for r in data_rows
        if np.isfinite(r.ci_lower) and np.isfinite(r.ci_upper) and not r.is_reference
    ]
    if not finite:
        return (0.25, 3.0) if log_x else (0.0, 3.0)

    lo = min(r.ci_lower for r in finite)
    hi = max(r.ci_upper for r in finite)
    # Always keep the null HR=1 inside the axis so reference markers / ref line stay visible.
    lo = min(lo, 1.0)
    hi = max(hi, 1.0)
    if log_x:
        xmin = max(0.25, lo * 0.85)
        xmax = max(3.0, hi * 1.12)
        return xmin, xmax
    xmax = max(3.0, hi * 1.15)
    return 0.0, xmax


def _style_text_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(ax.get_ylim())


def plot_cox_forest_table(
    plot_df: pd.DataFrame,
    *,
    title: str | None = None,
    group_col: str = "factor",
    n_col: str = "n_patients",
    log_x: bool = False,
    show_fdr: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a publication-style forest table and return ``(fig, forest_ax)``."""
    layout = build_table_layout(plot_df, group_col=group_col, n_col=n_col)
    data_rows = [r for r in layout if r.kind == "data"]
    if not data_rows:
        raise ValueError("plot_df produced no forest rows")

    y_max = max(r.y for r in layout) + 0.6
    n_groups = plot_df[group_col].nunique() if group_col in plot_df.columns else 1
    fig_h = max(3.2, 0.34 * len(data_rows) + 0.28 * n_groups + 1.0)
    fig_w = 9.8 if show_fdr else 8.6
    if figsize is None:
        figsize = (fig_w, fig_h)

    width_ratios = [2.4, 3.8, 2.0, 0.65, 0.65] if show_fdr else [2.4, 4.0, 2.1, 0.7]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.08,
        left=0.06,
        right=0.98,
        top=0.90 if title else 0.94,
        bottom=0.10,
    )
    ax_var = fig.add_subplot(gs[0, 0])
    ax_forest = fig.add_subplot(gs[0, 1])
    ax_hr = fig.add_subplot(gs[0, 2])
    if show_fdr:
        ax_fdr = fig.add_subplot(gs[0, 3])
        ax_p = fig.add_subplot(gs[0, 4])
    else:
        ax_fdr = None
        ax_p = fig.add_subplot(gs[0, 3])

    for ax in (ax_var, ax_hr, ax_p, ax_fdr):
        if ax is not None:
            _style_text_axis(ax)

    ax_forest.set_ylim(-0.35, y_max)
    ax_forest.set_yticks([])
    ax_forest.spines["left"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["bottom"].set_linewidth(0.8)
    ax_forest.tick_params(axis="x", labelsize=8, width=0.8, length=3)
    ax_forest.set_xlabel("Hazard ratio", fontsize=9)

    xmin, xmax = _forest_xlim(data_rows, log_x=log_x)
    if log_x:
        ax_forest.set_xscale("log")
        ax_forest.set_xlim(xmin, xmax)
        ticks = np.array([0.25, 0.5, 1, 2, 4, 8])
        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
        ax_forest.set_xticks(ticks)
        ax_forest.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        ax_forest.set_xlim(xmin, xmax)

    _draw_forest_markers(ax_forest, data_rows, log_x=log_x)

    # Alternating subtle bands per variable group
    headers = [r for r in layout if r.kind == "header"]
    for i, hdr in enumerate(headers):
        group_data = [r for r in layout if r.kind == "data" and r.group == hdr.group]
        if not group_data or i % 2 != 0:
            continue
        y0 = hdr.y - 0.12
        y1 = max(r.y for r in group_data) + 0.42
        ax_forest.axhspan(y0, y1, color=BAND_COLOR, zorder=0, linewidth=0)

    header_y = -0.12
    col_headers = [
        (ax_var, "Variable", "left"),
        (ax_hr, "Hazard ratio (95% CI)", "left"),
    ]
    if show_fdr and ax_fdr is not None:
        col_headers.extend([(ax_fdr, "FDR", "center"), (ax_p, "p", "center")])
    else:
        col_headers.append((ax_p, "p", "center"))

    for ax, text, ha in col_headers:
        ax.text(0.02 if ha == "left" else 0.5, header_y, text, ha=ha, va="bottom", fontsize=9, fontweight="bold", color=HEADER_TEXT)

    for row in layout:
        yi = row.y
        if row.kind == "header":
            ax_var.text(0.02, yi, row.label, ha="left", va="center", fontsize=9, fontweight="bold", color=HEADER_TEXT)
            continue

        ax_var.text(0.10, yi, row.label, ha="left", va="center", fontsize=8.5, color=BODY_TEXT)
        ax_hr.text(
            0.02,
            yi,
            format_hr_ci(row.hr, row.ci_lower, row.ci_upper, is_reference=row.is_reference),
            ha="left",
            va="center",
            fontsize=8.5,
            color=BODY_TEXT,
        )
        if not row.is_reference:
            if show_fdr and ax_fdr is not None:
                ax_fdr.text(0.5, yi, format_fdr(row.fdr), ha="center", va="center", fontsize=8.5, color=BODY_TEXT)
            ax_p.text(0.5, yi, format_p_value(row.p), ha="center", va="center", fontsize=8.5, color=BODY_TEXT)

    for ax in (ax_var, ax_hr, ax_p, ax_fdr):
        if ax is not None:
            ax.set_ylim(ax_forest.get_ylim())

    if title:
        fig.suptitle(title, fontsize=11, y=0.98, ha="left", x=0.06)

    return fig, ax_forest


def save_cox_forest_table(
    plot_df: pd.DataFrame,
    out_stem: Path,
    *,
    title: str | None = None,
    group_col: str = "factor",
    log_x: bool = False,
    show_fdr: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """Write forest table figure (svg/png) and return the annotated ``plot_df``."""
    out_stem = Path(out_stem)
    plot_df = add_fdr_column(plot_df)
    plot_df.to_csv(out_stem.parent / f"{out_stem.name}_table.csv", index=False)

    fig, _ = plot_cox_forest_table(
        plot_df,
        title=title,
        group_col=group_col,
        log_x=log_x,
        show_fdr=show_fdr,
    )
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return plot_df


COHORT_FOREST_ROW_STEP = 1.15
COHORT_FOREST_FONT_HEADER = 14
COHORT_FOREST_FONT_BODY = 13
COHORT_FOREST_FONT_AXIS = 12
COHORT_FOREST_FONT_TITLE = 16


def pool_hr_fixed_effect(
    hrs: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
) -> dict[str, float]:
    """Inverse-variance fixed-effect pooling on log hazard ratios."""
    hrs = np.asarray(hrs, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)
    if len(hrs) == 0:
        raise ValueError("cannot pool empty HR vector")

    log_hr = np.log(hrs)
    se = (np.log(ci_upper) - np.log(ci_lower)) / (2 * 1.96)
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("invalid confidence intervals for HR pooling")

    weights = 1.0 / se**2
    pooled_log = float(np.sum(weights * log_hr) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    z = pooled_log / pooled_se
    return {
        "HR": float(np.exp(pooled_log)),
        "CI_lower": float(np.exp(pooled_log - 1.96 * pooled_se)),
        "CI_upper": float(np.exp(pooled_log + 1.96 * pooled_se)),
        "p": float(2 * (1 - norm.cdf(abs(z)))),
    }


def append_pooled_cohort_row(
    plot_df: pd.DataFrame,
    *,
    label: str = "Overall",
    label_col: str = "Dataset",
    n_col: str = "n",
    events_col: str = "Events",
) -> pd.DataFrame:
    """Append a fixed-effect pooled HR row across cohort studies."""
    pooled_mask = plot_df.get("is_pooled", pd.Series(False, index=plot_df.index)).fillna(False)
    study_df = plot_df.loc[~pooled_mask].copy()
    if study_df.empty:
        raise ValueError("no study rows available for pooling")

    pooled = pool_hr_fixed_effect(
        study_df["HR"].to_numpy(),
        study_df["CI_lower"].to_numpy(),
        study_df["CI_upper"].to_numpy(),
    )
    pooled_row = {
        label_col: label,
        n_col: int(study_df[n_col].sum()),
        events_col: int(study_df[events_col].sum()),
        **pooled,
        "is_pooled": True,
    }
    if "p" in study_df.columns:
        pooled_row["p"] = pooled["p"]
    return pd.concat([study_df, pd.DataFrame([pooled_row])], ignore_index=True)


def plot_cohort_hr_forest(
    plot_df: pd.DataFrame,
    *,
    label_col: str = "Dataset",
    n_col: str = "n",
    events_col: str = "Events",
    title: str | None = None,
    subtitle: str | None = None,
    log_x: bool = True,
    figsize: tuple[float, float] | None = None,
    font_header: float = COHORT_FOREST_FONT_HEADER,
    font_body: float = COHORT_FOREST_FONT_BODY,
    font_axis: float = COHORT_FOREST_FONT_AXIS,
    font_title: float = COHORT_FOREST_FONT_TITLE,
    font_subtitle: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Meta-analysis-style forest plot: one row per cohort with n, events, HR, and p."""
    if plot_df.empty:
        raise ValueError("plot_df is empty")
    if font_subtitle is None:
        font_subtitle = max(font_title - 3.0, 11.0)

    data_rows: list[_TableRow] = []
    y = 0.0
    for _, record in plot_df.iterrows():
        is_pooled = bool(record.get("is_pooled", False))
        if is_pooled and data_rows:
            y += GROUP_GAP + 0.15
        data_rows.append(
            _TableRow(
                kind="data",
                y=y,
                label=str(record[label_col]),
                n_text="" if pd.isna(record.get(n_col)) else str(int(record[n_col])),
                hr=float(record["HR"]),
                ci_lower=float(record["CI_lower"]),
                ci_upper=float(record["CI_upper"]),
                p=float(record["p"]),
                is_pooled=is_pooled,
            )
        )
        y += COHORT_FOREST_ROW_STEP

    y_max = max(r.y for r in data_rows) + 0.75
    fig_h = max(3.2, 0.72 * len(data_rows) + 1.4)
    if figsize is None:
        figsize = (12.0, fig_h)

    width_ratios = [1.85, 0.5, 0.58, 4.0, 2.0, 0.58]
    has_heading = bool(title or subtitle)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.08,
        left=0.05,
        right=0.98,
        top=0.82 if has_heading else 0.92,
        bottom=0.14,
    )
    ax_label = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_events = fig.add_subplot(gs[0, 2])
    ax_forest = fig.add_subplot(gs[0, 3])
    ax_hr = fig.add_subplot(gs[0, 4])
    ax_p = fig.add_subplot(gs[0, 5])
    text_axes = (ax_label, ax_n, ax_events, ax_hr, ax_p)
    all_axes = (*text_axes, ax_forest)

    for ax in text_axes:
        _style_text_axis(ax)

    ax_forest.set_ylim(-0.45, y_max)
    ax_forest.set_yticks([])
    ax_forest.spines["left"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["bottom"].set_linewidth(0.8)
    ax_forest.tick_params(axis="x", labelsize=font_axis, width=0.8, length=4)
    ax_forest.set_xlabel("Hazard ratio", fontsize=font_axis)

    xmin, xmax = _forest_xlim(data_rows, log_x=log_x)
    if log_x:
        ax_forest.set_xscale("log")
        ax_forest.set_xlim(xmin, xmax)
        ticks = np.array([0.5, 1, 2, 4, 8])
        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
        ax_forest.set_xticks(ticks)
        ax_forest.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        ax_forest.set_xlim(xmin, xmax)

    # Grey band behind the pooled Overall row (full table width).
    for row in data_rows:
        if not row.is_pooled:
            continue
        y0 = row.y - COHORT_FOREST_ROW_STEP * 0.42
        y1 = row.y + COHORT_FOREST_ROW_STEP * 0.42
        for ax in all_axes:
            ax.axhspan(y0, y1, color=BAND_COLOR, zorder=0, linewidth=0)

    _draw_forest_markers(
        ax_forest,
        data_rows,
        log_x=log_x,
        study_marker="D",
        pooled_marker="D",
        study_marker_size=6.5,
        pooled_marker_size=10.0,
    )

    header_y = -0.18
    for ax, text, ha in (
        (ax_label, "Dataset", "left"),
        (ax_n, "n", "center"),
        (ax_events, "Events", "center"),
        (ax_hr, "HR (95% CI)", "left"),
        (ax_p, "p", "center"),
    ):
        ax.text(
            0.02 if ha == "left" else 0.5,
            header_y,
            text,
            ha=ha,
            va="bottom",
            fontsize=font_header,
            fontweight="bold",
            color=HEADER_TEXT,
        )

    for row, (_, record) in zip(data_rows, plot_df.iterrows(), strict=True):
        yi = row.y
        is_pooled = row.is_pooled
        weight = "bold" if is_pooled else "normal"
        events_val = record.get(events_col, np.nan)
        events_text = "" if pd.isna(events_val) else str(int(events_val))
        ax_label.text(
            0.02, yi, row.label, ha="left", va="center", fontsize=font_body, fontweight=weight, color=BODY_TEXT
        )
        ax_n.text(0.5, yi, row.n_text, ha="center", va="center", fontsize=font_body, fontweight=weight, color=BODY_TEXT)
        ax_events.text(
            0.5, yi, events_text, ha="center", va="center", fontsize=font_body, fontweight=weight, color=BODY_TEXT
        )
        ax_hr.text(
            0.02,
            yi,
            format_hr_ci(row.hr, row.ci_lower, row.ci_upper),
            ha="left",
            va="center",
            fontsize=font_body,
            fontweight=weight,
            color=BODY_TEXT,
        )
        ax_p.text(
            0.5, yi, format_p_value(row.p), ha="center", va="center", fontsize=font_body, fontweight=weight, color=BODY_TEXT
        )

    for ax in text_axes:
        ax.set_ylim(ax_forest.get_ylim())

    if title:
        fig.text(0.05, 0.97, title, fontsize=font_title, fontweight="bold", ha="left", va="top", color=HEADER_TEXT)
    if subtitle:
        fig.text(
            0.05,
            0.915 if title else 0.97,
            subtitle,
            fontsize=font_subtitle,
            ha="left",
            va="top",
            color=BODY_TEXT,
        )

    return fig, ax_forest


def save_cohort_hr_forest(
    plot_df: pd.DataFrame,
    out_stem: Path,
    *,
    label_col: str = "Dataset",
    n_col: str = "n",
    events_col: str = "Events",
    title: str | None = None,
    subtitle: str | None = None,
    log_x: bool = True,
    include_pooled: bool = True,
    pooled_label: str = "Overall",
    show: bool = False,
    font_header: float = COHORT_FOREST_FONT_HEADER,
    font_body: float = COHORT_FOREST_FONT_BODY,
    font_axis: float = COHORT_FOREST_FONT_AXIS,
    font_title: float = COHORT_FOREST_FONT_TITLE,
    font_subtitle: float | None = None,
) -> pd.DataFrame:
    """Write cohort forest figure (svg/png) and return ``plot_df``."""
    out_stem = Path(out_stem)
    export_df = plot_df.copy()
    if include_pooled and not export_df.get("is_pooled", pd.Series(False, index=export_df.index)).fillna(False).any():
        export_df = append_pooled_cohort_row(
            export_df,
            label=pooled_label,
            label_col=label_col,
            n_col=n_col,
            events_col=events_col,
        )
    export_df.to_csv(out_stem.parent / f"{out_stem.name}_table.csv", index=False)

    fig, _ = plot_cohort_hr_forest(
        export_df,
        label_col=label_col,
        n_col=n_col,
        events_col=events_col,
        title=title,
        subtitle=subtitle,
        log_x=log_x,
        font_header=font_header,
        font_body=font_body,
        font_axis=font_axis,
        font_title=font_title,
        font_subtitle=font_subtitle,
    )
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return export_df


LOCATION_VS_DI_CONTRAST_STYLE = {
    "CP vs DI": {"color": "#4575b4", "label": "CP vs DI"},
    "LO vs DI": {"color": "#d73027", "label": "LO vs DI"},
}

VS_DI_CONTRAST_ORDER = ("CP vs DI", "LO vs DI")


@dataclass
class _LocationVsDiRow:
    kind: Literal["header", "data"]
    y: float
    location: str = ""
    label: str = ""
    hr: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    fdr: float = np.nan
    color: str = FOREST_MARKER


def _build_location_vs_di_layout(
    plot_df: pd.DataFrame,
    *,
    location_col: str = "Location",
    contrast_col: str = "contrast",
    location_order: list[str] | tuple[str, ...],
) -> list[_LocationVsDiRow]:
    rows: list[_LocationVsDiRow] = []
    y = 0.0
    for loc in location_order:
        sub = plot_df.loc[plot_df[location_col].astype(str) == loc]
        if sub.empty:
            continue
        if rows:
            y += GROUP_GAP
        rows.append(_LocationVsDiRow(kind="header", y=y, location=str(loc), label=str(loc)))
        y += HEADER_STEP
        for contrast in VS_DI_CONTRAST_ORDER:
            match = sub.loc[sub[contrast_col].astype(str) == contrast]
            if match.empty:
                continue
            record = match.iloc[0]
            style = LOCATION_VS_DI_CONTRAST_STYLE.get(contrast, {"color": FOREST_MARKER, "label": contrast})
            rows.append(
                _LocationVsDiRow(
                    kind="data",
                    y=y,
                    location=str(loc),
                    label=str(style["label"]),
                    hr=float(record["HR"]),
                    ci_lower=float(record["CI_lower"]),
                    ci_upper=float(record["CI_upper"]),
                    fdr=float(record.get("FDR", np.nan)),
                    color=str(style["color"]),
                )
            )
            y += ROW_STEP
    return rows


def plot_location_archetype_vs_di_forest(
    plot_df: pd.DataFrame,
    *,
    location_col: str = "Location",
    contrast_col: str = "contrast",
    location_order: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Minimal site-stratified forest with formal HR / FDR columns (DI reference)."""
    if plot_df.empty:
        raise ValueError("plot_df is empty")

    if location_order is None:
        location_order = list(dict.fromkeys(plot_df[location_col].astype(str)))
    else:
        location_order = [loc for loc in location_order if loc in set(plot_df[location_col].astype(str))]
    if not location_order:
        raise ValueError("no locations available for forest plot")

    layout = _build_location_vs_di_layout(
        plot_df,
        location_col=location_col,
        contrast_col=contrast_col,
        location_order=location_order,
    )
    data_rows = [r for r in layout if r.kind == "data"]
    if not data_rows:
        raise ValueError("plot_df produced no forest rows")

    y_max = max(r.y for r in layout) + 0.55
    n_groups = sum(1 for r in layout if r.kind == "header")
    if figsize is None:
        figsize = (8.8, max(3.6, 0.34 * len(data_rows) + 0.28 * n_groups + 1.0))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.35, 3.4, 1.85, 0.55],
        wspace=0.08,
        left=0.07,
        right=0.98,
        top=0.90 if title else 0.94,
        bottom=0.12,
    )
    ax_label = fig.add_subplot(gs[0, 0])
    ax_forest = fig.add_subplot(gs[0, 1])
    ax_hr = fig.add_subplot(gs[0, 2])
    ax_fdr = fig.add_subplot(gs[0, 3])
    for ax in (ax_label, ax_hr, ax_fdr):
        _style_text_axis(ax)

    ax_forest.set_ylim(-0.25, y_max)
    ax_forest.set_yticks([])
    ax_forest.spines["left"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["bottom"].set_linewidth(0.8)
    ax_forest.tick_params(axis="x", labelsize=8, width=0.8, length=3)
    ax_forest.set_xlabel("Hazard ratio", fontsize=9)

    lo = min(r.ci_lower for r in data_rows if np.isfinite(r.ci_lower))
    hi = max(r.ci_upper for r in data_rows if np.isfinite(r.ci_upper))
    xmin = max(0.25, lo * 0.85)
    xmax = max(4.0, hi * 1.12)
    ax_forest.set_xscale("log")
    ax_forest.set_xlim(xmin, xmax)
    ticks = np.array([0.5, 1, 2, 4, 8, 16])
    ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
    ax_forest.set_xticks(ticks)
    ax_forest.set_xticklabels([f"{t:g}" for t in ticks])
    ax_forest.axvline(1.0, color="0.55", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    headers = [r for r in layout if r.kind == "header"]
    for i, hdr in enumerate(headers):
        group_data = [r for r in layout if r.kind == "data" and r.location == hdr.location]
        if not group_data or i % 2 != 0:
            continue
        y0 = hdr.y - 0.12
        y1 = max(r.y for r in group_data) + 0.42
        ax_forest.axhspan(y0, y1, color=BAND_COLOR, zorder=0, linewidth=0)

    for row in data_rows:
        ax_forest.errorbar(
            row.hr,
            row.y,
            xerr=[[row.hr - row.ci_lower], [row.ci_upper - row.hr]],
            fmt="s",
            color=row.color,
            ecolor=row.color,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.2,
            markersize=5.0,
            markeredgewidth=0.0,
            zorder=4,
        )

    ax_forest.invert_yaxis()

    header_y = -0.12
    ax_label.text(0.02, header_y, "Location", ha="left", va="bottom", fontsize=9, fontweight="bold", color=HEADER_TEXT)
    ax_hr.text(0.02, header_y, "Hazard ratio (95% CI)", ha="left", va="bottom", fontsize=9, fontweight="bold", color=HEADER_TEXT)
    ax_fdr.text(0.5, header_y, "FDR", ha="center", va="bottom", fontsize=9, fontweight="bold", color=HEADER_TEXT)

    for row in layout:
        yi = row.y
        if row.kind == "header":
            ax_label.text(0.02, yi, row.label, ha="left", va="center", fontsize=9, fontweight="bold", color=HEADER_TEXT)
            continue
        ax_label.text(0.12, yi, row.label, ha="left", va="center", fontsize=8.5, color=row.color)
        ax_hr.text(
            0.02,
            yi,
            format_hr_ci(row.hr, row.ci_lower, row.ci_upper),
            ha="left",
            va="center",
            fontsize=8.5,
            color=BODY_TEXT,
        )
        ax_fdr.text(0.5, yi, format_fdr(row.fdr), ha="center", va="center", fontsize=8.5, color=BODY_TEXT)

    for ax in (ax_label, ax_hr, ax_fdr):
        ax.set_ylim(ax_forest.get_ylim())

    if title:
        fig.suptitle(title, fontsize=11, y=0.98, ha="left", x=0.07)

    return fig, ax_forest


def save_location_archetype_vs_di_forest(
    plot_df: pd.DataFrame,
    out_stem: Path,
    *,
    location_order: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    show: bool = False,
) -> pd.DataFrame:
    """Write site-stratified CP/LO vs DI forest figure (svg/png) and return plot_df."""
    out_stem = Path(out_stem)
    plot_df.to_csv(out_stem.parent / f"{out_stem.name}_table.csv", index=False)

    fig, _ = plot_location_archetype_vs_di_forest(
        plot_df,
        location_order=location_order,
        title=title,
    )
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return plot_df
