"""Location × classifier pie patchwork grid (h5ad-backed)."""

from __future__ import annotations

from colorsys import rgb_to_hsv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .integration_figures import DONUT_COLOR_PALETTES, LOCATION_RECODE, configure_matplotlib

LOCATION_ORDER = ["PCNS", "bone", "nodal", "testis", "other"]
UNKNOWN_TOKENS = frozenset({"unknown", "unk", "na", "nan", ""})

# (metadata column, display title for column header / palette key)
MOLECULAR_COLUMN_SPECS: list[tuple[str, str]] = [
    ("abundance_cluster_30_label", "Spatial Protein"),
    ("COO_NanoString", "Cell of Origin"),
    ("Ciav_Cluster", "Ciavarella 2018\nCluster"),
    ("Lymphoma_Ecotype", "Steen 2021\nLymphoma_Ecotype"),
    ("KotlovSig", "Kotlov 2021\nLME"),
    ("lymphomap", "Li 2025\nLymphoMAP"),
    ("Lymphgen", "Wright 2020\nLymphgen"),
    ("DLBclass", "Chapuy 2025\nDLBclass"),
    ("HMRN", "Lacy 2020\nHMRN"),
    ("LymphPlex", "Shen 2023\nLymphPlex"),
]

# Subtype labels not in the shared donut palette
PALETTE_EXTRAS: dict[str, dict[str, str]] = {
    "Wright 2020\nLymphgen": {
        "MCD/ST2": "purple",
        "EZB/ST2": "darkorange",
        "N1": "#b3de69",
        "A53": "#bc80bd",
    },
    "Chapuy 2025\nDLBclass": {"C2": "#4daf4a"},
    "Shen 2023\nLymphPlex": {"Other": "lightgray"},
}

# Harmonize cross-classifier residual labels for display / legend.
LABEL_DISPLAY_ALIASES: dict[str, dict[str, str]] = {
    "Shen 2023\nLymphPlex": {"Other": "Others"},
    "Wright 2020\nLymphgen": {},
}

SUPERVISED_LEVEL_ORDER: dict[str, list[str]] = {
    "Wright 2020\nLymphgen": [
        "EZB",
        "ST2",
        "EZB/ST2",
        "Other",
        "BN2",
        "BN2/MCD",
        "MCD/ST2",
        "MCD",
    ],
}


def normalize_value(val: object) -> str:
    if pd.isna(val):
        return "unknown"
    return str(val).strip()


def is_purple_or_blue(color: str) -> bool:
    try:
        rgb = mcolors.to_rgb(color)
        h, _s, v = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        return (0.5 <= h <= 0.85) or (v < 0.4)
    except (ValueError, TypeError):
        return False


def build_palette(values: list[str]) -> dict[str, str]:
    uniq = list(values)
    n = len(uniq)
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    colors = []
    for idx, val in enumerate(uniq):
        if str(val).strip().lower() in UNKNOWN_TOKENS:
            colors.append("#f0f0f0")
        else:
            colors.append(mcolors.to_hex(cmap(idx / max(n - 1, 1))))
    return dict(zip(uniq, colors))


GENOMIC_CLASSIFIER_COLS = ("DLBclass", "Lymphgen", "HMRN", "LymphPlex")
GENOMIC_STACKED_BAR_COLS = ("Lymphgen", "DLBclass", "HMRN", "LymphPlex")
LYMPHGEN_DLBCLASS_COLS = ("Lymphgen", "DLBclass")
SUPPLEMENTARY_DROP_COLS = ("abundance_cluster", "abundance_cluster_30")
ARCHETYPE_LABEL_COL = "abundance_cluster_30_label"
ARCHETYPE_DISPLAY_COL = "tumor immune archetype (this work)"


def format_pie_patchwork_supplementary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fig 1E supplementary export: mask failed WES classifiers and tidy columns."""
    out = df.copy()
    if "patient_id" not in out.columns:
        out = out.reset_index(names="patient_id")
    out["patient_id"] = out["patient_id"].astype(str)

    if "genomic_tested" in out.columns:
        untested = ~out["genomic_tested"].fillna(False)
        for col in GENOMIC_CLASSIFIER_COLS:
            if col in out.columns:
                out.loc[untested, col] = pd.NA

    out = out.drop(columns=[c for c in SUPPLEMENTARY_DROP_COLS if c in out.columns])
    if ARCHETYPE_LABEL_COL in out.columns:
        out = out.rename(columns={ARCHETYPE_LABEL_COL: ARCHETYPE_DISPLAY_COL})

    return out


def prepare_pie_patchwork_table(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Return patient table, molecular column names, and column→display label map."""
    df = meta.copy()
    if "patient_id" in df.columns:
        df = df.set_index("patient_id")
    df.index = df.index.astype(str)

    if "Location" not in df.columns:
        raise KeyError("metadata must include Location")
    df["Location"] = df["Location"].replace(LOCATION_RECODE).map(normalize_value)

    molecular_cols: list[str] = []
    label_by_col: dict[str, str] = {}
    for col, label in MOLECULAR_COLUMN_SPECS:
        if col not in df.columns:
            continue
        if label in label_by_col.values():
            continue
        molecular_cols.append(col)
        label_by_col[col] = label
        df[col] = df[col].map(normalize_value)

    return df, molecular_cols, label_by_col


def _display_label(label: str, value: str) -> str:
    aliases = LABEL_DISPLAY_ALIASES.get(label, {})
    return aliases.get(str(value), str(value))


def _legend_entries_for_column(
    col: str,
    label: str,
    palette: dict[str, str],
    order: list[str],
    values: pd.Series,
) -> list[tuple[str, str]]:
    """Legend swatches: observed levels only, one row per display label."""
    observed = {str(v) for v in pd.unique(values) if str(v).lower() not in UNKNOWN_TOKENS and v not in {"NA"}}
    ordered = [v for v in order if v in observed]
    ordered += sorted(v for v in observed if v not in ordered)

    entries: list[tuple[str, str]] = []
    seen_display: set[str] = set()
    for val in ordered:
        display = _display_label(label, val)
        if display in seen_display:
            continue
        seen_display.add(display)
        color = palette.get(val, palette.get(display, "#cccccc"))
        entries.append((display, color))

    if label == "Spatial Protein":
        rank_order = ["cytotoxic predominant", "low immune", "complex immune"]
        rank = {name: i for i, name in enumerate(rank_order)}
        entries.sort(key=lambda x: (rank.get(str(x[0]), 999), str(x[0])))
    return entries


def _palette_for_column(col: str, label: str, values: pd.Series) -> tuple[dict[str, str], list[str]]:
    base = dict(DONUT_COLOR_PALETTES.get(label, {}))
    base.update(PALETTE_EXTRAS.get(label, {}))
    if base:
        supervised_order = SUPERVISED_LEVEL_ORDER.get(label, [])
        base_order = [k for k in base if str(k).lower() not in UNKNOWN_TOKENS and k not in {"NA"}]
        order = [k for k in supervised_order if k in base]
        order += [k for k in base_order if k not in order]
        return base, order
    built = build_palette(list(pd.unique(values)))
    return built, list(built.keys())


def plot_location_pie_patchwork(
    df: pd.DataFrame,
    molecular_cols: list[str],
    label_by_col: dict[str, str],
    *,
    out_svg: Path,
    legend_svg: Path | None = None,
    legend_png: Path | None = None,
    show: bool = False,
    cell_size: float = 2.6,
    title_fontsize: int = 18,
    legend_height_scale: float = 3.5,
    layout: str = "locations_rows",
) -> tuple[Path, Path | None]:
    """Pie grid for location × classifier composition.

    ``layout="locations_rows"`` (default): 4 rows = tumor location, columns =
    classifiers (legacy anatomy_matters wide grid).

    ``layout="classifiers_rows"``: rows = classifiers, columns = tumor location
    (transposed tall grid).
    """
    configure_matplotlib()

    locations = [loc for loc in LOCATION_ORDER if loc in set(df["Location"])]
    locations += sorted(loc for loc in pd.unique(df["Location"]) if loc not in locations)
    location_n = {loc: int((df["Location"] == loc).sum()) for loc in locations}

    col_palettes: dict[str, dict[str, str]] = {}
    col_orders: dict[str, list[str]] = {}
    for col in molecular_cols:
        palette, order = _palette_for_column(col, label_by_col[col], df[col])
        col_palettes[col] = palette
        col_orders[col] = order

    classifiers_rows = layout == "classifiers_rows"
    if layout not in {"classifiers_rows", "locations_rows"}:
        raise ValueError(f"layout must be 'classifiers_rows' or 'locations_rows', got {layout!r}")

    if classifiers_rows:
        row_items = molecular_cols
        col_items = locations
        n_rows, n_cols = len(molecular_cols), len(locations)
    else:
        row_items = locations
        col_items = molecular_cols
        n_rows, n_cols = len(locations), len(molecular_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        subplot_kw={"aspect": "equal"},
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)

    for row_idx, row_item in enumerate(row_items):
        for col_idx, col_item in enumerate(col_items):
            if classifiers_rows:
                col, location = row_item, col_item
            else:
                location, col = row_item, col_item
            ax = axes[row_idx, col_idx]
            loc_df = df[df["Location"] == location]
            counts = loc_df[col].value_counts(dropna=False)
            predefined_order = col_orders.get(col, [])
            values_in_order = [v for v in predefined_order if v in counts.index]
            values_rest = [v for v in counts.index if v not in values_in_order]
            values = values_in_order + values_rest
            counts_ordered = counts[values]
            colors = [col_palettes[col].get(val, "#cccccc") for val in values]
            total = counts_ordered.sum()
            percentages = (counts_ordered.values / total * 100).round(0) if total else np.array([])

            def autopct_func(pct: float) -> str:
                return "" if pct < 1 else f"{round(pct):.0f}%"

            _wedges, _texts, autotexts = ax.pie(
                counts_ordered.values,
                colors=colors,
                startangle=90,
                counterclock=False,
                wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
                autopct=autopct_func,
                textprops={"fontsize": 11, "color": "black", "weight": "bold"},
            )
            for pct, autotext, color in zip(percentages, autotexts, colors):
                if not autotext.get_text():
                    continue
                autotext.set_color("white" if is_purple_or_blue(color) else "black")
                autotext.set_fontsize(10 if pct < 5 else 11)

            if classifiers_rows:
                if row_idx == 0:
                    ax.set_title(
                        f"{location}\n(N={location_n[location]})",
                        fontsize=title_fontsize,
                    )
                if col_idx == 0:
                    ax.annotate(
                        label_by_col[col],
                        xy=(-1.25, 0),
                        xycoords="data",
                        ha="right",
                        va="center",
                        fontsize=title_fontsize,
                        fontweight="bold",
                    )
            else:
                if row_idx == 0:
                    ax.set_title(label_by_col[col], fontsize=title_fontsize)
                if col_idx == 0:
                    ax.annotate(
                        f"{location}\n(N={location_n[location]})",
                        xy=(-1.25, 0),
                        xycoords="data",
                        ha="right",
                        va="center",
                        fontsize=title_fontsize,
                        fontweight="bold",
                    )
            ax.set_axis_off()

    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    legend_path = None
    if legend_svg is not None or legend_png is not None:
        legend_sections: list[tuple[str, list[tuple[str, str]]]] = []
        for col in molecular_cols:
            label = label_by_col[col]
            entries = _legend_entries_for_column(
                col,
                label,
                col_palettes.get(col, {}),
                col_orders.get(col, []),
                df[col],
            )
            legend_sections.append((label, entries))

        n_lines = sum(1 + len(entries) for _label, entries in legend_sections)
        line_step = 0.92 / max(n_lines, 1)
        legend_h = 10.0 * legend_height_scale
        legend_fig, legend_ax = plt.subplots(figsize=(6, legend_h))
        y = 0.98
        for label, entries in legend_sections:
            legend_ax.text(0.01, y, label, fontsize=title_fontsize, fontweight="bold", va="top")
            y -= line_step
            for k, v in entries:
                legend_ax.add_patch(
                    plt.Rectangle(
                        (0.03, y - line_step * 0.35),
                        0.025,
                        min(0.025, line_step * 0.8),
                        color=v,
                        transform=legend_ax.transAxes,
                        clip_on=False,
                    )
                )
                legend_ax.text(
                    0.06,
                    y,
                    str(k),
                    fontsize=title_fontsize - 2,
                    va="center",
                    ha="left",
                    transform=legend_ax.transAxes,
                )
                y -= line_step
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis("off")
        if legend_svg is not None:
            legend_svg.parent.mkdir(parents=True, exist_ok=True)
            legend_fig.savefig(legend_svg, format="svg", bbox_inches="tight")
            legend_path = legend_svg
        if legend_png is not None:
            legend_png.parent.mkdir(parents=True, exist_ok=True)
            legend_fig.savefig(legend_png, dpi=300, bbox_inches="tight")
        plt.close(legend_fig)

    return out_svg, legend_path


def plot_location_stacked_bar_patchwork(
    df: pd.DataFrame,
    molecular_cols: list[str],
    label_by_col: dict[str, str],
    *,
    out_svg: Path,
    out_png: Path | None = None,
    show: bool = False,
    location_order: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    row_height: float = 1.7,
    width: float = 13.5,
    title: str | None = None,
) -> Path:
    """Row-normalized horizontal stacked bars for location × classifier composition.

    Rows are classifiers, y-axis bars are tumor locations, and each horizontal
    bar sums to one within the corresponding location.
    """
    configure_matplotlib()

    cols = [col for col in molecular_cols if col in df.columns]
    if not cols:
        raise ValueError("No requested molecular columns are available for stacked bars")
    locations = [loc for loc in location_order if loc in set(df["Location"])]
    locations += sorted(loc for loc in pd.unique(df["Location"]) if loc not in locations)
    if not locations:
        raise ValueError("No locations available for stacked bars")

    col_palettes: dict[str, dict[str, str]] = {}
    col_orders: dict[str, list[str]] = {}
    for col in cols:
        palette, order = _palette_for_column(col, label_by_col[col], df[col])
        col_palettes[col] = palette
        col_orders[col] = order

    fig, axes = plt.subplots(
        nrows=len(cols),
        ncols=1,
        figsize=(width, max(2.4, row_height * len(cols))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    y = np.arange(len(locations))
    location_n = {loc: int((df["Location"] == loc).sum()) for loc in locations}

    for ax, col in zip(axes, cols, strict=True):
        label = label_by_col[col]
        palette = col_palettes[col]
        observed_values: list[str] = []
        for location in locations:
            loc_values = df.loc[df["Location"].eq(location), col].map(normalize_value)
            observed_values.extend([str(v) for v in pd.unique(loc_values)])
        observed = [v for v in col_orders[col] if v in set(observed_values)]
        observed += sorted(v for v in set(observed_values) if v not in observed)
        lefts = np.zeros(len(locations), dtype=float)
        for value in observed:
            props = []
            counts = []
            for location in locations:
                loc_values = df.loc[df["Location"].eq(location), col].map(normalize_value)
                denom = int(len(loc_values))
                count = int(loc_values.eq(value).sum())
                props.append(count / denom if denom else 0.0)
                counts.append(count)
            color = palette.get(value, "#cccccc")
            bars = ax.barh(
                y,
                props,
                left=lefts,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, prop, count in zip(bars, props, counts, strict=True):
                if prop < 0.10 or count == 0:
                    continue
                text_color = "white" if is_purple_or_blue(color) else "black"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prop:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            lefts += np.asarray(props, dtype=float)

        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{loc} (N={location_n[loc]})" for loc in locations], fontsize=9)
        ax.invert_yaxis()
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=82)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[-1].set_xticklabels(["0", "25", "50", "75", "100"], fontsize=9)
    axes[-1].set_xlabel("Fraction within location (%)")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_svg
