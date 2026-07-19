"""Helpers for immune-archetype phenotype-density heatmaps (notebook 4 / Fig 2)."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.stats import zscore


def filter_phenotypes(df: pd.DataFrame, filter_terms=None) -> pd.DataFrame:
    if filter_terms:
        pattern = "|".join(filter_terms)
        df = df.loc[~df.index.str.contains(pattern, case=False, na=False)]
    return df


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(
        lambda row: zscore(row, nan_policy="omit") if row.nunique() > 1 else np.zeros(len(row)),
        axis=1,
        result_type="broadcast",
    )
    out.columns = df.columns
    out.index = df.index
    return out


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def make_continuous_color_series(series, cmap="viridis", missing_color="#d9d9d9"):
    s_num = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)

    valid = s_num.notna()
    if valid.sum() == 0:
        out[:] = missing_color
        return out

    vmin = s_num[valid].min()
    vmax = s_num[valid].max()

    if vmin == vmax:
        normed = pd.Series(0.5, index=series.index, dtype=float)
        normed.loc[~valid] = np.nan
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        normed = s_num.map(norm)

    cmap_obj = plt.get_cmap(cmap)
    out.loc[valid] = normed.loc[valid].map(lambda x: mcolors.to_hex(cmap_obj(x)))
    out.loc[~valid] = missing_color
    return out


def make_categorical_color_series(
    series,
    palette="tab20",
    missing_color="#d9d9d9",
    level_order=None,
    overrides=None,
):
    s = series.astype("object").copy()
    out = pd.Series(index=s.index, dtype=object)

    non_missing = s.dropna()
    if len(non_missing) == 0:
        out[:] = missing_color
        return out, {}

    if level_order is None:
        levels = list(pd.unique(non_missing))
    else:
        levels = [x for x in level_order if x in set(non_missing)]
        extra = [x for x in pd.unique(non_missing) if x not in levels]
        levels = levels + extra

    colors = sns.color_palette(palette, n_colors=max(len(levels), 1))
    color_map = {lev: rgb_to_hex(colors[i]) for i, lev in enumerate(levels)}

    if overrides is not None:
        color_map.update(overrides)

    out[:] = missing_color
    for lev, color in color_map.items():
        out.loc[s == lev] = color

    return out, color_map


def infer_annotation_type(
    series,
    force_continuous=None,
    force_categorical=None,
    max_unique_for_categorical=8,
):
    name = series.name

    if force_continuous and name in force_continuous:
        return "continuous"
    if force_categorical and name in force_categorical:
        return "categorical"

    s_num = pd.to_numeric(series, errors="coerce")
    numeric_fraction = s_num.notna().mean()

    if numeric_fraction > 0.9:
        n_unique = s_num.dropna().nunique()
        if n_unique <= max_unique_for_categorical:
            return "categorical"
        return "continuous"

    return "categorical"
