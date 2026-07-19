"""Segmentation mask overlays for spatial figure panels (Fig 3A–C, S2J)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from .phenotype_labels import COLORMAP_30_UNS_KEY, read_celltype_colormap_30


def slice_adata_for_fov(adata, fov):
    """Extract segmentation mask and spatial metadata for one FOV."""
    if "spatial" not in adata.uns:
        raise KeyError("adata.uns['spatial'] not found.")

    if fov not in adata.uns["spatial"]:
        raise KeyError(f"FOV {fov!r} not found in adata.uns['spatial'].")

    spatial_data = adata.uns["spatial"][fov]

    if "images" in spatial_data and "segmentation" in spatial_data["images"]:
        segmentation_mask = spatial_data["images"]["segmentation"]
    elif "segmentation" in spatial_data:
        segmentation_mask = spatial_data["segmentation"]
    else:
        raise KeyError(f"No segmentation mask found for FOV {fov!r}.")

    if not isinstance(segmentation_mask, np.ndarray):
        segmentation_mask = np.asarray(segmentation_mask)

    return segmentation_mask, spatial_data


def infer_segmentation_id_from_obs_index(cell_id):
    """Parse the numeric segmentation id from an obs index like ``{fov}_{seg_id}``."""
    try:
        return int(str(cell_id).split("_")[-1])
    except Exception as e:
        raise ValueError(
            f"Could not parse segmentation id from obs index {cell_id!r}. "
            "Expected something like 'T5_2_1234'."
        ) from e


def build_category_palette(obs_data, phenotype_key, color_dict, fallback_color="#BEBEBE"):
    """Build ordered categories and matching colors from obs_data and a flat dict."""
    if phenotype_key not in obs_data.columns:
        raise KeyError(f"{phenotype_key!r} not found in obs.")

    if not isinstance(obs_data[phenotype_key].dtype, pd.CategoricalDtype):
        obs_data = obs_data.copy()
        obs_data[phenotype_key] = obs_data[phenotype_key].astype("category")

    categories = list(obs_data[phenotype_key].cat.categories)
    palette_colors = [color_dict.get(cat, fallback_color) for cat in categories]

    missing = [cat for cat in categories if cat not in color_dict]
    if missing:
        print(f"Warning: missing colors for {len(missing)} categories in {phenotype_key}: {missing}")

    return obs_data, categories, palette_colors


def spatial_segment_full_mask_and_borders(
    segmentation_mask,
    obs_data,
    phenotype_key,
    color_dict,
    save_dir=None,
    fov=None,
    fallback_color="#BEBEBE",
    dpi=300,
):
    """Create colored border-only and full-cell mask overlays for one FOV."""
    obs_data, categories, palette_colors = build_category_palette(
        obs_data=obs_data,
        phenotype_key=phenotype_key,
        color_dict=color_dict,
        fallback_color=fallback_color,
    )

    cat_to_code = {cat: i + 1 for i, cat in enumerate(categories)}

    colored_mask = np.zeros(segmentation_mask.shape, dtype=np.uint16)
    colored_boundaries = np.zeros(segmentation_mask.shape, dtype=np.uint16)

    boundaries = find_boundaries(segmentation_mask, mode="inner")

    for cell_id, phenotype in obs_data[phenotype_key].items():
        if pd.isna(phenotype):
            continue

        seg_id = infer_segmentation_id_from_obs_index(cell_id)
        code = cat_to_code[phenotype]

        mask = segmentation_mask == seg_id
        if not np.any(mask):
            continue

        colored_mask[mask] = code
        colored_boundaries[boundaries & mask] = code

    masked_filled = np.ma.masked_where(colored_mask == 0, colored_mask)
    masked_borders = np.ma.masked_where(colored_boundaries == 0, colored_boundaries)

    cmap_colors = ["#00000000"] + palette_colors
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(cmap_colors) + 1) - 0.5,
        ncolors=len(cmap_colors),
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    fig_borders, ax_borders = plt.subplots(figsize=(10, 10))
    ax_borders.imshow(masked_borders, cmap=cmap, norm=norm, interpolation="nearest")
    ax_borders.axis("off")
    fig_borders.patch.set_alpha(0.0)
    ax_borders.set_facecolor((0, 0, 0, 0))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_dir is not None and fov is not None:
        base = save_dir / f"{fov}_{phenotype_key}_borders_no_axes"
        fig_borders.savefig(base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
        fig_borders.savefig(base.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.close(fig_borders)

    fig_filled, ax_filled = plt.subplots(figsize=(10, 10))
    ax_filled.imshow(masked_filled, cmap=cmap, norm=norm, interpolation="nearest")
    ax_filled.axis("off")
    fig_filled.patch.set_alpha(0.0)
    ax_filled.set_facecolor((0, 0, 0, 0))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_dir is not None and fov is not None:
        base = save_dir / f"{fov}_{phenotype_key}_full_mask_no_axes"
        fig_filled.savefig(base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
        fig_filled.savefig(base.with_suffix(".png"), format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.close(fig_filled)

    return categories, palette_colors


def run_for_all_fovs(
    adata,
    phenotype_key,
    color_map_key,
    save_dir,
    restrict_to_fovs=None,
    fallback_color="#BEBEBE",
):
    """Run segmentation overlay export for all or selected FOVs."""
    if color_map_key not in adata.uns:
        raise KeyError(f"{color_map_key!r} not found in adata.uns.")

    color_dict = (
        read_celltype_colormap_30(adata)
        if color_map_key == COLORMAP_30_UNS_KEY
        else adata.uns[color_map_key]
    )
    if not isinstance(color_dict, dict):
        raise TypeError(f"adata.uns[{color_map_key!r}] must be a flat dict of phenotype -> color.")

    fovs = pd.Index(adata.obs["fov"].astype(str).unique())
    if restrict_to_fovs is not None:
        restrict_to_fovs = set(map(str, restrict_to_fovs))
        fovs = [f for f in fovs if f in restrict_to_fovs]
    else:
        fovs = list(fovs)

    for fov in tqdm(fovs, desc="Processing FOVs"):
        try:
            segmentation_mask, _ = slice_adata_for_fov(adata, fov)

            obs_data = adata.obs.loc[adata.obs["fov"].astype(str) == str(fov), [phenotype_key]].copy()
            if obs_data.empty:
                print(f"Skipping {fov}: no obs rows found.")
                continue

            spatial_segment_full_mask_and_borders(
                segmentation_mask=segmentation_mask,
                obs_data=obs_data,
                phenotype_key=phenotype_key,
                color_dict=color_dict,
                save_dir=save_dir,
                fov=fov,
                fallback_color=fallback_color,
            )

        except Exception as e:
            print(f"Error processing FOV {fov}: {e}")
