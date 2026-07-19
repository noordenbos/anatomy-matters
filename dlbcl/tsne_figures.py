"""tSNE panel helpers for genomic classifier figures (notebook 3)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .dlbcl_io import rel_path

TSNE_BASIS = "tsne_all"
PLOT_SEED = 42
DOWNSAMPLE_FRAC = 1.0

# Set by ``tools/run_all_notebooks.py`` (default on) to skip huge vector tSNE SVGs.
# Override for internal vector exports: ``DLBCL_TSNE_WRITE_VECTOR=1``.
_TSNE_RASTER_ONLY_ENV = "DLBCL_TSNE_RASTER_ONLY"
_TSNE_WRITE_VECTOR_ENV = "DLBCL_TSNE_WRITE_VECTOR"


def tsne_raster_only() -> bool:
    """True when only rasterized tSNE exports should be written."""
    if os.environ.get(_TSNE_WRITE_VECTOR_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get(_TSNE_RASTER_ONLY_ENV, "").strip().lower() in {"1", "true", "yes"}

CLASSIFIER_PALETTES = {
    "Lymphgen": ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#8c564b", "lightgrey"],
    "DLBclass": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
    "HMRN": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"],
    "LymphPlex": ["#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c", "#b3de69", "#d62728", "lightgrey"],
}


def configure_tsne_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "figure.figsize": (6, 5),
        }
    )


def show_inline_preview(
    fig,
    *,
    dpi: int = 96,
    max_width_px: int = 900,
    jpeg_quality: int = 72,
) -> None:
    """Notebook-only preview: same figure, lower-res JPEG (not used for savefig)."""
    from io import BytesIO

    from IPython.display import Image, display
    from PIL import Image as PILImage

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        with PILImage.open(buf) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w > max_width_px:
                scale = max_width_px / w
                img = img.resize(
                    (max_width_px, max(1, int(h * scale))),
                    PILImage.Resampling.LANCZOS,
                )
            out = BytesIO()
            img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
            jpeg_bytes = out.getvalue()
    finally:
        buf.close()

    display(Image(data=jpeg_bytes, format="jpeg"))


def save_figure(fig, stem: Path, *, repo_root: Path | None = None, preview: bool = False) -> None:
    """Save tSNE figures.

    Default / interactive: vector SVG/PDF plus ``*_raster`` SVG/PDF/PNG.
    When ``DLBCL_TSNE_RASTER_ONLY=1`` (set by ``run_all_notebooks.py``): write only
    rasterized point layers to the main stem (svg/pdf/png) — much smaller/faster.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    if tsne_raster_only():
        _rasterize_scatter_layers(fig)
        fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), format="png", bbox_inches="tight")
        print(rel_path(stem.with_suffix(".svg"), repo_root))
    else:
        fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        print(rel_path(stem.with_suffix(".svg"), repo_root))

        _rasterize_scatter_layers(fig)
        raster_stem = stem.parent / f"{stem.name}_raster"
        fig.savefig(raster_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(raster_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(raster_stem.with_suffix(".png"), format="png", bbox_inches="tight")
        print(rel_path(raster_stem.with_suffix(".svg"), repo_root))

    if preview:
        show_inline_preview(fig)


def _rasterize_scatter_layers(fig) -> None:
    """Embed point layers as bitmap in SVG/PDF (Illustrator-friendly)."""
    for ax in fig.axes:
        for artist in ax.collections:
            artist.set_rasterized(True)


def sort_adata_by_obs_category(adata, col: str):
    """Stable row order for categorical obs columns (pandas 2.x safe)."""
    clusters = adata.obs[col].value_counts().index.tolist()
    codes = pd.Categorical(adata.obs[col], categories=clusters, ordered=True).codes
    return adata[np.argsort(codes, kind="stable")].copy()


def downsample_indices(n_obs: int, frac: float = DOWNSAMPLE_FRAC, seed: int = PLOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_plot = max(1, int(round(n_obs * frac)))
    return rng.choice(n_obs, size=n_plot, replace=False)


def valid_label_mask(series: pd.Series, extra_exclude=()) -> pd.Series:
    s = series.astype(str)
    mask = series.notna() & (s != "")
    for bad in ("NA", "unknown", "0", *extra_exclude):
        mask &= ~s.str.contains(rf"\b{bad}\b", case=False, na=False)
    return mask


def map_patient_column(adata, uns_key: str, obs_col: str):
    cc = adata.uns["case_classifications"]
    if isinstance(cc, pd.DataFrame):
        if uns_key in cc.columns:
            mapping = cc[uns_key]
        else:
            raise KeyError(f"{uns_key} not in case_classifications")
        if mapping.index.name != "patient_id" and "patient_id" not in cc.columns:
            mapping = mapping.copy()
            mapping.index = mapping.index.astype(str)
        else:
            if "patient_id" in cc.columns:
                mapping = cc.set_index("patient_id")[uns_key]
            mapping.index = mapping.index.astype(str)
    else:
        mapping = cc[uns_key]
    adata.obs[obs_col] = adata.obs["patient_id"].astype(str).map(mapping.to_dict())
    return adata


def plot_tsne(
    adata_plot,
    color: str,
    title: str,
    out_stem: Path,
    *,
    palette=None,
    sort_order=None,
    alpha: float = 0.7,
    size: float = 1,
    repo_root: Path | None = None,
    preview: bool = False,
):
    import gc

    kwargs = dict(
        basis=TSNE_BASIS,
        color=color,
        title=title,
        show=False,
        alpha=alpha,
        size=size,
        return_fig=True,
    )
    if palette is not None:
        kwargs["palette"] = palette
    if sort_order is not None:
        kwargs["sort_order"] = sort_order
    fig = sc.pl.embedding(adata_plot, **kwargs)
    save_figure(fig, out_stem, repo_root=repo_root, preview=preview)
    plt.close(fig)
    gc.collect()


def plot_tsne_panel(
    adata_subset,
    *,
    color: str,
    title: str,
    out_stem: Path,
    palette=None,
    repo_root: Path | None = None,
    preview: bool = True,
) -> None:
    """Convenience wrapper used by compiled Fig 2 tSNE cells."""
    adata_sorted = sort_adata_by_obs_category(adata_subset, color)
    adata_plot = adata_sorted[downsample_indices(adata_sorted.n_obs)]
    plot_tsne(
        adata_plot,
        color=color,
        title=title,
        out_stem=out_stem,
        palette=palette,
        repo_root=repo_root,
        preview=preview,
    )


def plot_classifier_tsne_panels(
    adata_subset,
    fig_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Fig 2C/2D and supp HMRN/LymphPlex tSNE panels."""
    panels = [
        ("fig2C_tsne_lymphgen", "Lymphgen", "Lymphgen", "tSNE — LymphGen"),
        ("fig2D_tsne_dlbclass", "DLBclass", "DLBclass", "tSNE — DLBclass"),
    ]
    for stem, uns_key, obs_col, title in panels:
        _plot_classifier_panel(adata_subset, fig_dir, stem, uns_key, obs_col, title, repo_root=repo_root)


def plot_s3_supplementary_classifier_tsne_panels(
    adata_subset,
    fig_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Supplemental Fig S3 — HMRN and LymphPlex classifier tSNE (not main-text panels)."""
    panels = [
        ("figS3I_tsne_hmrn", "HMRN", "HMRN", "tSNE — HMRN"),
        ("figS3J_tsne_lymphplex", "LymphPlex", "LymphPlex", "tSNE — LymphPlex"),
    ]
    for stem, uns_key, obs_col, title in panels:
        _plot_classifier_panel(adata_subset, fig_dir, stem, uns_key, obs_col, title, repo_root=repo_root)


def _plot_classifier_panel(
    adata_subset,
    fig_dir: Path,
    stem: str,
    uns_key: str,
    obs_col: str,
    title: str,
    *,
    repo_root: Path | None = None,
) -> None:
    map_patient_column(adata_subset, uns_key, obs_col)
    adata_filt = adata_subset[valid_label_mask(adata_subset.obs[obs_col])].copy()
    adata_plot = adata_filt[downsample_indices(adata_filt.n_obs)]
    rng = np.random.default_rng(PLOT_SEED)
    sort_order = rng.permutation(adata_plot.n_obs).tolist()
    plot_tsne(
        adata_plot,
        color=obs_col,
        title=title,
        out_stem=fig_dir / stem,
        palette=CLASSIFIER_PALETTES.get(obs_col),
        sort_order=sort_order,
        repo_root=repo_root,
    )
