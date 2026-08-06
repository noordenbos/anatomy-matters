"""Shared paths and data loaders for DLBCL reproduction notebooks."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

ADATA_BASENAME = "DLBCL_location_2026.h5ad"
LEGACY_ADATA_BASENAMES = ("adata3.h5ad",)

DISCOVERY_PATIENTS = [
    "T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9", "T10", "T12",
    "T13", "T14", "T15", "T16", "T17", "T18", "T19", "T20",
    "T21", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
    "T31", "T32", "T33", "T34", "T36", "T37", "T38", "T39", "T40",
    "T41", "T42", "T43", "T44", "T45", "T46", "T48", "T49", "T50",
    "T51", "T52", "T53", "T54", "T55", "T56", "T58", "T59", "T60",
    "T61", "T62", "T63", "T64", "T65", "T66", "T67", "T68", "T69", "T70",
]

# Reference tissues and withheld IDs excluded from immune-archetype discovery (Fig 2E/F).
ARCHETYPE_DISCOVERY_EXCLUDE_PATIENT_IDS = frozenset({
    "Tbrain",
    "Tspleen",
    "T5",
    "T35",
    "Tonsil394",
    "Tonsil409",
    "Tonsil447",
    "Tonsil470",
    "Tonsil481",
})

# Numeric tumor–immune archetype ids (1–3) → display labels.
CLUSTER_NAME_MAP = {
    1: "low immune",
    2: "cytotoxic predominant",
    3: "complex immune",
}

TUMORIMMUNE_ARCHETYPE_ID_COL = "tumorimmune_archetype_id"
TUMORIMMUNE_ARCHETYPE_COL = "tumorimmune_archetype"
TUMORIMMUNE_ARCHETYPE_UNS_KEY = "tumorimmune_archetype_id"
PRED_TUMORIMMUNE_ARCHETYPE_ID_COL = "pred_tumorimmune_archetype_id"
PRED_TUMORIMMUNE_ARCHETYPE_COL = "pred_tumorimmune_archetype"

# Legacy names accepted when reading older AnnData / CSV sidecars.
_LEGACY_ARCHETYPE_ID_COLS = ("tumorimmune_archetype_id", "abundance_cluster_30")
_LEGACY_ARCHETYPE_LABEL_COLS = ("tumorimmune_archetype", "abundance_cluster_30_label")


def is_archetype_discovery_excluded(patient_id: str) -> bool:
    """True for reference tissues (Tonsil*, Tspleen, Tbrain) and withheld IDs."""
    pid = str(patient_id).strip()
    if pid in ARCHETYPE_DISCOVERY_EXCLUDE_PATIENT_IDS:
        return True
    return pid.startswith("Tonsil")


@dataclass(frozen=True)
class NotebookPaths:
    repo_root: Path
    data_dir: Path
    processed_dir: Path
    fig_dir: Path
    adata_path: Path
    arch_path: Path


def repo_root_from_notebook_cwd() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def resolve_adata_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / ADATA_BASENAME,
        data_dir / "processed" / ADATA_BASENAME,
        *(data_dir / name for name in LEGACY_ADATA_BASENAMES),
        *(data_dir / "processed" / name for name in LEGACY_ADATA_BASENAMES),
    ]
    for path in candidates:
        if path.exists():
            return path
    return data_dir / ADATA_BASENAME


def get_paths(repo_root: Path | None = None, *, notebook: str) -> NotebookPaths:
    root = repo_root or repo_root_from_notebook_cwd()
    data_dir = Path(os.environ.get("DLBCL_DATA_DIR", root / "data"))
    processed = data_dir / "processed"
    return NotebookPaths(
        repo_root=root,
        data_dir=data_dir,
        processed_dir=processed,
        fig_dir=root / "figures" / notebook,
        adata_path=resolve_adata_path(data_dir),
        arch_path=processed / "tumorimmune_archetype_assignments.csv",
    )


def configure_notebook(repo_root: Path | None, notebook: str) -> NotebookPaths:
    """Resolve paths, create output dirs, and verify the AnnData bundle exists."""
    paths = get_paths(repo_root, notebook=notebook)
    paths.fig_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    # Legacy maintainer notebooks (nb1–nb17) still use notebook{N} folder names.
    if notebook == "notebook4":
        (paths.fig_dir / "cox_phenotype").mkdir(parents=True, exist_ok=True)
    if notebook == "notebook9":
        (paths.fig_dir / "cox_clinical").mkdir(parents=True, exist_ok=True)
    if notebook == "notebook11":
        (paths.fig_dir / "km").mkdir(parents=True, exist_ok=True)
    if notebook == "notebook12":
        (paths.fig_dir / "integration").mkdir(parents=True, exist_ok=True)
    if notebook == "notebook14":
        (paths.fig_dir / "cox_validation").mkdir(parents=True, exist_ok=True)
    if not paths.adata_path.exists():
        raise FileNotFoundError(
            f"AnnData bundle not found at {paths.adata_path}.\n"
            f"Download data/{ADATA_BASENAME} — see README.md (Data access)."
        )
    return paths


def load_adata(path: Path | None = None, *, paths: NotebookPaths | None = None) -> ad.AnnData:
    h5 = path or (paths.adata_path if paths else resolve_adata_path(repo_root_from_notebook_cwd() / "data"))
    if not h5.exists():
        raise FileNotFoundError(f"AnnData not found: {h5}")
    return sc.read_h5ad(h5)


def _patient_metadata_frame(uns: dict) -> pd.DataFrame:
    if "case_classifications" in uns:
        cc = pd.DataFrame(uns["case_classifications"]).copy()
        if "patient_id" in cc.columns:
            cc = cc.set_index("patient_id")
        meta = cc
        if "case_path" in uns:
            cp = pd.DataFrame(uns["case_path"]).copy()
            if "patient_id" in cp.columns:
                cp = cp.set_index("patient_id")
            meta = meta.join(cp, how="left", rsuffix="_path")
        if "case_clinical" in uns:
            cl = pd.DataFrame(uns["case_clinical"]).copy()
            if "patient_id" in cl.columns:
                cl = cl.set_index("patient_id")
            meta = meta.join(cl, how="left", rsuffix="_clinical")
        return meta.reset_index()
    if "patient_metadata" in uns:
        meta = pd.DataFrame(uns["patient_metadata"]).copy()
        if meta.index.name == "patient_id":
            meta = meta.reset_index()
        return meta
    raise KeyError(
        "Expected adata.uns['case_classifications'] (and case_path/case_clinical) "
        "or legacy adata.uns['patient_metadata']."
    )


def load_discovery_metadata(adata: ad.AnnData, patient_subset: list[str] | None = None) -> pd.DataFrame:
    """Patient-level table for cohort / PCA figures (notebook 1)."""
    subset = patient_subset or DISCOVERY_PATIENTS
    meta = _patient_metadata_frame(adata.uns)
    keep = set(map(str, subset))
    return meta.loc[meta["patient_id"].astype(str).isin(keep)].copy()


def load_omiq_tsne(adata: ad.AnnData, compartment: str) -> pd.DataFrame:
    """OMIQ opt-SNE coordinates with a cell_id column (notebook 8)."""
    if "omiq_tsne" not in adata.uns:
        raise KeyError(
            "adata.uns['omiq_tsne'] not found. "
            "Run: python scripts/inject_omiq_tsne.py"
        )
    omiq = adata.uns["omiq_tsne"]
    if compartment not in omiq:
        raise KeyError(
            f"compartment {compartment!r} not in adata.uns['omiq_tsne']; "
            f"available: {list(omiq)}"
        )
    df = pd.DataFrame(omiq[compartment]).reset_index(names="cell_id")
    df["cell_id"] = df["cell_id"].astype(str)
    required = ["cell_id", "optsne_1", "optsne_2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"omiq_tsne[{compartment!r}] missing columns: {missing}")
    return df



def normalize_archetype_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy archetype columns to tumorimmune_archetype(_id)."""
    out = df.copy()
    rename: dict[str, str] = {}
    if TUMORIMMUNE_ARCHETYPE_ID_COL not in out.columns:
        for col in _LEGACY_ARCHETYPE_ID_COLS:
            if col in out.columns and col != TUMORIMMUNE_ARCHETYPE_ID_COL:
                rename[col] = TUMORIMMUNE_ARCHETYPE_ID_COL
                break
    if TUMORIMMUNE_ARCHETYPE_COL not in out.columns:
        for col in _LEGACY_ARCHETYPE_LABEL_COLS:
            if col in out.columns and col != TUMORIMMUNE_ARCHETYPE_COL:
                rename[col] = TUMORIMMUNE_ARCHETYPE_COL
                break
    if rename:
        out = out.rename(columns=rename)
    # Drop stale misaligned string column if present alongside the canonical label.
    if "abundance_cluster" in out.columns and TUMORIMMUNE_ARCHETYPE_COL in out.columns:
        out = out.drop(columns=["abundance_cluster"])
    if TUMORIMMUNE_ARCHETYPE_ID_COL in out.columns and TUMORIMMUNE_ARCHETYPE_COL not in out.columns:
        out[TUMORIMMUNE_ARCHETYPE_COL] = pd.to_numeric(
            out[TUMORIMMUNE_ARCHETYPE_ID_COL], errors="coerce"
        ).map(CLUSTER_NAME_MAP)
    return out


def archetype_assignments_from_adata(adata: ad.AnnData) -> pd.DataFrame | None:
    """Build archetype assignment table from embedded ``adata.uns`` (Zenodo fallback)."""
    rows: list[dict[str, object]] | None = None
    if "case_classifications" in adata.uns:
        cc = normalize_archetype_frame(pd.DataFrame(adata.uns["case_classifications"]))
        if "tumorimmune_archetype_id" in cc.columns:
            if "patient_id" in cc.columns:
                out = cc[["patient_id", "tumorimmune_archetype_id"]].copy()
                out["patient_id"] = out["patient_id"].astype(str)
            else:
                out = cc[["tumorimmune_archetype_id"]].copy()
                out.insert(0, "patient_id", cc.index.astype(str))
            if "tumorimmune_archetype" in cc.columns:
                out["tumorimmune_archetype"] = cc["tumorimmune_archetype"]
            rows = out.dropna(subset=["tumorimmune_archetype_id"]).to_dict("records")

    if rows is None:
        cluster_dict = adata.uns.get("tumorimmune_archetype_id")
        if not isinstance(cluster_dict, dict):
            cluster_dict = adata.uns.get("abundance_cluster_30")
        if isinstance(cluster_dict, dict):
            rows = [
                {"patient_id": str(pid), "tumorimmune_archetype_id": int(clu)}
                for pid, clu in cluster_dict.items()
                if pd.notna(clu)
            ]

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["patient_id"] = df["patient_id"].astype(str)
    df["tumorimmune_archetype_id"] = pd.to_numeric(df["tumorimmune_archetype_id"], errors="coerce")
    df = df.dropna(subset=["tumorimmune_archetype_id"])
    df["tumorimmune_archetype_id"] = df["tumorimmune_archetype_id"].astype(int)
    if "tumorimmune_archetype" not in df.columns:
        df["tumorimmune_archetype"] = df["tumorimmune_archetype_id"].map(CLUSTER_NAME_MAP)
    return df.reset_index(drop=True)


def load_archetype_assignments(
    paths: NotebookPaths,
    *,
    adata: ad.AnnData | None = None,
    required: bool = True,
) -> pd.DataFrame:
    if adata is not None:
        embedded = archetype_assignments_from_adata(adata)
        if embedded is not None and not embedded.empty:
            return embedded
    # Public bundles embed archetypes in the h5ad; load them even when callers
    # only pass ``paths`` (no separate CSV / maintainer notebook required).
    if adata is None and paths.adata_path.exists():
        try:
            embedded_adata = sc.read_h5ad(paths.adata_path, backed="r")
            embedded = archetype_assignments_from_adata(embedded_adata)
            if embedded is not None and not embedded.empty:
                return embedded
        except Exception:
            pass
    if paths.arch_path.exists():
        return normalize_archetype_frame(pd.read_csv(paths.arch_path))
    legacy_csv = paths.processed_dir / "abundance_cluster_30_assignments.csv"
    if legacy_csv.exists():
        return normalize_archetype_frame(pd.read_csv(legacy_csv))
    if required:
        raise FileNotFoundError(
            "Archetype assignments not found in adata.uns['case_classifications'] "
            f"(columns tumorimmune_archetype_id / tumorimmune_archetype) "
            f"and no fallback CSV at {rel_path(paths.arch_path, paths.repo_root)}.\n"
            "Use the published DLBCL_location_2026.h5ad bundle, or pass adata= to "
            "load_archetype_assignments() / call ensure_discovery_archetypes()."
        )
    return pd.DataFrame()


def ensure_discovery_archetypes(
    adata: ad.AnnData,
    paths: NotebookPaths,
) -> pd.DataFrame:
    """Load archetypes from CSV or h5ad fallback and sync into ``adata.uns``."""
    arch_df = load_archetype_assignments(paths, adata=adata, required=True)
    inject_archetypes_into_adata(adata, arch_df)
    return arch_df


def inject_archetypes_into_adata(
    adata: ad.AnnData,
    arch_df: pd.DataFrame,
) -> ad.AnnData:
    """Sync archetype assignments into ``adata.uns`` for downstream panels."""
    cluster_dict = dict(zip(arch_df["patient_id"].astype(str), arch_df["tumorimmune_archetype_id"].astype(int)))
    adata.uns["tumorimmune_archetype_id"] = cluster_dict

    cc = pd.DataFrame(adata.uns["case_classifications"]).copy()
    if "patient_id" in cc.columns:
        idx = cc["patient_id"].astype(str)
    else:
        idx = cc.index.astype(str)
    cc["tumorimmune_archetype_id"] = idx.map(cluster_dict).values
    if "tumorimmune_archetype" not in cc.columns or cc["tumorimmune_archetype"].isna().all():
        cc["tumorimmune_archetype"] = cc["tumorimmune_archetype_id"].map(CLUSTER_NAME_MAP)
    adata.uns["case_classifications"] = cc
    return adata


def cast_obs_id_columns(obs: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Avoid pandas categorical arithmetic bugs in groupby workflows."""
    out = obs.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def ensure_parent_dir(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rel_path(path: Path | str, repo_root: Path | str | None = None) -> str:
    """Path relative to the repository root (for notebook stdout)."""
    root = Path(repo_root or repo_root_from_notebook_cwd()).resolve()
    p = Path(path)
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.name if p.is_absolute() else p.as_posix()


def log_wrote(path: Path | str, repo_root: Path | str | None = None) -> None:
    print(f"Wrote {rel_path(path, repo_root)}")


def log_saved(path: Path | str, repo_root: Path | str | None = None) -> None:
    print(f"Saved: {rel_path(path, repo_root)}")


def save_figure_formats(
    fig,
    stem: Path | str,
    *,
    formats: tuple[str, ...] = ("svg", "png"),
    dpi: int = 300,
    bbox_inches: str = "tight",
    **savefig_kw,
) -> list[Path]:
    """Write a paper panel as SVG/PNG (and optional PDF) next to ``stem``."""
    base = Path(stem)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        kw = dict(bbox_inches=bbox_inches, **savefig_kw)
        if fmt == "png":
            kw.setdefault("dpi", dpi)
        fig.savefig(out, format=fmt, **kw)
        written.append(out)
    return written


def promote_panel(
    src_stem: Path | str,
    dest_stem: Path | str,
    *,
    exts: tuple[str, ...] = (".svg", ".png"),
) -> list[Path]:
    """Copy analysis outputs to a manuscript panel stem under the figure root."""
    import shutil

    src_base = Path(src_stem)
    if src_base.suffix:
        src_base = src_base.with_suffix("")
    dest_base = Path(dest_stem)
    if dest_base.suffix:
        dest_base = dest_base.with_suffix("")
    dest_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in exts:
        src = src_base.with_suffix(ext)
        dest = dest_base.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, dest)
            written.append(dest)
    return written


def supplementary_dir(repo_root: Path | str) -> Path:
    """Root folder for plot source tables (``data/supplementary/``)."""
    path = Path(repo_root) / "data" / "supplementary"
    path.mkdir(parents=True, exist_ok=True)
    return path


_supplementary_fig_id: str | None = None


def set_supplementary_fig_id(fig_id: str | None) -> None:
    """Set active figure notebook id for supplementary CSV output paths."""
    global _supplementary_fig_id
    _supplementary_fig_id = fig_id


def supplementary_table_dir(repo_root: Path | str, *, fig_id: str | None = None) -> Path:
    """Per-notebook supplementary folder: ``data/supplementary/{fig_id}/`` when set."""
    fid = fig_id if fig_id is not None else _supplementary_fig_id
    path = supplementary_dir(repo_root)
    if fid:
        path = path / fid
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_supplementary_table(
    df: pd.DataFrame,
    repo_root: Path | str,
    stem: str,
    *,
    fig_id: str | None = None,
    index: bool = False,
) -> Path:
    """Write a plot source table to ``data/supplementary/{fig_id}/{stem}.csv``."""
    out = supplementary_table_dir(repo_root, fig_id=fig_id) / f"{stem}.csv"
    df.to_csv(out, index=index)
    return out


def write_registered_supplementary_table(
    df: pd.DataFrame,
    repo_root: Path | str,
    registry_key: str,
    *,
    index: bool = False,
) -> Path:
    """Write using a canonical stem from ``dlbcl.figure_registry``."""
    from .figure_registry import supplementary_stem

    return write_supplementary_table(
        df, repo_root, supplementary_stem(registry_key), index=index
    )


def _read_supplementary_csv(csv_path: Path) -> pd.DataFrame:
    try:
        preview = pd.read_csv(csv_path, nrows=0)
        index_col = 0 if len(preview.columns) and preview.columns[0] == "Unnamed: 0" else None
        return pd.read_csv(csv_path, index_col=index_col)
    except pd.errors.EmptyDataError:
        return pd.read_csv(csv_path)


def audit_supplementary_tables(repo_root: Path | str) -> dict[str, object]:
    """Report registry vs on-disk CSV status (orphans, missing, duplicates)."""
    from .figure_registry import (
        LEGACY_SUPPLEMENTARY_STEMS,
        SUPPLEMENTARY_TABLES,
        canonical_stem_from_path,
        registered_stem_set,
        stem_to_registry_key,
        workbook_id_for_key,
    )

    supp = supplementary_dir(repo_root)
    csv_paths = sorted(p for p in supp.rglob("*.csv") if p.is_file())
    by_stem: dict[str, list[Path]] = {}
    for path in csv_paths:
        stem = canonical_stem_from_path(path)
        by_stem.setdefault(stem, []).append(path)

    registered_canonical = {spec.stem for spec in SUPPLEMENTARY_TABLES.values()}
    orphans_on_disk = sorted(stem for stem in by_stem if stem not in registered_stem_set())
    missing_from_disk = sorted(stem for stem in registered_canonical if stem not in by_stem)
    duplicates = {stem: [str(p) for p in paths] for stem, paths in by_stem.items() if len(paths) > 1}
    key_by_stem = stem_to_registry_key()
    present_keys = [key_by_stem[stem] for stem in by_stem if stem in key_by_stem]

    return {
        "n_csv": len(csv_paths),
        "n_unique_stems": len(by_stem),
        "orphans_on_disk": orphans_on_disk,
        "missing_from_disk": missing_from_disk,
        "duplicates": duplicates,
        "by_workbook": {
            workbook_id_for_key(k): None for k in present_keys
        },
        "paths_by_stem": {stem: [str(p) for p in paths] for stem, paths in by_stem.items()},
    }


def _preferred_csv_path(stem: str, paths: list[Path], workbook_id: str) -> Path:
    from .figure_registry import WORKBOOK_FOLDER_PREFERENCE

    prefs = WORKBOOK_FOLDER_PREFERENCE.get(workbook_id, ())
    if prefs:
        ranked: list[tuple[int, Path]] = []
        for path in paths:
            parent = path.parent.name
            try:
                rank = prefs.index(parent)
            except ValueError:
                rank = len(prefs)
            ranked.append((rank, path))
        ranked.sort(key=lambda item: (item[0], str(item[1])))
        return ranked[0][1]
    return sorted(paths, key=lambda p: str(p))[0]


def build_supplementary_tables_by_figure(
    repo_root: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> list[Path]:
    """Write one ordered Excel workbook per manuscript figure.

    Sheets are prefixed ``Input_`` / ``Stats_`` from the registry role, ordered by
    ``SUPPLEMENTARY_TABLES`` insertion. An ``index`` sheet lists panel, role, stem,
    and source CSV. Missing registry tables are listed on ``index`` but omitted as
    data sheets. Duplicate on-disk stems prefer the folder for that figure book.
    """
    from .figure_registry import (
        SUPPLEMENTARY_TABLES,
        canonical_stem_from_path,
        excel_sheet_name,
        table_role,
        workbook_id_for_key,
    )

    root = Path(repo_root)
    supp = supplementary_dir(root)
    out_root = Path(out_dir) if out_dir is not None else supp / "xlsx"
    out_root.mkdir(parents=True, exist_ok=True)

    by_stem: dict[str, list[Path]] = {}
    for path in supp.rglob("*.csv"):
        if not path.is_file():
            continue
        stem = canonical_stem_from_path(path)
        by_stem.setdefault(stem, []).append(path)

    # workbook_id → list of (registry_key, spec, csv_path|None)
    books: dict[str, list[tuple[str, object, Path | None]]] = {}
    for key, spec in SUPPLEMENTARY_TABLES.items():
        book = workbook_id_for_key(key)
        paths = by_stem.get(spec.stem, [])
        csv_path = _preferred_csv_path(spec.stem, paths, book) if paths else None
        books.setdefault(book, []).append((key, spec, csv_path))

    written: list[Path] = []
    book_order = sorted(books.keys(), key=lambda b: (b.startswith("Other"), b.replace("FigS", "FigZ"), b))
    for book in book_order:
        entries = books[book]
        # Skip empty books (all missing) — still useful to know; write only if ≥1 CSV
        if not any(csv for _, _, csv in entries):
            continue
        out_path = out_root / f"{book}_supplementary_tables.xlsx"
        index_rows = []
        used_sheet_names: set[str] = set()
        frames: list[tuple[str, pd.DataFrame]] = []
        for key, spec, csv_path in entries:
            role = table_role(spec)
            sheet = excel_sheet_name(key, role)
            # Disambiguate collisions within the 31-char limit.
            base = sheet
            n = 2
            while sheet in used_sheet_names:
                suffix = f"_{n}"
                sheet = (base[: 31 - len(suffix)] + suffix)[:31]
                n += 1
            used_sheet_names.add(sheet)
            index_rows.append(
                {
                    "sheet": sheet,
                    "role": role,
                    "registry_key": key,
                    "stem": spec.stem,
                    "manuscript": spec.manuscript,
                    "description": spec.description,
                    "status": "ok" if csv_path is not None else "missing",
                    "source_csv": rel_path(csv_path, root) if csv_path is not None else "",
                }
            )
            if csv_path is not None:
                frames.append((sheet, _read_supplementary_csv(csv_path)))

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(index_rows).to_excel(writer, sheet_name="index", index=False)
            for sheet, frame in frames:
                frame.to_excel(writer, sheet_name=sheet, index=False)
        written.append(out_path)
    return written


def build_supplementary_tables_xlsx(
    repo_root: Path | str,
    *,
    out_name: str = "supplementary_tables.xlsx",
) -> Path:
    """Legacy all-in-one workbook (registry order, Input_/Stats_ sheet names).

    Prefer ``build_supplementary_tables_by_figure`` for publication. This keeps a
    single combined file for convenience.
    """
    from .figure_registry import (
        SUPPLEMENTARY_TABLES,
        canonical_stem_from_path,
        excel_sheet_name,
        table_role,
        workbook_id_for_key,
    )

    root = Path(repo_root)
    supp = supplementary_dir(root)
    by_stem: dict[str, list[Path]] = {}
    for path in supp.rglob("*.csv"):
        if path.is_file():
            by_stem.setdefault(canonical_stem_from_path(path), []).append(path)

    out_path = supp / out_name
    index_rows = []
    frames: list[tuple[str, pd.DataFrame]] = []
    used: set[str] = set()
    for key, spec in SUPPLEMENTARY_TABLES.items():
        paths = by_stem.get(spec.stem, [])
        if not paths:
            continue
        book = workbook_id_for_key(key)
        csv_path = _preferred_csv_path(spec.stem, paths, book)
        role = table_role(spec)
        sheet = excel_sheet_name(key, role)
        base = sheet
        n = 2
        while sheet in used:
            suffix = f"_{n}"
            sheet = (base[: 31 - len(suffix)] + suffix)[:31]
            n += 1
        used.add(sheet)
        index_rows.append(
            {
                "sheet": sheet,
                "role": role,
                "workbook": book,
                "registry_key": key,
                "stem": spec.stem,
                "manuscript": spec.manuscript,
                "description": spec.description,
                "source_csv": rel_path(csv_path, root),
            }
        )
        frames.append((sheet, _read_supplementary_csv(csv_path)))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="index", index=False)
        for sheet, frame in frames:
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return out_path


# Embedded genomic classifier columns → canonical names in case_classifications.
LYMPHOTYPER_TO_CANONICAL: dict[str, str] = {
    "LymphGen_lymphotyper": "Lymphgen",
    "DLBclass_lymphotyper": "DLBclass",
    "HMRN_lymphotyper": "HMRN",
    "LymphPlex_lymphotyper": "LymphPlex",
}

# Upstream *_update columns → canonical spatial / transcriptomic classifiers.
UPDATE_COLUMN_ALIASES: dict[str, str] = {
    "Lymphoma_Ecotype_original_update": "Lymphoma_Ecotype",
    "New_Ecotype_update": "New_Ecotype",
    "KotlovSig_update": "KotlovSig",
    "Ciav_Cluster_update": "Ciav_Cluster",
    "COO_NanoString_update": "COO_NanoString",
}

GENOMIC_LEGACY_FALLBACK: dict[str, str] = {
    "LymphGen_update": "Lymphgen",
    "Lymphgen_aug2024": "Lymphgen",
    "DLBclass_update": "DLBclass",
}

DROP_CLASSIFICATION_SOURCES = frozenset(
    set(LYMPHOTYPER_TO_CANONICAL)
    | set(UPDATE_COLUMN_ALIASES)
    | set(GENOMIC_LEGACY_FALLBACK)
    | {
        "LymphGen_lymphotyper_panel",
        # Stale / superseded archetype columns (prefer tumorimmune_archetype*).
        "abundance_cluster",
        "abundance_cluster_30",
        "abundance_cluster_30_label",
    }
)

def consolidate_case_classifications(
    case_classifications: pd.DataFrame,
) -> pd.DataFrame:
    """One canonical column per classifier category from embedded labels."""
    out = pd.DataFrame(case_classifications).copy()

    for src, dst in UPDATE_COLUMN_ALIASES.items():
        if src in out.columns:
            out[dst] = out[src]

    for src, dst in LYMPHOTYPER_TO_CANONICAL.items():
        if src in out.columns:
            out[dst] = out[src]

    for src, dst in GENOMIC_LEGACY_FALLBACK.items():
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]

    from .validation_classifications import finalize_lymphoma_ecotype_columns

    out = finalize_lymphoma_ecotype_columns(out)
    out = normalize_archetype_frame(out)
    drop = [c for c in DROP_CLASSIFICATION_SOURCES if c in out.columns]
    return out.drop(columns=drop)


def consolidate_embedded_case_classifications(
    adata: ad.AnnData,
    paths: NotebookPaths | None = None,
    *,
    repo_root: Path | str | None = None,
) -> ad.AnnData:
    """Canonicalize embedded ``uns['case_classifications']``."""
    del paths, repo_root  # retained for notebook call compatibility
    adata.uns["case_classifications"] = consolidate_case_classifications(
        adata.uns["case_classifications"]
    )
    return adata


def genomic_analysis_patient_ids(
    adata: ad.AnnData,
    patient_subset: list[str] | None = None,
) -> list[str]:
    """Discovery (or custom) patients with ≥1 source variant in ``genomic_profiling``."""
    from .genomic_profiling import is_nested_genomic_profiling, patients_without_source_variants

    subset = list(patient_subset or DISCOVERY_PATIENTS)
    gp = adata.uns.get("genomic_profiling")
    if not is_nested_genomic_profiling(gp):
        return subset
    drop = patients_without_source_variants(gp, patient_ids=subset)
    if drop:
        return [p for p in subset if p not in drop]
    return subset


def load_validation_cohort(adata):
    """Return embedded validation cohort ``uns`` payload (nb5 / nb14)."""
    from .validation_cohort import require_validation_cohort

    return require_validation_cohort(adata)


def elastic_net_classifier(**kwargs):
    """sklearn >=1.8 compatible elastic-net multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression

    kwargs.pop("penalty", None)
    kwargs.pop("multi_class", None)
    kwargs.pop("n_jobs", None)
    kwargs.setdefault("solver", "saga")
    kwargs.setdefault("l1_ratio", 0.5)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        return LogisticRegression(**kwargs)


def _write_gmt(genesets: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for term, genes in genesets.items():
            clean = [str(g) for g in genes if str(g)]
            fh.write(f"{term}\tna\t" + "\t".join(clean) + "\n")


def load_msigdb_gmt(
    category: str,
    *,
    data_dir: Path,
    dbver: str = "2023.1.Hs",
) -> dict[str, list[str]]:
    """Load an MSigDB GMT via gseapy, caching under ``data/gene_sets/``."""
    import gseapy as gp
    from gseapy import Msigdb

    cache_dir = data_dir / "gene_sets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = category.replace(".", "_")
    cache_path = cache_dir / f"{safe_name}.{dbver}.symbols.gmt"

    if cache_path.exists():
        return gp.read_gmt(str(cache_path))

    genesets = Msigdb().get_gmt(category=category, dbver=dbver, entrez=False)
    _write_gmt(genesets, cache_path)
    return genesets
