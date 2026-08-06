"""Combined discovery/validation transcriptome DE helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .dlbcl_io import DISCOVERY_PATIENTS, NotebookPaths, load_archetype_assignments
from .validation_cohort import cohort_notebook_inputs
from .validation_figures import DISEASE_TYPE_TO_LOCATION

ARCHETYPE_CANONICAL = {
    "complex immune": "diverse immune",
    "diverse immune": "diverse immune",
    "low immune": "low immune",
    "cytotoxic predominant": "cytotoxic predominant",
}
ARCHETYPE_SHORT = {
    "diverse immune": "DI",
    "low immune": "LO",
    "cytotoxic predominant": "CP",
}
ARCHETYPE_ORDER = ["diverse immune", "low immune", "cytotoxic predominant"]
ARCHETYPE_COLORS = {
    "diverse immune": "#2ca02c",
    "low immune": "#d62728",
    "cytotoxic predominant": "#1f77b4",
}
HLA_STATUS_COLORS = {
    "loss": "#b2182b",
    "retained": "#2166ac",
}
HLA_CLASS_GROUP_ORDER = [
    "HLA-I+/HLA-II+",
    "HLA-I+/HLA-II-",
    "HLA-I-/HLA-II+",
    "HLA-I-/HLA-II-",
]
HLA_CLASS_GROUP_SHORT = {
    "HLA-I+/HLA-II+": "Ipos_IIpos",
    "HLA-I+/HLA-II-": "Ipos_IIneg",
    "HLA-I-/HLA-II+": "Ineg_IIpos",
    "HLA-I-/HLA-II-": "Ineg_IIneg",
}
HLA_CLASS_GROUP_COLORS = {
    "HLA-I+/HLA-II+": "#b07d2b",
    "HLA-I+/HLA-II-": "#a3a3a3",
    "HLA-I-/HLA-II+": "#7b5aa6",
    "HLA-I-/HLA-II-": "#4b5563",
}
DISCOVERY_LOCATION_RECODE = {
    "Brain": "PCNS",
    "CNS": "PCNS",
    "PCNSL": "PCNS",
    "PCNS": "PCNS",
    "Nodal": "nodal",
    "nodal": "nodal",
    "Bone": "bone",
    "bone": "bone",
    "Testis": "testis",
    "testis": "testis",
}
LOCATION_ORDER = ["PCNS", "bone", "nodal", "testis"]
MARKER_GENE_SETS = {
    "cytotoxic": {
        "B2M",
        "CALHM6",
        "CD2",
        "CD247",
        "CD3D",
        "CD3E",
        "CD3G",
        "CD8A",
        "CD8B",
        "CX3CR1",
        "EOMES",
        "FGFBP2",
        "GNLY",
        "GZMA",
        "GZMB",
        "GZMH",
        "IFNG",
        "IL2RB",
        "KLRB1",
        "KLRC1",
        "KLRC2",
        "KLRD1",
        "KLRK1",
        "NKG7",
        "PRF1",
        "SH2D1A",
        "TBX21",
    },
    "myeloid": {
        "AIF1",
        "APOE",
        "C1QA",
        "C1QB",
        "C1QC",
        "CCR2",
        "CD14",
        "CD163",
        "CD33",
        "CD36",
        "CD68",
        "CSF1R",
        "CXCL9",
        "CXCL10",
        "CYBB",
        "FCER1G",
        "FCGR1A",
        "FCGR1B",
        "FCGR2A",
        "FCGR3A",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DRA",
        "HLA-DRB1",
        "IDO1",
        "IL1B",
        "IRF8",
        "ITGAM",
        "ITGAX",
        "LST1",
        "LYZ",
        "MAFB",
        "MRC1",
        "MRC2",
        "MS4A7",
        "S100A8",
        "S100A9",
        "SPI1",
        "TYROBP",
    },
    "ifng_antigen_presentation": {
        "B2M",
        "CD274",
        "CXCL9",
        "CXCL10",
        "GBP1",
        "GBP5",
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DRA",
        "HLA-DRB1",
        "IDO1",
        "IFNG",
        "IRF1",
        "STAT1",
    },
    "stromal_ecm": {
        "AEBP1",
        "BGN",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL4A1",
        "COL4A2",
        "COL5A1",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "DCN",
        "FAP",
        "FBLN1",
        "FBLN2",
        "FN1",
        "LAMA4",
        "LAMB1",
        "LUM",
        "MMP2",
        "MMP9",
        "MRC2",
        "PDGFRB",
        "POSTN",
        "SPARC",
        "TAGLN",
        "THBS1",
        "TIMP1",
        "TIMP2",
        "VCAN",
        "VIM",
    },
    "lymphoid_cd4_help": {
        "BATF",
        "BCL6",
        "CCR7",
        "CD2",
        "CD27",
        "CD28",
        "CD3D",
        "CD3E",
        "CD4",
        "CD40LG",
        "CXCL13",
        "CXCR5",
        "ICOS",
        "IL21",
        "IL21R",
        "IL7R",
        "PDCD1",
        "SH2D1A",
        "TCF7",
        "TOX",
    },
    "bcell_plasmablast": {
        "BACH2",
        "BCL11A",
        "CD19",
        "CD22",
        "CD27",
        "CD38",
        "CD79A",
        "CD79B",
        "EBF1",
        "IGHG1",
        "IGHM",
        "IRF4",
        "JCHAIN",
        "MZB1",
        "PAX5",
        "PRDM1",
        "SDC1",
        "TNFRSF13B",
        "TNFRSF13C",
        "XBP1",
    },
    "proliferation": {
        "AURKA",
        "BIRC5",
        "CCNA2",
        "CCNB1",
        "CCNB2",
        "CCNE1",
        "CDC20",
        "CDK1",
        "CENPF",
        "ESCO2",
        "MCM2",
        "MCM3",
        "MCM4",
        "MCM5",
        "MKI67",
        "PCNA",
        "PLK1",
        "RFC3",
        "TOP2A",
        "UBE2C",
    },
}

PROGRAM_GENE_SETS = {
    "Myeloid / macrophage": sorted(MARKER_GENE_SETS["myeloid"]),
    "Cytotoxic / NK": sorted(MARKER_GENE_SETS["cytotoxic"]),
    "IFN / antigen presentation": sorted(MARKER_GENE_SETS["ifng_antigen_presentation"]),
    "Stromal / ECM": sorted(MARKER_GENE_SETS["stromal_ecm"]),
    "Lymphoid organization / CD4-help": sorted(MARKER_GENE_SETS["lymphoid_cd4_help"]),
    "B-cell differentiation / plasmablast": sorted(MARKER_GENE_SETS["bcell_plasmablast"]),
    "Proliferation": sorted(MARKER_GENE_SETS["proliferation"]),
}

HLA_IMMUNE_PROGRAM_GENE_SETS = {
    "Class I machinery": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "ERAP1", "ERAP2",
    ],
    "Class II machinery": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "CD74", "CIITA", "RFX5", "RFXAP", "RFXANK",
    ],
    "HLA-DM peptide loading": ["HLA-DMA", "HLA-DMB", "CD74", "HLA-DOA", "HLA-DOB"],
    "CD74 / invariant chain": ["CD74"],
    "IFN-gamma response": [
        "IFNG", "STAT1", "IRF1", "IRF8", "CXCL9", "CXCL10", "CXCL11",
        "GBP1", "GBP5", "IDO1", "TAP1", "PSMB9",
    ],
    "IFN-alpha / ISG": ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "STAT1", "IRF7"],
    "CD4/Tfh organization": [
        "CD4", "CXCR5", "ICOS", "PDCD1", "CD40LG", "IL21", "IL21R",
        "BCL6", "CXCL13", "CCR7", "CCL19", "CCL21", "LTA", "LTB",
    ],
    "APC/CD4 co-stimulation": ["CD40", "CD80", "CD86", "ICOSLG", "TNFSF4", "TNFRSF4", "TNFSF9", "TNFRSF9"],
    "Cytotoxicity": ["CD8A", "CD8B", "GZMB", "GZMA", "PRF1", "GNLY", "NKG7", "KLRD1", "KLRK1"],
    "Exhaustion": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "CTLA4", "ENTPD1", "CXCL13"],
    "Suppressive counter-regulation": [
        "CD274", "PDCD1LG2", "IDO1", "TGFB1", "IL10", "LGALS9",
        "VSIR", "LILRB1", "LILRB2", "MRC1", "CD163",
    ],
    "APC-like myeloid / DC": ["CD74", "HLA-DRA", "HLA-DPA1", "ITGAX", "CD1C", "CLEC10A", "FCER1A", "LAMP3", "CCR7"],
    "Suppressive myeloid": ["CD68", "CD163", "MRC1", "MSR1", "LILRB1", "LILRB2", "C1QA", "C1QB", "C1QC", "SPP1", "APOE", "TREM2"],
    "Lymphoid organization": ["CXCL13", "CCR7", "CCL19", "CCL21", "LTA", "LTB", "CD40LG", "IL21", "IL21R"],
}
HLA_CLASS_GROUP_PROGRAM_HEATMAP_ORDER = [
    "Class I machinery",
    "Class II machinery",
    "HLA-DM peptide loading",
    "IFN-gamma response",
    "CD4/Tfh organization",
    "APC/CD4 co-stimulation",
    "Lymphoid organization",
]


def canonical_archetype(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    return ARCHETYPE_CANONICAL.get(key, str(value).strip())


def _as_frame(obj) -> pd.DataFrame:
    return obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj).copy()


def _clean_expression(expr: pd.DataFrame) -> pd.DataFrame:
    out = _as_frame(expr)
    out.index = out.index.astype(str).str.strip()
    out.columns = out.columns.astype(str).str.strip()
    return out.apply(pd.to_numeric, errors="coerce")


def clean_hla_call(values: pd.Series) -> pd.Series:
    """Normalize tumor HLA annotation to ``loss``/``retained``."""
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


def _discovery_metadata(adata, paths: NotebookPaths | None, patient_subset: list[str] | None) -> pd.DataFrame:
    meta = _as_frame(adata.uns["case_classifications"])
    if "patient_id" not in meta.columns:
        meta["patient_id"] = meta.index.astype(str)
    meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
    meta = meta.drop_duplicates("patient_id").set_index("patient_id")

    if "tumorimmune_archetype" not in meta.columns and "tumorimmune_archetype_id" not in meta.columns:
        if paths is None:
            raise KeyError(
                "Discovery metadata lacks tumorimmune_archetype / tumorimmune_archetype_id; pass paths with "
                "data/processed/tumorimmune_archetype_assignments.csv available."
            )
        arch = load_archetype_assignments(paths)
        arch = arch.copy()
        arch["patient_id"] = arch["patient_id"].astype(str)
        meta = meta.join(
            arch.set_index("patient_id")[["tumorimmune_archetype_id", "tumorimmune_archetype"]],
            how="left",
        )

    if "tumorimmune_archetype" in meta.columns:
        meta["Archetype"] = meta["tumorimmune_archetype"].map(canonical_archetype)
    else:
        label_map = {1: "low immune", 2: "cytotoxic predominant", 3: "complex immune"}
        meta["Archetype"] = pd.to_numeric(meta["tumorimmune_archetype_id"], errors="coerce").map(label_map)

    loc_source = "Location" if "Location" in meta.columns else "disease_type"
    meta["Location"] = meta[loc_source].map(DISCOVERY_LOCATION_RECODE).fillna(meta[loc_source])
    meta["Cohort"] = "Discovery"

    subset = list(patient_subset or DISCOVERY_PATIENTS)
    return meta.loc[meta.index.intersection(pd.Index(subset).astype(str))].copy()


def build_combined_transcriptome_inputs(
    adata,
    *,
    paths: NotebookPaths | None = None,
    patient_subset: list[str] | None = None,
    min_common_genes: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized combined expression (samples × genes) and aligned metadata.

    Discovery and validation expression are intersected on gene symbols, then each
    gene is z-scored within cohort before concatenation. This makes the volcano
    coefficient a cohort-standardized mean-expression difference.
    """
    if "gene_expression" not in adata.uns:
        raise KeyError("adata.uns['gene_expression'] is required for discovery transcriptome DE")

    discovery_expr = _clean_expression(adata.uns["gene_expression"])
    discovery_meta = _discovery_metadata(adata, paths, patient_subset)

    pred, validation_expr, validation_labels = cohort_notebook_inputs(adata)
    validation_expr = _clean_expression(validation_expr)
    validation_meta = pred.copy()
    validation_meta.index = validation_meta.index.astype(str)
    validation_meta["Archetype"] = validation_labels.map(canonical_archetype)
    validation_meta["Location"] = validation_meta["disease_type"].map(DISEASE_TYPE_TO_LOCATION)
    validation_meta["Cohort"] = "Validation"

    genes = discovery_expr.index.intersection(validation_expr.index)
    if len(genes) < min_common_genes:
        raise ValueError(f"Only {len(genes)} common transcriptome genes found; expected at least {min_common_genes}.")

    discovery_samples = discovery_expr.columns.intersection(discovery_meta.index)
    validation_samples = validation_expr.columns.intersection(validation_meta.index)

    disc_x = discovery_expr.loc[genes, discovery_samples].T
    val_x = validation_expr.loc[genes, validation_samples].T
    disc_x.index = "Discovery:" + disc_x.index.astype(str)
    val_x.index = "Validation:" + val_x.index.astype(str)

    disc_meta = discovery_meta.loc[discovery_samples, ["Location", "Archetype", "Cohort"]].copy()
    val_meta = validation_meta.loc[validation_samples, ["Location", "Archetype", "Cohort"]].copy()
    disc_meta.index = disc_x.index
    val_meta.index = val_x.index

    expr = pd.concat([disc_x, val_x], axis=0)
    meta = pd.concat([disc_meta, val_meta], axis=0)
    keep = meta["Location"].notna() & meta["Archetype"].isin(ARCHETYPE_ORDER)
    expr = expr.loc[keep].copy()
    meta = meta.loc[keep].copy()

    expr = _zscore_by_group(expr, meta["Cohort"])
    expr = expr.loc[:, expr.notna().any(axis=0)]
    expr = expr.loc[:, expr.var(axis=0, ddof=1).fillna(0) > 0]
    expr = expr.apply(lambda s: s.fillna(s.median()), axis=0)
    return expr, meta


def build_discovery_hla_transcriptome_inputs(
    adata,
    *,
    paths: NotebookPaths | None = None,
    patient_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return discovery-only z-scored transcriptome and HLA/archetype metadata.

    The expression matrix is samples × genes and each gene is standardized across
    discovery patients, so coefficients and Hedges g are on comparable scales.
    """
    if "gene_expression" not in adata.uns:
        raise KeyError("adata.uns['gene_expression'] is required for discovery HLA transcriptome DE")
    expr_raw = _clean_expression(adata.uns["gene_expression"])
    meta = _discovery_metadata(adata, paths, patient_subset)

    missing_hla = [col for col in ("HLAABC", "HLADR") if col not in meta.columns]
    if missing_hla and "case_path" in adata.uns:
        case_path = _as_frame(adata.uns["case_path"])
        if "patient_id" not in case_path.columns:
            case_path["patient_id"] = case_path.index.astype(str)
        case_path["patient_id"] = case_path["patient_id"].astype(str).str.strip()
        case_path = case_path.drop_duplicates("patient_id").set_index("patient_id")
        join_cols = [col for col in missing_hla if col in case_path.columns]
        if join_cols:
            meta = meta.join(case_path[join_cols], how="left")

    for col in ("HLAABC", "HLADR"):
        if col not in meta.columns:
            raise KeyError(f"Discovery metadata lacks {col!r}; expected HLA annotations in case_path or case_classifications.")
        meta[col] = clean_hla_call(meta[col])

    samples = expr_raw.columns.intersection(meta.index)
    expr = expr_raw.loc[:, samples].T
    meta = meta.loc[samples].copy()
    keep = meta["Location"].notna() & meta["Archetype"].isin(ARCHETYPE_ORDER)
    expr = expr.loc[keep].copy()
    meta = meta.loc[keep].copy()

    expr = _zscore_by_group(expr, pd.Series("Discovery", index=expr.index))
    expr = expr.loc[:, expr.notna().any(axis=0)]
    expr = expr.loc[:, expr.var(axis=0, ddof=1).fillna(0) > 0]
    expr = expr.apply(lambda s: s.fillna(s.median()), axis=0)
    return expr, meta


def add_hla_class_group(meta: pd.DataFrame) -> pd.DataFrame:
    """Add the combined HLA-I/HLA-II retained-loss four-state group."""
    out = meta.copy()
    hla_i = clean_hla_call(out["HLAABC"])
    hla_ii = clean_hla_call(out["HLADR"])
    i_part = hla_i.map({"retained": "HLA-I+", "loss": "HLA-I-"})
    ii_part = hla_ii.map({"retained": "HLA-II+", "loss": "HLA-II-"})
    out["hla_class_group"] = np.where(
        i_part.notna() & ii_part.notna(),
        i_part.astype(str) + "/" + ii_part.astype(str),
        pd.NA,
    )
    out["hla_class_group"] = pd.Categorical(
        out["hla_class_group"],
        categories=HLA_CLASS_GROUP_ORDER,
        ordered=True,
    )
    out["hla_class_group_short"] = out["hla_class_group"].astype("object").map(HLA_CLASS_GROUP_SHORT)
    return out


def run_discovery_hla_class_group_de(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """One-vs-rest discovery DEG for the four combined HLA-I/HLA-II states.

    Positive coefficients and Hedges g indicate higher expression in the tested
    HLA class group. Models adjust for anatomical location and archetype.
    """
    import statsmodels.api as sm

    meta = add_hla_class_group(meta.reindex(expr.index))
    use_meta = meta.loc[meta["hla_class_group"].notna()].copy()
    expr = expr.loc[use_meta.index]
    rows = []
    covars = pd.get_dummies(use_meta[["Location", "Archetype"]].astype(str), drop_first=True, dtype=float)

    for group in groups:
        group_flag = use_meta["hla_class_group"].astype("object").eq(group).astype(float)
        n_group = int(group_flag.sum())
        n_rest = int(len(group_flag) - n_group)
        if n_group < min_per_group or n_rest < min_per_group:
            continue
        design = pd.DataFrame({"Intercept": 1.0, "hla_group": group_flag}, index=use_meta.index)
        if not covars.empty:
            design = design.join(covars)
        design = design.loc[:, (design.nunique(dropna=False) > 1) | (design.columns == "Intercept")]
        if design.shape[0] <= design.shape[1]:
            continue
        target_idx = group_flag.loc[group_flag.eq(1)].index
        rest_idx = group_flag.loc[group_flag.eq(0)].index
        for gene in expr.columns:
            y = pd.to_numeric(expr[gene], errors="coerce")
            valid = y.notna()
            if int(valid.sum()) <= design.loc[valid].shape[1]:
                continue
            fit = sm.OLS(y.loc[valid].astype(float), design.loc[valid]).fit()
            target_vals = y.loc[target_idx].dropna()
            rest_vals = y.loc[rest_idx].dropna()
            if len(target_vals) < min_per_group or len(rest_vals) < min_per_group:
                continue
            t_stat, ttest_p = stats.ttest_ind(target_vals, rest_vals, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "hla_class_group": group,
                    "hla_class_group_short": HLA_CLASS_GROUP_SHORT[group],
                    "contrast": f"{group} vs rest",
                    "gene": gene,
                    "coef": float(fit.params["hla_group"]),
                    "p": float(fit.pvalues["hla_group"]),
                    "ttest_t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "ttest_p_unadjusted": float(ttest_p) if pd.notna(ttest_p) else np.nan,
                    "effect_g": _hedges_g(target_vals, rest_vals),
                    "mean_group": float(target_vals.mean()),
                    "mean_rest": float(rest_vals.mean()),
                    "diff_unadjusted": float(target_vals.mean() - rest_vals.mean()),
                    "n_group": n_group,
                    "n_rest": n_rest,
                    "n_samples": int(len(group_flag)),
                    "marker_tags": marker_tags_for_gene(gene),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = np.nan
    for _, idx in out.groupby("hla_class_group", observed=False).groups.items():
        out.loc[idx, "q"] = multipletests(out.loc[idx, "p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["hla_class_group", "q", "p", "coef"], ascending=[True, True, True, False])


def hla_class_group_de_summary(de: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Top genes per HLA class group in both one-vs-rest directions."""
    if de.empty:
        return de
    rows = []
    for group, sub in de.groupby("hla_class_group", sort=False, observed=False):
        up = sub.loc[sub["coef"] > 0].sort_values(["q", "p", "coef"], ascending=[True, True, False]).head(top_n)
        down = sub.loc[sub["coef"] < 0].sort_values(["q", "p", "coef"], ascending=[True, True, True]).head(top_n)
        for direction, piece in (("higher_in_group", up), ("higher_in_rest", down)):
            for rank, (_, row) in enumerate(piece.iterrows(), start=1):
                rows.append(
                    {
                        "hla_class_group": group,
                        "hla_class_group_short": row["hla_class_group_short"],
                        "direction": direction,
                        "rank": rank,
                        "gene": row["gene"],
                        "coef": row["coef"],
                        "effect_g": row["effect_g"],
                        "p": row["p"],
                        "q": row["q"],
                        "n_group": row["n_group"],
                        "n_rest": row["n_rest"],
                        "marker_tags": row["marker_tags"],
                    }
                )
    return pd.DataFrame(rows)


def _zscore_by_group(expr: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    pieces = []
    for group, idx in groups.groupby(groups).groups.items():
        sub = expr.loc[idx].copy()
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=0).replace(0, np.nan)
        pieces.append(sub.sub(mu, axis=1).div(sd, axis=1))
    out = pd.concat(pieces, axis=0).reindex(expr.index)
    return out.replace([np.inf, -np.inf], np.nan)


def _design_matrix(meta: pd.DataFrame, target: str, min_per_group: int) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    target_flag = meta["Archetype"].eq(target)
    loc_ok = []
    for loc, loc_meta in meta.groupby("Location", observed=False):
        n_target = int(loc_meta["Archetype"].eq(target).sum())
        n_rest = int((~loc_meta["Archetype"].eq(target)).sum())
        if n_target >= min_per_group and n_rest >= min_per_group:
            loc_ok.append(loc)
    use_meta = meta.loc[meta["Location"].isin(loc_ok)].copy()
    y_flag = use_meta["Archetype"].eq(target).astype(float)

    covars = pd.get_dummies(use_meta[["Location", "Cohort"]].astype(str), drop_first=True, dtype=float)
    design = pd.DataFrame({"Intercept": 1.0, "target": y_flag}, index=use_meta.index)
    if not covars.empty:
        design = design.join(covars)
    nunique = design.nunique(dropna=False)
    design = design.loc[:, (nunique > 1) | (design.columns == "Intercept")]
    counts = {
        "n_samples": int(len(use_meta)),
        "n_target": int(y_flag.sum()),
        "n_rest": int(len(use_meta) - y_flag.sum()),
        "n_locations": int(len(loc_ok)),
    }
    return design.astype(float), y_flag.astype(bool), counts


def _one_vs_rest_de_from_design(
    expr: pd.DataFrame,
    design: pd.DataFrame,
    target_mask: pd.Series,
    counts: dict[str, int],
    target: str,
    *,
    location: str | None = None,
) -> list[dict[str, object]]:
    import statsmodels.api as sm

    rows = []
    x = expr.loc[design.index]
    for gene in x.columns:
        y = pd.to_numeric(x[gene], errors="coerce")
        valid = y.notna()
        if int(valid.sum()) <= design.loc[valid].shape[1]:
            continue
        fit = sm.OLS(y.loc[valid].astype(float), design.loc[valid]).fit()
        target_vals = y.loc[target_mask[target_mask].index].dropna()
        rest_vals = y.loc[target_mask[~target_mask].index].dropna()
        if len(target_vals) < 2 or len(rest_vals) < 2:
            continue
        _, ttest_p = stats.ttest_ind(target_vals, rest_vals, equal_var=False, nan_policy="omit")
        row = {
            "target_archetype": target,
            "target_short": ARCHETYPE_SHORT[target],
            "contrast": f"{ARCHETYPE_SHORT[target]} vs rest",
            "gene": gene,
            "coef": float(fit.params["target"]),
            "p": float(fit.pvalues["target"]),
            "ttest_p_unadjusted": float(ttest_p) if pd.notna(ttest_p) else np.nan,
            "mean_target": float(target_vals.mean()),
            "mean_rest": float(rest_vals.mean()),
            "diff_unadjusted": float(target_vals.mean() - rest_vals.mean()),
            **counts,
        }
        if location is not None:
            row["Location"] = location
        rows.append(row)
    return rows


def run_location_adjusted_one_vs_rest_de(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    min_per_group_within_location: int = 2,
) -> pd.DataFrame:
    """Gene-wise OLS DE for each archetype vs rest, adjusted for location and cohort."""
    rows = []
    meta = meta.reindex(expr.index).copy()
    for raw_target in targets:
        target = canonical_archetype(raw_target)
        design, target_mask, counts = _design_matrix(meta, target, min_per_group_within_location)
        if counts["n_target"] < 2 or counts["n_rest"] < 2 or design.shape[0] <= design.shape[1]:
            continue
        rows.extend(_one_vs_rest_de_from_design(expr, design, target_mask, counts, target))
    de = pd.DataFrame(rows)
    if de.empty:
        return de
    de["q"] = np.nan
    for target, idx in de.groupby("target_archetype").groups.items():
        de.loc[idx, "q"] = multipletests(de.loc[idx, "p"].fillna(1), method="fdr_bh")[1]
    return de.sort_values(["target_archetype", "q", "p", "coef"], ascending=[True, True, True, False])


def _design_matrix_for_single_location(meta: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    y_flag = meta["Archetype"].eq(target).astype(float)
    covars = pd.get_dummies(meta[["Cohort"]].astype(str), drop_first=True, dtype=float)
    design = pd.DataFrame({"Intercept": 1.0, "target": y_flag}, index=meta.index)
    if not covars.empty:
        design = design.join(covars)
    nunique = design.nunique(dropna=False)
    design = design.loc[:, (nunique > 1) | (design.columns == "Intercept")]
    counts = {
        "n_samples": int(len(meta)),
        "n_target": int(y_flag.sum()),
        "n_rest": int(len(meta) - y_flag.sum()),
        "n_locations": 1,
    }
    return design.astype(float), y_flag.astype(bool), counts


def run_within_location_one_vs_rest_de(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    locations: list[str] | tuple[str, ...] | None = tuple(LOCATION_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Gene-wise one-vs-rest DE separately within each anatomical location.

    Within each location, the model is ``expression ~ archetype_one_vs_rest + cohort``.
    FDR correction is applied independently for each location × archetype contrast.
    """
    rows = []
    meta = meta.reindex(expr.index).copy()
    location_values = list(locations) if locations is not None else sorted(meta["Location"].dropna().unique())
    for location in location_values:
        loc_meta = meta.loc[meta["Location"].eq(location)].copy()
        if loc_meta.empty:
            continue
        for raw_target in targets:
            target = canonical_archetype(raw_target)
            n_target = int(loc_meta["Archetype"].eq(target).sum())
            n_rest = int((~loc_meta["Archetype"].eq(target)).sum())
            if n_target < min_per_group or n_rest < min_per_group:
                continue
            design, target_mask, counts = _design_matrix_for_single_location(loc_meta, target)
            if counts["n_target"] < 2 or counts["n_rest"] < 2 or design.shape[0] <= design.shape[1]:
                continue
            rows.extend(_one_vs_rest_de_from_design(expr, design, target_mask, counts, target, location=location))

    de = pd.DataFrame(rows)
    if de.empty:
        return de
    de["q"] = np.nan
    for _, idx in de.groupby(["Location", "target_archetype"]).groups.items():
        de.loc[idx, "q"] = multipletests(de.loc[idx, "p"].fillna(1), method="fdr_bh")[1]
    return de.sort_values(
        ["Location", "target_archetype", "q", "p", "coef"],
        ascending=[True, True, True, True, False],
    )


def _hla_design_matrix(meta: pd.DataFrame, hla_col: str, archetype: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if archetype is not None:
        meta = meta.loc[meta["Archetype"].eq(archetype)].copy()
        covar_cols = ["Location"]
    else:
        covar_cols = ["Location", "Archetype"]
    status = clean_hla_call(meta[hla_col])
    use_meta = meta.loc[status.isin(["loss", "retained"])].copy()
    status = status.loc[use_meta.index]
    loss_flag = status.eq("loss").astype(float)
    covars = pd.get_dummies(use_meta[covar_cols].astype(str), drop_first=True, dtype=float)
    design = pd.DataFrame({"Intercept": 1.0, "hla_loss": loss_flag}, index=use_meta.index)
    if not covars.empty:
        design = design.join(covars)
    design = design.loc[:, (design.nunique(dropna=False) > 1) | (design.columns == "Intercept")]
    return design.astype(float), status


def run_discovery_hla_status_de(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    hla_cols: list[str] | tuple[str, ...] = ("HLAABC", "HLADR"),
    archetypes: list[str] | tuple[str, ...] | None = None,
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Discovery-only DEG for tumor HLA loss versus retained status.

    Positive coefficients and Hedges g indicate higher expression in HLA-loss
    (aberrant) cases. Overall contrasts adjust for location and archetype;
    within-archetype contrasts adjust for location.
    """
    import statsmodels.api as sm

    meta = meta.reindex(expr.index).copy()
    rows = []
    contrast_archetypes: list[str | None] = [None]
    if archetypes is not None:
        contrast_archetypes.extend([canonical_archetype(a) for a in archetypes])

    for hla_col in hla_cols:
        if hla_col not in meta.columns:
            continue
        for archetype in contrast_archetypes:
            design, status = _hla_design_matrix(meta, hla_col, archetype)
            n_loss = int(status.eq("loss").sum())
            n_retained = int(status.eq("retained").sum())
            if n_loss < min_per_group or n_retained < min_per_group or design.shape[0] <= design.shape[1]:
                continue
            x = expr.loc[design.index]
            loss_idx = status.loc[status.eq("loss")].index
            retained_idx = status.loc[status.eq("retained")].index
            for gene in x.columns:
                y = pd.to_numeric(x[gene], errors="coerce")
                valid = y.notna()
                if int(valid.sum()) <= design.loc[valid].shape[1]:
                    continue
                fit = sm.OLS(y.loc[valid].astype(float), design.loc[valid]).fit()
                loss_vals = y.loc[loss_idx].dropna()
                retained_vals = y.loc[retained_idx].dropna()
                if len(loss_vals) < min_per_group or len(retained_vals) < min_per_group:
                    continue
                t_stat, ttest_p = stats.ttest_ind(loss_vals, retained_vals, equal_var=False, nan_policy="omit")
                rows.append(
                    {
                        "hla_col": hla_col,
                        "hla_axis": "HLA-I" if hla_col == "HLAABC" else "HLA-II",
                        "archetype": archetype if archetype is not None else "overall",
                        "archetype_short": ARCHETYPE_SHORT.get(archetype, "all"),
                        "contrast": f"{hla_col} loss vs retained"
                        if archetype is None
                        else f"{hla_col} loss vs retained within {ARCHETYPE_SHORT[archetype]}",
                        "gene": gene,
                        "coef": float(fit.params["hla_loss"]),
                        "p": float(fit.pvalues["hla_loss"]),
                        "ttest_t": float(t_stat) if pd.notna(t_stat) else np.nan,
                        "ttest_p_unadjusted": float(ttest_p) if pd.notna(ttest_p) else np.nan,
                        "effect_g": _hedges_g(loss_vals, retained_vals),
                        "mean_loss": float(loss_vals.mean()),
                        "mean_retained": float(retained_vals.mean()),
                        "diff_unadjusted": float(loss_vals.mean() - retained_vals.mean()),
                        "n_loss": n_loss,
                        "n_retained": n_retained,
                        "n_samples": int(len(status)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = np.nan
    for _, idx in out.groupby(["hla_col", "archetype"]).groups.items():
        out.loc[idx, "q"] = multipletests(out.loc[idx, "p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["hla_col", "archetype", "q", "p", "coef"], ascending=[True, True, True, True, False])


def hla_de_summary(de: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Compact per-contrast summaries of strongest HLA-loss associated genes."""
    if de.empty:
        return de
    rows = []
    for (hla_col, archetype), sub in de.groupby(["hla_col", "archetype"], sort=False):
        up = sub.loc[sub["coef"] > 0].sort_values(["q", "p", "coef"], ascending=[True, True, False]).head(top_n)
        down = sub.loc[sub["coef"] < 0].sort_values(["q", "p", "coef"], ascending=[True, True, True]).head(top_n)
        for direction, piece in (("higher_in_loss", up), ("higher_in_retained", down)):
            for rank, (_, row) in enumerate(piece.iterrows(), start=1):
                rows.append(
                    {
                        "hla_col": hla_col,
                        "hla_axis": row["hla_axis"],
                        "archetype": archetype,
                        "archetype_short": row["archetype_short"],
                        "direction": direction,
                        "rank": rank,
                        "gene": row["gene"],
                        "coef": row["coef"],
                        "effect_g": row["effect_g"],
                        "p": row["p"],
                        "q": row["q"],
                        "n_loss": row["n_loss"],
                        "n_retained": row["n_retained"],
                        "marker_tags": marker_tags_for_gene(row["gene"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_hla_status_volcanoes(
    de: pd.DataFrame,
    out_dir: Path | str,
    *,
    q_th: float = 0.05,
    p_th: float = 0.05,
    effect_th: float = 0.25,
    y_metric: str = "q",
    top_n_label: int = 12,
    show: bool = True,
) -> list[Path]:
    """Write HLA loss-vs-retained volcanoes for overall and within-archetype DEG."""
    import matplotlib.pyplot as plt

    if y_metric not in {"q", "p"}:
        raise ValueError("y_metric must be 'q' or 'p'")
    y_th = q_th if y_metric == "q" else p_th
    y_label = "-log10(FDR)" if y_metric == "q" else "-log10(p)"
    sig_label = "FDR" if y_metric == "q" else "p"
    suffix = "" if y_metric == "q" else "_pvalue"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (hla_col, archetype), plot_df in de.groupby(["hla_col", "archetype"], sort=False):
        plot_df = plot_df.copy()
        if plot_df.empty:
            continue
        plot_df[f"minus_log10_{y_metric}"] = -np.log10(plot_df[y_metric].clip(lower=1e-300))
        sig = (plot_df[y_metric] < y_th) & (plot_df["coef"].abs() >= effect_th)
        color = HLA_STATUS_COLORS["loss"]
        arch_short = str(plot_df["archetype_short"].iloc[0])
        title_arch = "overall" if archetype == "overall" else f"within {arch_short}"

        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ax.scatter(
            plot_df.loc[~sig, "coef"],
            plot_df.loc[~sig, f"minus_log10_{y_metric}"],
            s=9,
            alpha=0.35,
            linewidths=0,
            color="lightgray",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] > 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] > 0), f"minus_log10_{y_metric}"],
            s=12,
            alpha=0.85,
            linewidths=0,
            color=color,
            label="higher in loss",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] < 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] < 0), f"minus_log10_{y_metric}"],
            s=12,
            alpha=0.75,
            linewidths=0,
            color=HLA_STATUS_COLORS["retained"],
            label="higher in retained",
        )
        ax.axvline(effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axvline(-effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axhline(-np.log10(y_th), linestyle="--", linewidth=1, color="black", alpha=0.45)

        label_df = plot_df.loc[sig].sort_values([y_metric, "p"]).head(top_n_label)
        for _, row in label_df.iterrows():
            ax.text(row["coef"], row[f"minus_log10_{y_metric}"], row["gene"], fontsize=7)

        n_loss = int(plot_df["n_loss"].iloc[0])
        n_retained = int(plot_df["n_retained"].iloc[0])
        ax.set_title(f"{hla_col} loss vs retained, {title_arch}")
        ax.set_xlabel("Adjusted coefficient (loss minus retained)")
        ax.set_ylabel(y_label)
        ax.text(
            0.02,
            0.02,
            f"n={n_loss} loss vs {n_retained} retained; {sig_label}<{y_th:g}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()

        safe_arch = "overall" if archetype == "overall" else arch_short
        stem = f"volcano_{hla_col}_{safe_arch}_loss_vs_retained{suffix}"
        svg = out / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show(fig)
        plt.close(fig)
        written.append(svg)
    return written


def plot_hla_class_group_volcanoes(
    de: pd.DataFrame,
    out_dir: Path | str,
    *,
    q_th: float = 0.05,
    p_th: float = 0.05,
    effect_th: float = 0.25,
    y_metric: str = "q",
    top_n_label: int = 12,
    show: bool = True,
) -> list[Path]:
    """Write four HLA class-group one-vs-rest volcanoes."""
    import matplotlib.pyplot as plt

    if y_metric not in {"q", "p"}:
        raise ValueError("y_metric must be 'q' or 'p'")
    y_th = q_th if y_metric == "q" else p_th
    y_label = "-log10(FDR)" if y_metric == "q" else "-log10(p)"
    sig_label = "FDR" if y_metric == "q" else "p"
    suffix = "" if y_metric == "q" else "_pvalue"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for group in HLA_CLASS_GROUP_ORDER:
        plot_df = de.loc[de["hla_class_group"].eq(group)].copy()
        if plot_df.empty:
            continue
        short = str(plot_df["hla_class_group_short"].iloc[0])
        plot_df[f"minus_log10_{y_metric}"] = -np.log10(plot_df[y_metric].clip(lower=1e-300))
        sig = (plot_df[y_metric] < y_th) & (plot_df["coef"].abs() >= effect_th)

        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ax.scatter(
            plot_df.loc[~sig, "coef"],
            plot_df.loc[~sig, f"minus_log10_{y_metric}"],
            s=9,
            alpha=0.35,
            linewidths=0,
            color="lightgray",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] > 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] > 0), f"minus_log10_{y_metric}"],
            s=12,
            alpha=0.85,
            linewidths=0,
            color="#7b3294",
            label=f"higher in {group}",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] < 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] < 0), f"minus_log10_{y_metric}"],
            s=12,
            alpha=0.75,
            linewidths=0,
            color="#4d4d4d",
            label="higher in rest",
        )
        ax.axvline(effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axvline(-effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axhline(-np.log10(y_th), linestyle="--", linewidth=1, color="black", alpha=0.45)

        label_df = plot_df.loc[sig].sort_values([y_metric, "p"]).head(top_n_label)
        for _, row in label_df.iterrows():
            ax.text(row["coef"], row[f"minus_log10_{y_metric}"], row["gene"], fontsize=7)

        n_group = int(plot_df["n_group"].iloc[0])
        n_rest = int(plot_df["n_rest"].iloc[0])
        ax.set_title(f"{group} vs rest")
        ax.set_xlabel("Adjusted coefficient (group minus rest)")
        ax.set_ylabel(y_label)
        ax.text(
            0.02,
            0.02,
            f"n={n_group} vs {n_rest}; {sig_label}<{y_th:g}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()

        stem = f"volcano_{short}_vs_rest{suffix}"
        svg = out / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show(fig)
        plt.close(fig)
        written.append(svg)
    return written


def _fov_metadata_frame(adata) -> pd.DataFrame:
    if "fov_metadata" not in adata.uns:
        raise KeyError("adata.uns['fov_metadata'] is required for cell/mm2 density calculations")
    fov_meta = pd.DataFrame(adata.uns["fov_metadata"]).copy()
    if "fov" in fov_meta.columns:
        fov_meta["fov"] = fov_meta["fov"].astype(str)
        fov_meta = fov_meta.drop_duplicates("fov").set_index("fov")
    fov_meta.index = fov_meta.index.astype(str)
    if "total_pixel" not in fov_meta.columns:
        if "total_pixel" in fov_meta.index:
            fov_meta = fov_meta.T
        else:
            raise KeyError("fov_metadata lacks 'total_pixel'")
    fov_meta["total_pixel"] = pd.to_numeric(fov_meta["total_pixel"], errors="coerce")
    return fov_meta


def compute_patient_phenotype_densities(
    adata,
    meta: pd.DataFrame,
    *,
    phenotype_col: str = "phenotype_30_clean",
    filtering_status_col: str = "filtering_status",
    filtering_keep_value: str = "Unfiltered",
    pixel_size_um: float = 1.0,
    patient_subset: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level phenotype densities in cells/mm2.

    Area is the sum of ``fov_metadata['total_pixel']`` for included FOVs, converted
    to mm2 as ``pixels * pixel_size_um**2 / 1e6``.
    """
    obs_cols = ["patient_id", "fov", phenotype_col]
    if filtering_status_col in adata.obs.columns:
        obs_cols.append(filtering_status_col)
    missing = [col for col in obs_cols if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing obs columns for phenotype density: {missing}")

    obs = adata.obs[obs_cols].copy()
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs["fov"] = obs["fov"].astype(str)
    obs[phenotype_col] = obs[phenotype_col].astype(str)
    if filtering_status_col in obs.columns:
        obs = obs.loc[obs[filtering_status_col].astype(str).eq(filtering_keep_value)].copy()
    if patient_subset is not None:
        obs = obs.loc[obs["patient_id"].isin([str(p) for p in patient_subset])].copy()

    fov_meta = _fov_metadata_frame(adata)
    patient_fov = obs[["patient_id", "fov"]].drop_duplicates()
    patient_fov = patient_fov.join(fov_meta[["total_pixel"]], on="fov")
    patient_area = patient_fov.groupby("patient_id", observed=False)["total_pixel"].sum(min_count=1)
    patient_area_mm2 = patient_area * (float(pixel_size_um) ** 2) / 1_000_000.0

    counts = (
        obs.groupby(["patient_id", phenotype_col], observed=False)
        .size()
        .rename("cell_count")
        .reset_index()
        .rename(columns={phenotype_col: "phenotype"})
    )
    patients = pd.Index(sorted(obs["patient_id"].dropna().unique()), name="patient_id")
    phenotypes = pd.Index(sorted(counts["phenotype"].dropna().unique()), name="phenotype")
    full_index = pd.MultiIndex.from_product([patients, phenotypes], names=["patient_id", "phenotype"])
    counts = counts.set_index(["patient_id", "phenotype"]).reindex(full_index, fill_value=0).reset_index()
    counts["area_mm2"] = counts["patient_id"].map(patient_area_mm2)
    counts["density_cells_per_mm2"] = counts["cell_count"] / counts["area_mm2"]

    meta_group = add_hla_class_group(meta)
    meta_cols = ["Location", "Archetype", "HLAABC", "HLADR", "hla_class_group", "hla_class_group_short"]
    meta_cols = [col for col in meta_cols if col in meta_group.columns]
    counts = counts.join(meta_group[meta_cols], on="patient_id")
    counts = counts.loc[counts["hla_class_group"].notna() & counts["area_mm2"].gt(0)].copy()

    area_table = (
        patient_area_mm2.rename("area_mm2")
        .to_frame()
        .join(meta_group[meta_cols], how="left")
        .reset_index()
    )
    return counts, area_table


def run_hla_class_group_density_tests(
    density_long: pd.DataFrame,
    *,
    value_col: str = "density_cells_per_mm2",
    group_col: str = "hla_class_group",
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Kruskal-Wallis phenotype-density tests across HLA class groups."""
    rows = []
    for phenotype, sub in density_long.groupby("phenotype", observed=False):
        vals = []
        medians = {}
        ns = {}
        for group in groups:
            v = pd.to_numeric(sub.loc[sub[group_col].astype("object").eq(group), value_col], errors="coerce").dropna()
            vals.append(v)
            medians[group] = float(v.median()) if len(v) else np.nan
            ns[group] = int(len(v))
        usable = [v for v in vals if len(v) >= min_per_group]
        if len(usable) < 2:
            continue
        h_stat, p_val = stats.kruskal(*usable, nan_policy="omit")
        row = {
            "phenotype": phenotype,
            "kruskal_H": float(h_stat) if pd.notna(h_stat) else np.nan,
            "p": float(p_val) if pd.notna(p_val) else np.nan,
            "max_group": max(medians, key=lambda g: -np.inf if pd.isna(medians[g]) else medians[g]),
            "max_median_density": np.nanmax(list(medians.values())),
            "min_group": min(medians, key=lambda g: np.inf if pd.isna(medians[g]) else medians[g]),
            "min_median_density": np.nanmin(list(medians.values())),
        }
        for group in groups:
            short = HLA_CLASS_GROUP_SHORT[group]
            row[f"median_{short}"] = medians[group]
            row[f"n_{short}"] = ns[group]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["q", "p", "kruskal_H"], ascending=[True, True, False])


def hla_class_group_density_matrix(
    density_long: pd.DataFrame,
    *,
    value_col: str = "density_cells_per_mm2",
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
) -> pd.DataFrame:
    """Median phenotype-density matrix with HLA class groups as columns."""
    mat = density_long.pivot_table(
        index="phenotype",
        columns="hla_class_group",
        values=value_col,
        aggfunc="median",
        observed=True,
    )
    return mat.reindex(columns=list(groups)).dropna(axis=1, how="all")


def _format_p_for_row_label(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value < 0.001:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _add_row_stat_text(
    ax,
    row_index: pd.Index,
    stats_table: pd.DataFrame | None,
    *,
    feature_col: str,
    x_start: float,
) -> None:
    """Print row-level p/FDR next to a heatmap without implying cell-level tests."""
    if stats_table is None or stats_table.empty:
        return
    stat_idx = stats_table.set_index(feature_col)
    for i, feature in enumerate(row_index):
        p = stat_idx.loc[feature, "p"] if feature in stat_idx.index and "p" in stat_idx.columns else np.nan
        q = stat_idx.loc[feature, "q"] if feature in stat_idx.index and "q" in stat_idx.columns else np.nan
        ax.text(x_start, i, f"p={_format_p_for_row_label(p)}", ha="left", va="center", fontsize=6.5)
        ax.text(x_start + 1.25, i, f"FDR={_format_p_for_row_label(q)}", ha="left", va="center", fontsize=6.5)


def plot_hla_class_group_density_heatmap(
    density_matrix: pd.DataFrame,
    stats_table: pd.DataFrame,
    out_path: Path | str,
    *,
    top_n: int = 30,
    q_th: float = 0.05,
    row_stats_table: pd.DataFrame | None = None,
    cmap_name: str = "PuOr_r",
    show: bool = True,
) -> Path:
    """Heatmap of phenotype density z-scores across four HLA class groups."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = stats_table.sort_values(["q", "p", "kruskal_H"], ascending=[True, True, False])["phenotype"].head(top_n)
    mat = density_matrix.reindex(order).dropna(how="all").dropna(axis=1, how="all")
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1, ddof=0).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.28 * len(z))))
    im = ax.imshow(z.to_numpy(dtype=float), aspect="auto", cmap=cmap_name, vmin=-2.0, vmax=2.0)
    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=7)
    ax.set_title("Immune phenotype density by tumor HLA class state", pad=26)
    ax.set_xlabel("Tumor HLA class state")
    ax.set_ylabel("Phenotype")
    _add_row_stat_text(
        ax,
        z.index,
        row_stats_table if row_stats_table is not None else stats_table,
        feature_col="phenotype",
        x_start=len(z.columns) - 0.15,
    )
    ax.set_xlim(-0.5, len(z.columns) + 2.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Row z-score of median cells/mm2")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def _significance_to_dot_sizes(
    significance: np.ndarray,
    *,
    min_size: float = 18.0,
    max_size: float = 240.0,
    max_score: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map p/FDR values to dot sizes via clipped ``-log10`` scores."""
    sig = np.asarray(significance, dtype=float)
    safe = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    scores = -np.log10(safe)
    scores = np.clip(scores, 0, max_score)
    sizes = min_size + (np.nan_to_num(scores, nan=0.0) / max_score) * (max_size - min_size)
    return sizes, scores


def _plot_hla_class_group_dotplot(
    value_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    out_path: Path | str,
    *,
    value_label: str,
    significance_label: str,
    title: str,
    cmap_name: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = True,
) -> Path:
    """Dotplot with color as value and dot size as p/FDR strength."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mat = value_matrix.dropna(axis=1, how="all").dropna(how="all")
    sigmat = significance_matrix.reindex(index=mat.index, columns=mat.columns)
    values = mat.to_numpy(dtype=float)
    sig_values = sigmat.to_numpy(dtype=float)
    finite_value = np.isfinite(values)
    sizes, sig_scores = _significance_to_dot_sizes(sig_values)
    max_score = float(np.nanmax(sig_scores)) if sig_scores.size and np.isfinite(sig_scores).any() else 1.0
    max_score = max(max_score, 1.3)

    if vmax is None:
        vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
        vmax = max(vmax, 0.5)
    if vmin is None:
        vmin = -vmax

    fig_w = max(8.2, 1.25 * max(len(mat.columns), 1) + 4.6)
    fig_h = max(3.8, 0.42 * max(len(mat.index), 1) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x, y = np.meshgrid(np.arange(len(mat.columns)), np.arange(len(mat.index)))
    cmap = plt.get_cmap(cmap_name)
    scatter = ax.scatter(
        x[finite_value],
        y[finite_value],
        c=values[finite_value],
        s=sizes[finite_value],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="#333333",
        linewidths=0.35,
    )
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlim(-0.6, len(mat.columns) - 0.4)
    ax.set_ylim(len(mat.index) - 0.4, -0.6)
    ax.set_xlabel("Tumor HLA class state")
    ax.set_ylabel("")
    ax.set_title(title, pad=26)
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(value_label)

    legend_scores = [1.3, 2.0, 3.0]
    if max_score >= 4.0:
        legend_scores.append(4.0)
    legend_scores = [s for s in legend_scores if s <= max(max_score, 1.3)]
    legend_sizes = 18.0 + (np.asarray(legend_scores) / 4.0) * (240.0 - 18.0)
    handles = [
        ax.scatter([], [], s=size, facecolor="#bdbdbd", edgecolor="#333333", linewidth=0.35)
        for size in legend_sizes
    ]
    labels = [f"{10 ** (-score):.0e}" for score in legend_scores]
    ax.legend(
        handles,
        labels,
        title=significance_label,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.34, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_hla_class_group_density_dotplot(
    density_matrix: pd.DataFrame,
    stats_table: pd.DataFrame,
    out_path: Path | str,
    *,
    top_n: int = 30,
    significance_col: str = "p",
    significance_label: str | None = None,
    show: bool = True,
) -> Path:
    """Phenotype density dotplot; color is row-z median density, size is p/FDR."""
    if significance_col not in stats_table.columns:
        raise KeyError(f"{significance_col!r} not found in density stats table")
    order_cols = [significance_col, "p", "kruskal_H"]
    order_cols = [col for col in order_cols if col in stats_table.columns]
    ascending = [True] * len(order_cols)
    if order_cols and order_cols[-1] == "kruskal_H":
        ascending[-1] = False
    order = stats_table.sort_values(order_cols, ascending=ascending)["phenotype"].head(top_n)
    mat = density_matrix.reindex(order).dropna(how="all").dropna(axis=1, how="all")
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1, ddof=0).replace(0, np.nan), axis=0).fillna(0)
    sig = stats_table.set_index("phenotype").reindex(z.index)[significance_col]
    sig_matrix = pd.DataFrame(
        np.tile(sig.to_numpy(dtype=float)[:, None], (1, len(z.columns))),
        index=z.index,
        columns=z.columns,
    )
    sig_label = significance_label or ("FDR" if significance_col == "q" else "Nominal p")
    return _plot_hla_class_group_dotplot(
        z,
        sig_matrix,
        out_path,
        value_label="Row z-score of median cells/mm2",
        significance_label=sig_label,
        title="Immune phenotype density by tumor HLA class state",
        vmin=-2.0,
        vmax=2.0,
        show=show,
    )


def compute_hla_class_group_program_effects(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Program one-vs-rest Hedges g and p-values for HLA class states."""
    meta = add_hla_class_group(meta.reindex(program_scores.index))
    use_meta = meta.loc[meta["hla_class_group"].notna()].copy()
    scores = program_scores.loc[use_meta.index]
    rows = []
    for group in groups:
        target_mask = use_meta["hla_class_group"].astype("object").eq(group)
        n_group = int(target_mask.sum())
        n_rest = int((~target_mask).sum())
        if n_group < min_per_group or n_rest < min_per_group:
            continue
        for program in scores.columns:
            target_vals = pd.to_numeric(scores.loc[target_mask, program], errors="coerce")
            rest_vals = pd.to_numeric(scores.loc[~target_mask, program], errors="coerce")
            t_stat, p_val = stats.ttest_ind(target_vals, rest_vals, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "hla_class_group": group,
                    "hla_class_group_short": HLA_CLASS_GROUP_SHORT[group],
                    "program": program,
                    "effect_g": _hedges_g(target_vals, rest_vals),
                    "p": float(p_val) if pd.notna(p_val) else np.nan,
                    "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "mean_group": float(target_vals.mean()),
                    "mean_rest": float(rest_vals.mean()),
                    "diff_unadjusted": float(target_vals.mean() - rest_vals.mean()),
                    "n_group": n_group,
                    "n_rest": n_rest,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["program", "hla_class_group"])


def hla_class_group_program_matrices(
    effects: pd.DataFrame,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    value_col: str = "effect_g",
    significance_col: str = "q",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return HLA-state program effect and significance matrices."""
    if value_col not in effects.columns:
        raise KeyError(f"{value_col!r} not found in HLA class program effects")
    if significance_col not in effects.columns:
        raise KeyError(f"{significance_col!r} not found in HLA class program effects")
    program_order = list(programs or HLA_IMMUNE_PROGRAM_GENE_SETS.keys())
    observed_groups = [g for g in groups if g in set(effects["hla_class_group"].astype("object"))]
    effect_matrix = (
        effects.pivot_table(index="program", columns="hla_class_group", values=value_col, aggfunc="first", observed=True)
        .reindex(index=program_order, columns=observed_groups)
        .dropna(axis=1, how="all")
    )
    sig_matrix = (
        effects.pivot_table(index="program", columns="hla_class_group", values=significance_col, aggfunc="first", observed=True)
        .reindex(index=program_order, columns=effect_matrix.columns)
    )
    return effect_matrix, sig_matrix


def hla_class_group_program_mean_matrix(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
) -> pd.DataFrame:
    """Mean program score matrix by HLA class state."""
    meta = add_hla_class_group(meta.reindex(program_scores.index))
    scores = program_scores.loc[meta.index].copy()
    scores["hla_class_group"] = meta["hla_class_group"].astype("object")
    program_order = list(programs or HLA_IMMUNE_PROGRAM_GENE_SETS.keys())
    available_programs = [program for program in program_order if program in program_scores.columns]
    observed_groups = [g for g in groups if scores["hla_class_group"].eq(g).any()]
    if not available_programs:
        return pd.DataFrame(index=program_order, columns=observed_groups, dtype=float)
    return (
        scores.groupby("hla_class_group", observed=True)[available_programs]
        .mean()
        .T
        .reindex(index=program_order, columns=observed_groups)
        .dropna(axis=1, how="all")
    )


def plot_hla_class_group_program_heatmap(
    value_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame | None,
    out_path: Path | str,
    *,
    significance_th: float = 0.05,
    significance_label: str = "FDR",
    mask_by_significance: bool = False,
    value_label: str = "Mean z-scored expression",
    title: str = "HLA and immune program scores by tumor HLA class state",
    row_stats_table: pd.DataFrame | None = None,
    row_feature_col: str = "program",
    cmap_name: str = "PuOr_r",
    show: bool = True,
) -> Path:
    """Program score heatmap across observed HLA class states.

    By default, all finite values are shown so the heatmap matches the
    individual score boxplots. Set ``mask_by_significance=True`` for exploratory
    effect-style views that grey out nonsignificant cells.
    """
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mat = value_matrix.dropna(axis=1, how="all")
    values = mat.to_numpy(dtype=float)
    mask = ~np.isfinite(values)
    if mask_by_significance:
        if significance_matrix is None:
            raise ValueError("significance_matrix is required when mask_by_significance=True")
        sigmat = significance_matrix.reindex(index=mat.index, columns=mat.columns)
        mask = mask | ~(sigmat.to_numpy(dtype=float) < significance_th)
    masked_values = np.ma.array(values, mask=mask)

    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.38 * len(mat))))
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Tumor HLA class state")
    ax.set_ylabel("Curated program")
    _add_row_stat_text(
        ax,
        mat.index,
        row_stats_table,
        feature_col=row_feature_col,
        x_start=len(mat.columns) - 0.15,
    )
    ax.set_xlim(-0.5, len(mat.columns) + 2.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    grey_label = "unavailable"
    if mask_by_significance:
        grey_label = f"{significance_label} >= {significance_th:g}"
    cbar.set_label(f"{value_label} (grey: {grey_label})")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_hla_class_group_program_dotplot(
    value_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    out_path: Path | str,
    *,
    significance_label: str = "Nominal p",
    value_label: str = "Mean z-scored expression",
    title: str = "HLA and immune program mean expression by tumor HLA class state",
    show: bool = True,
) -> Path:
    """Program dotplot; color is program value and dot size is p/FDR."""
    return _plot_hla_class_group_dotplot(
        value_matrix,
        significance_matrix,
        out_path,
        value_label=value_label,
        significance_label=significance_label,
        title=title,
        show=show,
    )


def hla_class_group_program_score_long(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Long-form HLA class-state program scores for boxplots."""
    meta_group = add_hla_class_group(meta.reindex(program_scores.index))
    program_order = [p for p in list(programs or HLA_IMMUNE_PROGRAM_GENE_SETS.keys()) if p in program_scores.columns]
    if not program_order:
        return pd.DataFrame(
            columns=["patient_id", "program", "score", "hla_class_group", "hla_class_group_short", "Location", "Archetype"]
        )
    scores = program_scores.loc[meta_group.index, program_order].copy()
    scores.index.name = "patient_id"
    long = scores.reset_index().melt(id_vars="patient_id", var_name="program", value_name="score")
    meta_cols = ["hla_class_group", "hla_class_group_short", "Location", "Archetype"]
    meta_cols = [col for col in meta_cols if col in meta_group.columns]
    long = long.join(meta_group[meta_cols], on="patient_id")
    long = long.loc[long["hla_class_group"].notna() & long["score"].notna()].copy()
    long["program"] = pd.Categorical(long["program"], categories=program_order, ordered=True)
    return long


def run_hla_class_group_feature_tests(
    plot_long: pd.DataFrame,
    *,
    feature_col: str,
    value_col: str,
    group_col: str = "hla_class_group",
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Kruskal-Wallis tests for a long-form feature table across HLA class states."""
    rows = []
    for feature, sub in plot_long.groupby(feature_col, observed=False):
        vals = []
        medians = {}
        ns = {}
        for group in groups:
            v = pd.to_numeric(sub.loc[sub[group_col].astype("object").eq(group), value_col], errors="coerce").dropna()
            vals.append(v)
            medians[group] = float(v.median()) if len(v) else np.nan
            ns[group] = int(len(v))
        usable = [v for v in vals if len(v) >= min_per_group]
        if len(usable) < 2:
            continue
        h_stat, p_val = stats.kruskal(*usable, nan_policy="omit")
        row = {
            feature_col: feature,
            "kruskal_H": float(h_stat) if pd.notna(h_stat) else np.nan,
            "p": float(p_val) if pd.notna(p_val) else np.nan,
            "max_group": max(medians, key=lambda g: -np.inf if pd.isna(medians[g]) else medians[g]),
            "max_median": np.nanmax(list(medians.values())),
            "min_group": min(medians, key=lambda g: np.inf if pd.isna(medians[g]) else medians[g]),
            "min_median": np.nanmin(list(medians.values())),
        }
        for group in groups:
            short = HLA_CLASS_GROUP_SHORT[group]
            row[f"median_{short}"] = medians[group]
            row[f"n_{short}"] = ns[group]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["q", "p", "kruskal_H"], ascending=[True, True, False])


def _format_plot_pvalue(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def plot_hla_class_group_boxplots(
    plot_long: pd.DataFrame,
    stats_table: pd.DataFrame,
    features_to_plot: list[str] | tuple[str, ...],
    out_path: Path | str,
    *,
    feature_col: str,
    value_col: str,
    ylabel: str,
    suptitle: str,
    groups: list[str] | tuple[str, ...] = tuple(HLA_CLASS_GROUP_ORDER),
    colors: dict[str, str] | None = None,
    ncols: int = 3,
    figsize_per: tuple[float, float] = (4.2, 3.8),
    seed: int = 42,
    jitter_width: float = 0.09,
    box_width: float = 0.55,
    show: bool = True,
) -> Path:
    """Individual HLA class-state boxplots with overlaid patient points."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = colors or HLA_CLASS_GROUP_COLORS
    observed_groups = [g for g in groups if plot_long["hla_class_group"].astype("object").eq(g).any()]
    features = [f for f in features_to_plot if f in set(plot_long[feature_col].astype("object"))]
    if not features or not observed_groups:
        return out

    ncols = max(1, min(int(ncols), len(features)))
    nrows = int(np.ceil(len(features) / ncols))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        sharey=False,
    )
    axes = np.array(axes).reshape(-1)
    stat_idx = stats_table.set_index(feature_col) if feature_col in stats_table.columns and not stats_table.empty else pd.DataFrame()

    for ax, feature in zip(axes, features):
        sub = plot_long.loc[plot_long[feature_col].astype("object").eq(feature)].copy()
        x = np.arange(len(observed_groups))
        box_data = [
            pd.to_numeric(sub.loc[sub["hla_class_group"].astype("object").eq(group), value_col], errors="coerce").dropna().values
            for group in observed_groups
        ]
        bp = ax.boxplot(
            box_data,
            positions=x,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"linewidth": 0.8, "edgecolor": "black"},
        )
        for patch, group in zip(bp["boxes"], observed_groups):
            patch.set_facecolor(colors.get(group, "#9ca3af"))
            patch.set_alpha(0.65)

        for i, group in enumerate(observed_groups):
            vals = pd.to_numeric(sub.loc[sub["hla_class_group"].astype("object").eq(group), value_col], errors="coerce").dropna().values
            jitter = rng.normal(0, jitter_width, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=30,
                color=colors.get(group, "#9ca3af"),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )

        if not stat_idx.empty and feature in stat_idx.index:
            row = stat_idx.loc[feature]
            title = f"{feature}\np {_format_plot_pvalue(row['p'])} | FDR {_format_plot_pvalue(row['q'])}"
        else:
            title = str(feature)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(observed_groups, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)

    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_one_vs_rest_volcanoes(
    de: pd.DataFrame,
    out_dir: Path | str,
    *,
    q_th: float = 0.05,
    effect_th: float = 0.25,
    top_n_label: int = 12,
    show: bool = True,
) -> list[Path]:
    """Write one volcano SVG/PNG per archetype; returns SVG paths."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for target in ARCHETYPE_ORDER:
        plot_df = de.loc[de["target_archetype"] == target].copy()
        if plot_df.empty:
            continue
        short = ARCHETYPE_SHORT[target]
        color = ARCHETYPE_COLORS[target]
        plot_df["minus_log10_q"] = -np.log10(plot_df["q"].clip(lower=1e-300))
        sig = (plot_df["q"] < q_th) & (plot_df["coef"].abs() >= effect_th)

        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ax.scatter(
            plot_df.loc[~sig, "coef"],
            plot_df.loc[~sig, "minus_log10_q"],
            s=9,
            alpha=0.35,
            linewidths=0,
            color="lightgray",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] > 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] > 0), "minus_log10_q"],
            s=12,
            alpha=0.85,
            linewidths=0,
            color=color,
            label=f"higher in {short}",
        )
        ax.scatter(
            plot_df.loc[sig & (plot_df["coef"] < 0), "coef"],
            plot_df.loc[sig & (plot_df["coef"] < 0), "minus_log10_q"],
            s=12,
            alpha=0.75,
            linewidths=0,
            color="#4d4d4d",
            label="higher in rest",
        )
        ax.axvline(effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axvline(-effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
        ax.axhline(-np.log10(q_th), linestyle="--", linewidth=1, color="black", alpha=0.45)

        label_df = plot_df.loc[sig].sort_values(["q", "p"]).head(top_n_label)
        for _, row in label_df.iterrows():
            ax.text(row["coef"], row["minus_log10_q"], row["gene"], fontsize=7)

        n_target = int(plot_df["n_target"].iloc[0])
        n_rest = int(plot_df["n_rest"].iloc[0])
        n_locations = int(plot_df["n_locations"].iloc[0])
        ax.set_title(f"{short} vs rest within location")
        ax.set_xlabel("Location/cohort-adjusted coefficient")
        ax.set_ylabel("-log10(FDR)")
        ax.text(
            0.02,
            0.02,
            f"n={n_target} vs {n_rest}; {n_locations} locations",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()

        stem = f"volcano_{short}_vs_rest_location_adjusted"
        svg = out / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show(fig)
        plt.close(fig)
        written.append(svg)
    return written


def plot_within_location_volcanoes(
    de: pd.DataFrame,
    out_dir: Path | str,
    *,
    locations: list[str] | tuple[str, ...] | None = tuple(LOCATION_ORDER),
    q_th: float = 0.05,
    p_th: float = 0.05,
    effect_th: float = 0.25,
    y_metric: str = "q",
    top_n_label: int = 12,
    filename_suffix: str | None = None,
    show: bool = True,
) -> list[Path]:
    """Write volcano SVG/PNG files for each location × archetype contrast."""
    import matplotlib.pyplot as plt

    if y_metric not in {"q", "p"}:
        raise ValueError("y_metric must be 'q' or 'p'")
    y_th = q_th if y_metric == "q" else p_th
    y_label = "-log10(FDR)" if y_metric == "q" else "-log10(p)"
    sig_label = "FDR" if y_metric == "q" else "p"
    suffix = filename_suffix
    if suffix is None:
        suffix = "" if y_metric == "q" else "_pvalue"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    location_values = list(locations) if locations is not None else sorted(de["Location"].dropna().unique())
    for location in location_values:
        for target in ARCHETYPE_ORDER:
            plot_df = de.loc[(de["Location"] == location) & (de["target_archetype"] == target)].copy()
            if plot_df.empty:
                continue
            short = ARCHETYPE_SHORT[target]
            color = ARCHETYPE_COLORS[target]
            plot_df[f"minus_log10_{y_metric}"] = -np.log10(plot_df[y_metric].clip(lower=1e-300))
            sig = (plot_df[y_metric] < y_th) & (plot_df["coef"].abs() >= effect_th)

            fig, ax = plt.subplots(figsize=(5.2, 4.5))
            ax.scatter(
                plot_df.loc[~sig, "coef"],
                plot_df.loc[~sig, f"minus_log10_{y_metric}"],
                s=9,
                alpha=0.35,
                linewidths=0,
                color="lightgray",
            )
            ax.scatter(
                plot_df.loc[sig & (plot_df["coef"] > 0), "coef"],
                plot_df.loc[sig & (plot_df["coef"] > 0), f"minus_log10_{y_metric}"],
                s=12,
                alpha=0.85,
                linewidths=0,
                color=color,
                label=f"higher in {short}",
            )
            ax.scatter(
                plot_df.loc[sig & (plot_df["coef"] < 0), "coef"],
                plot_df.loc[sig & (plot_df["coef"] < 0), f"minus_log10_{y_metric}"],
                s=12,
                alpha=0.75,
                linewidths=0,
                color="#4d4d4d",
                label="higher in rest",
            )
            ax.axvline(effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
            ax.axvline(-effect_th, linestyle="--", linewidth=1, color="black", alpha=0.45)
            ax.axhline(-np.log10(y_th), linestyle="--", linewidth=1, color="black", alpha=0.45)

            label_df = plot_df.loc[sig].sort_values([y_metric, "p"]).head(top_n_label)
            for _, row in label_df.iterrows():
                ax.text(row["coef"], row[f"minus_log10_{y_metric}"], row["gene"], fontsize=7)

            n_target = int(plot_df["n_target"].iloc[0])
            n_rest = int(plot_df["n_rest"].iloc[0])
            ax.set_title(f"{location}: {short} vs rest")
            ax.set_xlabel("Cohort-adjusted coefficient")
            ax.set_ylabel(y_label)
            ax.text(
                0.02,
                0.02,
                f"n={n_target} vs {n_rest}; {sig_label}<{y_th:g}",
                transform=ax.transAxes,
                fontsize=8,
                ha="left",
                va="bottom",
            )
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            fig.tight_layout()

            safe_location = str(location).replace(" ", "_")
            stem = f"volcano_{safe_location}_{short}_vs_rest{suffix}"
            svg = out / f"{stem}.svg"
            fig.savefig(svg, bbox_inches="tight")
            fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
            if show:
                plt.show(fig)
            plt.close(fig)
            written.append(svg)
    return written


def marker_tags_for_gene(gene: object, marker_sets: dict[str, set[str]] | None = None) -> str:
    symbol = str(gene).strip().upper()
    sets = marker_sets or MARKER_GENE_SETS
    tags = [name for name, genes in sets.items() if symbol in genes]
    return ";".join(tags)


def cp_up_gene_location_rankings(
    de: pd.DataFrame,
    *,
    top_n: int = 20,
    q_th: float = 0.05,
    effect_th: float = 0.0,
    significant_only: bool = False,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank CP-up genes by location and summarize shared vs location-specific signal.

    Returns ``(top_long, summary, coef_matrix)`` where ``summary`` is restricted to
    the union of the top ``top_n`` CP-up genes from each location.
    """
    cp = de.loc[de["target_short"].eq("CP")].copy()
    cp = cp.loc[cp["Location"].isin(locations)].copy()
    cp["rank_score"] = cp["coef"] * -np.log10(cp["q"].clip(lower=1e-300))
    cp["up"] = cp["coef"] > effect_th
    cp["up_sig"] = cp["up"] & (cp["q"] < q_th)
    cp["eligible_top"] = cp["up_sig"] if significant_only else cp["up"]
    cp = cp.sort_values(["Location", "eligible_top", "rank_score", "coef"], ascending=[True, False, False, False])
    cp["rank_all"] = cp.groupby("Location").cumcount() + 1
    cp["rank_up"] = np.nan
    up_idx = cp.loc[cp["eligible_top"]].index
    cp.loc[up_idx, "rank_up"] = cp.loc[up_idx].groupby("Location").cumcount() + 1

    top_long = (
        cp.loc[cp["eligible_top"] & cp["rank_up"].le(top_n)]
        .sort_values(["Location", "rank_up"])
        .copy()
    )
    top_genes = pd.Index(top_long["gene"].drop_duplicates())

    coef_matrix = (
        cp.loc[cp["gene"].isin(top_genes)]
        .pivot_table(index="gene", columns="Location", values="coef", aggfunc="first")
        .reindex(columns=list(locations))
    )
    q_matrix = (
        cp.loc[cp["gene"].isin(top_genes)]
        .pivot_table(index="gene", columns="Location", values="q", aggfunc="first")
        .reindex(columns=list(locations))
    )
    rank_matrix = (
        cp.loc[cp["gene"].isin(top_genes)]
        .pivot_table(index="gene", columns="Location", values="rank_up", aggfunc="first")
        .reindex(columns=list(locations))
    )
    up_matrix = (coef_matrix > effect_th) & (q_matrix < q_th)
    positive_matrix = coef_matrix > effect_th

    summary = pd.DataFrame(index=coef_matrix.index)
    summary["gene"] = summary.index
    summary["n_locations_cp_up"] = up_matrix.sum(axis=1).astype(int)
    summary["n_locations_positive_coef"] = positive_matrix.sum(axis=1).astype(int)
    summary["locations_cp_up"] = [
        ",".join([loc for loc in locations if bool(up_matrix.loc[gene, loc])])
        for gene in summary.index
    ]
    summary["locations_positive_coef"] = [
        ",".join([loc for loc in locations if bool(positive_matrix.loc[gene, loc])])
        for gene in summary.index
    ]
    summary["mean_coef"] = coef_matrix.mean(axis=1)
    summary["coef_sd"] = coef_matrix.std(axis=1, ddof=1)
    summary["coef_range"] = coef_matrix.max(axis=1) - coef_matrix.min(axis=1)
    summary["max_location"] = coef_matrix.idxmax(axis=1)
    summary["max_coef"] = coef_matrix.max(axis=1)
    summary["pcns_coef"] = coef_matrix["PCNS"] if "PCNS" in coef_matrix.columns else np.nan
    other_cols = [c for c in coef_matrix.columns if c != "PCNS"]
    summary["pcns_minus_max_other"] = summary["pcns_coef"] - coef_matrix[other_cols].max(axis=1)
    summary["pcns_rank_up"] = rank_matrix["PCNS"] if "PCNS" in rank_matrix.columns else np.nan
    summary["marker_tags"] = summary["gene"].map(marker_tags_for_gene)
    summary = summary.sort_values(
        ["n_locations_cp_up", "coef_range", "mean_coef"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return top_long, summary, coef_matrix


def plot_cp_top_gene_location_heatmap(
    coef_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path | str,
    *,
    max_genes: int = 60,
    show: bool = True,
) -> Path:
    """Heatmap of CP-vs-rest coefficients for union of top location-ranked genes."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = summary.head(max_genes)["gene"].tolist()
    mat = coef_matrix.reindex(order).dropna(how="all")
    labels = []
    summary_idx = summary.set_index("gene")
    for gene in mat.index:
        tags = summary_idx.loc[gene, "marker_tags"]
        suffix = f" [{tags}]" if isinstance(tags, str) and tags else ""
        labels.append(f"{gene}{suffix}")

    fig_h = max(5.0, 0.22 * len(mat))
    fig, ax = plt.subplots(figsize=(6.6, fig_h))
    vmax = float(np.nanmax(np.abs(mat.values))) if mat.size else 1.0
    vmax = max(vmax, 0.5)
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("CP vs rest UP-gene coefficients by location")
    ax.set_xlabel("Anatomical location")
    ax.set_ylabel("Union of top CP-up genes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Cohort-adjusted coefficient")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def archetype_location_variance_summary(
    de: pd.DataFrame,
    *,
    target_short: str,
    top_n: int = 20,
    q_th: float = 0.05,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    require_positive_anywhere: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top genes whose one-vs-rest coefficients vary most between locations.

    Returns ``(summary, coef_matrix, q_matrix)`` for one archetype. The summary is
    restricted to the top ``top_n`` genes ranked by coefficient standard deviation
    across locations, with coefficient range used as a tie-breaker.
    """
    short = str(target_short).upper()
    sub = de.loc[de["target_short"].astype(str).str.upper().eq(short)].copy()
    sub = sub.loc[sub["Location"].isin(locations)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    coef_matrix = (
        sub.pivot_table(index="gene", columns="Location", values="coef", aggfunc="first")
        .reindex(columns=list(locations))
    )
    q_matrix = (
        sub.pivot_table(index="gene", columns="Location", values="q", aggfunc="first")
        .reindex(columns=list(locations))
    )
    coef_matrix = coef_matrix.dropna(how="all")
    q_matrix = q_matrix.reindex(index=coef_matrix.index)
    if require_positive_anywhere:
        coef_matrix = coef_matrix.loc[(coef_matrix > 0).any(axis=1)]
        q_matrix = q_matrix.reindex(index=coef_matrix.index)

    sig_matrix = q_matrix < q_th
    summary = pd.DataFrame(index=coef_matrix.index)
    summary["target_short"] = short
    summary["gene"] = summary.index
    summary["coef_mean"] = coef_matrix.mean(axis=1)
    summary["coef_sd"] = coef_matrix.std(axis=1, ddof=1)
    summary["coef_range"] = coef_matrix.max(axis=1) - coef_matrix.min(axis=1)
    summary["max_location"] = coef_matrix.idxmax(axis=1)
    summary["max_coef"] = coef_matrix.max(axis=1)
    summary["min_location"] = coef_matrix.idxmin(axis=1)
    summary["min_coef"] = coef_matrix.min(axis=1)
    summary["n_locations_significant"] = sig_matrix.sum(axis=1).astype(int)
    summary["locations_significant"] = [
        ",".join([loc for loc in locations if bool(sig_matrix.loc[gene, loc])])
        for gene in summary.index
    ]
    summary["n_locations_positive"] = (coef_matrix > 0).sum(axis=1).astype(int)
    summary["locations_positive"] = [
        ",".join([loc for loc in locations if bool(coef_matrix.loc[gene, loc] > 0)])
        for gene in summary.index
    ]
    summary["marker_tags"] = summary["gene"].map(marker_tags_for_gene)
    summary = summary.sort_values(["coef_sd", "coef_range", "max_coef"], ascending=[False, False, False])
    summary = summary.head(top_n).reset_index(drop=True)
    top_genes = summary["gene"].tolist()
    return summary, coef_matrix.reindex(top_genes), q_matrix.reindex(top_genes)


def _hedges_g(target_vals: pd.Series, rest_vals: pd.Series) -> float:
    target_vals = pd.to_numeric(target_vals, errors="coerce").dropna()
    rest_vals = pd.to_numeric(rest_vals, errors="coerce").dropna()
    n_target = len(target_vals)
    n_rest = len(rest_vals)
    if n_target < 2 or n_rest < 2:
        return np.nan
    df = n_target + n_rest - 2
    var_target = target_vals.var(ddof=1)
    var_rest = rest_vals.var(ddof=1)
    pooled = ((n_target - 1) * var_target + (n_rest - 1) * var_rest) / df
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    d = (target_vals.mean() - rest_vals.mean()) / np.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    return float(d * correction)


def compute_within_location_standardized_effects(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    de: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
) -> pd.DataFrame:
    """Compute Hedges g target-vs-rest effect sizes within each location.

    Input expression is already cohort-standardized by
    ``build_combined_transcriptome_inputs``. FDR/q-values from the matching
    within-location DE table are joined for heatmap masking and statistical context.
    """
    meta = meta.reindex(expr.index).copy()
    q_lookup_cols = ["Location", "target_short", "gene", "p", "q", "coef", "n_target", "n_rest"]
    q_lookup = de[[c for c in q_lookup_cols if c in de.columns]].copy()
    rows = []
    for location in locations:
        loc_meta = meta.loc[meta["Location"].eq(location)].copy()
        if loc_meta.empty:
            continue
        loc_expr = expr.loc[loc_meta.index]
        for raw_target in targets:
            target = canonical_archetype(raw_target)
            short = ARCHETYPE_SHORT[target]
            target_mask = loc_meta["Archetype"].eq(target)
            n_target = int(target_mask.sum())
            n_rest = int((~target_mask).sum())
            if n_target < 2 or n_rest < 2:
                continue
            for gene in loc_expr.columns:
                rows.append(
                    {
                        "Location": location,
                        "target_archetype": target,
                        "target_short": short,
                        "gene": gene,
                        "effect_g": _hedges_g(
                            loc_expr.loc[target_mask, gene],
                            loc_expr.loc[~target_mask, gene],
                        ),
                        "n_target_effect": n_target,
                        "n_rest_effect": n_rest,
                    }
                )
    effects = pd.DataFrame(rows)
    if effects.empty:
        return effects
    effects = effects.merge(q_lookup, on=["Location", "target_short", "gene"], how="left", suffixes=("", "_de"))
    return effects


def standardized_effect_location_variance_summary(
    effects: pd.DataFrame,
    *,
    target_short: str,
    top_n: int = 20,
    q_th: float = 0.05,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    require_positive_anywhere: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank genes by cross-location variance of standardized effect sizes."""
    short = str(target_short).upper()
    sub = effects.loc[effects["target_short"].astype(str).str.upper().eq(short)].copy()
    sub = sub.loc[sub["Location"].isin(locations)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    effect_matrix = (
        sub.pivot_table(index="gene", columns="Location", values="effect_g", aggfunc="first")
        .reindex(columns=list(locations))
    )
    q_matrix = (
        sub.pivot_table(index="gene", columns="Location", values="q", aggfunc="first")
        .reindex(columns=list(locations))
    )
    effect_matrix = effect_matrix.dropna(how="all")
    q_matrix = q_matrix.reindex(index=effect_matrix.index)
    if require_positive_anywhere:
        effect_matrix = effect_matrix.loc[(effect_matrix > 0).any(axis=1)]
        q_matrix = q_matrix.reindex(index=effect_matrix.index)

    sig_matrix = q_matrix < q_th
    summary = pd.DataFrame(index=effect_matrix.index)
    summary["target_short"] = short
    summary["gene"] = summary.index
    summary["effect_mean"] = effect_matrix.mean(axis=1)
    summary["effect_sd"] = effect_matrix.std(axis=1, ddof=1)
    summary["effect_range"] = effect_matrix.max(axis=1) - effect_matrix.min(axis=1)
    summary["max_location"] = effect_matrix.idxmax(axis=1)
    summary["max_effect"] = effect_matrix.max(axis=1)
    summary["min_location"] = effect_matrix.idxmin(axis=1)
    summary["min_effect"] = effect_matrix.min(axis=1)
    summary["n_locations_significant"] = sig_matrix.sum(axis=1).astype(int)
    summary["locations_significant"] = [
        ",".join([loc for loc in locations if bool(sig_matrix.loc[gene, loc])])
        for gene in summary.index
    ]
    summary["n_locations_positive"] = (effect_matrix > 0).sum(axis=1).astype(int)
    summary["locations_positive"] = [
        ",".join([loc for loc in locations if bool(effect_matrix.loc[gene, loc] > 0)])
        for gene in summary.index
    ]
    summary["marker_tags"] = summary["gene"].map(marker_tags_for_gene)
    summary = summary.sort_values(["effect_sd", "effect_range", "max_effect"], ascending=[False, False, False])
    summary = summary.head(top_n).reset_index(drop=True)
    top_genes = summary["gene"].tolist()
    return summary, effect_matrix.reindex(top_genes), q_matrix.reindex(top_genes)


def standardized_effect_location_variance_summaries(
    effects: pd.DataFrame,
    *,
    top_n: int = 20,
    q_th: float = 0.05,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "LO", "CP"),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run standardized-effect variance ranking for all requested archetypes."""
    summaries = []
    effect_matrices: dict[str, pd.DataFrame] = {}
    q_matrices: dict[str, pd.DataFrame] = {}
    for short in target_shorts:
        summary, effect_matrix, q_matrix = standardized_effect_location_variance_summary(
            effects,
            target_short=short,
            top_n=top_n,
            q_th=q_th,
            locations=locations,
        )
        if not summary.empty:
            summaries.append(summary)
        effect_matrices[str(short).upper()] = effect_matrix
        q_matrices[str(short).upper()] = q_matrix
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return all_summary, effect_matrices, q_matrices


def standardized_effect_significance_matrices(
    effects: pd.DataFrame,
    *,
    significance_col: str = "p",
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "LO", "CP"),
) -> dict[str, pd.DataFrame]:
    """Return per-archetype matrices of p/q values matching effect heatmaps."""
    if significance_col not in effects.columns:
        raise KeyError(f"{significance_col!r} not found in standardized effects")
    matrices: dict[str, pd.DataFrame] = {}
    for short in target_shorts:
        sub = effects.loc[effects["target_short"].astype(str).str.upper().eq(str(short).upper())].copy()
        matrices[str(short).upper()] = (
            sub.pivot_table(index="gene", columns="Location", values=significance_col, aggfunc="first")
            .reindex(columns=list(locations))
        )
    return matrices


def run_archetype_location_interaction_tests(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
) -> pd.DataFrame:
    """Omnibus gene-wise test for archetype × location interaction.

    Full model: ``expression ~ target + cohort + location + target:location``.
    Reduced model removes the interaction terms. P-values are F-tests comparing
    full and reduced OLS models, with FDR corrected within each archetype.
    """
    import statsmodels.api as sm

    meta = meta.reindex(expr.index).copy()
    meta = meta.loc[meta["Location"].isin(locations)].copy()
    expr = expr.loc[meta.index]
    rows = []
    loc_dummies = pd.get_dummies(meta["Location"].astype(str), drop_first=True, dtype=float)
    cohort_dummies = pd.get_dummies(meta["Cohort"].astype(str), drop_first=True, dtype=float)
    base_covars = pd.concat([cohort_dummies, loc_dummies], axis=1)

    for raw_target in targets:
        target = canonical_archetype(raw_target)
        short = ARCHETYPE_SHORT[target]
        target_flag = meta["Archetype"].eq(target).astype(float).rename("target")
        if target_flag.sum() < 2 or (1 - target_flag).sum() < 2:
            continue
        interaction = loc_dummies.mul(target_flag, axis=0)
        interaction = interaction.rename(columns={c: f"target_x_{c}" for c in interaction.columns})
        reduced = pd.concat(
            [pd.Series(1.0, index=meta.index, name="Intercept"), target_flag, base_covars],
            axis=1,
        )
        full = pd.concat([reduced, interaction], axis=1)
        reduced = reduced.loc[:, (reduced.nunique(dropna=False) > 1) | (reduced.columns == "Intercept")]
        full = full.loc[:, (full.nunique(dropna=False) > 1) | (full.columns == "Intercept")]
        if interaction.empty or full.shape[1] <= reduced.shape[1]:
            continue
        for gene in expr.columns:
            y = pd.to_numeric(expr[gene], errors="coerce")
            valid = y.notna()
            if int(valid.sum()) <= full.loc[valid].shape[1]:
                continue
            full_fit = sm.OLS(y.loc[valid].astype(float), full.loc[valid]).fit()
            reduced_fit = sm.OLS(y.loc[valid].astype(float), reduced.loc[valid]).fit()
            f_stat, p_val, df_diff = full_fit.compare_f_test(reduced_fit)
            rows.append(
                {
                    "target_archetype": target,
                    "target_short": short,
                    "gene": gene,
                    "interaction_F": float(f_stat) if pd.notna(f_stat) else np.nan,
                    "interaction_p": float(p_val) if pd.notna(p_val) else np.nan,
                    "interaction_df": float(df_diff) if pd.notna(df_diff) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["interaction_q"] = np.nan
    for _, idx in out.groupby("target_short").groups.items():
        out.loc[idx, "interaction_q"] = multipletests(out.loc[idx, "interaction_p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["target_short", "interaction_q", "interaction_p"])


def archetype_location_variance_summaries(
    de: pd.DataFrame,
    *,
    top_n: int = 20,
    q_th: float = 0.05,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "LO", "CP"),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run location-variance ranking for all requested archetypes."""
    summaries = []
    coef_matrices: dict[str, pd.DataFrame] = {}
    q_matrices: dict[str, pd.DataFrame] = {}
    for short in target_shorts:
        summary, coef_matrix, q_matrix = archetype_location_variance_summary(
            de,
            target_short=short,
            top_n=top_n,
            q_th=q_th,
            locations=locations,
        )
        if not summary.empty:
            summaries.append(summary)
        coef_matrices[str(short).upper()] = coef_matrix
        q_matrices[str(short).upper()] = q_matrix
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return all_summary, coef_matrices, q_matrices


def plot_archetype_location_variance_heatmap(
    coef_matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path | str,
    *,
    target_short: str,
    q_th: float = 0.05,
    show: bool = True,
) -> Path:
    """Heatmap of location-varying coefficients; non-significant cells are grey."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = summary["gene"].tolist()
    mat = coef_matrix.reindex(order).dropna(how="all")
    qmat = q_matrix.reindex(index=mat.index, columns=mat.columns)
    labels = []
    summary_idx = summary.set_index("gene")
    for gene in mat.index:
        tags = summary_idx.loc[gene, "marker_tags"]
        suffix = f" [{tags}]" if isinstance(tags, str) and tags else ""
        labels.append(f"{gene}{suffix}")

    values = mat.to_numpy(dtype=float)
    sig = (qmat.to_numpy(dtype=float) < q_th) & np.isfinite(values)
    masked_values = np.ma.array(values, mask=~sig)

    fig_h = max(5.0, 0.28 * len(mat))
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{target_short} vs rest: top location-variable genes")
    ax.set_xlabel("Anatomical location")
    ax.set_ylabel("Top genes by coefficient variance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Cohort-adjusted coefficient (grey: FDR >= 0.05)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_standardized_effect_variance_heatmap(
    effect_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path | str,
    *,
    target_short: str,
    significance_th: float = 0.05,
    significance_label: str = "FDR",
    show: bool = True,
) -> Path:
    """Heatmap of standardized effects; non-significant cells are grey."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = summary["gene"].tolist()
    mat = effect_matrix.reindex(order).dropna(how="all")
    sigmat = significance_matrix.reindex(index=mat.index, columns=mat.columns)
    labels = []
    summary_idx = summary.set_index("gene")
    for gene in mat.index:
        tags = summary_idx.loc[gene, "marker_tags"]
        suffix = f" [{tags}]" if isinstance(tags, str) and tags else ""
        labels.append(f"{gene}{suffix}")

    values = mat.to_numpy(dtype=float)
    sig = (sigmat.to_numpy(dtype=float) < significance_th) & np.isfinite(values)
    masked_values = np.ma.array(values, mask=~sig)

    fig_h = max(5.0, 0.28 * len(mat))
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{target_short} vs rest: top location-variable standardized effects")
    ax.set_xlabel("Anatomical location")
    ax.set_ylabel("Top genes by Hedges g variance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"Hedges g (grey: {significance_label} >= {significance_th:g})")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def program_gene_presence(
    expr: pd.DataFrame,
    programs: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Report requested/present genes for each mini-program."""
    programs = programs or PROGRAM_GENE_SETS
    present_genes = set(expr.columns.astype(str))
    rows = []
    for program, genes in programs.items():
        present = [g for g in genes if g in present_genes]
        rows.append(
            {
                "program": program,
                "n_requested": len(genes),
                "n_present": len(present),
                "present_genes": ",".join(present),
                "missing_genes": ",".join([g for g in genes if g not in present_genes]),
            }
        )
    return pd.DataFrame(rows)


def compute_program_scores(
    expr: pd.DataFrame,
    *,
    programs: dict[str, list[str]] | None = None,
    min_genes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average cohort-standardized expression over present genes in each program."""
    programs = programs or PROGRAM_GENE_SETS
    presence = program_gene_presence(expr, programs)
    scores = pd.DataFrame(index=expr.index)
    for _, row in presence.iterrows():
        if int(row["n_present"]) < min_genes:
            continue
        genes = [g for g in str(row["present_genes"]).split(",") if g]
        scores[row["program"]] = expr.loc[:, genes].mean(axis=1)
    return scores, presence


def compute_within_location_program_effects(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
) -> pd.DataFrame:
    """Hedges g program effects for each location × archetype-vs-rest contrast."""
    meta = meta.reindex(program_scores.index).copy()
    rows = []
    for location in locations:
        loc_meta = meta.loc[meta["Location"].eq(location)].copy()
        if loc_meta.empty:
            continue
        loc_scores = program_scores.loc[loc_meta.index]
        for raw_target in targets:
            target = canonical_archetype(raw_target)
            short = ARCHETYPE_SHORT[target]
            target_mask = loc_meta["Archetype"].eq(target)
            n_target = int(target_mask.sum())
            n_rest = int((~target_mask).sum())
            if n_target < 2 or n_rest < 2:
                continue
            for program in loc_scores.columns:
                target_vals = loc_scores.loc[target_mask, program]
                rest_vals = loc_scores.loc[~target_mask, program]
                t_stat, p_val = stats.ttest_ind(
                    pd.to_numeric(target_vals, errors="coerce"),
                    pd.to_numeric(rest_vals, errors="coerce"),
                    equal_var=False,
                    nan_policy="omit",
                )
                rows.append(
                    {
                        "Location": location,
                        "target_archetype": target,
                        "target_short": short,
                        "column_key": f"{location} | {short}",
                        "program": program,
                        "effect_g": _hedges_g(target_vals, rest_vals),
                        "p": float(p_val) if pd.notna(p_val) else np.nan,
                        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                        "n_target": n_target,
                        "n_rest": n_rest,
                    }
                )
    effects = pd.DataFrame(rows)
    if effects.empty:
        return effects
    effects["q"] = multipletests(effects["p"].fillna(1), method="fdr_bh")[1]
    return effects.sort_values(["program", "Location", "target_short"])


def run_program_location_interaction_tests(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    targets: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
) -> pd.DataFrame:
    """Omnibus archetype × location interaction tests for program scores."""
    return run_archetype_location_interaction_tests(
        program_scores,
        meta,
        targets=targets,
        locations=locations,
    ).rename(columns={"gene": "program"})


def program_score_long_by_location_archetype(
    program_scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    archetypes: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
) -> pd.DataFrame:
    """Long-form mini-program scores with ordered location × archetype labels."""
    meta = meta.reindex(program_scores.index).copy()
    program_order = [p for p in list(programs or PROGRAM_GENE_SETS.keys()) if p in program_scores.columns]
    if not program_order:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "program",
                "score",
                "Location",
                "Archetype",
                "archetype_short",
                "column_key",
                "Cohort",
            ]
        )
    scores = program_scores.loc[meta.index, program_order].copy()
    scores.index.name = "sample_id"
    long = scores.reset_index().melt(id_vars="sample_id", var_name="program", value_name="score")
    meta_cols = [col for col in ["Location", "Archetype", "Cohort"] if col in meta.columns]
    long = long.join(meta[meta_cols], on="sample_id")
    long["Archetype"] = long["Archetype"].map(canonical_archetype)
    short_map = {arch: ARCHETYPE_SHORT[arch] for arch in ARCHETYPE_ORDER}
    long["archetype_short"] = long["Archetype"].map(short_map)
    long["column_key"] = long["Location"].astype(str) + " | " + long["archetype_short"].astype(str)
    col_order = [f"{loc} | {ARCHETYPE_SHORT[canonical_archetype(arch)]}" for loc in locations for arch in archetypes]
    long = long.loc[
        long["score"].notna()
        & long["Location"].isin(locations)
        & long["Archetype"].isin([canonical_archetype(a) for a in archetypes])
    ].copy()
    long["program"] = pd.Categorical(long["program"], categories=program_order, ordered=True)
    long["column_key"] = pd.Categorical(long["column_key"], categories=col_order, ordered=True)
    return long.sort_values(["program", "column_key", "sample_id"])


def compute_program_score_posthoc_tests(
    score_long: pd.DataFrame,
    *,
    feature_col: str = "program",
    group_col: str = "column_key",
    value_col: str = "score",
    location_col: str = "Location",
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Pairwise within-location posthoc tests for mini-program score boxplots."""
    rows = []
    for feature, feature_df in score_long.groupby(feature_col, observed=False):
        for location, loc_df in feature_df.groupby(location_col, observed=False):
            groups = [g for g in loc_df[group_col].dropna().astype(str).unique().tolist() if g != "nan"]
            groups = sorted(groups, key=lambda g: list(score_long[group_col].cat.categories).index(g) if hasattr(score_long[group_col], "cat") and g in score_long[group_col].cat.categories else g)
            for i, group_a in enumerate(groups):
                vals_a = pd.to_numeric(loc_df.loc[loc_df[group_col].astype(str).eq(group_a), value_col], errors="coerce").dropna()
                if len(vals_a) < min_per_group:
                    continue
                for group_b in groups[i + 1 :]:
                    vals_b = pd.to_numeric(loc_df.loc[loc_df[group_col].astype(str).eq(group_b), value_col], errors="coerce").dropna()
                    if len(vals_b) < min_per_group:
                        continue
                    stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                    mean_a = float(vals_a.mean())
                    mean_b = float(vals_b.mean())
                    if mean_a >= mean_b:
                        high_group, low_group = group_a, group_b
                        diff = mean_a - mean_b
                    else:
                        high_group, low_group = group_b, group_a
                        diff = mean_b - mean_a
                    rows.append(
                        {
                            feature_col: feature,
                            location_col: location,
                            "group_a": group_a,
                            "group_b": group_b,
                            "high_group": high_group,
                            "low_group": low_group,
                            "mean_a": mean_a,
                            "mean_b": mean_b,
                            "mean_high_minus_low": diff,
                            "n_a": int(len(vals_a)),
                            "n_b": int(len(vals_b)),
                            "mannwhitney_U": float(stat),
                            "p": float(p_val) if pd.notna(p_val) else np.nan,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values([feature_col, location_col, "q", "p"])


def compute_program_score_anova_tests(
    score_long: pd.DataFrame,
    *,
    feature_col: str = "program",
    value_col: str = "score",
    location_col: str = "Location",
    group_col: str = "Archetype",
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    archetypes: list[str] | tuple[str, ...] = tuple(ARCHETYPE_ORDER),
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Within-location one-way ANOVA tests across archetypes for each mini-program."""
    rows = []
    archetype_order = [canonical_archetype(a) for a in archetypes]
    for feature, feature_df in score_long.groupby(feature_col, observed=False):
        for location in locations:
            loc_df = feature_df.loc[feature_df[location_col].eq(location)]
            vals = []
            ns = {}
            means = {}
            for archetype in archetype_order:
                group_vals = pd.to_numeric(
                    loc_df.loc[loc_df[group_col].eq(archetype), value_col],
                    errors="coerce",
                ).dropna()
                ns[ARCHETYPE_SHORT[archetype]] = int(len(group_vals))
                means[ARCHETYPE_SHORT[archetype]] = float(group_vals.mean()) if len(group_vals) else np.nan
                if len(group_vals) >= min_per_group:
                    vals.append(group_vals)
            if len(vals) < 2:
                continue
            f_stat, p_val = stats.f_oneway(*vals)
            row = {
                feature_col: feature,
                location_col: location,
                "anova_F": float(f_stat) if pd.notna(f_stat) else np.nan,
                "p": float(p_val) if pd.notna(p_val) else np.nan,
            }
            for short, n in ns.items():
                row[f"n_{short}"] = n
            for short, mean in means.items():
                row[f"mean_{short}"] = mean
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out.sort_values([location_col, "q", "p", feature_col])


def program_score_mean_matrix(
    score_long: pd.DataFrame,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "CP", "LO"),
) -> pd.DataFrame:
    """Mean mini-program score matrix ordered as location × archetype."""
    program_order = list(programs or PROGRAM_GENE_SETS.keys())
    col_order = [f"{loc} | {short}" for loc in locations for short in target_shorts]
    return (
        score_long.pivot_table(index="program", columns="column_key", values="score", aggfunc="mean", observed=True)
        .reindex(index=program_order, columns=col_order)
        .dropna(how="all")
    )


def program_effect_matrices(
    effects: pd.DataFrame,
    *,
    value_col: str = "effect_g",
    significance_col: str = "q",
    programs: list[str] | tuple[str, ...] | None = None,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "CP", "LO"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return effect and significance matrices ordered as location × archetype."""
    if value_col not in effects.columns:
        raise KeyError(f"{value_col!r} not found in program effects")
    if significance_col not in effects.columns:
        raise KeyError(f"{significance_col!r} not found in program effects")
    program_order = list(programs or PROGRAM_GENE_SETS.keys())
    col_order = [f"{loc} | {short}" for loc in locations for short in target_shorts]
    effect_matrix = (
        effects.pivot_table(index="program", columns="column_key", values=value_col, aggfunc="first")
        .reindex(index=program_order, columns=col_order)
    )
    significance_matrix = (
        effects.pivot_table(index="program", columns="column_key", values=significance_col, aggfunc="first")
        .reindex(index=program_order, columns=col_order)
    )
    return effect_matrix, significance_matrix


def plot_program_score_mean_heatmaps_by_location(
    mean_matrix: pd.DataFrame,
    anova_table: pd.DataFrame,
    out_path: Path | str,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    locations: list[str] | tuple[str, ...] = tuple(LOCATION_ORDER),
    target_shorts: list[str] | tuple[str, ...] = ("DI", "CP", "LO"),
    cmap_name: str = "PuOr_r",
    show: bool = True,
) -> Path:
    """Mean mini-program score heatmaps split by location with row ANOVA p/FDR."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    program_order = [p for p in list(programs or PROGRAM_GENE_SETS.keys()) if p in mean_matrix.index]
    if not program_order:
        raise ValueError("No requested programs found in mean_matrix")
    mat = mean_matrix.reindex(program_order)
    values = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)

    fig, axes = plt.subplots(
        1,
        len(locations),
        figsize=(max(13.5, 4.1 * len(locations)), max(4.0, 0.46 * len(program_order) + 1.2)),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(cmap_name).copy()
    last_im = None
    for ax, location in zip(axes, locations, strict=True):
        cols = [f"{location} | {short}" for short in target_shorts if f"{location} | {short}" in mat.columns]
        loc_mat = mat.reindex(columns=cols)
        last_im = ax.imshow(loc_mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(location)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([c.split(" | ")[1] for c in cols], rotation=0, ha="center", fontsize=8)
        ax.set_yticks(np.arange(len(program_order)))
        ax.set_yticklabels(program_order, fontsize=8)
        ax.set_xlabel("Archetype")
        loc_stats = anova_table.loc[anova_table["Location"].eq(location)].copy() if not anova_table.empty else anova_table
        _add_row_stat_text(
            ax,
            pd.Index(program_order),
            loc_stats,
            feature_col="program",
            x_start=len(cols) - 0.15,
        )
        ax.set_xlim(-0.5, len(cols) + 2.8)
    axes[0].set_ylabel("Curated mini-program")
    fig.suptitle("Mean mini-program scores by anatomical site and archetype", y=1.02)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Mean mini-program score")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_program_score_boxplots(
    score_long: pd.DataFrame,
    out_path: Path | str,
    *,
    programs: list[str] | tuple[str, ...] | None = None,
    column_order: list[str] | tuple[str, ...] | None = None,
    posthoc_tests: pd.DataFrame | None = None,
    q_th: float = 0.05,
    jitter: float = 0.16,
    point_size: float = 10,
    seed: int = 7,
    show: bool = True,
) -> Path:
    """Mini-program score boxplots with jittered samples across location × archetype."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = score_long.copy()
    program_order = [p for p in list(programs or PROGRAM_GENE_SETS.keys()) if p in set(data["program"].astype(str))]
    if not program_order:
        raise ValueError("No requested programs found in score_long")
    if column_order is None:
        column_order = [c for c in data["column_key"].cat.categories if c in set(data["column_key"].astype(str))]
    else:
        column_order = list(column_order)
    if not column_order:
        raise ValueError("No location-archetype groups found in score_long")
    sig_tests = pd.DataFrame()
    max_sig_per_program = 0
    if posthoc_tests is not None and not posthoc_tests.empty:
        sig_tests = posthoc_tests.loc[
            posthoc_tests["program"].astype(str).isin(program_order)
            & (pd.to_numeric(posthoc_tests["q"], errors="coerce") < q_th)
            & (pd.to_numeric(posthoc_tests.get("mean_high_minus_low", np.nan), errors="coerce") > 0)
        ].copy()
        sig_tests = sig_tests.loc[
            sig_tests["high_group"].astype(str).isin(column_order)
            & sig_tests["low_group"].astype(str).isin(column_order)
        ]
        if not sig_tests.empty:
            max_sig_per_program = int(sig_tests.groupby("program", observed=False).size().max())

    rng = np.random.default_rng(seed)
    row_height = 1.35 + min(max_sig_per_program, 12) * 0.18
    fig, axes = plt.subplots(
        len(program_order),
        1,
        figsize=(max(10.5, 0.62 * len(column_order) + 3.0), max(2.2, row_height * len(program_order))),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(len(column_order))
    colors = [ARCHETYPE_COLORS.get(c.split(" | ")[-1], "#6b7280") for c in column_order]
    short_to_arch = {v: k for k, v in ARCHETYPE_SHORT.items()}
    colors = [ARCHETYPE_COLORS.get(short_to_arch.get(c.split(" | ")[-1], ""), "#6b7280") for c in column_order]
    x_lookup = {col: i for i, col in enumerate(column_order)}

    for ax, program in zip(axes, program_order, strict=True):
        sub = data.loc[data["program"].astype(str).eq(program)].copy()
        vals = [
            pd.to_numeric(sub.loc[sub["column_key"].astype(str).eq(col), "score"], errors="coerce").dropna().to_numpy()
            for col in column_order
        ]
        bp = ax.boxplot(
            vals,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"color": "#4b5563", "linewidth": 0.8},
            capprops={"color": "#4b5563", "linewidth": 0.8},
        )
        for patch, color in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor("#4b5563")
            patch.set_linewidth(0.8)
        for x, vals_for_group, color in zip(positions, vals, colors, strict=True):
            if len(vals_for_group) == 0:
                continue
            xj = x + rng.uniform(-jitter, jitter, size=len(vals_for_group))
            ax.scatter(
                xj,
                vals_for_group,
                s=point_size,
                color=color,
                alpha=0.58,
                edgecolors="white",
                linewidths=0.25,
                zorder=3,
            )
        ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=0)
        ax.set_ylabel(program, rotation=0, ha="right", va="center", fontsize=8, labelpad=70)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        ax.set_axisbelow(True)
        for x in range(len(column_order) + 1):
            if x % 3 == 0:
                ax.axvline(x - 0.5, color="black", linewidth=0.7, alpha=0.28)
        prog_sig = sig_tests.loc[sig_tests["program"].astype(str).eq(program)].copy() if not sig_tests.empty else pd.DataFrame()
        if not prog_sig.empty:
            finite_vals = np.concatenate([v for v in vals if len(v)]) if any(len(v) for v in vals) else np.array([0.0])
            y_min = float(np.nanmin(finite_vals))
            y_max = float(np.nanmax(finite_vals))
            y_range = max(y_max - y_min, 1.0)
            step = y_range * 0.09
            y = y_max + step
            prog_sig = prog_sig.sort_values(["Location", "q", "p"])
            for _, stat_row in prog_sig.iterrows():
                x1 = x_lookup[str(stat_row["low_group"])]
                x2 = x_lookup[str(stat_row["high_group"])]
                if x1 > x2:
                    x1, x2 = x2, x1
                ax.plot([x1, x1, x2, x2], [y, y + step * 0.35, y + step * 0.35, y], color="#111111", linewidth=0.65)
                ax.text(
                    (x1 + x2) / 2,
                    y + step * 0.42,
                    f"FDR={_format_p_for_row_label(stat_row['q'])}",
                    ha="center",
                    va="bottom",
                    fontsize=5.2,
                    color="#111111",
                )
                y += step * 1.05
            ax.set_ylim(y_min - y_range * 0.12, y + step * 0.55)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(column_order, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Location | archetype")
    fig.supylabel("Mini-program score (mean z-scored expression)", x=0.01, fontsize=10)
    fig.suptitle("Mini-program scores by anatomical site and archetype", y=0.995)
    fig.tight_layout(rect=(0.06, 0.0, 1.0, 0.98))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_program_effect_heatmap(
    effect_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    out_path: Path | str,
    *,
    q_th: float | None = None,
    significance_th: float = 0.05,
    significance_label: str = "FDR",
    show: bool = True,
) -> Path:
    """Compact program × location-archetype heatmap; non-significant cells grey."""
    import matplotlib.pyplot as plt

    if q_th is not None:
        significance_th = q_th
        significance_label = "FDR"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mat = effect_matrix.dropna(how="all")
    sigmat = significance_matrix.reindex(index=mat.index, columns=mat.columns)
    values = mat.to_numpy(dtype=float)
    sig = (sigmat.to_numpy(dtype=float) < significance_th) & np.isfinite(values)
    masked_values = np.ma.array(values, mask=~sig)

    fig, ax = plt.subplots(figsize=(12.5, max(3.6, 0.52 * len(mat))))
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title("Mini-program implementation of archetypes across anatomical sites")
    ax.set_xlabel("Location | archetype-vs-rest")
    ax.set_ylabel("Curated mini-program")
    for x in range(len(mat.columns) + 1):
        if x % 3 == 0:
            ax.axvline(x - 0.5, color="black", linewidth=0.8, alpha=0.45)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"Hedges g (grey: {significance_label} >= {significance_th:g})")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out


def plot_combined_standardized_effect_gene_heatmap(
    effect_matrices: dict[str, pd.DataFrame],
    q_matrices: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    out_path: Path | str,
    *,
    target_shorts: list[str] | tuple[str, ...] = ("DI", "CP", "LO"),
    q_th: float = 0.05,
    show: bool = True,
) -> Path:
    """One gene-level supplement heatmap with rows grouped by archetype target."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_frames = []
    q_frames = []
    row_labels = []
    for short in target_shorts:
        sub = summary.loc[summary["target_short"].eq(short)].copy()
        genes = sub["gene"].tolist()
        mat = effect_matrices.get(short, pd.DataFrame()).reindex(genes)
        qmat = q_matrices.get(short, pd.DataFrame()).reindex(index=genes, columns=mat.columns)
        row_frames.append(mat)
        q_frames.append(qmat)
        sub_idx = sub.set_index("gene")
        for gene in genes:
            tags = sub_idx.loc[gene, "marker_tags"]
            suffix = f" [{tags}]" if isinstance(tags, str) and tags else ""
            row_labels.append(f"{short} | {gene}{suffix}")
    mat_all = pd.concat(row_frames, axis=0)
    q_all = pd.concat(q_frames, axis=0)
    values = mat_all.to_numpy(dtype=float)
    sig = (q_all.to_numpy(dtype=float) < q_th) & np.isfinite(values)
    masked_values = np.ma.array(values, mask=~sig)

    fig_h = max(8.0, 0.22 * len(mat_all))
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    vmax = float(np.nanmax(np.abs(values))) if values.size and np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.5)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(mat_all.columns)))
    ax.set_xticklabels(mat_all.columns, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    offset = 0
    for short in target_shorts:
        n = int(summary["target_short"].eq(short).sum())
        if n:
            ax.axhline(offset - 0.5, color="black", linewidth=0.8)
            offset += n
    ax.axhline(offset - 0.5, color="black", linewidth=0.8)
    ax.set_title("Top location-variable genes by archetype")
    ax.set_xlabel("Anatomical location")
    ax.set_ylabel("Archetype target | gene [annotation]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Hedges g (grey: FDR >= 0.05)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show(fig)
    plt.close(fig)
    return out
