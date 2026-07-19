"""Manuscript figure ↔ notebook artifact mapping (V4 captions, Jul 2026).

Canonical manuscript panel IDs and supplementary table CSV stems. Figure SVG/PNG
output stems are in ``dlbcl.figure_artifacts``; legacy internal names are
remapped via ``LEGACY_*_STEMS`` when building supplementary table bundles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TableRole = Literal["input", "stats"]


@dataclass(frozen=True, slots=True)
class SupplementaryTableSpec:
    """Metadata for one exported source table."""

    stem: str
    manuscript: str
    description: str
    notebook: str | None = None
    role: TableRole | None = None  # None → inferred from stem/description

    @property
    def sheet_name(self) -> str:
        """Excel sheet name (31-char limit). Prefer ``excel_sheet_name`` with key."""
        return self.stem[:31]


# Canonical supplementary CSV stems keyed by stable logical id.
# role="input" = plot-ready / patient-level source; role="stats" = tests / model output.
SUPPLEMENTARY_TABLES: dict[str, SupplementaryTableSpec] = {
    # --- Fig 1 ---
    "1D_pie_patients": SupplementaryTableSpec(
        "fig1D_pie_patients", "Fig 1D", "Classifier pie grid patient-level table", "nb1_1E",
        role="input",
    ),
    # --- Fig 3 discovery ---
    "3D_top_drivers": SupplementaryTableSpec(
        "fig3D_top_class_drivers_ranked", "Fig 3D", "Top class-driving DE genes", "nb6",
        role="input",
    ),
    "3D_heatmap_z": SupplementaryTableSpec(
        "fig3D_heatmap_zscores", "Fig 3D", "DE heatmap z-scores", "nb6",
        role="input",
    ),
    "3E-H_scores": SupplementaryTableSpec(
        "fig3E-H_module_scores_long", "Fig 3E–H", "Immune module scores (long)", "nb6",
        role="input",
    ),
    "3E-H_summary": SupplementaryTableSpec(
        "fig3E-H_module_group_summary", "Fig 3E–H", "Immune module Kruskal summary", "nb6",
        role="stats",
    ),
    "3I-L_scores": SupplementaryTableSpec(
        "fig3I-L_module_scores_long", "Fig 3I–L", "T-helper signature scores (long)", "nb6",
        role="input",
    ),
    "3I-L_summary": SupplementaryTableSpec(
        "fig3I-L_module_group_summary", "Fig 3I–L", "T-helper signature Kruskal summary", "nb6",
        role="stats",
    ),
    "3M_ecotyper_counts": SupplementaryTableSpec(
        "fig3M_ecotyper_counts", "Fig 3M", "EcoTyper B-cell state counts", "nb6",
        role="input",
    ),
    "3M_ecotyper_props": SupplementaryTableSpec(
        "fig3M_ecotyper_proportions", "Fig 3M", "EcoTyper B-cell state proportions", "nb6",
        role="input",
    ),
    "3N_hla_stats": SupplementaryTableSpec(
        "fig3N_HLA_global_stats", "Fig 3N", "HLA × archetype global chi-square stats", "nb8",
        role="stats",
    ),
    "3N_hla_counts": SupplementaryTableSpec(
        "fig3N_HLA_counts", "Fig 3N", "HLA class counts by archetype", "nb8",
        role="input",
    ),
    "3N_hla_pct": SupplementaryTableSpec(
        "fig3N_HLA_percentages", "Fig 3N", "HLA class percentages by archetype", "nb8",
        role="input",
    ),
    "3N_classifier_assoc": SupplementaryTableSpec(
        "fig3N_classifier_associations", "Fig 3N", "Classifier × archetype associations", "nb8",
        role="stats",
    ),
    "3O_gsea_long": SupplementaryTableSpec(
        "fig3O_gsea_long", "Fig 3O", "GSEA results (long)", "nb6",
        role="stats",
    ),
    "3O_gsea_nes": SupplementaryTableSpec(
        "fig3O_gsea_NES", "Fig 3O", "GSEA NES matrix", "nb6",
        role="input",
    ),
    "3O_gsea_fdr": SupplementaryTableSpec(
        "fig3O_gsea_FDR", "Fig 3O", "GSEA FDR matrix", "nb6",
        role="stats",
    ),
    # --- Fig 4 discovery integration ---
    "4B_loc_assoc": SupplementaryTableSpec(
        "fig4B_location_association", "Fig 4B", "Cramér's V vs location", "nb12",
        role="stats",
    ),
    "4B_arch_assoc": SupplementaryTableSpec(
        "fig4B_archetype_association", "Fig 4B", "Cramér's V vs archetype", "nb12",
        role="stats",
    ),
    "4B_pairwise": SupplementaryTableSpec(
        "fig4B_classifier_pairwise", "Fig 4B", "Classifier pairwise associations", "nb12",
        role="stats",
    ),
    "4C_enrichment": SupplementaryTableSpec(
        "fig4C_location_enrichment", "Fig 4C", "Archetype enrichment dotplot source", "nb12",
        role="input",
    ),
    "4D_enrichment": SupplementaryTableSpec(
        "fig4D_archetype_enrichment", "Fig 4D", "Location enrichment dotplot source", "nb12",
        role="input",
    ),
    # --- Fig 5 validation main ---
    "5A_val_km_arch": SupplementaryTableSpec(
        "fig5A_val_km_os_archetype", "Fig 5A", "Validation KM OS by archetype", "nb14",
        role="input",
    ),
    "5F_loc_arch_counts": SupplementaryTableSpec(
        "fig5F_location_archetype_counts", "Fig 5F", "Location × archetype counts", "nb14",
        role="input",
    ),
    "5F_loc_arch_props": SupplementaryTableSpec(
        "fig5F_location_archetype_row_props", "Fig 5F", "Location × archetype row %", "nb14",
        role="input",
    ),
    "5F_loc_arch_resid": SupplementaryTableSpec(
        "fig5F_location_archetype_residuals", "Fig 5F", "Location × archetype residuals", "nb14",
        role="stats",
    ),
    "5F_loc_arch_chi2": SupplementaryTableSpec(
        "fig5F_location_archetype_chi2_stats", "Fig 5F", "Location × archetype chi-square", "nb14",
        role="stats",
    ),
    "5G_mv_forest": SupplementaryTableSpec(
        "fig5G_cox_location_archetype_forest_OS", "Fig 5G", "Validation MV Cox forest", "nb14",
        role="stats",
    ),
    "5G_univar": SupplementaryTableSpec(
        "fig5_supp_cox_univar_OS", "Fig 5 / supp", "Validation univariable Cox (OS)", "nb14",
        role="stats",
    ),
    "5H_event_grid": SupplementaryTableSpec(
        "fig5H_event_rate_grid", "Fig 5H", "Validation 2-y OS event-rate grid", "nb14",
        role="input",
    ),
    "5I_sym_lrt": SupplementaryTableSpec(
        "fig5I_symmetric_lrt", "Fig 5I", "Validation symmetric LRT tests", "nb14",
        role="stats",
    ),
    "5J_partition": SupplementaryTableSpec(
        "fig5J_loglikelihood_partition", "Fig 5J", "Validation prognostic partition", "nb14",
        role="stats",
    ),
    # --- Fig 6 validation integration ---
    "6B_loc_assoc": SupplementaryTableSpec(
        "fig6B_val_location_association", "Fig 6B", "Validation Cramér's V vs location", "nb14",
        role="stats",
    ),
    "6B_arch_assoc": SupplementaryTableSpec(
        "fig6B_val_archetype_association", "Fig 6B", "Validation Cramér's V vs archetype", "nb14",
        role="stats",
    ),
    "6B_pairwise": SupplementaryTableSpec(
        "fig6B_val_classifier_pairwise", "Fig 6B", "Validation classifier pairwise", "nb14",
        role="stats",
    ),
    "6C_enrichment": SupplementaryTableSpec(
        "fig6C_archetype_enrichment", "Fig 6C", "Validation archetype enrichment", "nb14",
        role="input",
    ),
    "6D_enrichment": SupplementaryTableSpec(
        "fig6D_location_enrichment", "Fig 6D", "Validation location enrichment", "nb14",
        role="input",
    ),
    # --- Fig S3 (transcriptomic / HLA support) ---
    "S3D_hla_programs": SupplementaryTableSpec(
        "figS3D_hla_immune_program_effects", "Fig S3D", "HLA class immune program effects", "nb14",
        role="stats",
    ),
    "S3D_hla_de": SupplementaryTableSpec(
        "figS3D_hla_class_group_one_vs_rest_de", "Fig S3D", "HLA class group DE", "nb14",
        role="stats",
    ),
    "S3E_hla_density": SupplementaryTableSpec(
        "figS3E_hla_phenotype_density_kruskal", "Fig S3E", "HLA phenotype density Kruskal", "nb14",
        role="stats",
    ),
    "S3_hla_loss_summary": SupplementaryTableSpec(
        "figS3_hla_loss_vs_retained_top20_summary", "Fig S3D–E", "HLA loss vs retained top genes", "nb14",
        role="input",
    ),
    "S3_hla_loss_de": SupplementaryTableSpec(
        "figS3_hla_loss_vs_retained_transcriptome_de", "Fig S3D–E", "HLA loss vs retained DE", "nb14",
        role="stats",
    ),
    "combined_within_loc_de": SupplementaryTableSpec(
        "combined_transcriptome_within_location_one_vs_rest_de",
        "Fig S3 / analysis",
        "Within-location archetype DE (discovery + validation)",
        "nb14",
        role="stats",
    ),
    "combined_program_effects": SupplementaryTableSpec(
        "combined_transcriptome_curated_program_effects",
        "Fig S3 / analysis",
        "Curated immune program effects",
        "nb14",
        role="stats",
    ),
    "combined_loc_variance": SupplementaryTableSpec(
        "combined_transcriptome_archetype_top20_location_variance_standardized_effect",
        "Fig S3 / analysis",
        "Archetype top-20 location variance effects",
        "nb14",
        role="stats",
    ),
    # --- Fig S4 discovery survival supplements ---
    "S4A_univar_os": SupplementaryTableSpec(
        "figS4A_cox_univar_OS", "Fig S4A", "Discovery univariable Cox (OS)", "nb9",
        role="stats",
    ),
    "S4A_univar_dss": SupplementaryTableSpec(
        "figS4A_cox_univar_DSS", "Fig S4A", "Discovery univariable Cox (DSS)", "nb9",
        role="stats",
    ),
    "S4B_val_km_no_pcns": SupplementaryTableSpec(
        "figS4B_val_km_os_archetype_no_pcns", "Fig S4B", "Validation KM OS archetype (no PCNS)", "nb17",
        role="input",
    ),
    "S4D_event_grid": SupplementaryTableSpec(
        "figS4D_event_rate_grid", "Fig S4D", "Discovery 2-y OS event-rate grid", "nb9",
        role="input",
    ),
    "S4E_sym_lrt": SupplementaryTableSpec(
        "figS4E_symmetric_lrt", "Fig S4E", "Discovery symmetric LRT tests", "nb9",
        role="stats",
    ),
    "S4F_partition": SupplementaryTableSpec(
        "figS4F_loglikelihood_partition", "Fig S4F", "Discovery prognostic partition", "nb9",
        role="stats",
    ),
    "S4_mv_clinical": SupplementaryTableSpec(
        "figS4_supp_cox_os_clinical_multivariable_forest",
        "Fig S4 / supp",
        "Discovery MV Cox (clinical covariates)",
        "nb9",
        role="stats",
    ),
    "S4_mv_age_coo": SupplementaryTableSpec(
        "figS4_supp_cox_os_age_coo_location_archetype_forest",
        "Fig S4 / supp",
        "Discovery MV Cox (age, COO, location, archetype)",
        "nb9",
        role="stats",
    ),
    "S4_sens_os_no_outliers": SupplementaryTableSpec(
        "figS4_supp_km_os_archetype_no_outliers", "Fig S4 / supp", "Discovery KM OS sensitivity", "nb17",
        role="input",
    ),
    "S4_sens_dss_no_outliers": SupplementaryTableSpec(
        "figS4_supp_km_dss_archetype_no_outliers", "Fig S4 / supp", "Discovery KM DSS sensitivity", "nb17",
        role="input",
    ),
    # --- Fig S5 validation biology supplements ---
    "S5A_val_km_loc": SupplementaryTableSpec(
        "figS5A_val_km_os_location", "Fig S5A", "Validation KM OS by location", "nb14",
        role="input",
    ),
    "S5B_ecotyper_counts": SupplementaryTableSpec(
        "figS5B_ecotyper_counts", "Fig S5B", "Validation EcoTyper counts", "nb14",
        role="input",
    ),
    "S5B_ecotyper_props": SupplementaryTableSpec(
        "figS5B_ecotyper_proportions", "Fig S5B", "Validation EcoTyper proportions", "nb14",
        role="input",
    ),
    "S5C_module_scores": SupplementaryTableSpec(
        "figS5C_val_module_scores_long", "Fig S5C", "Validation module scores", "nb14",
        role="input",
    ),
    "S5C_module_stats": SupplementaryTableSpec(
        "figS5C_val_module_kruskal_stats", "Fig S5C", "Validation module Kruskal stats", "nb14",
        role="stats",
    ),
    "S5D_gsea": SupplementaryTableSpec(
        "figS5D_val_gsea_long", "Fig S5D", "Validation GSEA (long)", "nb14",
        role="stats",
    ),
    "S5_th_scores": SupplementaryTableSpec(
        "figS5C_val_thelper_scores_long", "Fig S5C", "Validation T-helper scores", "nb14",
        role="input",
    ),
    "S5_val_3d_heatmap": SupplementaryTableSpec(
        "figS5_supp_val_top_class_drivers_heatmap_zscores", "Fig S5 / supp", "Validation DE heatmap z", "nb14",
        role="input",
    ),
    # --- Other analyses ---
    "val16_benchmark": SupplementaryTableSpec(
        "figS5_supp_classifier_os_benchmark", "Fig S5 / supp", "Classifier OS benchmark", "nb16",
        role="stats",
    ),
}

# Legacy CSV stems from earlier notebook runs → canonical stem (for xlsx ingest).
LEGACY_SUPPLEMENTARY_STEMS: dict[str, str] = {
    "figS3A_cox_univar_OS": "figS4A_cox_univar_OS",
    "figS3A_cox_univar_DSS": "figS4A_cox_univar_DSS",
    "figS3B_cox_os_clinical_multivariable_forest": "figS4_supp_cox_os_clinical_multivariable_forest",
    "figS3B_cox_os_age_coo_location_archetype_forest": "figS4_supp_cox_os_age_coo_location_archetype_forest",
    "figS3D_symmetric_lrt": "figS4E_symmetric_lrt",
    "figValS3A_cox_univar_OS": "fig5_supp_cox_univar_OS",
    "figValS3B_cox_location_archetype_forest_OS": "fig5G_cox_location_archetype_forest_OS",
    "figValS3C_symmetric_lrt": "fig5I_symmetric_lrt",
    "figValS3C_event_rate_grid": "fig5H_event_rate_grid",
    "figValS3D_loglikelihood_partition": "fig5J_loglikelihood_partition",
    "figVal3M_ecotyper_counts": "figS5B_ecotyper_counts",
    "figVal3M_ecotyper_proportions": "figS5B_ecotyper_proportions",
    "figVal4C_archetype_enrichment": "fig6C_archetype_enrichment",
    "figVal4D_location_enrichment": "fig6D_location_enrichment",
    "fig4B_val_location_association": "fig6B_val_location_association",
    "fig4B_val_archetype_association": "fig6B_val_archetype_association",
    "fig4B_val_classifier_pairwise": "fig6B_val_classifier_pairwise",
    "fig1F_val_km_os_location": "figS5A_val_km_os_location",
    "fig1E_pie_patients": "fig1D_pie_patients",
    "fig2F_val_km_os_archetype": "fig5A_val_km_os_archetype",
    "fig3D_val_heatmap_zscores": "figS5_supp_val_top_class_drivers_heatmap_zscores",
    "fig3E-H_val_module_scores_long": "figS5C_val_module_scores_long",
    "fig3E-H_val_module_kruskal_stats": "figS5C_val_module_kruskal_stats",
    "fig3I-L_val_module_scores_long": "figS5C_val_thelper_scores_long",
    "fig3O_val_gsea_long": "figS5D_val_gsea_long",
    "fig5A_val_km_os_archetype_no_pcns": "figS4B_val_km_os_archetype_no_pcns",
    "fig2F_km_os_archetype_no_outliers": "figS4_supp_km_os_archetype_no_outliers",
    "fig2G_km_dss_archetype_no_outliers": "figS4_supp_km_dss_archetype_no_outliers",
    "figVal16_classifier_os_benchmark": "figS5_supp_classifier_os_benchmark",
    "val_location_archetype_counts": "fig5F_location_archetype_counts",
    "val_location_archetype_row_props": "fig5F_location_archetype_row_props",
    "val_location_archetype_residuals": "fig5F_location_archetype_residuals",
    "val_location_archetype_chi2_stats": "fig5F_location_archetype_chi2_stats",
    "discovery_hla_class_group_hla_immune_program_effects": "figS3D_hla_immune_program_effects",
    "discovery_hla_class_group_one_vs_rest_de": "figS3D_hla_class_group_one_vs_rest_de",
    "discovery_hla_class_group_phenotype_density_kruskal": "figS3E_hla_phenotype_density_kruskal",
    "discovery_hla_loss_vs_retained_top20_summary": "figS3_hla_loss_vs_retained_top20_summary",
    "discovery_hla_loss_vs_retained_transcriptome_de": "figS3_hla_loss_vs_retained_transcriptome_de",
}


def supplementary_stem(key: str) -> str:
    """Return canonical CSV stem for a registered supplementary table key."""
    try:
        return SUPPLEMENTARY_TABLES[key].stem
    except KeyError as exc:
        raise KeyError(f"Unknown supplementary table key: {key!r}") from exc


def canonical_stem_from_path(path: Path | str) -> str:
    """Map a CSV filename stem to the canonical manuscript-aligned stem."""
    stem = Path(path).stem
    return LEGACY_SUPPLEMENTARY_STEMS.get(stem, stem)


def registered_stem_set() -> set[str]:
    """All canonical stems from the registry plus legacy aliases."""
    stems = {spec.stem for spec in SUPPLEMENTARY_TABLES.values()}
    stems.update(LEGACY_SUPPLEMENTARY_STEMS.values())
    stems.update(LEGACY_SUPPLEMENTARY_STEMS.keys())
    return stems


_STATS_HINT = re.compile(
    r"(kruskal|chi2|chi.?square|cox|lrt|fdr|assoc|residual|partition|benchmark|de\b|effect)",
    re.I,
)


def table_role(spec: SupplementaryTableSpec) -> TableRole:
    """Return Input vs Stats role (explicit or inferred)."""
    if spec.role in {"input", "stats"}:
        return spec.role
    blob = f"{spec.stem} {spec.description}"
    return "stats" if _STATS_HINT.search(blob) else "input"


def manuscript_workbook_id(manuscript: str, stem: str = "") -> str:
    """Map a manuscript label to a workbook id (``Fig3``, ``FigS5``, …)."""
    text = str(manuscript)
    m = re.search(r"Fig\s*S\s*(\d+)", text, flags=re.I)
    if m:
        return f"FigS{m.group(1)}"
    m = re.search(r"Fig\s*(\d+)", text, flags=re.I)
    if m:
        return f"Fig{m.group(1)}"
    # Fall back to stem prefix (figS5… / fig5…).
    sm = re.match(r"fig(S?)(\d+)", stem, flags=re.I)
    if sm:
        return f"Fig{'S' if sm.group(1) else ''}{sm.group(2)}"
    return "Other"


def excel_sheet_name(registry_key: str, role: TableRole | None = None) -> str:
    """Build a prefixed Excel sheet name from the short registry key (≤31 chars)."""
    spec = SUPPLEMENTARY_TABLES[registry_key]
    resolved = role or table_role(spec)
    prefix = "Input_" if resolved == "input" else "Stats_"
    return f"{prefix}{registry_key}"[:31]


# Preferred on-disk notebook folders when the same stem appears twice.
WORKBOOK_FOLDER_PREFERENCE: dict[str, tuple[str, ...]] = {
    "Fig1": ("fig1",),
    "Fig3": ("fig3",),
    "Fig4": ("fig4",),
    "Fig5": ("fig5",),
    "Fig6": ("fig6",),
    "FigS3": ("supplemental_fig3", "fig3"),
    "FigS4": ("supplemental_fig4",),
    "FigS5": ("supplemental_fig5", "fig5"),
    "Other": (),
}


def workbook_id_for_key(registry_key: str) -> str:
    spec = SUPPLEMENTARY_TABLES[registry_key]
    return manuscript_workbook_id(spec.manuscript, spec.stem)


def stem_to_registry_key() -> dict[str, str]:
    """Canonical stem → first registry key (insertion order)."""
    out: dict[str, str] = {}
    for key, spec in SUPPLEMENTARY_TABLES.items():
        out.setdefault(spec.stem, key)
    return out
