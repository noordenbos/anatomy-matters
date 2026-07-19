"""Figure output artifact stems (SVG/PNG/PDF) aligned to manuscript panels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FigureArtifactSpec:
    """One exported figure file (stem without extension)."""

    stem: str
    manuscript: str
    description: str
    notebook: str | None = None


# Registry keys → canonical output stems under ``figures/notebook*/``.
FIGURE_ARTIFACTS: dict[str, FigureArtifactSpec] = {
  # Fig 1
  "1C_modality": FigureArtifactSpec("fig1C_cohort_modality_matrix", "Fig 1C", "Cohort modality matrix", "nb1_1C"),
  "S1A_oncoprint": FigureArtifactSpec("figS1A_oncoprint", "Fig S1A", "Discovery oncoprint", "nb1_1D"),
  "1D_classifier_pies": FigureArtifactSpec(
      "fig1D_classifier_pie_patchwork", "Fig 1D", "Classifier pie patchwork", "nb1_1E",
  ),
  "1E_km_os_location": FigureArtifactSpec("fig1E_km_os_location", "Fig 1E", "KM OS by location", "nb11"),
  "1F_km_pfs_location": FigureArtifactSpec("fig1F_km_pfs_location", "Fig 1F", "KM PFS by location", "nb11"),
  "S_supp_km_dss_location": FigureArtifactSpec(
      "figS_supp_km_dss_location", "Supp", "KM DSS by location", "nb11",
  ),
  # Fig 2
  "2A_tsne_phenotype16": FigureArtifactSpec("fig2A_tsne_phenotype16_bm", "Fig 2A", "tSNE 16-class phenotype", "nb3"),
  "2B_tsne_location": FigureArtifactSpec("fig2B_tsne_location", "Fig 2B", "tSNE location", "nb3"),
  "2C_tsne_lymphgen": FigureArtifactSpec("fig2C_tsne_lymphgen", "Fig 2C", "tSNE LymphGen", "nb3"),
  "2D_tsne_dlbclass": FigureArtifactSpec("fig2D_tsne_dlbclass", "Fig 2D", "tSNE DLBclass", "nb3"),
  "2E_archetype_heatmap": FigureArtifactSpec(
      "fig2E_archetype_density_heatmap", "Fig 2E", "Archetype phenotype-density heatmap", "nb4",
  ),
  "2F_km_os_archetype": FigureArtifactSpec("fig2F_km_os_archetype", "Fig 2F", "KM OS by archetype", "nb11"),
  "2G_km_dss_archetype": FigureArtifactSpec("fig2G_km_dss_archetype", "Fig 2G", "KM DSS by archetype", "nb11"),
  "S_supp_km_pfs_archetype": FigureArtifactSpec(
      "figS_supp_km_pfs_archetype", "Supp", "KM PFS by archetype", "nb11",
  ),
  # Fig 3
  "3A_spatial_example": FigureArtifactSpec("fig3A_spatial_example", "Fig 3A", "Spatial archetype example", "nb7"),
  "3D_de_heatmap": FigureArtifactSpec("fig3D_top_class_drivers_heatmap", "Fig 3D", "Top class-driver heatmap", "nb6"),
  "3E_L_module_scores": FigureArtifactSpec("fig3E-L_module_scores", "Fig 3E–L", "Immune / T-helper module scores", "fig3"),
  "3M_ecotyper": FigureArtifactSpec("fig3M_ecotyper_b_state_by_archetype", "Fig 3M", "EcoTyper B-cell states", "fig3"),
  "3O_gsea_hallmark": FigureArtifactSpec("fig3O_gsea_archetypes_hallmark", "Fig 3O", "Hallmark GSEA dotplot", "fig3"),
  "S3E_hla_stacked": FigureArtifactSpec("figS3E_HLA_archetype_stacked", "Fig S3E", "HLA status by archetype", "fig3"),
  "4A_integration_donut": FigureArtifactSpec("fig4A_integration_donut", "Fig 4A", "Discovery integration circos", "nb12"),
  "4B_association": FigureArtifactSpec("fig4B_association_dumbbell", "Fig 4B", "Association dumbbell", "nb12"),
  # Fig 5 validation
  "5A_val_km_archetype": FigureArtifactSpec("fig5A_val_km_os_archetype", "Fig 5A", "Validation KM OS archetype", "nb14"),
  "5F_val_loc_arch_heatmaps": FigureArtifactSpec("fig5F_val_location_archetype_heatmaps", "Fig 5F", "Location × archetype heatmaps", "fig5"),
  "5F_val_loc_arch_pies": FigureArtifactSpec("fig5F_val_location_archetype_pies", "Fig 5F", "Location × archetype pies", "fig5"),
  "5G_cox_forest": FigureArtifactSpec("fig5G_cox_location_archetype_forest_OS", "Fig 5G", "MV Cox forest OS", "fig5"),
  "5H_event_grid": FigureArtifactSpec("fig5H_event_rate_grid", "Fig 5H", "2-y event-rate grid", "fig5"),
  "5I_symmetric_lrt": FigureArtifactSpec("fig5I_symmetric_lrt", "Fig 5I", "Symmetric LRT", "fig5"),
  "5J_ll_partition": FigureArtifactSpec("fig5J_loglikelihood_partition", "Fig 5J", "Log-likelihood partition", "fig5"),
  "6A_val_donut": FigureArtifactSpec("fig6A_val_integration_donut", "Fig 6A", "Validation integration circos", "nb14"),
  "6B_val_dumbbell": FigureArtifactSpec("fig6B_val_association_dumbbell", "Fig 6B", "Validation association dumbbell", "nb14"),
  "6B_val_pairwise": FigureArtifactSpec(
      "fig6B_val_classifier_pairwise_heatmap", "Fig 6B", "Validation classifier pairwise heatmap", "nb14",
  ),
  "6C_val_enrichment_arch": FigureArtifactSpec(
      "fig6C_val_archetype_association_dotplot", "Fig 6C", "Validation archetype enrichment", "nb14",
  ),
  "6D_val_enrichment_loc": FigureArtifactSpec(
      "fig6D_val_location_association_dotplot", "Fig 6D", "Validation location enrichment", "nb14",
  ),
  # Supplements (tSNE / PCA)
  "S2E_tsne_qc": FigureArtifactSpec("figS2E_tsne_qc_filtering", "Fig S2E", "tSNE QC filtering", "nb3"),
  "S1L_tsne_phenotype30": FigureArtifactSpec("figS1L_tsne_phenotype30_bm", "Fig S1L", "tSNE 30-class phenotype", "nb3"),
  "S3B_pca": FigureArtifactSpec("figS3B_PCA", "Fig S3B", "Transcriptome PCA", "nb1_1C"),
  "S3F_tsne_coo": FigureArtifactSpec("figS3F_tsne_coo", "Fig S3F", "tSNE COO", "nb3"),
  "S3G_tsne_lymphomap": FigureArtifactSpec("figS3G_tsne_lymphomap", "Fig S3G", "tSNE LymphoMAP", "nb3"),
  "S3H_tsne_archetype": FigureArtifactSpec("figS3H_tsne_archetype", "Fig S3H", "tSNE archetype", "nb3"),
  "S3A_gene_protein": FigureArtifactSpec("figS3A_gene_protein_correlation", "Fig S3A", "RNA–protein concordance", "nb2"),
}

# Legacy figure filename stems → canonical (longest-match first when applying).
LEGACY_FIGURE_STEMS: dict[str, str] = {
    "clustered_heatmap_20032026": "fig2E_archetype_density_heatmap",
    "gsea_dotplot_abundance_cluster_30_patientlevel": "fig3O_gsea_archetypes_hallmark",
    "ecotyper_b_state_by_archetype": "fig3M_ecotyper_b_state_by_archetype",
    "HLAABC_HLADR_grouped_stacked_with_stats": "figS3E_HLA_archetype_stacked",
    "heatmap_top_class_driving_genes": "fig3D_top_class_drivers_heatmap",
    # Fig 1
    "fig1D_oncoprint": "figS1A_oncoprint",
    "_fig1D_r_work": "_figS1A_r_work",
    "fig1E_location_pie_patchwork_legend": "fig1D_classifier_pie_patchwork_legend",
    "fig1E_location_pie_patchwork": "fig1D_classifier_pie_patchwork",
    "fig1E_location_genomic_stacked_bars": "fig1D_location_genomic_stacked_bars",
    "fig1E_location_lymphgen_dlbclass_stacked_bars": "fig1D_location_lymphgen_dlbclass_stacked_bars",
    "fig1E_pie_patients": "fig1D_pie_patients",
    "fig1F_km_os_location": "fig1E_km_os_location",
    "fig1G_km_dss_location": "figS_supp_km_dss_location",
    "fig1H_km_pfs_location": "fig1F_km_pfs_location",
    "fig1F_val_km_os_location": "figS5A_val_km_os_location",
    # Fig 2
    "fig2H_km_pfs_archetype": "figS_supp_km_pfs_archetype",
    "fig2F_val_km_os_archetype": "fig5A_val_km_os_archetype",
    # Fig 3 / S3 tSNE panels
    "figS1E_tsne_qc_filtering": "figS2E_tsne_qc_filtering",
    "figS1J_tsne_phenotype30_bm": "figS1L_tsne_phenotype30_bm",
    "figS2B_PCA": "figS3B_PCA",
    "figS2D_tsne_coo": "figS3F_tsne_coo",
    "figS2E_tsne_lymphomap": "figS3G_tsne_lymphomap",
    "figS2F_tsne_archetype": "figS3H_tsne_archetype",
    "figS3_tsne_patchwork": "figS_supp_tsne_patchwork",
    "figS2A_gene_protein_correlation": "figS3A_gene_protein_correlation",
    "figS2G_tsne_hmrn": "figS3I_tsne_hmrn",
    "figS2H_tsne_lymphplex": "figS3J_tsne_lymphplex",
    # Validation integration (Fig 6)
    "fig4A_val_integration_donut_legend": "fig6A_val_integration_donut_legend",
    "fig4A_val_integration_donut": "fig6A_val_integration_donut",
    "fig4B_val_association_dumbbell": "fig6B_val_association_dumbbell",
    "fig4B_val_classifier_pairwise_heatmap": "fig6B_val_classifier_pairwise_heatmap",
    "fig4C_val_archetype_association_dotplot": "fig6C_val_archetype_association_dotplot",
    "fig4D_val_location_association_dotplot": "fig6D_val_location_association_dotplot",
    # Validation biology supplements (Fig S5)
    "fig3D_val_top_class_drivers_heatmap": "figS5_supp_val_top_class_drivers_heatmap",
    "fig3D_val_heatmap_zscores": "figS5_supp_val_top_class_drivers_heatmap_zscores",
    "fig3E-H_val_immune_thelper_module_scores": "figS5C_val_immune_thelper_module_scores",
    "fig3E-H_val_module_scores_long": "figS5C_val_module_scores_long",
    "fig3E-H_val_module_kruskal_stats": "figS5C_val_module_kruskal_stats",
    "fig3I-L_val_thelper_signature_scores": "figS5C_val_thelper_signature_scores",
    "fig3I-L_val_module_scores_long": "figS5C_val_thelper_scores_long",
    "fig3O_val_gsea_dotplot": "figS5D_val_gsea_dotplot",
    "fig3O_val_gsea_long": "figS5D_val_gsea_long",
    # Sensitivity KM SVG (supp table already renamed)
    "fig5A_val_km_os_archetype_no_pcns": "figS4B_val_km_os_archetype_no_pcns",
    "fig2F_km_os_archetype_no_outliers": "figS4_supp_km_os_archetype_no_outliers",
    "fig2G_km_dss_archetype_no_outliers": "figS4_supp_km_dss_archetype_no_outliers",
}

# Inline markdown / caption phrase fixes (apply after stem renames).
MARKDOWN_PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Reproduces **Fig 1E**:", "Reproduces **Fig 1D**:"),
    ("## Fig 1E —", "## Fig 1D —"),
    ("Fig 1E pie", "Fig 1D classifier"),
    ("| Fig 1E pie grid |", "| Fig 1D pie grid |"),
    ("# Fig 1E option", "# Fig 1D option"),
    ("Fig 1F / 1G / 1H — KM by location", "Fig 1E / 1F — KM by location (+ Supp DSS)"),
    ("## Fig 1F / 1G / 1H", "## Fig 1E / 1F (+ Supp DSS)"),
    ("Fig 2F / 2G / 2H — KM by immune archetype", "Fig 2F / 2G — KM by archetype (+ Supp PFS)"),
    ("## Fig 2F / 2G / 2H", "## Fig 2F / 2G (+ Supp PFS)"),
    ("Fig 4A_val", "Fig 6A"),
    ("Fig 4B_val", "Fig 6B"),
    ("Fig 4C_val", "Fig 6C"),
    ("Fig 4D_val", "Fig 6D"),
    ("**1D** oncoprint", "**S1A** oncoprint"),
    ("Fig 1D oncoprint", "Fig S1A oncoprint"),
    ("Fig S2B", "Fig S3B"),
    ("Fig S2F", "Fig S3H"),
    ("Fig S2D", "Fig S3F"),
    ("Fig S2E —", "Fig S3G —"),
    ("Fig S1E", "Fig S2E"),
    ("Fig S1J", "Fig S1L"),
    ("Fig S2A", "Fig S3A"),
    ("Fig **1D** (genomic oncoprint)", "Fig **S1A** (genomic oncoprint)"),
    ("Fig **1E** (location × classifier pies)", "Fig **1D** (location × classifier pies)"),
    ("## Figs 4A–4D", "## Figs 6A–6D"),
    ("mirrors Figs 1F, 2F, 3D–O, 4A/4B, S3A–S3D", "Figs 5, 6, and S5 (see figure_registry)"),
    ("| **1F** KM OS by location |", "| **S5A** KM OS by location |"),
    ("| **2F** KM OS by archetype |", "| **5A** KM OS by archetype |"),
    ("| **4A** integration donut |", "| **6A** integration circos |"),
    ("| **4B** association bars |", "| **6B** association dumbbell |"),
    ("| **4C/4D** enrichment dotplots |", "| **6C/6D** enrichment dotplots |"),
    ("| **S3A** univariable Cox", "| **5G–J** validation Cox"),
)


def figure_stem(key: str) -> str:
    """Return canonical figure output stem for a registry key."""
    try:
        return FIGURE_ARTIFACTS[key].stem
    except KeyError as exc:
        raise KeyError(f"Unknown figure artifact key: {key!r}") from exc


def all_stem_replacements() -> list[tuple[str, str]]:
    """Legacy→canonical stem pairs, longest keys first."""
    from .figure_registry import LEGACY_SUPPLEMENTARY_STEMS

    merged = dict(LEGACY_FIGURE_STEMS)
    merged.update(LEGACY_SUPPLEMENTARY_STEMS)
    return sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True)


# Notebook path constants (apply after stem renames; longest keys first).
CODE_IDENTIFIER_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("FIG_1E_LYMPHGEN_DLBCLASS_STACKED_PNG", "FIG_1D_LYMPHGEN_DLBCLASS_STACKED_PNG"),
    ("FIG_1E_LYMPHGEN_DLBCLASS_STACKED", "FIG_1D_LYMPHGEN_DLBCLASS_STACKED"),
    ("FIG_1E_GENOMIC_STACKED_PNG", "FIG_1D_GENOMIC_STACKED_PNG"),
    ("FIG_1E_GENOMIC_STACKED", "FIG_1D_GENOMIC_STACKED"),
    ("FIG_1E_LEGEND_PNG", "FIG_1D_LEGEND_PNG"),
    ("FIG_1E_LEGEND", "FIG_1D_LEGEND"),
    ("FIG_1E", "FIG_1D"),
)


def apply_artifact_renames(text: str, *, markdown: bool = True) -> str:
    """Replace legacy figure/supplementary stems in ``text``."""
    for old, new in all_stem_replacements():
        text = text.replace(old, new)
    for old, new in CODE_IDENTIFIER_REPLACEMENTS:
        text = text.replace(old, new)
    if markdown:
        for old, new in MARKDOWN_PHRASE_REPLACEMENTS:
            text = text.replace(old, new)
    return text
