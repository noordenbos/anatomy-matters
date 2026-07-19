"""Standard setup for compiled public figure notebooks (``fig*.ipynb``, ``supplemental_fig*.ipynb``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc

from .dlbcl_io import (
    DISCOVERY_PATIENTS,
    NotebookPaths,
    configure_notebook,
    ensure_discovery_archetypes,
    load_adata,
    repo_root_from_notebook_cwd,
    set_supplementary_fig_id,
)

# Optional subfolders created under ``figures/{fig_id}/`` at setup time.
FIGURE_SUBDIRS: dict[str, tuple[str, ...]] = {
    "fig1": ("km",),
    "fig2": ("km", "pies"),
    "fig3": (
        "de_gene_expression",
        "immune_thelper_module_scores",
        "bcell_state_ecotyper",
        "gsea_abundance_cluster_30_patientlevel",
        "classifier_associations",
    ),
    "fig4": ("integration",),
    "fig5": (
        "elasticnet",
        "elasticnet_roc",
        "cox_validation",
        "location_archetype_distribution",
        "classifier_os_benchmark",
    ),
    "supplemental_fig4": ("cox_clinical",),
}

# Compiled notebooks that copy nb3 tSNE cells (need ADATA_PATH and filtering constants).
TSNE_PANEL_FIG_IDS = frozenset({"fig2", "supplemental_fig2", "supplemental_fig3"})

# Discovery KM panels (fig1 location, fig2 archetype) — helpers live in discovery_km.py.
DISCOVERY_KM_FIG_IDS = frozenset({"fig1", "fig2"})

# nb17 KM sensitivity slices (compiled into supplemental_fig4 / fig5).
KM_SENSITIVITY_FIG_IDS = frozenset({"supplemental_fig4", "fig5"})

# supplemental_fig4 validation Cox (nb14 archetype + IPI models).
COX_SUPPLEMENT_FIG_IDS = frozenset({"supplemental_fig4"})

# nb1 expression PCA slice (supplemental_fig3 Fig S3B).
DISCOVERY_PCA_FIG_IDS = frozenset({"supplemental_fig3"})

# nb5 elastic-net classifier (fig5 discovery classifier; S3J plot in supplemental_fig3).
ELASTIC_NET_FIG_IDS: frozenset[str] = frozenset()

# nb8 sub-compartment tSNE panels in supplemental_fig2 (S2G–S2I).
NB8_SUB_TSNE_FIG_IDS = frozenset({"supplemental_fig2"})


@dataclass
class NotebookContext:
    """Runtime objects injected as notebook globals by the setup cell."""

    repo_root: Path
    paths: NotebookPaths
    fig_dir: Path
    adata: Any | None = None
    arch_df: pd.DataFrame | None = None
    pred: pd.DataFrame | None = None
    gep: pd.DataFrame | None = None
    surv: pd.DataFrame | None = None
    archetype_label: pd.Series | None = None
    patient_subset: list[str] | None = None

    @property
    def FIG_DIR(self) -> Path:  # noqa: N802 — notebooks expect this name
        return self.fig_dir

    @property
    def REPO_ROOT(self) -> Path:  # noqa: N802
        return self.repo_root

    @property
    def _paths(self) -> NotebookPaths:  # noqa: SLF001 — notebooks use _paths
        return self.paths


def figure_output_dir(fig_id: str) -> str:
    """Return ``figures/`` subdirectory name (one folder per compiled notebook)."""
    return fig_id


def run_notebook_setup(
    profile: str,
    fig_id: str,
    *,
    repo_root: Path | None = None,
) -> NotebookContext:
    """Load AnnData and paths for a compiled figure notebook.

    Profiles:
    - ``discovery`` / ``discovery_archetype``: patient-level h5ad (+ archetypes)
    - ``discovery_sc``: same h5ad (single-cell panels use the main bundle)
    - ``validation``: validation cohort from ``adata.uns['validation_cohort']``
    - ``validation_and_discovery``: archetypes + validation predictions (fig5)
    """
    root = repo_root or repo_root_from_notebook_cwd()
    set_supplementary_fig_id(fig_id)
    paths = configure_notebook(root, figure_output_dir(fig_id))
    for sub in FIGURE_SUBDIRS.get(fig_id, ()):
        (paths.fig_dir / sub).mkdir(parents=True, exist_ok=True)

    adata = load_adata(paths=paths)
    ctx = NotebookContext(
        repo_root=root,
        paths=paths,
        fig_dir=paths.fig_dir,
        adata=adata,
        patient_subset=list(DISCOVERY_PATIENTS),
    )

    if profile in {"discovery", "discovery_archetype", "discovery_sc", "validation_and_discovery"}:
        ctx.arch_df = ensure_discovery_archetypes(adata, paths)

    if fig_id in DISCOVERY_KM_FIG_IDS and ctx.arch_df is not None:
        from .discovery_km import build_discovery_survival_table, configure_km_runtime

        configure_km_runtime(root)
        ctx.surv = build_discovery_survival_table(
            adata, ctx.arch_df, km_dir=paths.fig_dir / "km"
        )

    if profile in {"validation", "validation_and_discovery"}:
        import dlbcl.validation_figures as vf

        ctx.pred, ctx.gep, ctx.archetype_label = vf.load_from_adata(adata)
        ctx.surv = vf.build_survival_table(ctx.pred)

    return ctx


def setup_cell_source(profile: str, fig_id: str) -> list[str]:
    """Return ipynb source lines for the injected setup code cell."""
    lines = [
        "%matplotlib inline\n",
        "\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "NOTEBOOK_DIR = Path.cwd().resolve()\n",
        "REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == \"notebooks\" else NOTEBOOK_DIR\n",
        "if str(REPO_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(REPO_ROOT))\n",
        "\n",
        "from dlbcl.notebook_setup import run_notebook_setup\n",
        "\n",
        f"_ctx = run_notebook_setup({profile!r}, {fig_id!r})\n",
        "REPO_ROOT = _ctx.repo_root\n",
        "_paths = _ctx.paths\n",
        "FIG_DIR = _ctx.fig_dir\n",
        "adata = _ctx.adata\n",
        "arch_df = _ctx.arch_df\n",
        "pred = _ctx.pred\n",
        "gep = _ctx.gep\n",
        "surv = _ctx.surv\n",
        "archetype_label = _ctx.archetype_label\n",
        "PATIENT_SUBSET = _ctx.patient_subset\n",
        "OUTDIR = FIG_DIR\n",
        "OUTDIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Override data root: export DLBCL_DATA_DIR=/path/to/data before running.\n",
        "from dlbcl.dlbcl_io import rel_path\n",
        "print(f\"AnnData: {rel_path(_paths.adata_path, REPO_ROOT)}\")\n",
        "print(f\"Figures: {rel_path(FIG_DIR, REPO_ROOT)}\")\n",
        "\n",
        "import gc\n",
        "\n",
        "import pandas as pd\n",
        "import scanpy as sc\n",
        "\n",
        "ADATA_PATH = _paths.adata_path\n",
    ]
    if fig_id == "fig1":
        lines.extend(
            [
                "\n",
                "from dlbcl.dlbcl_io import (\n",
                "    DISCOVERY_PATIENTS,\n",
                "    consolidate_embedded_case_classifications,\n",
                "    ensure_discovery_archetypes,\n",
                ")\n",
                "from dlbcl.integration_figures import build_integration_metadata\n",
                "from dlbcl.location_pie_patchwork import (\n",
                "    GENOMIC_STACKED_BAR_COLS,\n",
                "    LYMPHGEN_DLBCLASS_COLS,\n",
                "    format_pie_patchwork_supplementary_table,\n",
                "    prepare_pie_patchwork_table,\n",
                "    plot_location_pie_patchwork,\n",
                "    plot_location_stacked_bar_patchwork,\n",
                ")\n",
                "\n",
                "FIG_1D = FIG_DIR / \"fig1D_classifier_pie_patchwork.svg\"\n",
                "FIG_1D_LEGEND = FIG_DIR / \"fig1D_classifier_pie_patchwork_legend.svg\"\n",
                "FIG_1D_LEGEND_PNG = FIG_DIR / \"fig1D_classifier_pie_patchwork_legend.png\"\n",
                "FIG_1D_GENOMIC_STACKED = FIG_DIR / \"fig1D_location_genomic_stacked_bars.svg\"\n",
                "FIG_1D_GENOMIC_STACKED_PNG = FIG_DIR / \"fig1D_location_genomic_stacked_bars.png\"\n",
                "FIG_1D_LYMPHGEN_DLBCLASS_STACKED = FIG_DIR / \"fig1D_location_lymphgen_dlbclass_stacked_bars.svg\"\n",
                "FIG_1D_LYMPHGEN_DLBCLASS_STACKED_PNG = FIG_DIR / \"fig1D_location_lymphgen_dlbclass_stacked_bars.png\"\n",
            ]
        )
    if fig_id == "fig5":
        lines.extend(
            [
                "\n",
                "import math\n",
                "import re\n",
                "\n",
                "import matplotlib as mpl\n",
                "import seaborn as sns\n",
                "\n",
                "from dlbcl.dlbcl_io import DISCOVERY_PATIENTS, elastic_net_classifier\n",
                "from dlbcl.validation_classifications import load_case_classification_validation\n",
                "from dlbcl.validation_classifier_survival import BASELINE_CORE, run_classifier_os_benchmark\n",
                "\n",
                'mpl.rcParams["svg.fonttype"] = "none"\n',
                'mpl.rcParams["pdf.fonttype"] = 42\n',
                'mpl.rcParams["ps.fonttype"] = 42\n',
                'mpl.rcParams["font.family"] = "DejaVu Sans"\n',
                "RUN_OOF_BOOTSTRAP_CI = True\n",
                "OOF_BOOTSTRAP_N = 2000\n",
                "OOF_BOOTSTRAP_SEED = 0\n",
                "RUN_TRAINING_ROC = False\n",
                "case_cc = load_case_classification_validation(adata.uns[\"validation_cohort\"], pred)\n",
            ]
        )
    if fig_id == "fig4":
        lines.extend(
            [
                "\n",
                "from dlbcl.dlbcl_io import (\n",
                "    DISCOVERY_PATIENTS,\n",
                "    consolidate_embedded_case_classifications,\n",
                "    ensure_discovery_archetypes,\n",
                ")\n",
                "from dlbcl.integration_figures import (\n",
                "    build_integration_metadata,\n",
                "    compute_classifier_pairwise_associations,\n",
                "    compute_group_enrichment_table,\n",
                "    configure_matplotlib,\n",
                "    order_patients_hierarchical,\n",
                "    plot_association_enrichment_dotplot_by_classifier,\n",
                "    plot_classifier_pairwise_heatmap,\n",
                "    plot_combined_association_dumbbell,\n",
                "    plot_donut_circos,\n",
                "    run_association_analysis,\n",
                ")\n",
                "\n",
                "OUTDIR = FIG_DIR / \"integration\"\n",
                "OUTDIR.mkdir(parents=True, exist_ok=True)\n",
                "FIG_4A = FIG_DIR / \"fig4A_integration_donut.svg\"\n",
                "FIG_4A_LEGEND = FIG_DIR / \"fig4A_integration_donut_legend.svg\"\n",
                "FIG_4B = FIG_DIR / \"fig4B_association_dumbbell.svg\"\n",
                "FIG_4B_PAIRWISE = FIG_DIR / \"fig4B_classifier_pairwise_heatmap.svg\"\n",
                "FIG_4C = FIG_DIR / \"fig4C_archetype_association_dotplot.svg\"\n",
                "FIG_4D = FIG_DIR / \"fig4D_location_association_dotplot.svg\"\n",
                "DOTPLOT_GREY_NONSIGNIFICANT = True\n",
                "DOTPLOT_FDR_THRESH = 0.25\n",
                "DOTPLOT_X_SPACING = 0.35\n",
                "DOTPLOT_FIG_W = None\n",
                "DOTPLOT_SHOW_GRID = False\n",
                "configure_matplotlib()\n",
            ]
        )
    if fig_id == "supplemental_fig1":
        lines.extend(
            [
                "\n",
                "import shutil\n",
                "import subprocess\n",
                "\n",
                "from dlbcl.genomic_profiling import (\n",
                "    GENOMIC_PROFILING_EXCLUDE_PATIENTS,\n",
                "    apply_genomic_profiling_exclusions,\n",
                "    export_for_r,\n",
                "    is_nested_genomic_profiling,\n",
                ")\n",
                "\n",
                "OUT_SVG = FIG_DIR / \"figS1A_oncoprint.svg\"\n",
                "OUT_PNG = FIG_DIR / \"figS1A_oncoprint.png\"\n",
                "WORK_DIR = FIG_DIR / \"_figS1A_r_work\"\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                "RSCRIPT = shutil.which(\"Rscript\")\n",
                "if RSCRIPT is None:\n",
                "    raise RuntimeError(\n",
                "        \"Rscript not found on PATH. Install R and the packages listed in README.md (Fig S1A).\"\n",
                "    )\n",
            ]
        )
    if fig_id in TSNE_PANEL_FIG_IDS:
        lines.extend(
            [
                "\n",
                "# tSNE panel constants and filtered single-cell subset.\n",
                "import gc\n",
                "\n",
                "import anndata as ad\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scanpy as sc\n",
                "\n",
                "from dlbcl.dlbcl_io import consolidate_embedded_case_classifications, log_wrote, log_saved, rel_path\n",
                "from dlbcl.phenotype_labels import ensure_phenotype_30_labels, read_celltype_colormap_30\n",
                "from dlbcl.tsne_figures import show_inline_preview\n",
                "\n",
                "DATA_DIR = _paths.data_dir\n",
                "ADATA_PATH = _paths.adata_path\n",
                "ARCH_PATH = _paths.arch_path\n",
                'FILTERING_STATUS_COL = "filtering_status"\n',
                'FILTERING_KEEP_VALUE = "Unfiltered"\n',
                'TSNE_BASIS = "tsne_all"\n',
                "DOWNSAMPLE_FRAC = 1\n",
                "PLOT_SEED = 42\n",
                "INLINE_PREVIEW_DPI = 96\n",
                "INLINE_PREVIEW_MAX_WIDTH = 900\n",
                "INLINE_PREVIEW_JPEG_QUALITY = 72\n",
            ]
        )
    if fig_id == "fig2":
        lines.extend(
            [
                "\n",
                "from dlbcl.archetype_heatmap import (\n",
                "    filter_phenotypes,\n",
                "    infer_annotation_type,\n",
                "    make_categorical_color_series,\n",
                "    make_continuous_color_series,\n",
                "    rgb_to_hex,\n",
                "    zscore_rows,\n",
                ")\n",
                "import dlbcl.transcriptome_de as tde\n",
                "from dlbcl.dlbcl_io import CLUSTER_NAME_MAP, ARCHETYPE_DISCOVERY_EXCLUDE_PATIENT_IDS\n",
                "from dlbcl.phenotype_labels import PHENOTYPE_30_HEATMAP_EXCLUDE_TERMS as filter_terms\n",
                "from dlbcl.tsne_figures import (\n",
                "    configure_tsne_matplotlib,\n",
                "    downsample_indices,\n",
                "    map_patient_column,\n",
                "    plot_tsne,\n",
                "    sort_adata_by_obs_category,\n",
                "    valid_label_mask,\n",
                ")\n",
                "\n",
                "configure_tsne_matplotlib()\n",
                "exclude_patient_ids = sorted(ARCHETYPE_DISCOVERY_EXCLUDE_PATIENT_IDS)\n",
            ]
        )
    if fig_id == "supplemental_fig3":
        lines.extend(
            [
                "\n",
                "import math\n",
                "import re\n",
                "\n",
                "from scipy import sparse\n",
                "\n",
                'IMC_LAYER = "sum_unhuddle"\n',
                "GENE_TO_PROTEIN = {\n",
                '    "CD3E": "CD3", "CD3G": "CD3", "CD4": "CD4", "CD8A": "CD8a", "FOXP3": "FOXP3",\n',
                '    "IDO1": "IDO", "GZMB": "GranzymeB", "LAG3": "LAG3", "MS4A1": "CD20", "TBX21": "Tbet",\n',
                '    "PDCD1": "PD1", "VSIR": "VISTA", "ACTA2": "SMA", "TIGIT": "TIGIT", "CD163": "CD163",\n',
                '    "HLA-DRB1": "HLADR", "CD68": "CD68", "ICOS": "ICOS", "HAVCR2": "TIM3", "KI67": "Ki67",\n',
                '    "ITGAX": "CD11c",\n',
                "}\n",
                'DROP_GENES = {"B2M", "HLA-A", "HLA-B"}\n',
                "\n",
                "import dlbcl.transcriptome_de as tde\n",
                "\n",
                "from dlbcl.tsne_figures import (\n",
                "    configure_tsne_matplotlib,\n",
                "    downsample_indices,\n",
                "    map_patient_column,\n",
                "    plot_tsne,\n",
                "    sort_adata_by_obs_category,\n",
                "    valid_label_mask,\n",
                ")\n",
                "\n",
                "configure_tsne_matplotlib()\n",
            ]
        )
    if fig_id == "supplemental_fig2":
        lines.extend(
            [
                "\n",
                "from scipy import sparse\n",
                "\n",
                "import matplotlib as mpl\n",
                "\n",
                'mpl.rcParams["svg.fonttype"] = "none"\n',
                'mpl.rcParams["font.family"] = "DejaVu Sans"\n',
                'IMC_LAYER = "sum_unhuddle"\n',
            ]
        )
    if fig_id in {"fig2", "supplemental_fig2", "supplemental_fig3"}:
        lines.extend(
            [
                "\n",
                "adata_backed = ad.read_h5ad(_paths.adata_path, backed=\"r\")\n",
                "mask = (\n",
                "    adata_backed.obs[\"patient_id\"].isin(PATIENT_SUBSET)\n",
                "    & (adata_backed.obs[FILTERING_STATUS_COL] == FILTERING_KEEP_VALUE)\n",
                ")\n",
                "adata_subset = adata_backed[mask].to_memory()\n",
                "ensure_phenotype_30_labels(adata_subset)\n",
                "ensure_phenotype_30_labels(adata)\n",
                "del adata_backed\n",
                "gc.collect()\n",
                "\n",
                "if TSNE_BASIS not in adata_subset.obsm:\n",
                "    raise KeyError(f\"Missing obsm['{TSNE_BASIS}']; keys: {list(adata_subset.obsm.keys())}\")\n",
                "\n",
                "adata_subset = consolidate_embedded_case_classifications(\n",
                "    adata_subset, _paths, repo_root=REPO_ROOT\n",
                ")\n",
                "print(\n",
                "    f\"Cells: {adata_subset.n_obs:,} | patients: {adata_subset.obs['patient_id'].nunique()}\"\n",
                ")\n",
            ]
        )
    if fig_id == "supplemental_fig3":
        lines.extend(
            [
                "\n",
                'if "gene_expression" not in adata_subset.uns:\n',
                '    raise KeyError("Expected adata_subset.uns[\'gene_expression\']")\n',
                "\n",
                'GE = adata_subset.uns["gene_expression"]\n',
                "if not isinstance(GE, pd.DataFrame):\n",
                '    raise TypeError("uns[\'gene_expression\'] must be a DataFrame (genes × patients)")\n',
            ]
        )
    if fig_id == "fig3":
        lines.extend(
            [
                "\n",
                "import gc\n",
                "import math\n",
                "import re\n",
                "\n",
                "import matplotlib as mpl\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scanpy as sc\n",
                "import seaborn as sns\n",
                "from scipy import stats\n",
                "from scipy.cluster.hierarchy import linkage, leaves_list\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from statsmodels.stats.multitest import multipletests\n",
                "\n",
                "from dlbcl.phenotype_labels import ensure_phenotype_30_labels, read_celltype_colormap_30\n",
                "from dlbcl.segmentation_overlays import run_for_all_fovs\n",
                "\n",
                'mpl.rcParams["svg.fonttype"] = "none"\n',
                'mpl.rcParams["font.family"] = "DejaVu Sans"\n',
                "\n",
                "DATA_DIR = _paths.data_dir\n",
                "adata_full = adata\n",
            ]
        )
    if profile in {"discovery", "discovery_archetype", "discovery_sc", "validation_and_discovery"}:
        lines.extend(
            [
                "\n",
                "from IPython.display import display\n",
                "from dlbcl.dlbcl_io import log_wrote, log_saved, rel_path, write_supplementary_table\n",
                "subset = None  # nb8 classifier associations: filtered AnnData or None for all patients\n",
            ]
        )
    if profile in {"validation", "validation_and_discovery"}:
        lines.extend(
            [
                "\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "\n",
                "from IPython.display import display\n",
                "from dlbcl.dlbcl_io import log_wrote, log_saved, write_supplementary_table, rel_path\n",
                "import dlbcl.validation_figures as vf\n",
                "\n",
                'plt.rcParams["svg.fonttype"] = "none"\n',
                'plt.rcParams["font.family"] = "DejaVu Sans"\n',
            ]
        )
    if fig_id in DISCOVERY_KM_FIG_IDS:
        lines.extend(
            [
                "\n",
                "from dlbcl.discovery_km import (\n",
                "    ENDPOINTS,\n",
                "    KM_HORIZON_YEARS,\n",
                "    plot_km_by_archetype,\n",
                "    plot_km_by_location,\n",
                ")\n",
            ]
        )
    if fig_id in DISCOVERY_PCA_FIG_IDS:
        lines.extend(
            [
                "\n",
                "# nb1 expression PCA constants (Fig S3B cells expect these names).\n",
                "import gc\n",
                "\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scanpy as sc\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "\n",
                "from dlbcl.dlbcl_io import load_discovery_metadata\n",
                "\n",
                "ADATA_PATH = _paths.adata_path\n",
                "RANDOM_SEED = 0\n",
                "N_PCS_TO_FIT = 10\n",
                "PLOT_PCS = [(1, 2), (1, 3)]\n",
                "\n",
                'plt.rcParams["svg.fonttype"] = "none"\n',
                'plt.rcParams["font.family"] = "DejaVu Sans"\n',
                "\n",
                "meta = load_discovery_metadata(adata_subset, PATIENT_SUBSET)\n",
                'adata_subset.uns["gene_expression"] = GE.drop(index="EBER2", errors="ignore")\n',
                'GE = adata_subset.uns["gene_expression"]\n',
            ]
        )
    if fig_id in NB8_SUB_TSNE_FIG_IDS:
        lines.extend(
            [
                "\n",
                "from dlbcl.dlbcl_io import load_omiq_tsne\n",
                "from dlbcl.phenotype_labels import ensure_phenotype_30_labels\n",
            ]
        )
    if fig_id in ELASTIC_NET_FIG_IDS:
        lines.extend(
            [
                "\n",
                "import gc\n",
                "import math\n",
                "import re\n",
                "\n",
                "import matplotlib as mpl\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scanpy as sc\n",
                "import seaborn as sns\n",
                "\n",
                "from dlbcl.dlbcl_io import elastic_net_classifier, log_saved, rel_path\n",
                "\n",
                "ADATA_PATH = _paths.adata_path\n",
                "ARCH_PATH = _paths.arch_path\n",
                "RUN_OOF_BOOTSTRAP_CI = True\n",
                "OOF_BOOTSTRAP_N = 2000\n",
                "OOF_BOOTSTRAP_SEED = 0\n",
                "RUN_TRAINING_ROC = False\n",
                'mpl.rcParams["svg.fonttype"] = "none"\n',
                'mpl.rcParams["font.family"] = "DejaVu Sans"\n',
            ]
        )
    if fig_id in COX_SUPPLEMENT_FIG_IDS:
        lines.extend(
            [
                "\n",
                "import scanpy as sc\n",
                "\n",
                "import dlbcl.validation_cox as vc\n",
            ]
        )
    if fig_id in KM_SENSITIVITY_FIG_IDS:
        if fig_id == "supplemental_fig4":
            lines.extend(
                [
                    "\n",
                    "from dlbcl.dlbcl_io import log_wrote, log_saved, write_supplementary_table\n",
                    "import dlbcl.validation_figures as vf\n",
                ]
            )
        lines.extend(
            [
                "from dlbcl.km_sensitivity import (\n",
                "    DISCOVERY_ARCHETYPE_OUTLIERS,\n",
                "    DISCOVERY_ENDPOINTS,\n",
                "    VALIDATION_EXCLUDE_LOCATIONS,\n",
                "    build_discovery_km_survival_table,\n",
                "    discovery_to_km_frame,\n",
                "    exclude_locations,\n",
                "    exclude_patients,\n",
                "    load_validation_survival_table,\n",
                ")\n",
            ]
        )
    return lines
