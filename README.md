# Anatomical Origin Defines Tumor–Immune Archetypes in Diffuse Large B-cell Lymphoma

Code, executed notebooks and versioned AnnData files for the manuscript. This checkout is the **v1.1.0 release candidate**. Its data checksums are frozen in `release.json`; the matching Zenodo record will be added when published. The older v1.0.0 data are not interchangeable with this version.

## Install and reproduce

Use Python **3.12 or later**. The recorded execution environment is in `config/environment.json`; Python dependencies are pinned in `requirements.txt`. Native classifier audits have additional requirements below.

```bash
git clone https://github.com/noordenbos/anatomy-matters.git
cd anatomy-matters
# Use the v1.1.0 release checkout/archive when available.
./setup_dev.sh --data-dir /path/to/release-data
source .venv/bin/activate
PYTHONPATH=. python tools/run_all_notebooks.py
```

Windows PowerShell:

```powershell
.\setup_dev.ps1 -DataDir C:\path\to\release-data
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
python tools/run_all_notebooks.py
```

The release-data directory contains both files listed below. Once this candidate is published, omit `--data-dir` / `-DataDir` to download them from the fixed Zenodo record in `release.json`. Setup checks SHA-256 and size, leaves an existing mismatched file untouched, and never substitutes the latest or an older release. It downloads no raw IMC archive. To verify installed files: `python tools/fetch_release_data.py --check`.

For interactive work, run `cd notebooks && jupyter notebook` after activating the environment. `--execute-figures` (PowerShell `-ExecuteFigures`) runs the manuscript notebooks during setup. Figure 2 can take substantially longer than the other notebooks. The runner executes one notebook per process and saves inline outputs. Resume with `python tools/run_all_notebooks.py --only fig2 fig3`.

## Paired data files

| Release asset | Installed path | Contents |
|---|---|---|
| `DLBCL_location_2026.publish.h5ad` | `data/DLBCL_location_2026.h5ad` | Study cell measurements, clinical/expression/genomic data, spatial information, canonical classifier calls, and selected genomic classifier inputs |
| `panel_call_confidence.public.h5ad` | `data/public/panel_call_confidence.public.h5ad` | External benchmark results, feature inventories, and the public source matrices included for those analyses |

The study bundle contains **64 discovery cases and 303 validation cases**. Genomic source inputs support 61 discovery and 221 validation cases. T29, T41 and T48 have `unknown` genomic calls because their source inputs are unavailable. Other missing calls are distinct from classifier residual categories (`Other`, `Others`, `Unclassified`) and from missing-feature abstention (`Unassigned`).

The two authoritative assignment tables are:

- `uns["case_classifications"]` — discovery.
- `uns["validation_cohort"]["case_classification_validation"]` — validation.

LymphPlex coverage and completion sensitivity are recorded separately in `uns["lymphplex_call_audit"]` and `uns["validation_cohort"]["lymphplex_call_audit"]`. The canonical `LymphPlex` columns contain native subtype labels; the sensitivity flag does not act as an abstention rule.

Clinical metadata and model scores are separate. Figure helpers read assignments from these tables. Figure 2 and Figure 5 can recalculate archetypes for transparency; `APPLY_RECOMPUTED_ASSIGNMENTS` defaults to false, and opting in affects only that notebook's in-memory data. It does not rewrite the released file.

External cohorts have individual source/access records. If an unavailable source matrix is requested, the confidence notebooks raise an error identifying the source publication and access route. Archived benchmark metrics are not represented as rerun native model inference. The external-reference bundle does not contain this study's discovery or validation cohorts.

## Notebooks

| Notebook | Output |
|---|---|
| `fig1.ipynb` | Classifier composition and location survival |
| `fig2.ipynb` | Cell landscape, spatial archetypes and archetype survival |
| `fig3.ipynb` | Spatial segmentation, expression programs and HLA |
| `fig4.ipynb` | Discovery integration and associations |
| `fig5.ipynb` | Expression classifier, validation survival and Cox analyses |
| `fig6.ipynb` | Validation integration |
| `table1.ipynb` | Discovery and validation clinical characteristics |
| `supplemental_fig1.ipynb` | Genomic oncoprint and paired classifier summaries |
| `supplemental_fig2.ipynb` | Modality, QC, cell compartments and protein heatmaps |
| `supplemental_fig3.ipynb` | Gene–protein comparisons, PCA, HLA and classifier landscapes |
| `supplemental_fig4.ipynb` | Supplementary Cox analyses |
| `supplemental_fig5.ipynb` | Validation survival, EcoTyper, module scores and GSEA |
| `genomic_panel_call_confidence.ipynb` | Genomic panel feature coverage and external call retention |
| `transcriptomic_panel_call_confidence.ipynb` | Expression panel feature coverage and external call retention |
| `genomic_classification_audit.ipynb` | Fresh genomic classifier inference and comparison with supplied study calls |
| `transcriptomic_classification_audit.ipynb` | Explicit preprocessing, fresh GEP inference and comparison with supplied study calls |

The default batch runner handles the twelve manuscript notebooks. The four classifier notebooks are run explicitly, because native inference has additional dependencies. Each confidence notebook has separate coverage and performance cells for each classifier. Declared feature coverage can run without expression datasets, using `data/public/panel_feature_requirements.json`.

Figure files are written to `figures/`; figure source tables to `data/supplementary/`; audit outputs to `outputs/classification_audit/`. These generated directories are not committed.

## Classifier audit resources and write behavior

See [classification resources](docs/classification-resources.md) for upstream software, models, packages and reference fingerprints. Audit notebooks use portable defaults in `config/classification_audit/default.json`. Optional machine-specific paths belong in the ignored `config/classification_audit/local.json`. Missing native resources cause an explicit error; saved study calls are never substituted for an inference run.

**Audits default to review-only (`promote: false`).** They export inputs, scores, provenance, fresh calls and before/after comparisons. Explicitly enabling promotion creates a backup, verifies the source hash, updates the canonical calls and companion score/state summaries atomically, and verifies unrelated data are unchanged. Run promotion sequentially and on a working copy. A modified AnnData intentionally no longer matches the release checksum.

Genomic preparation starts with selected events, panel assignments and FISH measurements in `uns["classification_inputs"]`. It does not rerun variant calling or upstream pathogenicity/AF selection. Four discovery cases use BLYMF200 and 57 use BLYMFv2; the 221 validation assignments to BLYMFv2 remain **provisional**. Panel membership is not equivalent to sample-specific callable-base coverage. There are no measured regional CN inputs. DLBclass imputes unmeasured inputs using encoded reference means; LymphGen A53 is not evaluated.

| Method | Implementation and qualification |
|---|---|
| LymphGen / DLBclass | Native reference model / author ensemble, with explicit panel masks and method-specific missingness |
| HMRN | Reproduced author reference fit; observed-feature Bernoulli marginalization is a research extension |
| LymphPlex | Native author function with numeric NA and PAM available-feature distances; coverage and completion sensitivity retained separately; incomplete-panel accuracy not separately author-validated |
| LymphoMAP / EcoTyper | Native author software with the declared input scale and cohort handling |
| Kotlov / Ciavarella | Explicit cohort-dependent clustering reconstructions; Kotlov subtype naming uses external reference centroids |
| Lymph2Cx | Full 15-predictor plus five-housekeeping-gene research reconstruction with externally frozen NanoString thresholds; not the proprietary clinical assay |

Validation NanoString normalization is replayed from raw probes and checked against the stored matrix. Discovery starts from the stored normalized log2 matrix; its raw-control preprocessing has not been independently replayed. Do not apply the NanoString Lymph2Cx thresholds automatically to TPM or microarrays.

## Figure labels and colors

`config/classifier_colors.json` stores named palettes, source provenance, aliases and explicit local fallbacks. Plotting helpers use `dlbcl/classifier_colors.py`. Current HMRN calls use biological names; historical C1–C6 display aliases are not reference-mixture component numbers. Unknown palette labels raise an error.

`dlbcl/display_labels.py` centralizes manuscript display names, including the diverse-immune archetype. LymphGen composites are striped in the oncoprint; composition pies retain the documented flat-color fallback. Restart the kernel and rerun affected figures after changing palette configuration.

## R oncoprint

Supplemental Figure 1 requires R ≥4.2 with `svglite` and Bioconductor `ComplexHeatmap`. The batch runner skips this notebook if Rscript is unavailable; a skipped notebook is not a fully reproduced figure suite. The classifier audits have additional R packages listed in the resource guide.

## Verification and release record

`tools/check_release.py` verifies manifests, configuration, notebook execution and public-path hygiene. `tools/publish_notebooks.py` sanitizes saved textual outputs without discarding plots; run its `--check` mode afterward. Tests can be run with `python tools/run_tests.py`.

Release changes are described in [CHANGELOG.md](CHANGELOG.md). The frozen data hashes, model/resource fingerprints and recorded environment identify the inputs used for this version. The original working dataset and maintainer-local configuration are not included in the code archive.

## Citation and license

Noordenbos et al. *Anatomical Origin Defines Tumor–Immune Archetypes in Diffuse Large B-Cell Lymphoma*. [Preprint](https://doi.org/10.2139/ssrn.7279068).

Repository code: [LICENSE](LICENSE). External software and reference data retain their original terms and citations; the repository license does not relicense third-party resources.

Discovery clinical staging and prognostic corrections used by Table 1 are supplied as `uns["discovery_clinical_elements"]`, using only pseudonymised T identifiers and the relevant clinical fields. No local clinical sidecar is required.
