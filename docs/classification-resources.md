# Resources for fresh classifier inference

The manuscript figures use supplied canonical calls. The two audit notebooks run inference afresh and require upstream software beyond the Python figure environment. Install `requirements-audit.txt` and R, then configure optional resource paths in `config/classification_audit/local.json`. Relative paths are resolved against the repository root; defaults use `vendor/`.

```json
{
  "vendor": "/path/to/native-classifiers",
  "ecotyper_source": "/path/to/prepared-ecotyper",
  "kotlov_utils": "/path/to/BostonGene_LME/lme/utils.py",
  "rscript": "Rscript",
  "promote": false
}
```

Do not commit this local file. `default.json` supplies the other settings, including the source AnnData, output directory, preprocessing scale, random seed and two Lymph2Cx thresholds.

| Resource | Expected location relative to `vendor` / configuration | Upstream |
|---|---|---|
| LymphGen 2.0 | `LymphGen2/src/Model10_8.1.22.R` and `LymphGen2/corefiles/` | [LymphGen portal](https://llmpp.nih.gov/lymphgen/index.php) |
| DLBclass | `DLBclass-tool/src/` and `saved_models/FINALMODEL_NN_evaluation_seeds1_100_folds5_reducedV3.4_removeN5/` with 500 networks | [Author implementation](https://github.com/getzlab/DLBCL-Classifier) |
| HMRN | `DLBCLGenomicSubtyping/data/genomic_data.csv` | [Author reference and fitting code](https://github.com/ecsg-uoy/DLBCLGenomicSubtyping) |
| LymphPlex | `LymphPlex/Functions.R`, `Centers.txt`, `Seeds.txt`; `Matrix.txt` and `TP53.txt` for the public example check | [Author implementation](https://github.com/difuSJTU/LymphPlex) |
| LymphoMAP | Installed R package `LymphoMapR`, including its reference data | [Author package](https://github.com/Green-Lab-MDACC/LymphoMapR) |
| EcoTyper | `ecotyper_source`: recovery script, `pipeline/`, and prepared Lymphoma reference tree | [Author recovery code and instructions](https://github.com/digitalcytometry/ecotyper) |
| Kotlov | `kotlov_utils`: `lme/utils.py` | [Author repository](https://github.com/BostonGene/LME) |

The required EcoTyper reference tree is `EcoTyper/Lymphoma/Lymphoma_Fractions/` containing `Analysis/`, `Cell_States/discovery/` and `Ecotypes/discovery/`. Install/download its lymphoma reference using the author's instructions. The audit creates its own recovery workspace; it does not overwrite the reference.

R packages include `LymphoMapR`, its dependencies, the packages required by EcoTyper's lymphoma recovery workflow, and `cluster`, `dplyr`, `reshape2` for native LymphPlex inference and example masking checks. `flexmix` is needed only to repeat the 1,000-start HMRN reference fit. `tools/classification_audit/fit_hmrn.R` records that procedure; the notebook normally verifies and uses the supplied frozen reference centers, priors and predictions. Reference sample identifiers belong to the published HMRN cohort.

`config/classification_audit/resource_manifest.json` records resource-relative fingerprints of the local software and model files used for this release. R package versions are recorded in the accompanying session metadata. No machine-specific paths or model binaries are supplied in that manifest. Obtain native resources from their original sources; missing files cause a resource-specific error instead of reuse of existing calls.

HGNC symbol/alias annotations and reference gene inventories are included in `config/classification_audit/`. They preserve the release's gene mapping: [HGNC archive](https://www.genenames.org/download/archive/), [LymphoMapR](https://github.com/Green-Lab-MDACC/LymphoMapR), and [EcoTyper](https://github.com/digitalcytometry/ecotyper). Public benchmark source availability is recorded separately inside the external-reference AnnData.

## Input boundary and limitations

Genomic reconstruction starts from selected mutation events and FISH values embedded in the study bundle. It recreates all classifier feature matrices, including measured negatives and unavailable features, without using the current classifier labels as inputs. It does not claim to rerun upstream sequencing, variant calling, pathogenicity filtering, or an unprovided CN assay. Validation BLYMFv2 panel assignment remains provisional.

The expression audit states its preprocessing contracts explicitly. Validation raw normalization can be replayed; discovery starts at the supplied normalized matrix. The reconstructed Kotlov, Ciavarella and Lymph2Cx procedures and partial-input HMRN rules are research implementations with the qualifications described in each notebook.

Run either notebook with `promote: false` to compare fresh predictions against the release. To update a working copy, explicitly enable promotion; a source-hash check, backup and transaction protect against stale or partial writes. Do not modify the frozen release asset itself.

## Preparing a new public bundle

`tools/prepare_release_bundle.py SOURCE OUTPUT --version VERSION` creates a new file, removes local paths from classifier audit provenance, and verifies that all scientific payloads and assignments remain unchanged. It refuses to overwrite an existing output or change the source. For an older source lacking embedded genomic inputs, supply `--selected-inputs DIR` with the verified selected-event, case/panel and FISH tables. Recalculate the paired asset hashes in `release.json` for every new data version; never reuse a previous version's checksum.

The study bundle also includes the pseudonymised discovery staging and prognostic components used by Table 1 in `uns["discovery_clinical_elements"]`. To package an older source, `--clinical-elements FILE` imports only the explicitly allowed T identifier, stage and score/component fields from an existing clinical correction table; other columns are discarded. The clinical helpers prefer this embedded table, so no local clinical sidecar is needed.

LymphPlex uses the unmodified author function with numeric missing values and native PAM available-feature distances, followed by the TP53 priority rule. No missing feature is filled as negative. The case audit records missing features, measured-feature counts and all-completion sensitivity separately from the subtype call. The paper does not independently validate incomplete-panel accuracy. The genomic audit notebook also reruns the public example matrix with selected genes/FISH masked, reporting overall, non-TP53 and per-label retention against full-input native calls.
