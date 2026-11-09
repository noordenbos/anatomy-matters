# Supplementary tables and data ownership

Run `notebooks/supplementary_tables.ipynb` to extract the submission tables from
`data/DLBCL_location_2026.h5ad`. `DLBCL_DATA_DIR` can select another prepared
bundle. The notebook does not read original spreadsheets or figure CSVs, and
does not modify AnnData. It writes a single Excel workbook, CSV exports and a
machine-readable workbook specification under `data/supplementary/submission/`.

| Material | Authoritative location | Export behaviour |
|---|---|---|
| Discovery classifications | `uns["case_classifications"]` | Join by case identifier; never read copied figure labels |
| Validation classifications | `uns["validation_cohort"]["case_classification_validation"]` | Join by case identifier |
| Clinical measurements | `uns["case_clinical"]`, `uns["validation_cohort"]["meta"]` | Generate patient views and clinical summaries |
| Discovery analysis overrides | `uns["discovery_clinical_elements"]` | Apply explicitly through the Table 1 preparation function; export recorded and analysis fields separately |
| BLYM777 assay definition | `uns["validation_cohort"]["gep_raw_probes"]` annotation columns | Generate target/control lists; require agreement with processed-probe annotations |
| IMC laboratory records | `uns["supplementary"]["tables"]["imc_antibody_panel"]` and `["imc_staining_protocol"]` | Export identities, dilution and incubation; no volume amounts or volume-only buffer rows |
| Classifier reference information | `uns["supplementary"]` resources and reference memberships | Store definitions once; compute panel membership and coverage at export |
| Numerical figure results | Catalog entries with `kind="derived_result"` | Export captured outputs only while scientific-input and producer-code fingerprints match |

The catalog records each table's canonical path, original source, producer
notebook and role. Stored tables contain an inspectable `data` DataFrame and a
null mask. Masks preserve missingness without turning missing values into zero.
The `numeric_text_mask` preserves numeric cells in columns that also contain a
literal `inf` result. Hashes protect the stored tables and reference metadata.

## A single current authority, with explicit provenance

Raw counts and normalized expression are different processing stages. Historical
classification audits describe past changes; they are not current assignment
tables. Source pathology records cover a different sample set and remain source
records. These distinctions preserve provenance without making them fallback
sources of current calls.

Two redundant entries have been removed: `uns["gene_expression_columns"]` is
replaced by the expression-matrix index, and genomic `location_class` is generated
from the canonical discovery case table. The oncoprint notebook uses
`genomic_profiling_view(adata)` to construct that annotation in memory. Table 1
uses embedded clinical components rather than an optional external CSV.

Some clinical source fields require adjudication: recorded versus analysis stage
for T31, and descriptive versus numeric survival events for the cases listed in
the workbook. This migration does not choose clinical values or change any
classification. The recorded fields and the explicit analysis overrides remain
distinguishable. They must be reviewed before consolidating those clinical fields.

## Capturing and refreshing numerical results

The initial migration imported the reviewed notebook outputs and laboratory
records. It did not recompute the statistical models. Their original source
descriptions and the reviewed-material hash are retained in AnnData. The
scientific payload and producer code are fingerprinted at capture time.

If scientific inputs or producer code change, export fails rather than silently
reusing old results. Recompute the affected analyses, assemble and review a
complete updated workbook specification, then refresh into a **new** AnnData
file using `tools/migrate_supplementary_sources.py`. Supply the expected source
SHA-256 and use `--replace-supplementary` only for an explicit refresh of an
existing namespace. The tool validates the original source hash, verifies that
unrelated scientific content is unchanged and writes atomically. It does not
overwrite its source file or silently import CSVs during ordinary extraction.

The generated Excel/CSV files are exports. Edits to an export do not update
AnnData. Patient corrections belong in the canonical clinical tables, and
reviewed classification changes belong in the canonical assignment tables.
