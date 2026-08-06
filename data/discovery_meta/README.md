# Discovery clinical meta

Optional sidecar used by `table1.ipynb` / `dlbcl.validation_clinical_table` when
`adata.uns['case_clinical']` lacks a complete primary `ipi_ielsg` bucket for a
patient.

- `discovery_clinical_elements.csv` — partial IPI/IELSG component scores keyed by `IMC_ID` (= `patient_id`)
