# Release changes

## 1.1.0 — release candidate

- Consolidate discovery and validation calls into two authoritative AnnData tables.
- Use native LymphPlex PAM missing-value handling to resolve 17 discovery and 84 validation calls; retain coverage and completion sensitivity separately, without changing previously assigned labels or other classifiers.
- Include the verified genomic and GEP reclassification, panel-specific input handling, biological HMRN labels, and explicit missing-input abstention. Unsupported discovery genomic calls are unknown.
- Add four public notebooks for genomic/transcriptomic call confidence and auditable inference. Keep exploratory development notebooks outside the release.
- Include selected genomic measurement inputs in the study AnnData, allowing feature reconstruction without maintainer workbooks or private script paths.
- Distribute a separate external-reference benchmark AnnData and declared panel feature inventories.
- Filter genomic pies and stacked bars to the same source-supported cohort, label each pie denominator, and explicitly embed inline bar/KM previews.
- Order Wright/LymphGen stacked bars and legends from MCD on the left to EZB on the right, with Other between them.
- Exclude LymphGen Other from association tests and enrichment dotplots, retaining it in descriptive composition plots.
- Match global dumbbell populations to the dotplots, retaining observed rare categories and native confidence rules; use reproducible Monte Carlo p-values and show assignment coverage. Classifier-pair heatmaps and legacy pairwise-group tables retain their >5-case cutoff; survival statistics are unchanged.
- Retain a GC-like row in Kotlov enrichment dotplots when no cases are assigned, using small grey display-only placeholders.
- Centralize classifier colors and manuscript labels; fix the Figure 1 missing location-count definition.
- Make audit configuration portable and review-only by default. Record native resource fingerprints and dependency versions.
- Pair code and both AnnData files through a release manifest; verify SHA-256 on download and reject mismatched existing data without overwriting it.
- Remove local filesystem paths from public provenance and saved textual notebook outputs while retaining method parameters, source hashes and figures.

- Embed the existing pseudonymised discovery clinical corrections used by Table 1.
- Generate Supplemental Figure 1 benchmark metrics directly from the public benchmark AnnData.

Validation panel assignments and discovery raw-normalization provenance retain the qualifications documented in the README. This release does not introduce measured CN data or convert research reconstructions into validated clinical assays.

All 16 notebooks executed successfully in a clean environment; 77 automated checks passed. Fresh classifier calls match the supplied assignments. The publication date and fixed Zenodo record will be added after the data upload.
