"""Authoritative case assignments and derived, non-persisted analysis views.

Discovery labels live in uns['case_classifications']; validation labels live in
uns['validation_cohort']['case_classification_validation']. Clinical metadata,
model probabilities and raw measurements are not alternate label sources.
"""
from __future__ import annotations

import pandas as pd

ARCHETYPE_LABELS = {1: 'low immune', 2: 'cytotoxic predominant', 3: 'complex immune'}
ASSIGNMENT_COLUMNS = frozenset({
    'tumorimmune_archetype_id', 'tumorimmune_archetype',
    'pred_tumorimmune_archetype_id', 'pred_tumorimmune_archetype',
    'COO_NanoString', 'KotlovSig', 'KotlovssGSEA', 'Ciav_Cluster', 'lymphomap',
    'Lymphoma_Ecotype', 'New_Ecotype', 'Lymphoma_Ecotype_confident',
    'Lymphoma_Ecotype_imputed', 'Lymphgen', 'DLBclass', 'HMRN', 'LymphPlex',
})
SCORE_COLUMNS = ('prob_1', 'prob_2', 'prob_3', 'max_prob')


def patient_frame(value: object) -> pd.DataFrame:
    """Copy a patient table and validate its join key before aligning any labels."""
    out = pd.DataFrame(value).copy()
    for key in ('patient_id', 'patient_alias'):
        if key in out.columns:
            out = out.set_index(key)
            break
    if out.index.isna().any():
        raise ValueError('Missing patient IDs in case metadata')
    out.index = out.index.astype(str)
    if out.index.has_duplicates:
        raise ValueError('Duplicate patient IDs in case metadata')
    return out


def discovery_assignments(adata) -> pd.DataFrame:
    """Read the discovery table, with no legacy dictionary/CSV fallback."""
    if 'case_classifications' not in adata.uns:
        raise KeyError("Missing authoritative uns['case_classifications']")
    return patient_frame(adata.uns['case_classifications'])


def validation_assignments(validation_uns: dict) -> pd.DataFrame:
    if 'case_classification_validation' not in validation_uns:
        raise KeyError("Missing authoritative validation_cohort['case_classification_validation']")
    return patient_frame(validation_uns['case_classification_validation'])


def discovery_archetype_dict(adata) -> dict[str, int]:
    frame = discovery_assignments(adata)
    ids = pd.to_numeric(frame['tumorimmune_archetype_id'], errors='raise')
    if not ids.isin(ARCHETYPE_LABELS).all():
        raise ValueError('Discovery archetype IDs must be 1, 2 or 3')
    return ids.astype(int).to_dict()


def validation_analysis_frame(validation_uns: dict) -> pd.DataFrame:
    """Join clinical metadata to canonical labels; aliases exist only in this view.

    The ``pred_*`` names are retained for plotting API compatibility. They are
    generated from the case table on every load, never from saved predictions.
    """
    cc = validation_assignments(validation_uns)
    meta = patient_frame(validation_uns['meta'])
    if set(cc.index) != set(meta.index):
        raise ValueError('Validation clinical metadata and case assignments must cover the same patients')
    out = meta.drop(columns=list(ASSIGNMENT_COLUMNS | set(cc.columns)), errors='ignore')
    out = out.join(cc.reindex(out.index), how='left')
    out['pred_tumorimmune_archetype_id'] = out['tumorimmune_archetype_id']
    out['pred_tumorimmune_archetype'] = out['tumorimmune_archetype']
    out.index.name = 'patient_alias'
    return out


def clean_validation_assignment_storage(validation_uns: dict) -> dict:
    """Remove duplicate label payloads, preserving model probabilities separately.

    Returns a new mapping; the canonical case table is not changed. This also
    normalizes older bundles during explicit migration or new bundle creation.
    """
    out = dict(validation_uns)
    cc = validation_assignments(out)
    # Validate clinical alignment before removing anything.
    validation_analysis_frame(out)
    if 'predictions' in out:
        old = patient_frame(out.pop('predictions'))
        if set(old.index) != set(cc.index):
            raise ValueError('Saved predictions and case assignments have different patient IDs')
        scores = old.loc[:, [c for c in SCORE_COLUMNS if c in old]].copy()
        if 'prediction_scores' in out:
            existing = patient_frame(out['prediction_scores'])
            pd.testing.assert_frame_equal(existing.sort_index(), scores.sort_index(), check_names=False)
        out['prediction_scores'] = scores
    meta = pd.DataFrame(out['meta']).copy()
    out['meta'] = meta.drop(columns=list(ASSIGNMENT_COLUMNS), errors='ignore')
    return out


def clean_assignment_storage(uns: dict) -> dict:
    """Remove redundant assignments outside the two canonical case tables."""
    out = dict(uns)
    if 'case_classifications' not in out:
        raise KeyError("Missing authoritative uns['case_classifications']")
    patient_frame(out['case_classifications'])
    for key in ('tumorimmune_archetype_id', 'tumorimmune_archetype_id_raw',
                'abundance_cluster_30', 'abundance_cluster'):
        out.pop(key, None)
    if 'case_clinical' in out:
        out['case_clinical'] = pd.DataFrame(out['case_clinical']).drop(
            columns=list(ASSIGNMENT_COLUMNS), errors='ignore')
    if 'validation_cohort' in out:
        out['validation_cohort'] = clean_validation_assignment_storage(out['validation_cohort'])
    return out


def apply_archetype_assignments(adata, assignments: pd.DataFrame, *, cohort: str) -> None:
    """Explicitly promote reviewed candidates into one canonical table in memory.

    Requires all existing cases exactly once. Does not write the H5AD file.
    Callers must rebuild derived analysis frames after an update. Validation
    probabilities are refreshed when provided, otherwise removed as stale.
    """
    if cohort == 'discovery':
        owner, key = adata.uns, 'case_classifications'
    elif cohort == 'validation':
        owner, key = adata.uns['validation_cohort'], 'case_classification_validation'
    else:
        raise ValueError("cohort must be 'discovery' or 'validation'")
    original = pd.DataFrame(owner[key])
    current = patient_frame(original)
    candidate = patient_frame(assignments).rename(columns={
        'pred_tumorimmune_archetype_id': 'tumorimmune_archetype_id',
        'pred_tumorimmune_archetype': 'tumorimmune_archetype'})
    if candidate.columns.has_duplicates:
        raise ValueError('Ambiguous candidate assignment columns')
    if set(candidate.index) != set(current.index):
        raise ValueError('Candidate assignments must cover exactly the canonical cases')
    ids = pd.to_numeric(candidate['tumorimmune_archetype_id'], errors='raise')
    if not ids.isin(ARCHETYPE_LABELS).all():
        raise ValueError('Candidate archetype IDs must be 1, 2 or 3')
    labels = ids.map(ARCHETYPE_LABELS)
    if 'tumorimmune_archetype' in candidate:
        if not candidate['tumorimmune_archetype'].eq(labels).all():
            raise ValueError('Candidate archetype IDs and labels disagree')
    canonical_ids = current.index.copy()
    current['tumorimmune_archetype_id'] = ids.reindex(current.index).astype(int)
    current['tumorimmune_archetype'] = labels.reindex(current.index)
    if 'patient_id' in original.columns:
        current.index.name = 'patient_id'
        current = current.reset_index()
        current.index = original.index
    owner[key] = current
    if cohort == 'validation':
        owner.pop('predictions', None)
        owner.pop('prediction_scores', None)
        if all(c in candidate for c in SCORE_COLUMNS):
            owner['prediction_scores'] = candidate.loc[canonical_ids, list(SCORE_COLUMNS)].copy()
    else:
        adata.uns.pop('tumorimmune_archetype_id', None)
        adata.uns.pop('tumorimmune_archetype_id_raw', None)
