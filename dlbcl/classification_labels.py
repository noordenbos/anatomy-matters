"""Canonical classifier labels, established when calls or source data are produced."""
import pandas as pd

CIAVARELLA_LABELS = ('Hot', 'Intermediate', 'Cold')
CIAVARELLA_FIELD_DEFINITION = 'Expression-based microenvironment group: Hot, Intermediate or Cold.'
_MISSING = {'', 'unknown', 'unassigned', 'nan', 'none', '<na>'}


def validate_ciavarella_labels(values):
    """Reject legacy names in current study calls; do not recode on read/display."""
    series = pd.Series(values, copy=False).dropna().astype(str)
    bad = sorted(set(series) - set(CIAVARELLA_LABELS) - {
        value for value in series if value.casefold() in _MISSING})
    if bad:
        raise ValueError('Noncanonical Ciavarella labels: '+', '.join(bad)+
                         '. Normalize external source labels at ingestion; current labels are Hot, Intermediate and Cold.')


def ingest_ciavarella_labels(values):
    """Normalize an external legacy file once, before storing canonical calls."""
    series = pd.Series(values, copy=True).astype(object)
    series = series.map(lambda value: value.strip() if isinstance(value, str) else value)
    series = series.replace({'High': 'Hot', 'Low': 'Cold'})
    validate_ciavarella_labels(series)
    return series


def ingest_ciavarella_table(frame):
    """Canonicalize Ciavarella label fields when importing legacy benchmark CSVs."""
    out = frame.copy()
    if 'classifier' not in out:
        return out
    selected = out['classifier'].eq('Ciavarella')
    for column in ('label', 'subtype', 'full_label', 'panel_label',
                   'full_cluster', 'panel_cluster', 'panel_cluster_aligned'):
        if column in out and selected.any():
            out[column] = out[column].astype(object)
            rows = selected & out[column].ne('All')
            out.loc[rows, column] = ingest_ciavarella_labels(out.loc[rows, column])
    return out


ARCHETYPE_LABELS = {1: 'low immune', 2: 'cytotoxic predominant', 3: 'diverse immune'}
ARCHETYPE_FIELD_DEFINITION = 'Spatial archetype: low immune (LI), cytotoxic predominant (CP) or diverse immune (DI).'


def validate_archetype_labels(values):
    """Reject obsolete or unsupported names instead of changing them at display."""
    series = pd.Series(values, copy=False).dropna().astype(str)
    bad = sorted(set(series) - set(ARCHETYPE_LABELS.values()) - {
        value for value in series if value.casefold() in _MISSING})
    if bad:
        raise ValueError('Noncanonical spatial archetype labels: '+', '.join(bad)+
                         '. Current source labels are low immune, cytotoxic predominant and diverse immune.')
