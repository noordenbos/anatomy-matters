"""Summarize saved exploratory scores within each classifier's assigned subtypes.

LymphGen Other has no dedicated score. Composite calls contribute one score per
constituent, retained under their composite assignment (never pooled as one score).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dlbcl.classifier_colors import classifier_color

KEY = ['cohort', 'patient_id']
LYMPHGEN_LABELS = ['BN2', 'EZB', 'MCD', 'N1', 'ST2']
DLBCLASS_LABELS = ['C1', 'C2', 'C3', 'C4', 'C5']


def assigned_scores(cases):
    """Return long-form assigned-label scores, preserving missing Other scores."""
    if cases.duplicated(KEY + ['classifier']).any():
        raise ValueError('Duplicate patient within cohort and classifier')
    rows = []
    for _, row in cases.iterrows():
        label = row.assigned_label
        if not isinstance(label, str) or not label:
            raise ValueError('Missing assigned label')
        labels = label.split('/') if label != 'Other' else [None]
        valid = LYMPHGEN_LABELS if row.classifier == 'LymphGen' else DLBCLASS_LABELS
        if row.classifier not in ['LymphGen', 'DLBclass']:
            raise ValueError('Unknown classifier')
        if label == 'Other' and row.classifier != 'LymphGen':
            raise ValueError('Other is only defined for LymphGen')
        for score_label in labels:
            score = np.nan
            if score_label is not None:
                if score_label not in valid:
                    raise ValueError(f'Unknown assigned subtype {score_label}')
                score = float(row[f'score_{score_label}'])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Assigned subtype score must be finite in [0,1]')
            rows.append({**{c: row[c] for c in KEY + ['classifier', 'panel', 'assigned_label']},
                         'score_label': score_label or 'not applicable', 'confidence': score,
                         'no_positive_measured_features': row.get('dlbclass_positive_measured_features', np.nan) == 0})
    return pd.DataFrame(rows)


def summarize_scores(long, by_panel=False):
    groups = ['cohort', 'classifier', 'assigned_label', 'score_label']
    if by_panel:
        groups.insert(2, 'panel')
    return long.groupby(groups, sort=False, dropna=False).agg(
        n_cases=('patient_id', 'nunique'), n_scored=('confidence', 'count'),
        median=('confidence', 'median'), q1=('confidence', lambda x: x.quantile(.25)),
        q3=('confidence', lambda x: x.quantile(.75)), minimum=('confidence', 'min'),
        maximum=('confidence', 'max')).reset_index()


def plot_confidence(cases, long):
    """Patient points and median/IQR, separate panels for classifier and cohort."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.7), sharey=True)
    rng = np.random.default_rng(20261014)
    composites = sorted(set(cases.loc[cases.classifier.eq('LymphGen'), 'assigned_label'])
                        - set(LYMPHGEN_LABELS) - {'Other'})
    for col, cohort in enumerate(['discovery', 'validation']):
        for row, classifier in enumerate(['LymphGen', 'DLBclass']):
            ax = axes[row, col]
            subset = long.loc[long.cohort.eq(cohort) & long.classifier.eq(classifier)]
            labels = LYMPHGEN_LABELS + composites + ['Other'] if row == 0 else DLBCLASS_LABELS
            ticks = []
            for x, label in enumerate(labels):
                group = subset.loc[subset.assigned_label.eq(label)]
                n = group.patient_id.nunique()
                ticks.append(f'{label}\n(n={n})')
                if label == 'Other':
                    ax.text(x, .43, 'No subtype\nscore', ha='center', va='center',
                            fontsize=8, color='#555555')
                    continue
                parts = label.split('/')
                for part_index, part in enumerate(parts):
                    points = group.loc[group.score_label.eq(part) & group.confidence.notna()]
                    values = points.confidence.to_numpy()
                    if not len(values):
                        continue
                    offset = 0 if len(parts) == 1 else (part_index - (len(parts)-1)/2)*.32
                    center = x + offset
                    color = classifier_color(classifier, part)
                    xs = center + rng.uniform(-.10, .10, len(values))
                    hollow = points.no_positive_measured_features.to_numpy() if classifier == 'DLBclass' else np.zeros(len(points), dtype=bool)
                    ax.scatter(xs[~hollow], values[~hollow], s=25, color=color, alpha=.55, edgecolors='none', zorder=3)
                    ax.scatter(xs[hollow], values[hollow], s=30, facecolors='none', edgecolors=color, linewidths=1.2, zorder=5)
                    q1, median, q3 = np.quantile(values, [.25, .5, .75])
                    ax.plot([center, center], [q1, q3], color='#222222', lw=2, zorder=4)
                    ax.plot([center-.10, center+.10], [median, median], color='#222222', lw=2, zorder=4)
                    if len(parts) > 1:
                        ax.text(center, 1.055, part, rotation=90, ha='center', va='bottom',
                                fontsize=7, color=color)
            total = cases.loc[cases.cohort.eq(cohort) & cases.classifier.eq(classifier)].shape[0]
            ax.set_title(f'{cohort.title()} · {classifier} (N={total})', loc='left', fontsize=12, pad=13)
            ax.set_xticks(range(len(labels)), ticks, fontsize=9)
            ax.set_xlim(-.6, len(labels)-.4)
            ax.set_ylim(-.025, 1.20)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='x', length=0, pad=8)
            if col == 0:
                ax.set_ylabel('Assigned-subtype score' if row == 0 else 'Calibrated winning score')
    fig.suptitle('Confidence within assigned genomic subtypes', fontsize=17, x=.07, ha='left', y=.98)
    fig.text(.07, .935, 'Points: individual cases. Black marks: median and interquartile range. Composite LymphGen calls retain both scores.', fontsize=10)
    empty = long.loc[long.classifier.eq('DLBclass') & long.no_positive_measured_features].groupby('cohort').patient_id.nunique()
    fig.text(.07, .018, f"Open circles: no positive measured DLBclass features ({empty.get('discovery', 0)} discovery, {empty.get('validation', 0)} validation cases).\n"
             'Discovery: 57 BLYMFv2 + 4 BLYMF200. Validation: 221 BLYMFv2 (provisional panel assignment).\n'
             'No regional copy-number data; LymphGen A53 is not evaluated. Scores have classifier-specific definitions and are not accuracy estimates.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.07, right=.985, top=.86, bottom=.13, hspace=.43, wspace=.17)
    return fig


def analyze(folder, output):
    """Regenerate tables and figures from a frozen case-score export."""
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(folder/'case_scores.csv')
    long = assigned_scores(cases)
    summary = summarize_scores(long)
    long.to_csv(output/'assigned_label_scores.csv', index=False)
    summary.to_csv(output/'confidence_summary.csv', index=False)
    summarize_scores(long, by_panel=True).to_csv(output/'confidence_summary_by_panel.csv', index=False)
    fig = plot_confidence(cases, long)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'classifier_confidence.{ext}', dpi=200)
    return cases, long, summary, fig
