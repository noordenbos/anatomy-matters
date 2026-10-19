"""Exploratory partition stability under RNA panel restriction.

Cluster numbers are cohort-specific, not published biological subtype calls.
Kotlov's published Louvain outline does not specify a threshold grid or how to
combine its three selection criteria. Those choices are explicit below.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             silhouette_score)

SEED = 20261014
THRESHOLDS = np.round(np.arange(0, .901, .05), 2)


def author_median_scale(scores):
    """Port of BostonGene median_scale: denominator is pandas' former mean MAD.

    DataFrame.mad() meant MEAN absolute deviation about the MEAN, not median
    absolute deviation. Preserve the archived author's implementation explicitly.
    A constant signature contributes zero after centering.
    """
    denominator = (scores - scores.mean()).abs().mean()
    return ((scores - scores.median()) / denominator.replace(0, np.nan)).fillna(0).clip(-3, 3)


def ward_partition(expression, standardize=False):
    """Samples x genes, Euclidean/Ward; three groups, ordered by mean expression."""
    x = expression.copy()
    if standardize:
        x = ((x - x.mean()) / x.std(ddof=0).replace(0, np.nan)).fillna(0)
    labels = cut_tree(linkage(x.to_numpy(), method='ward', metric='euclidean'), n_clusters=3).ravel()
    order = sorted(np.unique(labels), key=lambda k: (-float(x.loc[labels == k].to_numpy().mean()), int(k)))
    names = dict(zip(order, ['High', 'Intermediate', 'Low']))
    return np.array([names[k] for k in labels])


def louvain_partition(scores, threshold, seed=SEED):
    corr = np.corrcoef(np.asarray(scores))
    if not np.isfinite(corr).all():
        raise ValueError('Undefined inter-sample correlation')
    weights = np.where((corr >= threshold) & (corr > 0), corr, 0.)
    np.fill_diagonal(weights, 0)
    graph = nx.from_numpy_array(weights)
    communities = (list(nx.community.louvain_communities(graph, weight='weight', seed=seed, resolution=1))
                   if graph.number_of_edges() else [{i} for i in range(len(scores))])
    communities.sort(key=lambda s: min(s))
    labels = np.empty(len(scores), dtype=int)
    for k, nodes in enumerate(communities):
        labels[list(nodes)] = k + 1
    return labels


def select_louvain(scores, seed=SEED, thresholds=THRESHOLDS):
    rows, partitions = [], {}
    for threshold in thresholds:
        labels = louvain_partition(scores, float(threshold), seed)
        counts = np.unique(labels, return_counts=True)[1]
        valid = 1 < len(counts) < len(scores) and counts.min() / len(scores) >= .05
        row = dict(threshold=float(threshold), clusters=len(counts), min_cluster_n=int(counts.min()), valid=valid)
        if valid:
            row.update(davies_bouldin=davies_bouldin_score(scores, labels),
                       calinski_harabasz=calinski_harabasz_score(scores, labels),
                       silhouette=silhouette_score(scores, labels, metric='euclidean'))
        rows.append(row)
        partitions[float(threshold)] = labels
    table = pd.DataFrame(rows)
    valid = table[table.valid].copy()
    if valid.empty:
        raise ValueError('No threshold yields eligible clusters; do not silently force a solution')
    valid['criterion_rank_sum'] = (valid.davies_bouldin.rank() +
                                    valid.calinski_harabasz.rank(ascending=False) +
                                    valid.silhouette.rank(ascending=False))
    best = valid.sort_values(['criterion_rank_sum', 'threshold']).iloc[0]
    table = table.merge(valid[['threshold', 'criterion_rank_sum']], on='threshold', how='left')
    table['selected'] = table.threshold.eq(best.threshold)
    return partitions[float(best.threshold)], float(best.threshold), table


def compare_partitions(full, panel):
    """Permutation-invariant metrics and one-to-one display alignment.

    Unmatched panel clusters stay explicit. No many-to-one merger inflation.
    """
    full, panel = np.asarray(full).astype(str), np.asarray(panel).astype(str)
    if len(full) != len(panel) or len(full) == 0:
        raise ValueError('Partitions must contain the same nonzero number of samples')
    left, right = sorted(set(full)), sorted(set(panel))
    counts = np.array([[np.sum((full == a) & (panel == b)) for b in right] for a in left])
    ii, jj = linear_sum_assignment(-counts)
    mapping = {right[j]: left[i] for i, j in zip(ii, jj)}
    aligned = np.array([mapping.get(b, 'Unmatched:' + b) for b in panel])
    return dict(n=len(full), full_clusters=len(left), panel_clusters=len(right),
                adjusted_rand=adjusted_rand_score(full, panel),
                normalized_mutual_information=normalized_mutual_info_score(full, panel),
                aligned_agreement=float(np.mean(full == aligned))), aligned, mapping


def cluster_label_metrics(pairs):
    """Recall/precision after the recorded one-to-one cluster alignment.

    A lost full-input cluster has zero recall and undefined precision. An extra
    unmatched panel cluster has zero precision and undefined recall.
    """
    rows = []
    selected = pairs[pairs.comparison.isin(['independently_selected', 'log_expression'])]
    if selected.duplicated(['classifier', 'cohort', 'query_id']).any():
        raise ValueError('Duplicate paired cluster sample')
    for (classifier, cohort), d in selected.groupby(['classifier', 'cohort'], sort=False):
        for label in sorted(set(d.full_cluster) | set(d.panel_cluster_aligned)):
            full, panel = d.full_cluster.eq(label), d.panel_cluster_aligned.eq(label)
            tp, nf, npanel = int((full & panel).sum()), int(full.sum()), int(panel.sum())
            rows.append(dict(classifier=classifier, cohort=cohort, label=label, tp=tp,
                             full_n=nf, panel_n=npanel, recall=tp/nf if nf else np.nan,
                             precision=tp/npanel if npanel else np.nan))
    return pd.DataFrame(rows)


def plot_cluster_metrics(metrics, config, cohorts=('Schmitz', 'Morin', 'Li')):
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    colors = json.loads(Path(config).read_text())['classifiers']['ciavarella']['reference_colors']
    ciav_colors = dict(zip(['High', 'Intermediate', 'Low'], [colors['Cluster 1'], colors['Cluster 2'], colors['Cluster 3']]))
    fig, axes = plt.subplots(4, len(cohorts), figsize=(max(11, 5.3*len(cohorts)), 12), squeeze=False)
    for block, classifier in enumerate(['Kotlov', 'Ciavarella']):
        for col, cohort in enumerate(cohorts):
            d = metrics[(metrics.classifier == classifier) & (metrics.cohort == cohort)].set_index('label')
            labels = sorted(d.index) if classifier == 'Kotlov' else ['High', 'Intermediate', 'Low']
            d = d.reindex(labels)
            for r, metric in enumerate(['recall', 'precision']):
                ax = axes[2*block+r, col]
                y = d[metric].to_numpy(); x = np.arange(len(labels)); good = np.isfinite(y)
                ax.bar(x[good], y[good], color='#274D70', width=.68)
                for j in x:
                    ax.text(j, y[j]+.025 if good[j] else .04, f'{y[j]:.0%}' if good[j] else 'NA', ha='center', fontsize=9)
                ax.set_xticks(x, labels, fontsize=9)
                ax.set_xlim(-.6,len(labels)-.4)
                if classifier == 'Ciavarella':
                    for tick, label in zip(ax.get_xticklabels(), labels):
                        tick.set_color(ciav_colors[label]); tick.set_fontweight('bold')
                ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
                ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True); ax.spines[['right','top']].set_visible(False)
                if col == 0: ax.set_ylabel(f'{classifier}\n{metric.title()} vs full-input clusters')
                if r == 0: ax.set_title(f'{cohort} · n={int(d.full_n.sum())}', loc='left', fontsize=11)
    platform = 'Chapuy microarray sensitivity' if list(cohorts) == ['Chapuy'] else 'RNA-seq clustering'
    fig.suptitle(f'{platform}: recall and precision after BLYM777 restriction', x=.08, ha='left', fontsize=14, y=.985)
    fig.text(.08,.945,'Full-input partitions are the comparator. Kotlov clusters are aligned one-to-one by maximum overlap.',fontsize=9)
    fig.text(.08,.025,'Recall: retained fraction of a full-input cluster. Precision: matching fraction of an aligned panel cluster.\n'
             'Kotlov IDs are cohort-specific, not published biological subtypes; alignment is optimized on these samples. NA = no denominator.\n'
             'Ciavarella inputs are unchanged because all 45 targets are on-panel. This is feature retention, not platform validation.',fontsize=8)
    fig.subplots_adjust(left=.08,right=.985,top=.89,bottom=.13,hspace=.47,wspace=.15)
    return fig


def plot_clustering(summary, transitions, cohorts=('Schmitz', 'Morin', 'Li')):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(cohorts), figsize=(max(11,4.7*len(cohorts)), 8), squeeze=False)
    for col, cohort in enumerate(cohorts):
        row = summary[(summary.classifier == 'Kotlov') & (summary.cohort == cohort) &
                      (summary.comparison == 'independently_selected')].iloc[0]
        ax = axes[0, col]
        vals = [row.adjusted_rand, row.normalized_mutual_information, row.aligned_agreement]
        ax.bar(['ARI', 'NMI', 'Aligned\nagreement'], vals, color=['#38658C', '#6195AF', '#A0BCC4'])
        ax.set_ylim(min(-.05, min(vals) - .05), 1.12)
        ax.axhline(0, color='black', lw=.5)
        for i, value in enumerate(vals):
            ax.text(i, value + .025, f'{value:.3f}', ha='center')
        ax.set_title(f'{cohort} (n={int(row.n)})\n{int(row.full_clusters)} → {int(row.panel_clusters)} clusters')
        ax = axes[1, col]
        d = transitions[(transitions.classifier == 'Kotlov') & (transitions.cohort == cohort)]
        counts = d.pivot(index='full_cluster', columns='panel_cluster_aligned', values='count').fillna(0)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        ax.imshow(fractions, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(len(counts)):
            for j in range(len(counts.columns)):
                value = fractions.iloc[i, j]
                ax.text(j, i, f'{value:.0%}\n(n={int(counts.iloc[i,j])})', ha='center', va='center',
                        color='white' if value > .55 else 'black', fontsize=9)
        ax.set_xticks(range(len(counts.columns)), counts.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(counts)), counts.index)
        ax.set_xlabel('Panel cluster (one-to-one aligned)')
        if col == 0:
            ax.set_ylabel('Full-input cluster')
    fig.suptitle('Kotlov signature-based Louvain clustering: BLYM777 restriction', fontsize=16)
    fig.text(.5, .015, 'C1, C2, … are cohort-specific partitions, not GC/ME/IN/DE labels. Row percentages show splits and mergers.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94))
    return fig
