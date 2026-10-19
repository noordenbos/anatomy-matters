import numpy as np
import pandas as pd
from dlbcl.rna_clustering_benchmark import compare_partitions, author_median_scale, ward_partition, cluster_label_metrics


def test_lost_and_extra_clusters_are_not_dropped():
    d=pd.DataFrame(dict(classifier=['Kotlov']*4,cohort=['test']*4,comparison=['independently_selected']*4,
                        query_id=['a','b','c','d'],full_cluster=['A','B','B','C'],
                        panel_cluster_aligned=['A','A','Unmatched:Z','A']))
    m=cluster_label_metrics(d).set_index('label')
    assert m.loc['B','recall']==0 and np.isnan(m.loc['B','precision'])
    assert m.loc['C','recall']==0 and np.isnan(m.loc['C','precision'])
    assert m.loc['Unmatched:Z','precision']==0 and np.isnan(m.loc['Unmatched:Z','recall'])
    assert m.loc['A','recall']==1 and m.loc['A','precision']==1/3


def test_label_permutation_and_cluster_split():
    metrics, aligned, _ = compare_partitions(['A', 'A', 'B', 'B'], ['2', '2', '1', '1'])
    assert metrics['adjusted_rand'] == metrics['normalized_mutual_information'] == metrics['aligned_agreement'] == 1
    assert list(aligned) == ['A', 'A', 'B', 'B']
    metrics, aligned, _ = compare_partitions(['A', 'A', 'B', 'B'], ['X', 'Y', 'Z', 'Z'])
    assert metrics['aligned_agreement'] == .75 and metrics['adjusted_rand'] < 1
    assert sum(str(x).startswith('Unmatched:') for x in aligned) == 1


def test_author_scaling_uses_mean_absolute_deviation():
    # Distinguishes historical pandas MAD from median absolute deviation.
    x = pd.DataFrame({'signature': [0., 1., 2., 9.], 'constant': [4.] * 4})
    expected = np.clip((x.signature - 1.5) / 3, -3, 3)
    assert np.allclose(author_median_scale(x).signature, expected)
    assert np.array_equal(author_median_scale(x).constant.to_numpy(), np.zeros(4))


def test_ward_keeps_partition_when_irrelevant_genes_are_removed():
    x = pd.DataFrame([[1., 1.], [1.1, 1.2], [4., 4.], [4.1, 4.2], [9., 9.], [9.1, 9.2]])
    for zscore in [False, True]:
        labels = ward_partition(x, zscore)
        assert len(set(labels)) == 3
        assert labels[0] == labels[1] == 'Low'
        assert labels[-1] == labels[-2] == 'High'


if __name__ == '__main__':
    test_lost_and_extra_clusters_are_not_dropped()
    test_label_permutation_and_cluster_split()
    test_author_scaling_uses_mean_absolute_deviation()
    test_ward_keeps_partition_when_irrelevant_genes_are_removed()
    print('4 clustering metric/scaling checks passed')
