import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dlbcl.integration_figures import (
    _prepare_classifier_series,
    run_association_analysis,
    _association_dumbbell_rows,
)


class AssociationCutoffTests(unittest.TestCase):
    def test_same_strict_cutoff_for_every_classifier(self):
        values = pd.Series(['A'] * 5 + ['B'] * 6 + ['C'] * 10 + ['unknown'] * 20)
        for label in ['Wright 2020 Lymphgen', 'Steen 2021 EcoTyper (confident)',
                      'Chapuy 2025 DLBclass', 'Li 2025 LymphoMAP']:
            with self.subTest(classifier=label):
                kept = _prepare_classifier_series(values, label)
                self.assertEqual(kept.value_counts().to_dict(), {'C': 10, 'B': 6})

    def test_sparse_ecotypes_produce_a_visible_association(self):
        frame = pd.DataFrame({
            'Lymphoma_Ecotype': ['LE6'] * 17 + ['LE8'] * 6 + ['LE3'] * 5,
            'Location': ['bone', 'nodal', 'PCNS', 'testis'] * 7,
            'tumorimmune_archetype': (['A', 'B', 'C'] * 10)[:28],
            'genomic_tested': True,
        })
        with tempfile.TemporaryDirectory() as d:
            location = run_association_analysis(frame, stratifier='location', out_dir=Path(d), pairwise_csv='loc.csv')
            archetype = run_association_analysis(frame, stratifier='archetype', out_dir=Path(d), pairwise_csv='arch.csv')
        label = 'Steen 2021 EcoTyper (confident)'
        for results in [location, archetype]:
            row = results.set_index('Metric').loc[label]
            self.assertEqual(row.n_tested, 23)
            self.assertEqual(row.n_levels, 2)
            self.assertTrue(np.isfinite(row.Cramer_V))
        groups = _association_dumbbell_rows(location, archetype)
        self.assertIn(label, [r['metric'] for group in groups for r in group])
