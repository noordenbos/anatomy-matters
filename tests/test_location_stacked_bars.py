import unittest
import pandas as pd
from dlbcl.location_pie_patchwork import stacked_bar_cohort, GENOMIC_CLASSIFIER_COLS


class StackedBarAvailabilityTests(unittest.TestCase):
    def test_common_genomic_cohort_keeps_other_and_missing_calls(self):
        frame = pd.DataFrame({
            'genomic_tested': [True, True, False],
            'Location': ['bone'] * 3,
            'Lymphgen': ['Other', 'unknown', 'MCD'],
            'DLBclass': ['C3', 'C4', 'C5'],
        }, index=['T1', 'T2', 'T3'])
        original = frame.copy(deep=True)
        for classifier in GENOMIC_CLASSIFIER_COLS:
            kept = stacked_bar_cohort(frame, classifier)
            self.assertEqual(kept.index.tolist(), ['T1', 'T2'])
            self.assertEqual(kept.Lymphgen.tolist(), ['Other', 'unknown'])
        pd.testing.assert_frame_equal(frame, original)

    def test_non_genomic_cohort_is_not_restricted(self):
        frame = pd.DataFrame({'genomic_tested': [True, False]})
        self.assertEqual(len(stacked_bar_cohort(frame, 'lymphomap')), 2)

    def test_absent_or_string_flags_are_not_silently_used(self):
        with self.assertRaises(KeyError):
            stacked_bar_cohort(pd.DataFrame({'Location': ['bone']}), 'Lymphgen')
        with self.assertRaises(ValueError):
            stacked_bar_cohort(pd.DataFrame({'genomic_tested': ['False']}), 'DLBclass')
