import unittest
import pandas as pd
from dlbcl.location_pie_patchwork import stacked_bar_cohort
from dlbcl.display_labels import classifier_display_label


class RerunFigureTests(unittest.TestCase):
    def test_shared_complete_cohort_preserves_other_and_unclassified(self):
        source = pd.DataFrame({'genomic_tested':[True,True,True,False],
                               'Lymphgen':['Other','unknown','MCD','EZB'],
                               'DLBclass':['0','C5','C5','C3']})
        before = source.copy(deep=True)
        for col in ['Lymphgen','DLBclass']:
            actual = stacked_bar_cohort(source,col,required_calls=('Lymphgen','DLBclass'))
            self.assertEqual(actual.index.tolist(),[0,2])
        pd.testing.assert_frame_equal(source,before)

    def test_hmrn_uses_configured_manuscript_names_only_in_hmrn_context(self):
        labels = ['NEC','SOCS1/SGK1','TET2/SGK1','MYD88','BCL2','NOTCH2']
        for n,label in enumerate(labels,start=1):
            self.assertEqual(classifier_display_label('Lacy 2020\nHMRN',f'C{n}'),label)
            self.assertEqual(classifier_display_label('DLBclass',f'C{n}'),f'C{n}')
