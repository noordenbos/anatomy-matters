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

    def test_pies_and_bars_exclude_unavailable_cases_and_report_actual_n(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import matplotlib.pyplot as plt
        from dlbcl.location_pie_patchwork import plot_location_pie_patchwork, plot_location_stacked_bar_patchwork
        frame = pd.DataFrame({
            'Location':['bone','bone','bone','nodal'],
            'genomic_tested':[True,True,False,False],
            'Lymphgen':['Other','MCD','EZB','ST2'],
            'COO_NanoString':['ABC','GCB','ABC','GCB'],
        })
        labels={'Lymphgen':'Wright 2020\nLymphgen','COO_NanoString':'Cell of Origin'}
        close=plt.close
        with TemporaryDirectory() as d:
            with patch('dlbcl.location_pie_patchwork.plt.close') as closed:
                plot_location_pie_patchwork(frame,list(labels),labels,
                    out_svg=Path(d)/'pies.svg',layout='classifiers_rows')
                fig=closed.call_args.args[0]
                self.assertEqual([[t.get_text() for t in ax.texts if t.get_text().startswith('N=')]
                                  for ax in fig.axes], [['N=2'],['N=0'],['N=3'],['N=1']])
                self.assertEqual(len(fig.axes[0].patches),2)
                self.assertIn('50%', [t.get_text() for t in fig.axes[0].texts])
                self.assertIn('No data',[t.get_text() for t in fig.axes[1].texts])
                close(fig)
            with patch('dlbcl.location_pie_patchwork.plt.close') as closed:
                plot_location_stacked_bar_patchwork(frame,['Lymphgen'],labels,
                    out_svg=Path(d)/'bars.svg')
                fig=closed.call_args.args[0]
                self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()],
                                 ['bone (N=2)','nodal (N=0)'])
                self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()],['MCD','Other'])
                close(fig)

    def test_lymphgen_stack_and_legend_put_mcd_left_and_ezb_right(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import matplotlib.pyplot as plt
        from dlbcl.location_pie_patchwork import plot_location_stacked_bar_patchwork
        frame=pd.DataFrame({'Location':['bone']*3,'genomic_tested':[True]*3,
                            'Lymphgen':['EZB','Other','MCD']})
        close=plt.close
        with TemporaryDirectory() as d, patch('dlbcl.location_pie_patchwork.plt.close') as closed:
            plot_location_stacked_bar_patchwork(frame,['Lymphgen'],{'Lymphgen':'Wright 2020\nLymphgen'},out_svg=Path(d)/'bars.svg')
            fig=closed.call_args.args[0];ax=fig.axes[0]
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],['MCD','Other','EZB'])
            self.assertAlmostEqual(ax.patches[0].get_x(),0.)
            self.assertAlmostEqual(ax.patches[-1].get_x(),2/3)
            close(fig)
