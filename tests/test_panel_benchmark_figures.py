import unittest
import pandas as pd
from dlbcl.panel_benchmark_figures import summarize_transitions

class TransitionTests(unittest.TestCase):
    def test_direction_and_composite_pooling_preserve_counts(self):
        frame=pd.DataFrame({'classifier':['LymphGen v2']*3,'panel':['BLYMFv2']*3,'fish':['available']*3,
            'full_label':['MCD','MCD','BN2/MCD'],'panel_label':['Other','MCD','MCD/ST2']})
        exact,display=summarize_transitions(frame)
        self.assertEqual(exact['count'].sum(),3)
        self.assertEqual(display['count'].sum(),3)
        loss=exact.query("full_label == 'MCD' and panel_label == 'Other'").iloc[0]
        self.assertEqual(loss.row_fraction,.5)
        self.assertEqual(display.query("full_label == 'Composite' and panel_label == 'Composite'")['count'].iloc[0],1)
        self.assertEqual(exact.query("full_label == 'BN2/MCD'").panel_label.iloc[0],'MCD/ST2')

if __name__=='__main__':unittest.main()
