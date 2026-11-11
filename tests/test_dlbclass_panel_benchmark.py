import unittest
import numpy as np
import pandas as pd
from dlbcl.dlbclass_panel_benchmark import evaluate, proportion_interval, compare_to_full, evaluate_against_full


class PanelBenchmarkTests(unittest.TestCase):
    def cases(self):
        records=[]
        for config,predictions in [('Full_features',['C1','C2','C3','C4','C5']),('panel',['C2','C2','C2','C4','C5'])]:
            for i,pred in enumerate(predictions):
                records.append(dict(configuration=config,patient_id=f'S{i}',reference_label=f'C{i+1}',
                                    DLBclass=pred,panel=config,fish='available',mean_source='none'))
        return pd.DataFrame(records)

    def test_sensitivity_ppv_and_confusion_direction(self):
        metrics,overall,counts=evaluate(self.cases())
        c2=metrics.query("configuration == 'panel' and subtype == 'C2'").iloc[0]
        self.assertEqual((c2.tp,c2.fp,c2.fn),(1,2,0))
        self.assertEqual(c2.sensitivity,1)
        self.assertAlmostEqual(c2.ppv,1/3)
        self.assertEqual(overall.query("configuration == 'panel'").accuracy.iloc[0],.6)
        self.assertEqual(counts.query("configuration == 'panel'")['count'].sum(),5)

    def test_no_predicted_cases_has_undefined_ppv(self):
        metrics,_,_=evaluate(self.cases())
        c1=metrics.query("configuration == 'panel' and subtype == 'C1'").iloc[0]
        self.assertEqual(c1.sensitivity,0)
        self.assertTrue(np.isnan(c1.ppv))

    def test_wilson_interval(self):
        p,lo,hi=proportion_interval(5,10)
        self.assertEqual(p,.5)
        self.assertAlmostEqual(lo,.23659309,places=7)
        self.assertAlmostEqual(hi,.76340691,places=7)

    def test_duplicate_case_rejected(self):
        frame=self.cases()
        with self.assertRaises(ValueError): evaluate(pd.concat([frame,frame.iloc[:1]]))

    def test_mismatched_reference_rejected(self):
        frame=self.cases();frame.loc[5,'reference_label']='C4'
        with self.assertRaises(ValueError): evaluate(frame)

    def test_missing_case_rejected(self):
        with self.assertRaises(ValueError): evaluate(self.cases().iloc[:-1])

    def test_relative_metrics_use_subtype_baseline(self):
        metrics,overall,_=evaluate(self.cases())
        relative,overall_relative=compare_to_full(metrics,overall)
        c2=relative.query("configuration == 'panel' and subtype == 'C2'").iloc[0]
        self.assertEqual(c2.sensitivity_percent_of_full,100)
        self.assertAlmostEqual(c2.ppv_percent_of_full,100/3)
        self.assertAlmostEqual(c2.ppv_change_pp,-200/3)
        panel=overall_relative.query("configuration == 'panel'").iloc[0]
        self.assertEqual(panel.accuracy_percent_of_full,60)
        self.assertEqual(panel.accuracy_change_pp,-40)

    def test_zero_baseline_has_no_defined_ratio(self):
        metrics,overall,_=evaluate(self.cases())
        metrics.loc[metrics.configuration.eq('Full_features')&metrics.subtype.eq('C2'),'sensitivity']=0
        relative,_=compare_to_full(metrics,overall)
        self.assertTrue(relative.loc[relative.subtype.eq('C2'),'sensitivity_percent_of_full'].isna().all())

    def test_full_prediction_comparator_is_bounded_even_when_panel_corrects_error(self):
        frame=self.cases()
        frame.loc[0,'DLBclass']='C2'  # baseline misclassifies C1 as C2
        frame.loc[frame.configuration.eq('panel'),'DLBclass']=['C1','C2','C3','C4','C5']
        metrics,overall,_=evaluate_against_full(frame.sample(frac=1,random_state=1))
        c2=metrics.query("configuration == 'panel' and subtype == 'C2'").iloc[0]
        self.assertEqual(c2.full_feature_n,2)
        self.assertEqual(c2.recall,.5)
        self.assertEqual(c2.precision,1)
        self.assertEqual(overall.query("configuration == 'panel'").agreement.iloc[0],.8)
        self.assertTrue(metrics[['recall','precision']].stack().between(0,1).all())


if __name__=='__main__':unittest.main()
