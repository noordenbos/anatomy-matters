import unittest
import numpy as np
import pandas as pd
from dlbcl.classifier_confidence import assigned_scores, summarize_scores


class ConfidenceTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(cohort='discovery', patient_id='T1', classifier='LymphGen', panel='P', assigned_label='Other', score_MCD=.3, score_ST2=.2),
            dict(cohort='discovery', patient_id='T2', classifier='LymphGen', panel='P', assigned_label='MCD/ST2', score_MCD=.95, score_ST2=.91),
            dict(cohort='validation', patient_id='T2', classifier='LymphGen', panel='P', assigned_label='MCD', score_MCD=.72, score_ST2=.2),
            dict(cohort='discovery', patient_id='T1', classifier='DLBclass', panel='P', assigned_label='C2', score_C2=.81),
        ])

    def test_other_does_not_invent_confidence(self):
        out = assigned_scores(self.frame())
        other = summarize_scores(out).query("assigned_label == 'Other'").iloc[0]
        self.assertEqual(other.n_cases, 1)
        self.assertEqual(other.n_scored, 0)
        self.assertTrue(np.isnan(other['median']))

    def test_composite_preserves_both_scores_and_case_count(self):
        out = assigned_scores(self.frame())
        part = out.query("assigned_label == 'MCD/ST2'")
        self.assertEqual(part.score_label.tolist(), ['MCD','ST2'])
        self.assertEqual(part.confidence.tolist(), [.95,.91])
        self.assertEqual(part.patient_id.nunique(), 1)

    def test_cohorts_and_classifiers_remain_separate(self):
        out = assigned_scores(self.frame())
        self.assertEqual(len(summarize_scores(out)), 5)
        self.assertEqual(out.query("classifier == 'DLBclass'").confidence.iloc[0], .81)

    def test_duplicate_case_fails(self):
        frame=self.frame()
        with self.assertRaises(ValueError):
            assigned_scores(pd.concat([frame,frame.iloc[:1]]))

    def test_missing_assigned_score_fails(self):
        frame=self.frame();frame.loc[1,'score_MCD']=np.nan
        with self.assertRaises(ValueError):
            assigned_scores(frame)


if __name__ == '__main__':
    unittest.main()
