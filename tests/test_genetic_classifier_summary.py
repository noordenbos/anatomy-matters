import unittest
from types import SimpleNamespace
import pandas as pd
from dlbcl.genetic_classifier_summary import annotation_table,component_colors,cohort_pair_table,pair_counts,display_assignment

class GeneticSummaryTests(unittest.TestCase):
    def fixture(self):
        cc=pd.DataFrame({'patient_id':['T1','T2','T3'],'Lymphgen':['BN2/MCD','Other','unknown'],'DLBclass':['C5','0','']})
        return SimpleNamespace(uns={'case_classifications':cc,'genomic_profiling':{'events':pd.DataFrame({'patient_id':['T1','T2'],'gene':['MYD88','TP53'],'alteration':['HOT','NON']})},'validation_cohort':{}})
    def test_annotations_follow_explicit_mutation_order(self):
        table=annotation_table(self.fixture(),['T2','T1'])
        self.assertEqual(table.index.tolist(),['T2','T1'])
        self.assertEqual(table.LymphGen.tolist(),['Other','BN2/MCD'])
        self.assertEqual(len(component_colors('BN2/MCD','Lymphgen')),2)
        with self.assertRaises(ValueError):annotation_table(self.fixture(),['T4'])
        with self.assertRaises(ValueError):annotation_table(self.fixture(),['T1','T1'])
    def test_distinct_missing_and_residual_categories(self):
        self.assertEqual(display_assignment('Other','Lymphgen'),'Other')
        self.assertEqual(display_assignment('unknown','Lymphgen'),'Unknown assignment')
        self.assertEqual(display_assignment('','Lymphgen'),'Unavailable')
        self.assertEqual(display_assignment('0','DLBclass'),'Unclassified')
    def test_denominators_preserve_missing_composite_and_residual_calls(self):
        frame,info=cohort_pair_table(self.fixture(),'discovery')
        counts=pair_counts(frame[frame.has_embedded_genomic_data])
        self.assertEqual(info['n_all_cases'],3);self.assertEqual(info['n_with_embedded_genomic_data'],2)
        self.assertEqual(info['n_with_both_outputs_and_embedded_genomic_data'],1)
        self.assertEqual(counts.to_numpy().sum(),2)
        self.assertEqual(counts.loc['BN2/MCD','C5'],1)
        self.assertEqual(counts.loc['Other','Unclassified'],1)
