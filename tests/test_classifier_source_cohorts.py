import tempfile
import unittest
from pathlib import Path
import pandas as pd
from dlbcl.integration_figures import (run_association_analysis,compute_group_enrichment_table,
 compute_classifier_pairwise_associations,HLA_CLASS_STATE_COL)

class ClassifierSourceCohortTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'Location':['bone','nodal','PCNS','testis']*16,
            'tumorimmune_archetype':['low immune','complex immune']*32,
            'genomic_tested':[True]*61+[False]*3,
            'Lymphgen':['MCD','EZB']*32,
            'lymphomap':['FMAC','TEX']*32,
            'Ciav_Cluster':['Hot','Cold']*32,
            'Lymphoma_Ecotype':['LE1','LE2']*26+['unknown']*12,
            HLA_CLASS_STATE_COL:[None]+['HLA-I+/HLA-II+','HLA-I-/HLA-II-']*31+['HLA-I+/HLA-II+'],
        })

    def test_availability_denominators_precede_confidence_and_label_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(self.frame(),stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999).set_index('Metric')
        expected={'Li 2025 LymphoMAP':(64,64),'Wright 2020 Lymphgen':(61,61),
                  'Tumor HLA class state':(63,63),'Steen 2021 EcoTyper (confident)':(52,64)}
        for label,(tested,eligible) in expected.items():
            row=result.loc[label]
            self.assertEqual((row.n_tested,row.n_eligible,row.n_cohort),(tested,eligible,64))

    def test_non_genomic_enrichment_uses_same_cases_regardless_of_genomic_availability(self):
        x=self.frame();other=x.copy();other['genomic_tested']=True
        for stratifier in ['location','archetype']:
            a=compute_group_enrichment_table(x,stratifier=stratifier)
            b=compute_group_enrichment_table(other,stratifier=stratifier)
            a=a[~a.Metric.eq('Wright 2020 Lymphgen')].reset_index(drop=True)
            b=b[~b.Metric.eq('Wright 2020 Lymphgen')].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)

    def test_pairwise_matrix_uses_intersection_of_relevant_assays(self):
        assoc,_,_=compute_classifier_pairwise_associations(self.frame(),columns=[
            ('lymphomap','Li 2025 LymphoMAP'),('Ciav_Cluster','Ciavarella 2018 Cluster'),
            ('Lymphgen','Wright 2020 Lymphgen'),(HLA_CLASS_STATE_COL,'Tumor HLA class state')])
        def n(a,b):return int(assoc[(assoc.classifier_a==a)&(assoc.classifier_b==b)].iloc[0]['n'])
        self.assertEqual(n('Li 2025 LymphoMAP','Ciavarella 2018 Cluster'),64)
        self.assertEqual(n('Li 2025 LymphoMAP','Wright 2020 Lymphgen'),61)
        self.assertEqual(n('Li 2025 LymphoMAP','Tumor HLA class state'),63)
        self.assertEqual(n('Wright 2020 Lymphgen','Tumor HLA class state'),60)
