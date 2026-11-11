import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, MonteCarloMethod
from dlbcl.integration_figures import run_association_analysis, compute_group_enrichment_table

class GlobalEnrichmentAlignmentTests(unittest.TestCase):
    def test_rare_subtypes_retained_but_other_excluded_in_both_panels(self):
        x=pd.DataFrame({'Location':['bone','nodal']*6,
                        'Lymphgen':['MCD']*5+['BN2']+['Other']*4+['unknown']*2,
                        'genomic_tested':[True]*11+[False]})
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(x,stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999,random_seed=7)
        row=result.iloc[0]
        self.assertEqual((row.n_tested,row.n_eligible,row.n_levels),(6,11,2))
        self.assertAlmostEqual(row.assignment_coverage,6/11)
        dots=compute_group_enrichment_table(x,stratifier='location')
        self.assertEqual(set(dots.Feature),{'MCD','BN2'})
        table=pd.crosstab(x.iloc[:6].Lymphgen,x.iloc[:6].Location)
        reference=chi2_contingency(table,correction=False,method=MonteCarloMethod(n_resamples=999,batch=2000,rng=np.random.default_rng(7)))
        self.assertEqual(row.p_value,reference.pvalue)
        self.assertAlmostEqual(row.Cramer_V,np.sqrt(reference.statistic/6))

    def test_two_by_two_strength_is_not_zero_when_exact_style_inference_used(self):
        x=pd.DataFrame({'Location':['bone']*4+['nodal']*4,'KotlovSig':['Depleted']*4+['Inflammatory']*4})
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(x,stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999,random_seed=1)
        self.assertEqual(result.iloc[0].Cramer_V,1.)
        self.assertEqual(result.iloc[0].Test_Type,'Chi-square (Monte Carlo)')

    def test_single_class_reports_coverage_but_no_strength_or_pvalue(self):
        x=pd.DataFrame({'Location':['bone','nodal']*3,'Lymphoma_Ecotype':['LE1']*4+['unknown']*2})
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(x,stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999)
        row=result.iloc[0]
        self.assertEqual((row.n_tested,row.n_eligible),(4,6))
        self.assertTrue(np.isnan(row.Cramer_V) and np.isnan(row.p_value))

    def test_remainder_outputs_excluded_without_removing_biological_intermediate(self):
        x=pd.DataFrame({
            'Location':['bone','nodal']*4,
            'HMRN':['MYD88','BCL2']*3+['NEC','unknown'],
            'COO_NanoString':['ABC','GCB']*2+['UNC','Unclassified','Intermediate','unknown'],
            'Ciav_Cluster':['Cold','Hot','Intermediate','Cold','Hot','Intermediate','Cold','Hot'],
            'DLBclass':['C1','C2']*3+['0','Unclassified'],
            'genomic_tested':[True]*8,
        })
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(x,stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999).set_index('Metric')
        dots=compute_group_enrichment_table(x,stratifier='location')
        expected={
            'Lacy 2020 HMRN':(6,{'MYD88','BCL2'}),
            'Cell of Origin':(4,{'ABC','GCB'}),
            'Ciavarella 2018 Cluster':(8,{'Cold','Hot','Intermediate'}),
            'Chapuy 2025 DLBclass':(6,{'C1','C2'}),
        }
        for metric,(n,features) in expected.items():
            with self.subTest(metric=metric):
                self.assertEqual((result.loc[metric,'n_tested'],result.loc[metric,'n_eligible']),(n,8))
                self.assertEqual(set(dots.loc[dots.Metric==metric,'Feature']),features)
