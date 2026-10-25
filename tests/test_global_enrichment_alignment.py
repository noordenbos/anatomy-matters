import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, MonteCarloMethod
from dlbcl.integration_figures import run_association_analysis, compute_group_enrichment_table

class GlobalEnrichmentAlignmentTests(unittest.TestCase):
    def test_rare_and_other_labels_share_dotplot_population_and_coverage(self):
        x=pd.DataFrame({'Location':['bone','nodal']*6,
                        'Lymphgen':['MCD']*5+['BN2']+['Other']*4+['unknown']*2,
                        'genomic_tested':[True]*11+[False]})
        with tempfile.TemporaryDirectory() as tmp:
            result=run_association_analysis(x,stratifier='location',out_dir=Path(tmp),pairwise_csv='pairs.csv',n_resamples=999,random_seed=7)
        row=result.iloc[0]
        self.assertEqual((row.n_tested,row.n_eligible,row.n_levels),(10,11,3))
        self.assertAlmostEqual(row.assignment_coverage,10/11)
        dots=compute_group_enrichment_table(x,stratifier='location')
        self.assertEqual(set(dots.Feature),{'MCD','BN2','Other'})
        table=pd.crosstab(x.iloc[:10].Lymphgen,x.iloc[:10].Location)
        reference=chi2_contingency(table,correction=False,method=MonteCarloMethod(n_resamples=999,batch=2000,rng=np.random.default_rng(7)))
        self.assertEqual(row.p_value,reference.pvalue)
        self.assertAlmostEqual(row.Cramer_V,np.sqrt(reference.statistic/10))

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
