import unittest
import numpy as np
import pandas as pd
from dlbcl.dlbclass_targeted import CONFIG,prepare_inputs

class TargetedDLBclassTests(unittest.TestCase):
    def fixture(self):
        ref=pd.read_csv(CONFIG/'feature_reference.csv',index_col=0)
        gsm=pd.DataFrame(0,index=ref.index,columns=['T2','T1'])
        fish=pd.DataFrame({'myc_final':[np.nan,0],'bcl2_final':[1,0],'bcl6_final':[0,np.nan]},index=['T1','T2'])
        return gsm,fish,ref
    def test_missing_not_negative_and_float_reduction(self):
        g,f,r=self.fixture();g.loc['TP53','T2']=1
        result,mask,reduced=prepare_inputs(g,f)
        self.assertEqual(result.loc['TP53','T2'],2)
        self.assertEqual(result.loc['TP53','T1'],0)
        self.assertEqual(result.loc['SV.BCL2','T1'],3)
        self.assertEqual(result.loc['SV.MYC','T2'],0)
        self.assertAlmostEqual(result.loc['SV.MYC','T1'],r.loc['SV.MYC','reference_mean'])
        self.assertFalse(mask.loc['X17P.DEL','T1'])
        self.assertAlmostEqual(reduced.loc['T2','TP53_biallelic'],2+r.loc['X17P.DEL','reference_mean'])
        self.assertEqual(reduced.index.tolist(),['T2','T1'])
        off=r.query("type == 'Mutation' and not blymfv2_mutation_measured").index[0]
        self.assertAlmostEqual(result.loc[off,'T1'],r.loc[off,'reference_mean'])
        g.loc[off,'T1']=1
        result,mask,_=prepare_inputs(g,f)
        self.assertEqual(result.loc[off,'T1'],2);self.assertTrue(mask.loc[off,'T1'])
    def test_bad_ids_or_unexpected_cn_rejected(self):
        g,f,r=self.fixture()
        with self.assertRaises(ValueError):prepare_inputs(g,f.iloc[:1])
        g.loc['X17P.DEL','T1']=1
        with self.assertRaises(ValueError):prepare_inputs(g,f)
    def test_reference_and_linear_weights(self):
        r=pd.read_csv(CONFIG/'feature_reference.csv',index_col=0)
        w=pd.read_csv(CONFIG/'reduction_weights.csv',index_col=0)
        self.assertEqual(w.shape,(163,21));self.assertTrue(r.reference_mean.between(0,3).all())
        self.assertTrue((w>=0).all().all());self.assertEqual(w.loc['TP53','TP53_biallelic'],1)
