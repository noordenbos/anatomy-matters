import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from anndata.io import write_elem, read_elem
from dlbcl.classification_audit import preprocess_expression, promote_calls, sha256, CASE_KEYS
from dlbcl.genomic_classification_audit import bernoulli_posterior, hmrn_features, lymphplex_partial

class AuditTests(unittest.TestCase):
    def test_scale_and_no_double_correction(self):
        x=pd.DataFrame([[0.,3.],[7.,15.]],index=['a','b'],columns=['p','q'])
        transformed,_=preprocess_expression(x,'rna_tpm')
        np.testing.assert_allclose(transformed,[[0,2],[3,4]])
        retained,_=preprocess_expression(transformed,'rna_log2_tpm1')
        pd.testing.assert_frame_equal(retained,transformed)
        with self.assertRaises(ValueError):preprocess_expression(x,'nanostring_log2_normalized',background='none')
        with self.assertRaises(ValueError):preprocess_expression(x,'rna_raw_counts')
        with self.assertRaises(ValueError):preprocess_expression(x,'microarray_raw')
    def test_raw_controls_explicit_background(self):
        x=pd.DataFrame({'Probe':['pos','neg1','neg2','hk','gene'],'Code.Class':['Positive','Negative','Negative','Housekeeping','Endogenous'],'s':[100,2,4,20,40]})
        with self.assertRaises(ValueError):preprocess_expression(x,'nanostring_raw_counts')
        y,a=preprocess_expression(x,'nanostring_raw_counts',background='negative_mean',housekeeping=['hk'],pseudocount=0)
        self.assertEqual(a.loc['s','background'],3)
        self.assertAlmostEqual(y.loc['gene','s'],np.log2(37/17))
        self.assertEqual(y.loc['hk','s'],0)
    def test_marginalization_equals_enumeration(self):
        centers=pd.DataFrame({'A':[.8,.4],'B':[.2,.9]},index=['x','y'])
        x=pd.DataFrame([[1,np.nan],[np.nan,np.nan]],columns=centers.index)
        p=bernoulli_posterior(x,centers,[.5,.5])
        np.testing.assert_allclose(p,[[.8,.2],[.5,.5]])
        self.assertFalse(np.allclose(p.iloc[0],bernoulli_posterior(x.fillna(0),centers,[.5,.5]).iloc[0]))
    def test_or_feature_mutation_is_positive_unknown_cn_not_negative(self):
        v=pd.DataFrame({'patient_id':['a'],'gene':['TP53'],'effect':['MUTATION'],'aa_change':['p.R175H']})
        c=pd.DataFrame({'patient_id':['a','b'],'panel':['panel','panel']})
        panel=pd.DataFrame({'approved_symbol':['TP53'],'panel':[True]})
        x,_=hmrn_features(v,c,panel,['TP53_OR_del'],None,lambda g:g)
        self.assertEqual(x.loc['a','TP53_OR_del'],1)
        self.assertTrue(pd.isna(x.loc['b','TP53_OR_del']))
    def test_lymphplex_does_not_invent_missing_negative(self):
        centers=pd.DataFrame({'g':[0.,1.]},index=['A','B'])
        seeds=pd.DataFrame({'Cluster':['B'],'Seed':['g']})
        x=pd.DataFrame({'g':[np.nan,0],'TP53':[0,1]},index=['unknown','tp53'])
        calls,_=lymphplex_partial(x,centers,seeds)
        self.assertEqual(calls.loc['unknown','call'],'Unassigned')
        self.assertEqual(calls.loc['tp53','call'],'TP53')
    def test_transaction_and_stale_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.h5ad'
            with h5py.File(path,'w') as f:
                f.create_dataset('X',data=np.arange(12).reshape(3,4))
                dtype=np.dtype([('gene',h5py.string_dtype()),('value','f8')])
                f.create_dataset('compound',data=np.array([('gene',0.25)],dtype=dtype))
                for c,k in CASE_KEYS.items():
                    write_elem(f,k,pd.DataFrame({'call':['old','old'],'clinical':[3,4]},index=['a','b']))
                write_elem(f,'uns/other',{'keep':'exactly'})
            digest=sha256(path)
            updates={'Discovery':pd.DataFrame({'call':['fresh']},index=['b'])}
            changes=promote_calls(path,updates,Path(tmp)/'audit',audit_name='test',run_provenance={},expected_source_sha256=digest)
            self.assertTrue(changes.changed.all())
            with h5py.File(path) as f:
                self.assertEqual(read_elem(f[CASE_KEYS['Discovery']]).loc['a','call'],'old')
                self.assertEqual(read_elem(f[CASE_KEYS['Discovery']]).loc['b','call'],'fresh')
                self.assertEqual(read_elem(f['uns/other'])['keep'],'exactly')
                np.testing.assert_array_equal(f['X'][:],np.arange(12).reshape(3,4))
            self.assertEqual(len(list(Path(tmp).glob('*.before_test_*.h5ad'))),1)
            with self.assertRaises(RuntimeError):promote_calls(path,updates,Path(tmp)/'audit',audit_name='test',run_provenance={},expected_source_sha256=digest)

if __name__=='__main__': unittest.main()
