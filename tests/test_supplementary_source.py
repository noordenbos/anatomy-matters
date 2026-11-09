"""Regression checks for source ownership and stale-output protection."""
from pathlib import Path
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import unittest
import tempfile
from anndata.io import write_elem

from dlbcl.supplementary_source import encode_table, decode_table, probe_annotations, load_source
from dlbcl.classification_audit import payload_digest
from dlbcl.genomic_profiling import genomic_profiling_view


class SupplementarySourceTests(unittest.TestCase):
    def test_table_storage_preserves_nulls_numbers_and_literal_infinity(self):
        data=pd.DataFrame({'OR':[1.25,'inf',None], 'available':[True,False,None], 'gene':['SEPTIN1','MARCHF1',None]})
        restored=decode_table(encode_table(data))
        assert restored.to_dict('list')==data.to_dict('list')
        assert isinstance(restored.loc[0,'OR'],float)


    def test_probe_annotations_refuse_a_second_conflicting_definition(self):
        raw=pd.DataFrame({'Probe':['A','HK','POS'], 'Accession':['1','2','3'],'Code.Class':['Endogenous','Housekeeping','Positive']})
        processed=raw.iloc[:2].copy();processed.loc[1,'Code.Class']='Endogenous'
        with self.assertRaises(AssertionError):
            probe_annotations({'validation_cohort':{'gep_raw_probes':raw,'gep_normalized_probes':processed}})


    def test_stale_numeric_results_rejected_before_export(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        tmp_path=Path(temporary.name)
        path=tmp_path/'data.h5ad'
        with h5py.File(path,'w') as f:
            uns=f.create_group('uns')
            for name in ['case_clinical','case_classifications','discovery_clinical_elements','classification_inputs','genomic_profiling']:
                write_elem(uns,name,{'value':1})
            vc=uns.create_group('validation_cohort')
            for name in ['meta','case_classification_validation','gep_raw_probes','gep_normalized_probes']:write_elem(vc,name,{'value':1})
        digest=payload_digest(path,['uns/supplementary'])
        with h5py.File(path,'r+') as f:
            write_elem(f['uns'],'supplementary',{'schema':'supplementary-source-v1','scientific_payload_sha256':digest})
            f['uns/case_clinical/value'][...]=2
        with self.assertRaisesRegex(ValueError,'Scientific inputs changed'):
            load_source(path)


    def test_oncoprint_annotation_uses_canonical_location(self):
        calls=pd.DataFrame({'patient_id':['T1','T2'],'Location':['bone','PCNS']})
        adata=SimpleNamespace(uns={'case_classifications':calls,'genomic_profiling':{'events':pd.DataFrame({'patient_id':['T1','T2'],'gene':['X','Y']}),
            'location_class':pd.DataFrame({'patient_id':['T1','T2'],'Location':['testis','testis'],'Location_class':['Testis','Testis']})}})
        actual=genomic_profiling_view(adata)['location_class'].set_index('patient_id')
        assert actual.loc['T1','Location']=='bone'
        assert actual.loc['T2','Location']=='PCNS'

if __name__ == '__main__':
    unittest.main()
