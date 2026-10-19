import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dlbcl.classification_audit import load_audit_config

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fetch_release_data',ROOT/'tools/fetch_release_data.py')
fetch=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetch)

class ReleasePackagingTests(unittest.TestCase):
    def test_defaults_resolve_from_checkout_and_cannot_implicitly_enable_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'config/classification_audit';p.mkdir(parents=True)
            original=json.loads((ROOT/'config/classification_audit/default.json').read_text())
            (p/'default.json').write_text(json.dumps(original))
            c,_=load_audit_config(root)
            self.assertFalse(c['promote'])
            self.assertEqual(c['source_adata'],str(root.resolve()/'data/DLBCL_location_2026.h5ad'))
            (p/'local.json').write_text('{"promote":"false"}')
            with self.assertRaises(ValueError):load_audit_config(root)

    def test_data_install_checks_hash_and_preserves_mismatched_local_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'code';root.mkdir();source=Path(d)/'assets';source.mkdir()
            data=source/'example.h5ad';data.write_bytes(b'\x89HDF\r\n\x1a\nexample')
            item={'name':data.name,'path':'data/example.h5ad','sha256':fetch.digest(data),'bytes':data.stat().st_size}
            (root/'release.json').write_text(json.dumps({'version':'test','zenodo_record_id':None,'files':[item]}))
            with self.assertRaises(FileNotFoundError):fetch.install(root)
            fetch.install(root,source_dir=source)
            fetch.install(root,check_only=True)
            destination=root/item['path'];destination.write_bytes(b'local working data')
            with self.assertRaises(ValueError):fetch.install(root,source_dir=source)
            self.assertEqual(destination.read_bytes(),b'local working data')
            destination.unlink();data.write_bytes(b'corrupt source')
            with self.assertRaises(ValueError):fetch.install(root,source_dir=source)
            self.assertFalse(destination.exists())
            self.assertFalse(list(destination.parent.glob('*.partial')))

    def test_public_bundle_preserves_payload_and_never_overwrites_source(self):
        import h5py
        import pandas as pd
        from anndata.io import read_elem, write_elem
        spec=importlib.util.spec_from_file_location('prepare_release_bundle',ROOT/'tools/prepare_release_bundle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'working.h5ad';output=Path(d)/'public.h5ad'
            calls=pd.DataFrame({'classifier':['A','unknown']},index=['T1','T2'])
            with h5py.File(source,'w') as f:
                write_elem(f,'uns',{'case_classifications':calls,'classification_inputs':{'schema':'selected-genomic-inputs-v1'},
                    'classification_audit_genomic':{'provenance_json':json.dumps({'source':'/Users/example/private/variants.csv','sha256':'abc'})}})
            original=source.read_bytes()
            clinical=Path(d)/'clinical.csv'
            fields={name:['T1' if name=='IMC_ID' else 'II'] for name in module.CLINICAL_COLUMNS}
            fields['hospital_id']=['private-identifier']
            pd.DataFrame(fields).to_csv(clinical,index=False)
            module.prepare(source,output,'test',clinical_elements=clinical)
            self.assertEqual(source.read_bytes(),original)
            with h5py.File(output,'r') as f:
                pd.testing.assert_frame_equal(read_elem(f['uns/case_classifications']),calls)
                self.assertEqual(list(read_elem(f['uns/discovery_clinical_elements']).columns),module.CLINICAL_COLUMNS)
                provenance=json.loads(f['uns/classification_audit_genomic/provenance_json'].asstr()[()])
                self.assertEqual(provenance,{'source':'resource:variants.csv','sha256':'abc'})
            with self.assertRaises(FileExistsError):module.prepare(source,source,'test')
            with self.assertRaises(FileExistsError):module.prepare(source,output,'test')

    def test_embedded_clinical_corrections_match_explicit_sidecar(self):
        import pandas as pd
        from types import SimpleNamespace
        from dlbcl.validation_clinical_table import prepare_discovery_clinical_frame
        clinical=pd.DataFrame({'patient_id':['T1'],'Location_group':['Bone'],'Age':[62],
            'Gender':[0],'Ann_Arbor_at_Dx':['IV'],'Treatment category':['RCHOP'],
            'tMYC':['negative'],'tBCL2':['negative'],'tBCL6':['negative'],'EBER':['negative'],
            'ipi_mskcc':['low']})
        elements=pd.DataFrame({'IMC_ID':['T1'],'Ann_Arbor_at_Dx':['II'],'ipi_score':[1]})
        adata=SimpleNamespace(uns={'case_clinical':clinical,'case_classifications':pd.DataFrame(
            {'patient_id':['T1'],'COO_NanoString':['GCB']}),'discovery_clinical_elements':elements})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'elements.csv';elements.to_csv(path,index=False)
            explicit=prepare_discovery_clinical_frame(adata,elements_csv=path)
            embedded=prepare_discovery_clinical_frame(adata,repo_root=d)
            pd.testing.assert_frame_equal(embedded,explicit)
            self.assertEqual(embedded.loc['T1','Ann_Arbor_at_Dx'],'II')
