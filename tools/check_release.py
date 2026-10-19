#!/usr/bin/env python3
"""Read-only checks on public code, notebook outputs and paired data assets."""
import argparse
import json
from pathlib import Path
import re
import sys
import h5py
import nbformat
from anndata.io import read_elem
from fetch_release_data import install
from prepare_release_bundle import CLINICAL_COLUMNS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dlbcl.public_panel_confidence import load_public_adata


def check(root=ROOT,require_published=False):
    root=Path(root);manifest=json.loads((root/'release.json').read_text());errors=[]
    install(root,check_only=True)
    if require_published and (manifest['status']!='published' or not manifest['zenodo_record_id']):
        errors.append('Publication metadata is not finalized')
    config=json.loads((root/'config/classification_audit/default.json').read_text())
    if config['promote'] is not False:errors.append('Public audit default must be review-only')
    private=re.compile(r'/Users/|/opt/homebrew/')
    for path in (root/'config').rglob('*'):
        if path.is_file() and path.name!='local.json' and private.search(path.read_text(errors='replace')):
            errors.append(f'Local path in {path.relative_to(root)}')
    notebooks=[]
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        if private.search(path.read_text()):errors.append(f'Local path in {path.name}')
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code' and c.source.strip():
                if c.get('execution_count') is None:errors.append(f'Unexecuted cell {path.name}:{i}')
                if any(o.output_type=='error' for o in c.get('outputs',[])):errors.append(f'Error output {path.name}:{i}')
        notebooks.append(path.name)
    if len(notebooks)!=16:errors.append(f'Expected 16 public notebooks, found {len(notebooks)}')
    with h5py.File(root/'data/DLBCL_location_2026.h5ad','r') as f:
        for key,n in [('uns/case_classifications',64),('uns/validation_cohort/case_classification_validation',303)]:
            x=read_elem(f[key]);ids=x.patient_id if 'patient_id' in x else x.index
            if len(x)!=n or not ids.is_unique:errors.append(f'Canonical case count/identity invalid: {key}')
        clinical=read_elem(f['uns/discovery_clinical_elements'])
        if list(clinical.columns)!=CLINICAL_COLUMNS or not clinical['IMC_ID'].str.fullmatch(r'T\d+').all():
            errors.append('Invalid public clinical input fields or identifiers')
        inputs=read_elem(f['uns/classification_inputs'])
        if inputs['cases'].groupby('cohort').size().to_dict()!={'discovery':61,'validation':221}:errors.append('Incorrect genomic input case counts')
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'OUS' and obj.size<100000:
                if private.search(str(obj.asstr()[()])):errors.append(f'Local path in AnnData:{name}')
        f.visititems(visit)
    load_public_adata(root/'data/public/panel_call_confidence.public.h5ad')
    result={'version':manifest['version'],'status':manifest['status'],'notebooks_checked':notebooks,'errors':errors}
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-published',action='store_true')
    check(require_published=parser.parse_args().require_published)
