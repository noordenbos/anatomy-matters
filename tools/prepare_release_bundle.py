#!/usr/bin/env python3
"""Prepare a separate public AnnData copy, preserving scientific data and calls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import os
import h5py
import pandas as pd
from anndata.io import read_elem, write_elem
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dlbcl.classification_audit import payload_digest, sha256

CLINICAL_COLUMNS=['IMC_ID','Ann_Arbor_at_Dx','ipi_score','ipi_extranodal','ipi_ldh',
                  'ipi_who','ielsg_ecog','ielsg_ldh','ielsg_csf','ielsg_deepbrain']
EXCLUDED=['uns/classification_inputs','uns/discovery_clinical_elements','uns/release_version',
          'uns/classification_audit_genomic/provenance_json',
          'uns/classification_audit_transcriptomic/provenance_json']


def public_provenance(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            new=public_provenance(key)
            if new in result:raise ValueError(f'Provenance resource-name collision: {new}')
            result[new]=public_provenance(item)
        return result
    if isinstance(value,list):return [public_provenance(x) for x in value]
    if isinstance(value,str) and value.startswith(('/Users/','/home/','/opt/homebrew/')):
        return 'resource:'+Path(value).name
    return value


def prepare(source,output,version,selected_inputs=None,clinical_elements=None):
    source=Path(source).resolve();output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Output must be a new file; working and frozen data are never overwritten')
    original_hash=sha256(source);original_payload=payload_digest(source,EXCLUDED)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.partial',dir=output.parent);os.close(fd);tmp=Path(name)
    try:
        shutil.copyfile(source,tmp)
        with h5py.File(tmp,'r+') as f:
            for group in ['classification_audit_genomic','classification_audit_transcriptomic']:
                key='uns/'+group+'/provenance_json'
                if key in f:
                    value=json.loads(f[key].asstr()[()]);del f[key]
                    write_elem(f['uns/'+group],'provenance_json',json.dumps(public_provenance(value),sort_keys=True))
            if selected_inputs:
                directory=Path(selected_inputs)
                inputs={'schema':'selected-genomic-inputs-v1',
                    'scope':'Pathogenic-selected variants with AF > 0.02 or unavailable AF. Classifier inputs, not raw reads. No CN measurements supplied.',
                    'selected_variants':pd.read_csv(directory/'selected_variants.tsv',sep='\t').fillna(''),
                    'cases':pd.read_csv(directory/'case_metadata.csv')[['cohort','patient_id','panel','panel_evidence']],
                    'panels':pd.read_csv(directory/'panel_coverage.tsv',sep='\t'),
                    'discovery_fish':pd.read_csv(directory/'discovery_fish.csv',index_col=0),
                    'validation_fish':pd.read_csv(directory/'validation_fish.csv',index_col=0)}
                if 'classification_inputs' in f['uns']:del f['uns/classification_inputs']
                write_elem(f['uns'],'classification_inputs',inputs)
            if 'classification_inputs' not in f['uns']:
                raise ValueError('Selected genomic inputs are required; provide --selected-inputs')
            if clinical_elements is not None:
                clinical=pd.read_csv(clinical_elements).loc[:,CLINICAL_COLUMNS].dropna(subset=['IMC_ID']).copy()
                if not clinical['IMC_ID'].astype(str).str.fullmatch(r'T\d+').all():
                    raise ValueError('Clinical input requires pseudonymised discovery T identifiers')
                for column in clinical.select_dtypes(include=['object','string']).columns:
                    clinical[column]=clinical[column].fillna('').astype(str)
                clinical=clinical.reset_index(drop=True)
                clinical.index=clinical.index.astype(str)
                if 'discovery_clinical_elements' in f['uns']:del f['uns/discovery_clinical_elements']
                write_elem(f['uns'],'discovery_clinical_elements',clinical)
            inputs=read_elem(f['uns/classification_inputs'])
            if 'cases' in inputs:
                inputs['cases']['panel_evidence']=inputs['cases']['panel_evidence'].str.replace(
                    'provisional per user','provisional BLYMFv2 assignment',regex=False)
                del f['uns/classification_inputs']
                write_elem(f['uns'],'classification_inputs',inputs)
            if 'release_version' in f['uns']:del f['uns/release_version']
            write_elem(f['uns'],'release_version',version)
        if payload_digest(tmp,EXCLUDED)!=original_payload:raise RuntimeError('Scientific payload changed during packaging')
        if sha256(source)!=original_hash:raise RuntimeError('Source changed during packaging')
        os.replace(tmp,output)
    finally:tmp.unlink(missing_ok=True)
    return {'version':version,'source_sha256':original_hash,'release_sha256':sha256(output),
            'unchanged_scientific_payload_sha256':original_payload,'allowed_changes':EXCLUDED}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--version',required=True);parser.add_argument('--selected-inputs',type=Path)
    parser.add_argument('--clinical-elements',type=Path,help='Existing discovery corrections; only pseudonymised clinical fields are retained')
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.output,args.version,args.selected_inputs,args.clinical_elements),indent=2))
