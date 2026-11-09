#!/usr/bin/env python3
"""Explicit, transactional import of reviewed supplementary material into AnnData.

This is a one-time migration, not an automatic CSV fallback used by notebooks.
The source file is never overwritten. Scientific values are preserved; only
verified redundant storage is removed. Review the returned audit before promotion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, write_elem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dlbcl.classification_audit import payload_digest, sha256
from dlbcl.supplementary_source import SCHEMA, NAMESPACE, encode_table, decode_table, table_digest, load_source, producer_digest

REMOVABLE=['uns/gene_expression_columns','uns/genomic_profiling/location_class']


def migrate(source, reviewed_json, output, *, expected_source_sha256, replace_existing=False):
    source=Path(source);output=Path(output);reviewed_json=Path(reviewed_json)
    if output.exists():raise FileExistsError('Choose a new output; migration never overwrites a data file')
    if sha256(source)!=expected_source_sha256:raise ValueError('Source SHA-256 differs from reviewed release')
    sheets=json.loads(reviewed_json.read_text())['sheets']
    by_name={s['name']:s for s in sheets}
    def frame(name):
        s=by_name[name];return pd.DataFrame(s['rows'],columns=s['headers'])
    catalog=[];stored={}
    dynamic={name for name in ['S1 Discovery','S1 Validation','S1 Cohort summary','S1 Field definitions','S3 1D_pie_patients','S4 BLYM777 targets','S4 Assay controls','S4 Gene symbol mapping','S5 RNA reference genes','S5 RNA coverage','S5 Kotlov signatures']}
    roots={'S1 Discovery':'uns/case_clinical + uns/case_classifications + uns/discovery_clinical_elements (explicit analysis overrides)',
           'S1 Validation':'uns/validation_cohort/meta + uns/validation_cohort/case_classification_validation',
           'S1 Cohort summary':'Canonical clinical tables, aggregated by table1 code',
           'S1 Field definitions':'uns/supplementary/field_definitions_json + current canonical patient views',
           'S3 1D_pie_patients':'uns/case_classifications + uns/classification_inputs/cases (derived figure view)',
           'S4 BLYM777 targets':'uns/validation_cohort/gep_raw_probes: Probe, Accession, Code.Class',
           'S4 Assay controls':'uns/validation_cohort/gep_raw_probes: Probe, Accession, Code.Class',
           'S4 Gene symbol mapping':'Raw-probe identifiers + uns/supplementary/gene_aliases_json',
           'S5 RNA reference genes':'uns/supplementary/reference_gene_memberships + canonical assay annotations',
           'S5 RNA coverage':'Reference memberships + canonical assay annotations (computed at export)',
           'S5 Kotlov signatures':'uns/supplementary/kotlov_signatures + canonical assay annotations'}
    for i,sheet in enumerate(sheets):
        name=sheet['name']
        if name in {'Contents','Read me'}:continue
        item={k:sheet[k] for k in ['name','title','section','notebook','note']}
        item['original_source']=sheet['source']
        if name in dynamic:
            item.update(kind='view',canonical_source=roots[name])
        else:
            key='imc_staining_protocol' if name=='S2 Staining protocol' else 'imc_antibody_panel' if name=='S2 IMC antibody panel' else 'table_'+str(i)
            data=frame(name)
            if name=='S2 Staining protocol':
                data=data.loc[~data['Target or component'].eq('Buffer'),[c for c in data if 'volume' not in c.lower()]].reset_index(drop=True)
                item['note']='ON=overnight; RT=room temperature; 4C=4 °C. Antibody identity, dilution and incubation conditions are retained. CD30 has no recorded protocol entry; SMACT supplier differs from the panel record. CD57 was excluded after staining failure.'
            kind='resource' if sheet['section'] in {'S2','S5'} and name not in {'S5 DNA features','S5 LymphoMAP selected'} else 'derived_result'
            entry=encode_table(data)
            entry['table_sha256']=table_digest(decode_table(entry))
            entry['provenance_json']=json.dumps({'kind':kind,'original_source':sheet['source'],'producer_notebook':sheet['notebook'],
                'imported_from_reviewed_material_sha256':sha256(reviewed_json),'capture':'Imported reviewed output; not recomputed during migration'},sort_keys=True)
            stored[key]=entry
            item.update(kind=kind,key=key,canonical_source=f'uns/supplementary/tables/{key}/data')
        if name in {'S4 BLYM777 targets','S4 Assay controls'}:
            item['note']='Assay identifiers and code classes are extracted from the canonical raw-probe annotations. Processed-probe annotations are checked for agreement. Gene aliases appear in S4 Gene symbol mapping.'
        if name=='S5 RNA reference genes':item['kind']='view'
        catalog.append(item)
    aliases=frame('S4 Gene symbol mapping')
    aliases=dict(zip(aliases.loc[aliases['Assay gene label'].ne(aliases['Classifier gene symbol']),'Assay gene label'],aliases.loc[aliases['Assay gene label'].ne(aliases['Classifier gene symbol']),'Classifier gene symbol']))
    definitions={r['Field']:r['Definition'] for r in frame('S1 Field definitions').to_dict('records')}
    memberships=frame('S5 RNA reference genes').drop(columns=['On BLYM777'])
    coverage=frame('S5 RNA coverage')[['Classifier','Subtype','Feature definition','Scope']].to_dict('records')
    readme=[list(row) for row in by_name['Read me']['rows']]
    for row in readme:
        if row[0]=='Laboratory source notes':row[1]='SMACT supplier differs between panel and protocol; CD30 is absent from the recorded protocol. Entries are retained without adjudication.'
        if row[0]=='Data source':row[1]='Prepared public-release AnnData with canonical case assignments, clinical inputs, assay annotations and supplementary resources/results.'
    audit={'source_sha256':expected_source_sha256,'reviewed_material_sha256':sha256(reviewed_json),'removals':[],
           'clinical_values_changed':False,'classification_values_changed':False,
           'role_policy':{'case_classifications':'canonical discovery assignments','validation_cohort/case_classification_validation':'canonical validation assignments',
                          'case_clinical':'recorded discovery clinical inputs','discovery_clinical_elements':'explicit clinical-analysis overrides; unresolved T31 discrepancy retained',
                          'case_path':'source pathology records; different cohort coverage, not a replacement assignment table',
                          'classification_audit_*':'historical audit records, not active calls','lymphplex_call_audit':'derived audit checked against canonical calls',
                          'gene_expression':'processed expression for analysis','gep_raw_probes':'raw assay measurements and authoritative probe annotations',
                          'gep_normalized_probes':'processed measurements; overlapping probe annotations must match raw metadata'},
           'unresolved_source_items':['T31: recorded stage IV versus analysis stage II','OS description/event mismatch: T52','DSS description/event mismatch: T23, T33, T40, T53, T58','SMACT supplier differs between laboratory tables; CD30 missing from protocol']}
    with h5py.File(source,'r') as f:
        if NAMESPACE in f and not replace_existing:raise ValueError('Supplementary namespace already exists; use --replace-supplementary with newly reviewed outputs to refresh it explicitly')
        if REMOVABLE[0] in f:
            expression=read_elem(f['uns/gene_expression'])
            assert np.array_equal(expression.index.astype(str),read_elem(f[REMOVABLE[0]]))
            audit['removals'].append({'path':REMOVABLE[0],'replacement':'uns/gene_expression index','reason':'Exact duplicate gene index; no code consumers'})
        if REMOVABLE[1] in f:
            from dlbcl.genomic_profiling import build_location_class
            expected=build_location_class(read_elem(f['uns/case_classifications']))
            actual=read_elem(f[REMOVABLE[1]])
            pd.testing.assert_frame_equal(actual.sort_values('patient_id').reset_index(drop=True),expected.sort_values('patient_id').reset_index(drop=True),check_dtype=False)
            audit['removals'].append({'path':REMOVABLE[1],'replacement':'build_location_class(uns/case_classifications)','reason':'Deterministic duplicate annotation; regenerated at use'})
    allowed=[NAMESPACE]+[x['path'] for x in audit['removals']]
    before=payload_digest(source,allowed)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.partial',dir=output.parent);os.close(fd);tmp=Path(name)
    try:
        shutil.copyfile(source,tmp)
        with h5py.File(tmp,'r+') as f:
            if NAMESPACE in f:del f[NAMESPACE]
            for row in audit['removals']:del f[row['path']]
        scientific_digest=payload_digest(tmp,[NAMESPACE])
        producer_paths={Path('notebooks')/Path(row['notebook']).name for row in catalog if row['kind']=='derived_result' and row['notebook']}
        tracked=subprocess.run(['git','ls-files','dlbcl/*.py'],cwd=ROOT,capture_output=True,text=True)
        producer_paths.update(Path(p) for p in tracked.stdout.splitlines() if not Path(p).name.startswith('supplementary_'))
        producer_hashes={str(p):producer_digest(ROOT/p) for p in sorted(producer_paths) if (ROOT/p).is_file()}
        store={'schema':SCHEMA,'scientific_payload_sha256':scientific_digest,'tables':stored,'catalog_json':json.dumps(catalog),
               'gene_aliases_json':json.dumps(aliases),'field_definitions_json':json.dumps(definitions),
               'reference_gene_memberships':encode_table(memberships),'coverage_spec_json':json.dumps(coverage),
               'kotlov_signatures':encode_table(frame('S5 Kotlov signatures').drop(columns=['On BLYM777'])),
               'readme_json':json.dumps(readme),'migration_audit_json':json.dumps(audit,sort_keys=True),
               'analysis_code_fingerprints_json':json.dumps(producer_hashes,sort_keys=True)}
        for key in ['reference_gene_memberships','kotlov_signatures']:
            store[key]['table_sha256']=table_digest(decode_table(store[key]))
        protected=['catalog_json','gene_aliases_json','field_definitions_json','coverage_spec_json']
        store['reference_json_sha256']=json.dumps({key:hashlib.sha256(store[key].encode()).hexdigest() for key in protected},sort_keys=True)
        with h5py.File(tmp,'r+') as f:write_elem(f['uns'],'supplementary',store)
        if payload_digest(tmp,allowed)!=before:raise RuntimeError('Unrelated scientific payload changed')
        load_source(tmp)
        if sha256(source)!=expected_source_sha256:raise RuntimeError('Source changed during migration')
        os.replace(tmp,output)
    finally:tmp.unlink(missing_ok=True)
    audit['output_sha256']=sha256(output);audit['unchanged_payload_sha256']=before
    audit['canonical_resource_or_result_tables']=len(stored);audit['dynamic_views']=len(dynamic)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('reviewed_json',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--expected-source-sha256',required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--replace-supplementary',action='store_true',help='Explicitly refresh the single existing namespace from newly reviewed outputs; source file remains untouched')
    a=p.parse_args();result=migrate(a.source,a.reviewed_json,a.output,expected_source_sha256=a.expected_source_sha256,replace_existing=a.replace_supplementary)
    a.audit.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
