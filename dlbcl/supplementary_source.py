"""Supplementary tables read exclusively from a versioned AnnData bundle.

Case tables are views of the established canonical inputs. Laboratory and model
documentation is stored once. Numerical outputs are immutable, fingerprinted
derived results, never replacement patient data or classifier calls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from .case_assignments import discovery_assignments, validation_assignments
from .classification_audit import payload_digest, sha256

SCHEMA = 'supplementary-source-v1'
NAMESPACE = 'uns/supplementary'


def scalar(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    return str(value)


def table_payload(frame):
    frame = pd.DataFrame(frame)
    return {'headers':list(map(str, frame.columns)),
            'rows':[[scalar(x) for x in row] for row in frame.itertuples(index=False, name=None)]}


def table_digest(frame):
    return hashlib.sha256(json.dumps(table_payload(frame), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def producer_digest(path):
    path=Path(path)
    if path.suffix=='.ipynb':
        notebook=json.loads(path.read_text())
        source=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        return hashlib.sha256(json.dumps(source).encode()).hexdigest()
    return sha256(path)


def encode_table(frame):
    """Keep an inspectable DataFrame plus a null mask; no stringified numbers."""
    frame = pd.DataFrame(frame).copy().reset_index(drop=True)
    nulls = frame.isna().to_numpy()
    numeric_text = np.zeros(frame.shape, dtype=bool)
    for col in frame:
        values = frame[col].dropna()
        if len(values) and values.map(lambda x:isinstance(x, (int,float,np.number)) and not isinstance(x,(bool,np.bool_))).all():
            frame[col] = pd.to_numeric(frame[col])
        elif len(values) and values.map(lambda x:isinstance(x,(bool,np.bool_))).all():
            frame[col] = frame[col].fillna(False).astype(bool)
        else:
            numeric_text[:,frame.columns.get_loc(col)] = frame[col].map(lambda x:isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)) and not pd.isna(x)).to_numpy()
            frame[col] = frame[col].map(lambda x:'' if pd.isna(x) else str(x))
    frame.index = frame.index.astype(str)
    return {'data':frame, 'null_mask':nulls, 'numeric_text_mask':numeric_text,
            'schema_json':json.dumps(table_payload(pd.DataFrame(frame)).get('headers'))}


def decode_table(entry):
    frame = pd.DataFrame(entry['data']).copy().reset_index(drop=True)
    mask = np.asarray(entry['null_mask'], dtype=bool)
    if mask.shape != frame.shape:
        raise ValueError('Supplementary null mask does not match table')
    frame=frame.astype(object).mask(mask, None)
    for i,j in np.argwhere(np.asarray(entry.get('numeric_text_mask',np.zeros(frame.shape)),dtype=bool)):
        frame.iat[i,j]=float(frame.iat[i,j])
    return frame


def probe_annotations(uns):
    """Raw probe annotations own the assay definition; processed views must agree."""
    vc = uns['validation_cohort']
    cols = ['Probe','Accession','Code.Class']
    raw = pd.DataFrame(vc['gep_raw_probes'])[cols].copy()
    if not raw.Probe.is_unique:
        raise ValueError('Duplicate assay probe identifiers')
    normalized = pd.DataFrame(vc['gep_normalized_probes'])[cols].copy()
    expected = raw.set_index('Probe').reindex(normalized.Probe)
    pd.testing.assert_frame_equal(expected.reset_index(), normalized.reset_index(drop=True), check_dtype=False)
    return raw.reset_index(drop=True)


def load_source(path, *, validate=True):
    """Read required .uns entries only, without allocating the single-cell matrix."""
    path = Path(path)
    with h5py.File(path, 'r') as f:
        if NAMESPACE not in f:
            raise ValueError('AnnData lacks the supplementary-source schema. Use the prepared supplementary release; CSV/XLSX fallback is disabled.')
        names = ['case_clinical','case_classifications','discovery_clinical_elements','classification_inputs','genomic_profiling','supplementary']
        uns = {k:read_elem(f['uns'][k]) for k in names}
        vc_keys = ['meta','case_classification_validation','gep_raw_probes','gep_normalized_probes']
        uns['validation_cohort'] = {k:read_elem(f['uns/validation_cohort'][k]) for k in vc_keys}
    if uns['supplementary']['schema'] != SCHEMA:
        raise ValueError('Unsupported supplementary-source schema')
    if validate:
        observed = payload_digest(path, [NAMESPACE])
        if observed != uns['supplementary']['scientific_payload_sha256']:
            raise ValueError('Scientific inputs changed after the numerical results were captured. Recompute and explicitly refresh the results before exporting; stale tables are refused.')
        validate_store(uns)
        root=Path(__file__).resolve().parents[1]
        for relative,expected in json.loads(uns['supplementary'].get('analysis_code_fingerprints_json','{}')).items():
            producer=root/relative
            if not producer.exists() or producer_digest(producer)!=expected:
                raise ValueError('Numerical-result producer changed: '+relative+'. Recompute and refresh the stored results before export.')
    return SimpleNamespace(uns=uns), sha256(path)


def validate_store(uns):
    store = uns['supplementary']
    for key,expected in json.loads(store.get('reference_json_sha256','{}')).items():
        if hashlib.sha256(store[key].encode()).hexdigest()!=expected:
            raise ValueError('Reference metadata changed without an explicit refresh: '+key)
    for key in ['reference_gene_memberships','kotlov_signatures']:
        entry=store[key]
        if 'table_sha256' in entry and table_digest(decode_table(entry))!=entry['table_sha256']:
            raise ValueError('Reference table changed without an explicit refresh: '+key)
    catalog = json.loads(store['catalog_json'])
    names = [r['name'] for r in catalog]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate supplementary worksheet identifiers')
    referenced = []
    for row in catalog:
        if row['kind'] in {'resource','derived_result'}:
            referenced.append(row['key'])
            entry = store['tables'][row['key']]
            data = decode_table(entry)
            if table_digest(data) != entry['table_sha256']:
                raise ValueError('Stored table changed without a reviewed refresh: '+row['name'])
    if len(referenced) != len(set(referenced)) or set(referenced) != set(store['tables']):
        raise ValueError('Repeated or orphan supplementary table storage')
    for forbidden in ['S1 Discovery','S1 Validation','S3 1D_pie_patients','S4 BLYM777 targets','S4 Assay controls','S4 Gene symbol mapping','S5 RNA coverage']:
        row = next(r for r in catalog if r['name']==forbidden)
        if row['kind'] in {'resource','derived_result'}:
            raise ValueError('Derived input view must not be stored again: '+forbidden)
    for cohort,frame,n in [('Discovery',discovery_assignments(SimpleNamespace(uns=uns)),64),
                           ('Validation',validation_assignments(uns['validation_cohort']),303)]:
        if len(frame)!=n or not frame.index.is_unique:
            raise ValueError('Invalid canonical '+cohort+' cohort')
    protocol = decode_table(store['tables']['imc_staining_protocol'])
    panel = decode_table(store['tables']['imc_antibody_panel'])
    targets = set(panel['Target'].str.strip().str.upper())
    if 'CD30' in targets or 'SMACT' not in targets:
        raise ValueError('IMC panel membership differs from the confirmed study panel: SMACT is required and CD30 is not included')
    if any('volume' in str(c).lower() for c in protocol.columns) or protocol['Target or component'].eq('Buffer').any():
        raise ValueError('Staining protocol must not contain volume amounts or volume-only buffer rows')
    probe_annotations(uns)


def patient_tables(adata):
    from .validation_clinical_table import prepare_discovery_clinical_frame, attach_primary_ipi_buckets
    uns = adata.uns
    disc = pd.DataFrame(uns['case_clinical']).copy()
    dc = discovery_assignments(adata)
    disc.index.name='patient_id'
    for c in dc:
        if c not in disc:disc[c]=dc[c].reindex(disc.index)
    analysis = prepare_discovery_clinical_frame(adata, elements_csv=uns['discovery_clinical_elements'])
    disc['Ann_Arbor_analysis']=analysis['Ann_Arbor_at_Dx']
    disc['Primary_clinical_risk_analysis']=analysis['ipi_ielsg_primary']
    for c in ['ipi_score','ipi_extranodal','ipi_ldh','ipi_who','ielsg_ecog','ielsg_ldh','ielsg_csf','ielsg_deepbrain']:
        if c in analysis:disc[c+'_analysis']=analysis[c]
    vc=uns['validation_cohort'];val=pd.DataFrame(vc['meta']).copy();cc=validation_assignments(vc)
    val.index.name='patient_alias'
    for c in cc:
        if c not in val:val[c]=cc[c].reindex(val.index)
    val['Primary_clinical_risk_analysis']=attach_primary_ipi_buckets(vc['meta'])['ipi_ielsg_primary']
    return disc.reset_index(),val.reset_index()


def cohort_summary(adata):
    from .validation_clinical_table import build_discovery_clinical_table_from_adata, build_clinical_table_dataframe, build_combined_clinical_table
    d=build_discovery_clinical_table_from_adata(adata, elements_csv=adata.uns['discovery_clinical_elements'])
    v=pd.DataFrame(adata.uns['validation_cohort']['meta']).copy()
    v['COO_NanoString']=validation_assignments(adata.uns['validation_cohort'])['COO_NanoString'].reindex(v.index)
    result=build_combined_clinical_table(d,build_clinical_table_dataframe(v))
    result.columns=[c.replace('d_','Discovery — ',1) if c.startswith('d_') else c.replace('v_','Validation — ',1) if c.startswith('v_') else c for c in result]
    return result


def supplementary_tables(adata, data_sha256, *, export_date):
    """Regenerate every workbook sheet from .uns; no original workbook is opened."""
    uns=adata.uns;store=uns['supplementary'];catalog=json.loads(store['catalog_json'])
    disc,val=patient_tables(adata);annotations=probe_annotations(uns)
    aliases=json.loads(store['gene_aliases_json'])
    targets=annotations[annotations['Code.Class'].isin(['Endogenous','Housekeeping'])].sort_values('Probe').copy()
    panelgenes={aliases.get(g,g) for g in targets.Probe}
    tables=[]
    for item in catalog:
        name=item['name'];kind=item['kind'];item=dict(item)
        if kind in {'resource','derived_result'}:
            frame=decode_table(store['tables'][item['key']])
        elif name=='S1 Discovery':frame=disc
        elif name=='S1 Validation':frame=val
        elif name=='S1 Cohort summary':frame=cohort_summary(adata)
        elif name=='S1 Field definitions':
            meanings=json.loads(store['field_definitions_json']);rows=[]
            for label,data in [('S1 Discovery',disc),('S1 Validation',val)]:
                for col in data:
                    observed=list(dict.fromkeys(scalar(x) for x in data[col] if not pd.isna(x)))[:8]
                    rows.append([label,col,meanings.get(col,'Source field retained without recoding.'),'; '.join(map(str,observed))])
            frame=pd.DataFrame(rows,columns=['Worksheet','Field','Definition','Example values'])
        elif name=='S3 1D_pie_patients':
            from .location_pie_patchwork import prepare_pie_patchwork_table, format_pie_patchwork_supplementary_table
            from .integration_figures import build_integration_metadata
            cc=discovery_assignments(adata)
            meta=build_integration_metadata(adata,cc.reset_index()[['patient_id','tumorimmune_archetype_id']],patient_subset=cc.index.tolist())
            cases=pd.DataFrame(uns['classification_inputs']['cases'])
            tested=set(cases.loc[cases.cohort.str.lower().eq('discovery'),'patient_id'])
            meta['genomic_tested']=meta.index.isin(tested)
            frame=format_pie_patchwork_supplementary_table(prepare_pie_patchwork_table(meta)[0])
        elif name in {'S4 BLYM777 targets','S4 Assay controls'}:
            frame=targets if name=='S4 BLYM777 targets' else annotations[annotations['Code.Class'].isin(['Positive','Negative'])]
            frame=frame.rename(columns={'Probe':'Probe or gene symbol','Accession':'Transcript accession','Code.Class':'Code class'})
        elif name=='S4 Gene symbol mapping':
            frame=pd.DataFrame([[g,aliases.get(g,g),'Discovery, Validation'] for g in targets.Probe],columns=['Assay gene label','Classifier gene symbol','Cohorts'])
        elif name=='S5 RNA reference genes':
            frame=decode_table(store['reference_gene_memberships'])
            frame['On BLYM777']=frame['Gene symbol'].isin(panelgenes)
        elif name=='S5 RNA coverage':
            genes=decode_table(store['reference_gene_memberships']);rows=[]
            for spec in json.loads(store['coverage_spec_json']):
                part=genes[genes['Classifier'].eq(spec['Classifier']) & genes['Feature definition'].eq(spec['Feature definition'])]
                if spec['Subtype']!='All':part=part[part['Subtype memberships'].map(lambda s:spec['Subtype'] in s.split(', '))]
                requested=set(part['Gene symbol']);covered=requested&panelgenes
                rows.append([spec['Classifier'],spec['Subtype'],spec['Feature definition'],len(requested),len(covered),len(requested-covered),len(covered)/len(requested),spec['Scope']])
            frame=pd.DataFrame(rows,columns=['Classifier','Subtype','Feature definition','Requested genes','On-panel genes','Off-panel genes','On-panel fraction','Scope'])
        elif name=='S5 Kotlov signatures':
            frame=decode_table(store['kotlov_signatures']);frame.insert(3,'On BLYM777',frame['Gene symbol'].isin(panelgenes))
        else:raise ValueError('Unknown supplementary view: '+name)
        item.update(table_payload(frame));item['source']=item['canonical_source']
        tables.append(item)
    readme=json.loads(store['readme_json'])
    for row in readme:
        if row[0]=='Data SHA-256':row[1]=data_sha256
        if row[0]=='Export date':row[1]=export_date
    readme += [['Source of truth','All sheets are extracted from this AnnData. Patient views reuse canonical tables; assay annotations reuse the raw-probe metadata. Laboratory/model documentation is stored once. Numerical tables are captured derived outputs, not alternative patient inputs.'],
               ['Numerical result provenance','Imported from the reviewed notebook exports during migration. No statistical model was rerun during migration. Source hashes and producer notebooks are retained. Export refuses results if the scientific payload has changed.'],
               ['Staining protocol','Antibody identities, incubation, temperature and dilution are retained. Volume amounts and volume-only buffer rows are omitted.'],
               ['Probe classes','Code class is read from AnnData raw-probe annotations: 784 endogenous probes, 13 housekeeping probes and 14 assay controls. Processed-probe annotations must agree for shared probes.']]
    index=[[s['section'],s['name'],s['title'],len(s['rows']),len(s['headers']),s['source'],s['notebook'],s['note']] for s in tables]
    front=[]
    for name,title,headers,rows in [('Contents','Worksheet index and provenance',['Table group','Worksheet','Contents','Data rows','Columns','Data source','Generating notebook','Notes'],index),('Read me','Supplementary tables',['Topic','Description'],readme)]:
        front.append(dict(name=name,title=title,headers=headers,rows=rows,section='Guide',source='AnnData supplementary source catalog',notebook='',note=''))
    return front+tables


def export_source_tables(path, output_dir, *, export_date):
    adata,digest=load_source(path)
    tables=supplementary_tables(adata,digest,export_date=export_date)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    csvdir=out/'csv';csvdir.mkdir(exist_ok=True)
    for table in tables:
        pd.DataFrame(table['rows'],columns=table['headers']).to_csv(csvdir/(table['name'].replace(' ','_')+'.csv'),index=False)
    (out/'workbook_data.json').write_text(json.dumps({'sheets':tables},ensure_ascii=False,allow_nan=False))
    (out/'source_validation.json').write_text(json.dumps({'schema':SCHEMA,'adata_sha256':digest,'worksheets':len(tables),'scientific_payload_verified':True,'source':'AnnData only'},indent=2))
    return tables
