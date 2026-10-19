"""Explicit expression contracts, process provenance, and transactional AnnData updates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, write_elem

CASE_KEYS = {'Discovery':'uns/case_classifications',
             'Validation':'uns/validation_cohort/case_classification_validation'}
EXPRESSION_KEYS = {'Discovery':'uns/gene_expression',
                  'Validation':'uns/validation_cohort/gene_expression'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def run(command, log, cwd=None):
    """Run, never silently substitute cached calls; preserve the exact command and log."""
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with log.open('w') as f:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log.with_suffix('.command.json').write_text(json.dumps(dict(command=list(map(str,command)), cwd=str(cwd), returncode=result.returncode), indent=2))
    if result.returncode: raise RuntimeError(f'Classifier failed ({result.returncode}): inspect {log}')


def validate_expression(x):
    if x.empty or not x.index.is_unique or not x.columns.is_unique:
        raise ValueError('Require unique gene rows and sample columns in a nonempty matrix')
    x = x.astype(float)
    if not np.isfinite(x.to_numpy()).all(): raise ValueError('Missing/nonfinite expression; no implicit imputation')
    return x


def preprocess_expression(x, kind, *, background=None, housekeeping=None, floor=1., pseudocount=None):
    """Return (log2 gene matrix, per-sample normalization audit).

    Raw NanoString requires Code.Class and Probe columns, explicit background
    choice, housekeeping genes, floor and pseudocount. This is a transparent
    research normalization, not a replacement for manufacturer's assay QC.
    Already normalized inputs NEVER receive background correction a second time.
    """
    if kind == 'nanostring_raw_counts':
        if background not in {'none','negative_mean','negative_mean_plus_2sd'} or not housekeeping:
            raise ValueError('Raw counts require explicit background and housekeeping choices')
        if pseudocount is None or pseudocount < 0 or floor <= 0:
            raise ValueError('Declare nonnegative pseudocount and positive background floor')
        if not {'Code.Class','Probe'} <= set(x): raise ValueError('Missing probe annotations')
        values=x.drop(columns=['Probe','Code.Class','Accession'],errors='ignore').copy()
        values.index=x.Probe.astype(str); values=validate_expression(values)
        if (values<0).any().any(): raise ValueError('Negative raw counts')
        annotation=pd.Series(x['Code.Class'].to_numpy(),index=values.index)
        pos=values.loc[annotation.eq('Positive')]
        if pos.empty or (pos<=0).any().any(): raise ValueError('Positive controls must be measured and positive')
        geo=np.exp(np.log(pos).mean()); scale=geo.mean()/geo
        scaled=values.mul(scale,axis=1)
        bg=pd.Series(0.,index=values.columns)
        if background!='none':
            neg=scaled.loc[annotation.eq('Negative')]
            if len(neg)<2: raise ValueError('At least two negative controls required')
            bg=neg.mean()+(2*neg.std(ddof=1) if background.endswith('2sd') else 0)
        endogenous=scaled.loc[annotation.isin(['Endogenous','Housekeeping'])]
        if not set(housekeeping)<=set(endogenous.index): raise ValueError('Missing housekeeping genes')
        corrected=endogenous.sub(bg,axis=1)
        clipped=(corrected<floor).sum() if background!='none' else pd.Series(0,index=values.columns)
        if background!='none': corrected=corrected.clip(lower=floor)
        if (corrected+pseudocount<=0).any().any(): raise ValueError('Nonpositive values before log2')
        logged=np.log2(corrected+pseudocount)
        hk=logged.loc[housekeeping].mean()
        normalized=logged.sub(hk,axis=1)
        return normalized, pd.DataFrame(dict(positive_control_factor=scale,background=bg,
                housekeeping_log2_mean=hk,clipped_genes=clipped,input_kind=kind))
    if background is not None or housekeeping is not None:
        raise ValueError('Do not reapply raw normalization to preprocessed expression')
    x=validate_expression(x)
    retained={'nanostring_log2_normalized','rna_log2_tpm1','microarray_log2_normalized'}
    if kind in retained: out=x.copy()
    elif kind in {'rna_tpm','nanostring_normalized_counts'}:
        if (x<0).any().any(): raise ValueError('Linear expression must be nonnegative')
        if kind=='nanostring_normalized_counts':
            if pseudocount is None: raise ValueError('Declare count log2 pseudocount explicitly')
            if pseudocount<0 or (x+pseudocount<=0).any().any(): raise ValueError('Invalid count log transform')
            out=np.log2(x+pseudocount)
        else: out=np.log2(x+1)
    elif kind in {'rna_raw_counts','microarray_raw','microarray_linear'}:
        raise ValueError('Use an assay-specific upstream normalization first: gene lengths/library metadata for RNA-seq, or platform-specific probe/background/RMA processing for arrays. These are not TPM.')
    else: raise ValueError(f'Unknown input kind: {kind}')
    return out,pd.DataFrame({'input_kind':kind,'operation':'retained' if kind in retained else 'explicit log2 transform'},index=x.columns)


def canonicalizer(hgnc):
    h=pd.read_csv(hgnc,sep='\t',dtype=str).fillna(''); approved=set(h.symbol); aliases={}
    for field in ['prev_symbol','alias_symbol']:
        lookup={}
        for row in h.to_dict('records'):
            for g in row[field].split('|'):
                if g: lookup.setdefault(g,set()).add(row['symbol'])
        for g,v in lookup.items():
            if len(v)==1 and g not in aliases: aliases[g]=next(iter(v))
    def canon(g):
        g={'AMICA1 (JAML)':'JAML','NSD2 (WHSC1)':'NSD2'}.get(str(g).strip(),str(g).strip())
        if g.startswith('HLA.'): g=g.replace('.','-',1)
        return g if g in approved else aliases.get(g,g)
    return canon


def read_study(path):
    tables,expression={},{}
    with h5py.File(path,'r') as f:
        for cohort,key in CASE_KEYS.items():
            t=read_elem(f[key]); ids=pd.Index(t.patient_id if 'patient_id' in t else t.index)
            if not ids.is_unique: raise ValueError('Duplicate canonical patient IDs')
            x=read_elem(f[EXPRESSION_KEYS[cohort]])
            if not ids.isin(x.columns).all(): raise ValueError('Missing expression for canonical cases')
            tables[cohort]=t; expression[cohort]=x.loc[:,ids]
    return tables,expression


def payload_digest(path, excluded):
    """Semantic fingerprint of every dataset outside the intended write paths."""
    h=hashlib.sha256()
    with h5py.File(path,'r') as f:
        def visit(name,obj):
            if any(name==x or name.startswith(x+'/') for x in excluded): return
            h.update(name.encode());h.update(repr(sorted((k,str(v)) for k,v in obj.attrs.items())).encode())
            if isinstance(obj,h5py.Dataset):
                h.update(str(obj.shape).encode());h.update(str(obj.dtype).encode())
                chunks=[obj[()]] if obj.ndim==0 else (obj[i:i+100000] for i in range(0,len(obj),100000))
                for chunk in chunks:
                    arr=np.asarray(chunk)
                    h.update(repr(arr.tolist()).encode() if arr.dtype.hasobject or arr.dtype.kind in 'OUSVT' else arr.tobytes())
        f.visititems(visit)
    return h.hexdigest()


def promote_calls(source, updates, output, *, audit_name, run_provenance, expected_source_sha256, auxiliary_updates=None):
    """Replace only declared columns and IDs, with backup, payload check and atomic commit.

    Updates are cohort -> sample-indexed DataFrames. Subsets are permitted only
    when their exact case set has already passed the method's eligibility checks.
    Missing labels are rejected. Explicit Unassigned is a legitimate method output.
    """
    source=Path(source).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if sha256(source)!=expected_source_sha256: raise RuntimeError('Source changed since the audit began; rerun before promotion')
    audit_key='uns/classification_audit_'+audit_name
    auxiliary_updates=auxiliary_updates or {}
    if any(not k.startswith('uns/') or k in CASE_KEYS.values() or k==audit_key for k in auxiliary_updates):
        raise ValueError('Auxiliary writes must be explicit uns paths separate from case tables and audit')
    excluded=list(CASE_KEYS.values())+[audit_key]+list(auxiliary_updates)
    untouched=payload_digest(source,excluded)
    expected={};changes=[]
    with h5py.File(source,'r') as f:
        for cohort,u in updates.items():
            old=read_elem(f[CASE_KEYS[cohort]]);new=old.copy()
            ids=pd.Index(old.patient_id if 'patient_id' in old else old.index)
            if not u.index.is_unique or not u.index.isin(ids).all(): raise ValueError('Invalid patient crosswalk')
            if u.isna().any().any(): raise ValueError('Missing calls cannot overwrite classifications')
            rowlookup=pd.Series(old.index,index=ids)
            for col in u:
                if col not in new: new[col]=''
                if isinstance(new[col].dtype,pd.CategoricalDtype): new[col]=new[col].astype(object)
                for pid,value in u[col].items():
                    idx=rowlookup[pid]; previous=new.at[idx,col]
                    changes.append(dict(cohort=cohort,patient_id=pid,column=col,previous=str(previous),fresh=str(value),changed=str(previous)!=str(value)))
                    new.at[idx,col]=value
            pd.testing.assert_frame_equal(old.drop(columns=list(u),errors='ignore'),new[old.columns].drop(columns=list(u),errors='ignore'))
            pd.testing.assert_frame_equal(old.loc[~ids.isin(u.index)],new.loc[~ids.isin(u.index),old.columns])
            expected[cohort]=new
            old.to_csv(output/f'{cohort}_before.csv');new.to_csv(output/f'{cohort}_after.csv')
    changes=pd.DataFrame(changes);changes.to_csv(output/'call_changes.csv',index=False)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=source.with_name(source.stem+'.before_'+audit_name+'_'+stamp+'.h5ad')
    shutil.copy2(source,backup)
    temp=source.with_name(source.stem+'.'+audit_name+'.tmp.h5ad')
    if temp.exists(): raise FileExistsError(temp)
    shutil.copy2(source,temp)
    provenance=dict(run_provenance,source_sha256=expected_source_sha256,backup=str(backup),timestamp_utc=stamp,auxiliary_paths=list(auxiliary_updates))
    try:
        with h5py.File(temp,'r+') as f:
            for cohort,t in expected.items():
                key=CASE_KEYS[cohort];del f[key];write_elem(f,key,t)
            for key,value in auxiliary_updates.items():
                if key in f: del f[key]
                write_elem(f,key,value)
            if audit_key in f: del f[audit_key]
            write_elem(f,audit_key,{'provenance_json':json.dumps(provenance,sort_keys=True),'changes':changes})
        with h5py.File(temp,'r') as f:
            for cohort,t in expected.items(): pd.testing.assert_frame_equal(read_elem(f[CASE_KEYS[cohort]]),t)
            for key,value in auxiliary_updates.items():
                recovered=read_elem(f[key])
                if isinstance(value,pd.DataFrame): pd.testing.assert_frame_equal(recovered,value)
                elif recovered!=value: raise RuntimeError('Auxiliary metadata failed round-trip verification')
        if payload_digest(temp,excluded)!=untouched: raise RuntimeError('Unrelated AnnData content changed')
        if sha256(source)!=expected_source_sha256: raise RuntimeError('Source changed during write')
        os.replace(temp,source)
    except Exception:
        temp.unlink(missing_ok=True);raise
    provenance['result_sha256']=sha256(source);provenance['unrelated_payload_verified']=True
    (output/'promotion.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return changes


AUDIT_PATH_KEYS = {'source_adata', 'output_root', 'vendor', 'hgnc', 'ecotyper_source',
                   'kotlov_utils', 'lymphomap_reference_genes', 'ecotyper_reference_genes'}

def load_audit_config(root, override=None):
    """Resolve portable defaults and optional local overrides without enabling writes."""
    root = Path(root).resolve()
    config_path = root/'config/classification_audit/default.json'
    config = json.loads(config_path.read_text())
    local = Path(override) if override else root/'config/classification_audit/local.json'
    if local.exists():
        extra = json.loads(local.read_text())
        unknown = set(extra) - set(config)
        if unknown: raise ValueError(f'Unknown audit configuration keys: {sorted(unknown)}')
        config.update(extra)
    if not isinstance(config.get('promote'), bool):
        raise ValueError('promote must be a JSON boolean')
    for key in AUDIT_PATH_KEYS:
        path = Path(config[key]).expanduser()
        config[key] = str(path.resolve() if path.is_absolute() else (root/path).resolve())
    return config, config_path


def require_audit_resources(config, kind):
    """Fail with explicit upstream sources; never substitute archived classifier calls."""
    required = [(Path(config['hgnc']), 'HGNC: https://www.genenames.org/download/archive/')]
    vendor = Path(config['vendor'])
    if kind == 'genomic':
        required += [
            (vendor/'LymphGen2/src/Model10_8.1.22.R', 'LymphGen: https://llmpp.nih.gov/lymphgen/index.php'),
            (vendor/'DLBclass-tool/saved_models/FINALMODEL_NN_evaluation_seeds1_100_folds5_reducedV3.4_removeN5', 'DLBclass: https://github.com/getzlab/DLBCL-Classifier'),
            (vendor/'DLBCLGenomicSubtyping/data/genomic_data.csv', 'HMRN: https://github.com/ecsg-uoy/DLBCLGenomicSubtyping'),
            (vendor/'LymphPlex/Functions.R', 'LymphPlex: https://github.com/difuSJTU/LymphPlex')]
    elif kind == 'transcriptomic':
        required += [(Path(config['ecotyper_source'])/'EcoTyper_recovery_bulk.R', 'EcoTyper: https://github.com/digitalcytometry/ecotyper'),
                     (Path(config['kotlov_utils']), 'Kotlov: https://github.com/BostonGene/LME')]
    else: raise ValueError(f'Unknown audit kind: {kind}')
    missing = [f'{p.name}: {citation}' for p,citation in required if not p.exists()]
    if not shutil.which(config['rscript']): missing.append('Rscript: install R and the packages listed in docs/classification-resources.md')
    if missing:
        raise FileNotFoundError('Classifier resources are not installed. Configure config/classification_audit/local.json. See docs/classification-resources.md.\n'+'\n'.join(missing))
    verify_native_resources(config, kind)


def verify_native_resources(config, kind):
    """Verify downloaded native resources against the recorded release fingerprints."""
    manifest_path=Path(__file__).resolve().parents[1]/'config/classification_audit/resource_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    vendor=Path(config['vendor'])
    roots={'LymphGen2':vendor/'LymphGen2','DLBclass-tool':vendor/'DLBclass-tool',
           'HMRN':vendor/'DLBCLGenomicSubtyping','LymphPlex':vendor/'LymphPlex',
           'EcoTyper':Path(config['ecotyper_source'])}
    selected=['LymphGen2','DLBclass-tool','HMRN','LymphPlex'] if kind=='genomic' else ['EcoTyper','Kotlov']
    checked=0
    for resource in selected:
        for item in manifest[resource]:
            path=Path(config['kotlov_utils']) if resource=='Kotlov' else roots[resource]/item['path']
            if not path.is_file() or sha256(path)!=item['sha256']:
                raise ValueError(f'Native resource differs from release fingerprint: {resource}/{item["path"]}. Obtain the matching resources listed in docs/classification-resources.md; do not silently substitute another model version.')
            checked+=1
    return checked
