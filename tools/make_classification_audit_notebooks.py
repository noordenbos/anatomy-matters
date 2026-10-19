"""Build executable audit notebooks; keep scientific decisions visible in cells."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def notebook(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/name)
M=nbf.v4.new_markdown_cell;C=nbf.v4.new_code_cell
setup='''from pathlib import Path
import sys, json, shutil, importlib.util, inspect
import numpy as np
import pandas as pd
import h5py
from anndata.io import read_elem
ROOT = Path.cwd() if (Path.cwd()/'dlbcl').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
from dlbcl.classification_audit import (sha256, run, read_study, canonicalizer,
    preprocess_expression, promote_calls, CASE_KEYS)
from dlbcl.classification_audit import load_audit_config, require_audit_resources
config, CONFIG_PATH = load_audit_config(ROOT)
SOURCE = Path(config['source_adata']).resolve()
SOURCE_HASH = sha256(SOURCE)
tables, raw_expression = read_study(SOURCE)
canon = canonicalizer(config['hgnc'])
UPDATE_ADATA = config['promote']  # Explicit final-cell write switch; backup and atomic replacement.
'''
trans=[M('''# Transcriptomic classification audit

Execute from top to bottom to reclassify the discovery and validation cohorts from their gene-expression matrices. Each method has a separate code cell, exported inputs, scores, labels, source hashes and a before/after comparison. The final cell can replace the authoritative case tables after explicit opt-in and verification; it creates a timestamped AnnData backup.

LymphoMAP and EcoTyper call native author implementations. Kotlov and Ciavarella reproduce the study's explicitly specified clustering procedures; these are cohort-dependent reconstructions. Lymph2Cx is a **research reconstruction**, using the full 15 predictors plus five housekeeping genes, with externally frozen NanoString borders. It is not the proprietary clinical assay.

Portable defaults are declared in `config/classification_audit/default.json`; optional machine-specific overrides belong in the ignored `local.json` beside it. `promote` defaults to false. See `docs/classification-resources.md` for native software and model resources. Missing native software or source data is an error, never a reason to reuse saved labels.'''),C(setup+'''
require_audit_resources(config, 'transcriptomic')
OUT=Path(config['output_root'])/'transcriptomic'
OUT.mkdir(parents=True,exist_ok=True)
SEED=config['seed']; N_FEATURES=config['lymphomap_features']
POSTERIOR_CUTOFF=config['lymphomap_posterior_cutoff']
GCB_UNC_THRESHOLD=config['lymph2cx_lower']; UNC_ABC_THRESHOLD=config['lymph2cx_upper']
assert GCB_UNC_THRESHOLD < UNC_ABC_THRESHOLD
METHOD_FILES={str(p):sha256(p) for p in [CONFIG_PATH,ROOT/'dlbcl/classification_audit.py',ROOT/'dlbcl/rna_clustering_benchmark.py',ROOT/'dlbcl/lymph2cx_research.py',*sorted((ROOT/'config/classification_audit').glob('*.csv')),*sorted((ROOT/'config/classification_audit').glob('*.json'))]}
display(pd.DataFrame({'cohort':tables.keys(),'cases':[len(x) for x in tables.values()]}))
'''),M('''## 1. Input scale and preprocessing

| Declared input | Processing before classification |
|---|---|
| Raw NanoString counts | Require probe classes and controls; positive-control scaling; explicitly chosen negative-control background correction; clipping only when background correction is selected; housekeeping normalization; explicit log2 offset. Save every sample's factors. |
| Normalized NanoString counts | No second background/HK normalization; explicit log2 transform for score/clustering methods. Native LymphoMapR also offers `NanoString_norm_counts`, which performs its own VST. |
| Normalized NanoString log2 | Retain supplied values. This is the study input; negative log ratios are valid. |
| TPM | `log2(TPM+1)`; never infer TPM from read counts. |
| log2(TPM+1) | Retain values. |
| Normalized microarray log2 | Retain values. Raw arrays require platform-specific preprocessing and probe-to-gene mapping upstream. |

The current study's validation pipeline uses positive-control scaling and log2 housekeeping centering **without negative-control subtraction**. Reproduce and verify that pipeline below. Discovery raw-control processing is not reconstructed here; its stored normalized matrix is the declared starting point. Adding background correction would be a separate preprocessing analysis, not a retrospective description of these matrices.

Native LymphoMapR's already-log branch is named `RNAseq_TPM_log2p1`; using it for normalized NanoString log ratios is the study's explicit cross-platform choice. Do not pass these negative values to `NanoString_norm_counts`. Microarrays use `microarray_lognorm`. Lymph2Cx NanoString thresholds must not be transferred automatically to RNA-seq or arrays.'''),C('''# Inspect the exact supported transforms, including the raw-control implementation.
print(inspect.getsource(preprocess_expression))
# Verify current validation preprocessing from its embedded raw probes.
from dlbcl.validation_cohort import normalize_nanos_string_gep, gene_expression_matrix
with h5py.File(SOURCE,'r') as handle:
    raw_probes=read_elem(handle['uns/validation_cohort/gep_raw_probes'])
replayed=gene_expression_matrix(normalize_nanos_string_gep(raw_probes))
stored=raw_expression['Validation']
np.testing.assert_allclose(replayed.loc[stored.index,stored.columns], stored, atol=1e-10)
(OUT/'preprocessing_verification.json').write_text(json.dumps({
    'validation_raw_replay_max_abs_difference':float(np.max(np.abs(replayed.loc[stored.index,stored.columns].values-stored.values))),
    'background_subtraction':'not performed in the verified validation pipeline',
    'discovery_starting_point':'stored normalized log2 matrix; raw-control provenance not independently replayed'},indent=2))
expression={}; symbol_rows=[]; input_hashes={}
for cohort,x in raw_expression.items():
    x,normalization=preprocess_expression(x,config['study_input_kind'])
    normalization.to_csv(OUT/f'{cohort}_normalization.csv')
    symbol_rows.extend({'cohort':cohort,'input_gene':g,'canonical_gene':canon(g)} for g in x.index)
    x.index=[canon(g) for g in x.index]
    x=x.groupby(level=0,sort=False).mean() # Explicit alias-collision rule on log scale.
    x.columns=[f'{cohort}__{s}' for s in x.columns]
    expression[cohort]=x
    path=OUT/f'{cohort}_canonical_log.tsv';x.to_csv(path,sep='\\t',float_format='%.17g')
    input_hashes[cohort]=sha256(path)
pd.DataFrame(symbol_rows).to_csv(OUT/'gene_symbol_mapping.csv',index=False)
updates={c:pd.DataFrame(index=[s.split('__',1)[1] for s in x.columns]) for c,x in expression.items()}
'''),M('''## 2. LymphoMAP: native feature selection and naive Bayes

Run LymphoMapR afresh with the declared seed, feature budget and posterior cutoff. The package selects DEGs from genes shared with its reference, performs its native batch adjustment and fits/predicts its naive Bayes model. Export the selected features and full R result. [Li et al., Cancer Cell 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12417684/).'''),C('''lymphomap_r = r\'''args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(LymphoMapR))
set.seed(as.integer(args[4]))
x <- as.matrix(read.delim(args[1],row.names=1,check.names=FALSE))
r <- run_LymphoMap(x,data_type=args[3],n_features_per_archetype=as.integer(args[5]),
                  LymphoMAP_prob_cutoff=as.numeric(args[6]),verbose=FALSE)
write.table(r$pred,file.path(args[2],'predictions.tsv'),sep="\\t",quote=FALSE,row.names=FALSE)
write.table(r$data$feature,file.path(args[2],'selected_features.tsv'),sep="\\t",quote=FALSE,row.names=FALSE)
saveRDS(r,file.path(args[2],'result.rds'))
capture.output(sessionInfo(),file=file.path(args[2],'sessionInfo.txt'))
capture.output(LymphoMapR:::data_prep_expr,file=file.path(args[2],'native_preprocessing.txt'))
\'''
script=OUT/'run_lymphomap.R';script.write_text(lymphomap_r)
reference=pd.read_csv(config['lymphomap_reference_genes'],sep='\\t').gene.tolist()
for cohort,x in expression.items():
    work=OUT/'LymphoMAP'/cohort;work.mkdir(parents=True,exist_ok=True)
    genes=[g for g in reference if canon(g) in x.index]
    native=x.loc[[canon(g) for g in genes]].copy();native.index=genes
    native_input_type='microarray_lognorm' if config['study_input_kind']=='microarray_log2_normalized' else 'RNAseq_TPM_log2p1'
    path=work/'input.tsv';native.to_csv(path,sep='\\t',index_label='Gene',float_format='%.12g')
    # Already-log input: retain values, with native ComBat and reference feature selection.
    run([config['rscript'],script,path,work,native_input_type,SEED,N_FEATURES,POSTERIOR_CUTOFF],work/'run.log')
    pred=pd.read_csv(work/'predictions.tsv',sep='\\t').set_index('sample_id')
    assert set(pred.index)==set(x.columns)
    updates[cohort]['lymphomap']=pred.loc[x.columns,'LymphoMAP'].to_numpy()
    display(pred.LymphoMAP.value_counts().rename(cohort))
'''),M('''## 3. EcoTyper: native state/ecotype recovery

The author recovery code scales reference genes within the query cohort and fills absent reference genes with zero. It is therefore cohort dependent. Native ecotype assignment can remain Unassigned; maximum abundance is stored separately and is not a probability or a confident assignment. Each run has an isolated workspace and exact source/model fingerprints. [EcoTyper code](https://github.com/digitalcytometry/ecotyper).'''),C('''source=Path(config['ecotyper_source']); work=OUT/'ecotyper_work'
work.mkdir(exist_ok=True)
shutil.copytree(source/'pipeline',work/'pipeline',dirs_exist_ok=True)
shutil.copy2(source/'EcoTyper_recovery_bulk.R',work/'EcoTyper_recovery_bulk.R')
for part in ['Analysis','Cell_States/discovery','Ecotypes/discovery']:
    target=work/'EcoTyper/Lymphoma/Lymphoma_Fractions'/part
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.symlink_to((source/'EcoTyper/Lymphoma/Lymphoma_Fractions'/part).resolve(),target_is_directory=True)
model_hashes={str(p.relative_to(source)):sha256(p) for folder in ['pipeline','EcoTyper/Lymphoma/Lymphoma_Fractions/Cell_States/discovery','EcoTyper/Lymphoma/Lymphoma_Fractions/Ecotypes/discovery'] for p in (source/folder).rglob('*') if p.is_file()}
(OUT/'ecotyper_source_hashes.json').write_text(json.dumps(model_hashes,indent=2))
reference=sorted(pd.read_csv(config['ecotyper_reference_genes']).reference_gene.unique())
for cohort,x in expression.items():
    genes=[g for g in reference if canon(g) in x.index]
    native=x.loc[[canon(g) for g in genes]].copy();native.index=genes
    path=OUT/f'audit_{cohort}.tsv';native.to_csv(path,sep='\\t',index_label='Gene',float_format='%.12g')
    destination=OUT/'EcoTyper'/cohort;destination.mkdir(parents=True,exist_ok=True)
    run([config['rscript'],work/'EcoTyper_recovery_bulk.R','-d','Lymphoma','-m',path,'-t','2','-o',destination],destination/'run.log',cwd=work)
    recovery=work/'EcoTyper/Lymphoma/Lymphoma_Fractions/Ecotypes/recovery'/path.stem
    assignment=pd.read_csv(recovery/'initial_ecotype_assignment.txt',sep='\\t').set_index('ID')
    abundance=pd.read_csv(recovery/'ecotype_abundance.txt',sep='\\t',index_col=0)
    normalized_ids=[s.replace('-','.') for s in x.columns]
    assert len(set(normalized_ids))==len(normalized_ids)
    assert set(abundance.columns)==set(normalized_ids)
    call=assignment.Ecotype.reindex(normalized_ids).fillna('Unassigned')
    updates[cohort]['Lymphoma_Ecotype']=call.to_numpy()
    updates[cohort]['Lymphoma_Ecotype_confident']=call.where(call.ne('Unassigned'),'').to_numpy()
    updates[cohort]['Lymphoma_Ecotype_imputed']=False
    updates[cohort]['New_Ecotype']=abundance[normalized_ids].idxmax().where(abundance[normalized_ids].max().gt(0),'').to_numpy()
    assignment.to_csv(destination/'assignment_with_q.csv');abundance.to_csv(destination/'abundance.csv')
    display(call.value_counts().rename(cohort))
'''),M('''## 4. Kotlov and Ciavarella: explicit clustering and subtype annotation

Kotlov uses 22 author ssGSEA signatures plus NFkB/p53/PI3K PROGENy scores. Missing PROGENy genes contribute zero; signature coverage is exported. Median scaling uses the author's **mean absolute deviation about the mean**, then clips to ±3. The Louvain threshold grid and selection criteria are printed below. Cluster profiles are mapped to the highest-correlated external published subtype centroid, permitting many-to-one mapping. The stored centroids were built from panel-restricted Schmitz, Morin and Li references; provenance is supplied alongside them. This is not the author's KNN classifier.

Ciavarella uses the 45-gene Euclidean/Ward reconstruction, cut into three groups and ordered by average expression. These procedures depend on the query cohort; adding or removing patients can change assignments. [Kotlov et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8178179/).'''),C('''from dlbcl.rna_clustering_benchmark import author_median_scale, select_louvain, ward_partition
print(inspect.getsource(select_louvain))
spec=importlib.util.spec_from_file_location('kotlov_author_utils',config['kotlov_utils'])
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
features=pd.read_csv(ROOT/'config/classification_audit/clustering_features.csv')
reference=pd.read_csv(ROOT/'config/classification_audit/kotlov_reference_centroids.csv',index_col=0)
display_names={'LME-DE':'Depleted','LME-GC':'GC-like','LME-ME':'Mesenchymal','LME-IN':'Inflammatory'}
coverage=[]; mappings=[]
for cohort,x in expression.items():
    raw={}
    for name,group in features[features.classifier.eq('Kotlov')].groupby('signature',sort=False):
        g=group.copy();g['gene']=g.gene.map(canon)
        genes=list(dict.fromkeys(g.gene)); overlap=[v for v in genes if v in x.index]
        coverage.append(dict(cohort=cohort,signature=name,required=len(genes),measured=len(overlap)))
        if g.weight.isna().all():
            raw[name]=author.ssgsea_formula(x.T,{name:author.GeneSet(name,'author signature',genes)})[name]
        else:
            weights=g.groupby('gene',sort=False).weight.sum()
            raw[name]=x.T.reindex(columns=weights.index,fill_value=0).dot(weights)
    scores=pd.DataFrame(raw,index=x.columns)
    scaled=author_median_scale(scores)
    labels,threshold,selection=select_louvain(scaled,seed=SEED)
    selection.to_csv(OUT/f'{cohort}_kotlov_threshold_selection.csv',index=False)
    scores.to_csv(OUT/f'{cohort}_kotlov_scores.csv')
    clusters=pd.Series(['C'+str(v) for v in labels],index=x.columns)
    named={}
    for cluster,profile in scaled.groupby(clusters).mean().iterrows():
        correlations=reference.apply(lambda r:np.corrcoef(profile.reindex(reference.columns),r)[0,1],axis=1).sort_values(ascending=False)
        assert np.isfinite(correlations).all()
        named[cluster]=display_names[correlations.index[0]]
        mappings.append(dict(cohort=cohort,cluster=cluster,label=named[cluster],correlation=correlations.iloc[0],margin=correlations.iloc[0]-correlations.iloc[1]))
    updates[cohort]['KotlovSig']=clusters.map(named).to_numpy()
    updates[cohort]['Kotlov_cluster_id']=clusters.to_numpy()
    genes=list(dict.fromkeys(features[features.classifier.eq('Ciavarella')].gene.map(canon)))
    if not set(genes)<=set(x.index): raise ValueError('Ciavarella requires all declared 45 features')
    calls=ward_partition(x.loc[genes].T,standardize=False)
    updates[cohort]['Ciav_Cluster']=[{'High':'Hot','Intermediate':'Intermediate','Low':'Cold'}[v] for v in calls]
pd.DataFrame(coverage).to_csv(OUT/'kotlov_feature_coverage.csv',index=False)
pd.DataFrame(mappings).to_csv(OUT/'kotlov_subtype_mapping.csv',index=False)
display(pd.DataFrame(mappings))
'''),M('''## 5. Full Lymph2Cx research reconstruction

Apply the fixed 15-predictor coefficients after subtracting the mean log2 expression of five housekeeping genes. Every input gene is required. This study matrix contains ISY1 and RAB29, so no read-through-gene proxy or dropped predictor is used. The two displayed borders were calibrated externally on GSE178394 NanoString data; no study labels enter the fit. Gene-specific codeset calibration and proprietary QC remain unavailable. [Public calibration cohort](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178394).'''),C('''model=json.loads((ROOT/'config/classification_audit/lymph2cx_full.json').read_text())
f=pd.DataFrame(model['features']).set_index('gene')
assert len(f)==20 and int(f.normalization_weight.ne(0).sum())==5
score_rows=[]
for cohort,x in expression.items():
    if config['study_input_kind'] not in {'nanostring_log2_normalized','nanostring_normalized_counts','nanostring_raw_counts'}:
        raise ValueError('NanoString borders cannot automatically classify TPM/microarray inputs')
    if not set(f.index)<=set(x.index): raise ValueError('Missing full Lymph2Cx features; do not silently drop predictors')
    q=x.loc[f.index].T
    hk=q.mul(f.normalization_weight,axis=1).sum(axis=1)
    contribution=q.sub(hk,axis=0).mul(f.coefficient,axis=1)
    score=contribution.sum(axis=1)
    call=np.where(score<GCB_UNC_THRESHOLD,'GCB',np.where(score<UNC_ABC_THRESHOLD,'UNC','ABC'))
    updates[cohort]['COO_NanoString']=call
    contribution.to_csv(OUT/f'{cohort}_lymph2cx_gene_contributions.csv')
    score_rows.append(pd.DataFrame({'cohort':cohort,'query_id':q.index,'score':score.values,'call':call,'lower':GCB_UNC_THRESHOLD,'upper':UNC_ABC_THRESHOLD}))
scores=pd.concat(score_rows);scores.to_csv(OUT/'lymph2cx_scores.csv',index=False)
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,(cohort,d) in zip(axes,scores.groupby('cohort',sort=False)):
    ax.hist(d.score,bins=25,color='#497999');ax.axvline(GCB_UNC_THRESHOLD,color='black',ls='--');ax.axvline(UNC_ABC_THRESHOLD,color='black',ls='--')
    ax.set(title=cohort,xlabel='Full Lymph2Cx research score',ylabel='Cases')
fig.tight_layout();fig.savefig(OUT/'lymph2cx_thresholds.pdf');plt.show()
'''),M('''## 6. Review and replace the authoritative calls

Updates use current patient IDs directly from the canonical expression matrices. The transaction verifies unchanged clinical fields, genomic calls, spatial archetypes, expression and cell-level payloads. The native EcoTyper confidence and maximum-abundance fields are refreshed together. The source hash prevents promotion after another process changes the AnnData.'''),C('''comparisons=[]
for cohort,new in updates.items():
    old=tables[cohort].copy()
    if 'patient_id' in old: old=old.set_index('patient_id',drop=False)
    assert set(new.index)==set(old.index) and not new.isna().any().any()
    new.to_csv(OUT/f'{cohort}_fresh_calls.csv')
    for col in new:
        prior=old[col].astype(str) if col in old else pd.Series('',index=old.index)
        comparisons.append(dict(cohort=cohort,column=col,n=len(new),changed=int(prior.reindex(new.index).ne(new[col].astype(str)).sum())))
display(pd.DataFrame(comparisons))
# Keep classifier-derived B-cell state summaries synchronized with native recovery.
auxiliary={}
for cohort in updates:
    prefix='uns' if cohort=='Discovery' else 'uns/validation_cohort'
    key=prefix+'/ecotyper_b_state'
    with h5py.File(SOURCE,'r') as h: old_state=read_elem(h[key])
    base=OUT/'ecotyper_work/EcoTyper/Lymphoma/Lymphoma_Fractions/Cell_States/recovery'/f'audit_{cohort}'/'B.cells/6'
    abundance=pd.read_csv(base/'state_abundances.txt',sep='\\t',index_col=0)
    ids=[f'{cohort}__{pid}' for pid in updates[cohort].index]
    a=abundance[ids]; ordered=np.sort(a.to_numpy(),axis=0)
    fresh=pd.DataFrame({'Dominant_B_cell_state':a.idxmax().values,'Dominant_score':ordered[-1],
        'Second_score':ordered[-2],'Margin':ordered[-1]-ordered[-2]},index=updates[cohort].index)
    fresh.index.name=old_state.index.name
    # Keep ancillary/control samples outside the canonical cohort unchanged.
    replacement=pd.concat([old_state.loc[~old_state.index.isin(fresh.index)],fresh])
    replacement=replacement.reindex(old_state.index)
    for col in ['Dominant_score','Second_score','Margin']:
        replacement[col]=pd.to_numeric(replacement[col],errors='raise')
    auxiliary[key]=replacement

provenance={'config':config,'inputs':input_hashes,'method_files':METHOD_FILES,
            'native_LymphoMAP_code':sha256(OUT/'run_lymphomap.R'),
            'ecotyper_manifest':sha256(OUT/'ecotyper_source_hashes.json'),
            'method_status':{'Kotlov':'study clustering reconstruction + external centroid naming','Ciavarella':'study Ward reconstruction','Lymph2Cx':'full research reconstruction; frozen external NanoString borders'}}
(OUT/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
if UPDATE_ADATA:
    changes=promote_calls(SOURCE,updates,OUT/'promotion',audit_name='transcriptomic',run_provenance=provenance,expected_source_sha256=SOURCE_HASH,auxiliary_updates=auxiliary)
    display(changes.groupby(['cohort','column']).changed.agg(['size','sum']))
else:
    print('Review-only run; UPDATE_ADATA is False.')
''')]
notebook('transcriptomic_classification_audit.ipynb',trans)
# Genomic notebook is assembled below after the feature contracts.
gen=[M('''# Genomic classification audit

Four classifiers, separate feature matrices and missing-data rules. Rebuild from selected variants, FISH measurements and declared panels embedded in the release AnnData, run native LymphGen/DLBclass/LymphPlex and an auditable HMRN adapter, compare with current calls, then replace the authoritative AnnData fields with a backup and verified atomic write.

**Input scope:** existing pathogenic-variant selection and AF >2% (or missing AF) are retained. This is not a new somatic-variant calling pipeline. Discovery panel assignments were recorded in the source workbook and are embedded with these inputs; validation BLYMFv2 remains a provisional study assumption. Absence of variants is not sufficient evidence that a patient was sequenced. Cases outside the explicit source manifest are listed. The explicitly declared discovery cases with unavailable source inputs have their unsupported calls cleared to unknown; other noneligible cases are retained.

**Method status:** HMRN observed-feature prediction is a research extension. LymphPlex runs the unmodified author function with numeric NA inputs, using native PAM missing-value handling; incomplete-panel performance was not separately validated in the publication. Coverage, exact-completion sensitivity and public example masking results are reported separately from subtype labels.'''),C(setup+'''
OUT=Path(config['output_root'])/'genomic';OUT.mkdir(parents=True,exist_ok=True)
VENDOR=Path(config['vendor'])
require_audit_resources(config, 'genomic')
METHOD_FILES={str(p):sha256(p) for p in [CONFIG_PATH,ROOT/'dlbcl/classification_audit.py',ROOT/'dlbcl/genomic_classification_audit.py',ROOT/'tools/classification_audit/prepare_genomic.py',ROOT/'tools/classification_audit/run_lymphgen.R',ROOT/'tools/classification_audit/run_lymphplex.R',*sorted((ROOT/'config/dlbclass').glob('*')),*sorted((ROOT/'config/classification_audit').glob('hmrn*'))] if p.is_file()}
'''),M('''## 1. Rebuild classifier inputs from released selected variants, panel masks and measured FISH

The release contains selected mutation events and measurement availability in `uns["classification_inputs"]`, independently of the saved classifier calls. Raw variant calling and the upstream pathogenic/AF selection are outside this notebook. The preparation code is printed in full before execution. It retains approved gene/Entrez mappings, hotspot and truncating consequences, source variant coordinates, and per-case panel assignments. Measured-negative FISH is 0, positive is 1, unavailable is missing. No regional copy number is inferred. DLBclass encodes synonymous/nonsynonymous/SV events as 1/2/3 before feature-specific mean imputation. HMRN and LymphPlex have their own binary encodings in later cells.'''),C('''prepare=ROOT/'tools/classification_audit/prepare_genomic.py'
print(prepare.read_text())
run([sys.executable,prepare],OUT/'prepare.log',cwd=ROOT)
variants=pd.read_csv(OUT/'selected_variants.tsv',sep='\\t')
cases=pd.read_csv(OUT/'case_metadata.csv')
panel=pd.read_csv(OUT/'panel_coverage.tsv',sep='\\t')
updates={};eligibility=[]
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    ids=cases.loc[cases.cohort.eq(short),'patient_id']
    assert ids.is_unique
    canonical=pd.Index(tables[cohort].patient_id if 'patient_id' in tables[cohort] else tables[cohort].index)
    assert ids.isin(canonical).all()
    updates[cohort]=pd.DataFrame(index=ids)
    eligibility.extend(dict(cohort=cohort,patient_id=pid,eligible=pid in set(ids)) for pid in canonical)
pd.DataFrame(eligibility).to_csv(OUT/'eligibility.csv',index=False)
display(cases.groupby(['cohort','panel','panel_evidence']).size().rename('cases'))
'''),M('''## 2. LymphGen 2.0: native reference model

Run separately for each panel using its complete measured-gene list, including genes with zero observed variants. Unknown FISH stays missing. No CN data means A53 is unavailable, not negative. MYD88 L265P and truncating mutations retain their explicit encoding. The native local model was previously verified against the official portal on identical inputs. [Wright et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8459709/).'''),C('''rscript=ROOT/'tools/classification_audit/run_lymphgen.R'
print(rscript.read_text())
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    result=[]
    for panel_name in cases.loc[cases.cohort.eq(short),'panel'].unique():
        source=OUT/'inputs'/f'{short}_{panel_name}'
        dest=OUT/'LymphGen'/f'{short}_{panel_name}'
        run([config['rscript'],rscript,VENDOR/'LymphGen2',source,dest],dest/'run.log')
        d=pd.read_csv(dest/'Result.tsv',sep='\\t').set_index('Sample.Name')
        expected=pd.read_csv(source/'sample_annotation.txt',sep='\\t').Sample
        assert d.index.is_unique and set(d.index)==set(expected)
        result.append(d)
    d=pd.concat(result)
    assert set(d.index)==set(updates[cohort].index)
    updates[cohort]['Lymphgen']=d.loc[updates[cohort].index,'Subtype.Prediction']
    d.to_csv(OUT/f'{cohort}_lymphgen_scores.csv')
    display(updates[cohort].Lymphgen.value_counts().rename(cohort))
'''),M('''## 3. DLBclass: author ensemble with reference-mean imputation

Unmeasured mutation, CN and rearrangement features receive their original encoded-feature means from the public 699-case reference. Mutation-negative is zero only for a measured gene. FISH positivity is a structural-variant input; merely sequencing MYC/BCL2/BCL6 does not measure a rearrangement. The native 163-to-21 reduction is evaluated with floating-point values so imputation is not truncated to integers. Execute all 500 author networks and their original confidence transformation. [DLBclass paper](https://doi.org/10.1182/blood.2024025652).'''),C('''from dlbcl.dlbclass_targeted import run_official_ensemble
ref=pd.read_csv(ROOT/'config/dlbclass/feature_reference.csv',index_col=0)
weights=pd.read_csv(ROOT/'config/dlbclass/reduction_weights.csv',index_col=0)
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    folder=OUT/'dlbclass'/short
    gsm=pd.read_csv(folder/'imputed_gsm.tsv',sep='\\t',index_col=0)
    mask=pd.read_csv(folder/'measured_mask.tsv',sep='\\t',index_col=0)
    assert not mask.loc[ref.type.eq('Regional CNV')].any().any()
    for pid in gsm:
        np.testing.assert_allclose(gsm.loc[~mask[pid],pid],ref.loc[~mask[pid],'reference_mean'])
    reduced=gsm.T @ weights.loc[gsm.index]
    scores,model_hash=run_official_ensemble(reduced,VENDOR/'DLBclass-tool')
    scores.to_csv(folder/'fresh_results.tsv',sep='\\t')
    assert set(scores.index)==set(updates[cohort].index)
    updates[cohort]['DLBclass']=scores.loc[updates[cohort].index,'DLBclass']
    METHOD_FILES['DLBclass_500_networks']=model_hash
    display(scores.DLBclass.value_counts().rename(cohort))
'''),M('''## 4. HMRN: restore the author reference fit and explicit missingness

The author reference fit is reproduced with RNG 3.5.3, seed 17, k=1…10, 1,000 starts and AIC. The six cluster sizes and enriched genes identify the published biological labels; we store names, not arbitrary component numbers. The fit and its complete-input predictions are frozen reference artifacts, independent of study labels. The executable reference-fit script is displayed below and can be rerun.

Corrected features include historical gene symbols, EZH2 Y641/Y646 isoforms, MYD88 L265P, and **mutation OR homozygous deletion**. A detected mutation proves the OR feature positive; no mutation with unmeasured CN does not prove it negative. The supplement describes broader SHM ascertainment for selected genes. A pathogenic-only export cannot establish those negatives: `_S` features retain observed mutations and otherwise remain missing.

For missing features, integrate them out of the Bernoulli likelihood (omit their terms). This is an explicit mathematical extension of the author's complete-input model, not its published validation. Export every feature's evidence and posterior; no zero filling. With no measured features, report Unassigned rather than classify from priors alone. [Author code](https://github.com/ecsg-uoy/DLBCLGenomicSubtyping/blob/master/fit_clusters.R), [supplement](https://api.repository.cam.ac.uk/server/api/core/bitstreams/8ecf06ef-3ab8-4356-8775-f6e7522cabcf/content).'''),C('''from dlbcl.genomic_classification_audit import hmrn_features, bernoulli_posterior
print((ROOT/'tools/classification_audit/fit_hmrn.R').read_text())
print(inspect.getsource(hmrn_features)); print(inspect.getsource(bernoulli_posterior))
hmrn_meta=json.loads((ROOT/'config/classification_audit/hmrn_reference.json').read_text())
assert sha256(VENDOR/'DLBCLGenomicSubtyping/data/genomic_data.csv')==hmrn_meta['reference_data_sha256']
centers=pd.read_csv(ROOT/'config/classification_audit/hmrn_reference.centers.csv',index_col=0)
priors=pd.read_csv(ROOT/'config/classification_audit/hmrn_reference.priors.csv').set_index('component').prior.reindex(centers.columns)
reference=pd.read_csv(VENDOR/'DLBCLGenomicSubtyping/data/genomic_data.csv').set_index('PID')
ref_calls=pd.read_csv(ROOT/'config/classification_audit/hmrn_reference.calls.csv').set_index('PID').component
reference_posterior=bernoulli_posterior(reference[centers.index],centers,priors)
assert reference_posterior.idxmax(axis=1).eq(ref_calls).all()
native_posterior=pd.read_csv(ROOT/'config/classification_audit/hmrn_reference.posterior.csv').set_index('PID')
np.testing.assert_allclose(reference_posterior,native_posterior,atol=1e-12)
display(ref_calls.map(hmrn_meta['labels']).value_counts())
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    c=cases[cases.cohort.eq(short)]
    v=variants[variants.cohort.eq(short)]
    fish=pd.read_csv(OUT/f'{short}_fish.csv',index_col=0)
    x,evidence=hmrn_features(v,c,panel,centers.index,fish,canon)
    posterior=bernoulli_posterior(x,centers,priors).rename(columns=hmrn_meta['labels'])
    calls=posterior.idxmax(axis=1).where(x.notna().any(axis=1),'Unassigned')
    updates[cohort]['HMRN']=calls.reindex(updates[cohort].index)
    posterior.to_csv(OUT/f'{cohort}_hmrn_posteriors.csv')
    x.to_csv(OUT/f'{cohort}_hmrn_features.csv');evidence.to_csv(OUT/f'{cohort}_hmrn_feature_evidence.csv',index=False)
    display(calls.value_counts().rename(cohort))
'''),M('''## 5. LymphPlex: native assignment with missing inputs

Run the unmodified author `GetCluster` function with numeric `NA` values. A unique observed seed pattern takes precedence; otherwise native PAM compares available features with the fixed centers. TP53 mutation overrides the result. The missing-distance rescaling is common across centers for each case. Measured negatives remain 0; off-panel genes and unavailable FISH remain missing. No zero or mean imputation is performed. TP53 must be measured, and non-TP53 cases need at least one measured center feature; this is an execution safeguard, not a validated minimum-coverage threshold.

The paper does not separately prescribe or validate incomplete-panel handling. Native PAM supports missing values. We retain feature coverage and exhaustive binary-completion sensitivity as annotations; disagreement among hypothetical completions does **not** prevent a native subtype call. `Others` is the native residual category. Cases with no genomic source data remain unknown outside this eligible cohort.

[Author code](https://github.com/difuSJTU/LymphPlex), [paper](https://www.nature.com/articles/s41392-023-01358-y), [PAM missing-value support](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html).'''),C('''from dlbcl.genomic_classification_audit import (lymphplex_features, lymphplex_partial,
    lymphplex_observed, lymphplex_native)
print(inspect.getsource(lymphplex_native)); print(inspect.getsource(lymphplex_observed))
print((ROOT/'tools/classification_audit/run_lymphplex.R').read_text())
centers=pd.read_csv(VENDOR/'LymphPlex/Centers.txt',sep='\\t',index_col=0)
seeds=pd.read_csv(VENDOR/'LymphPlex/Seeds.txt',sep='\\t')
lymphplex_auxiliary={}
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    c=cases[cases.cohort.eq(short)];v=variants[variants.cohort.eq(short)]
    fish=pd.read_csv(OUT/f'{short}_fish.csv',index_col=0)
    x=lymphplex_features(v,c,panel,centers,fish,canon)
    native=lymphplex_native(x,VENDOR/'LymphPlex',config['rscript'],OUT/f'{cohort}_lymphplex_native')
    sensitivity,completions=lymphplex_partial(x,centers,seeds)
    calls=sensitivity.rename(columns={'call':'completion_consensus','status':'completion_status'})
    calls.insert(0,'call',native)
    calls['completion_invariant']=calls.completion_consensus.ne('Unassigned')
    calls['n_measured']=x.notna().sum(axis=1);calls['n_required']=len(x.columns)
    calls['status']=np.where(x.isna().any(axis=1),'native_PAM_partial_inputs','native_complete_inputs')
    updates[cohort]['LymphPlex']=native.reindex(updates[cohort].index)
    x.to_csv(OUT/f'{cohort}_lymphplex_features.csv');calls.to_csv(OUT/f'{cohort}_lymphplex_audit.csv')
    completions.to_csv(OUT/f'{cohort}_lymphplex_completions.csv',index=False)
    prefix='uns' if cohort=='Discovery' else 'uns/validation_cohort'
    lymphplex_auxiliary[prefix+'/lymphplex_call_audit']=calls
    lymphplex_auxiliary[prefix+'/lymphplex_input_method']={
        'algorithm':'Author GetCluster with fixed PAM centers and TP53 override',
        'missing':'Numeric NA; native available-feature distances; no imputation',
        'sensitivity':'All binary completions are retained as a robustness annotation, not a call gate',
        'scope':'Cases with explicit genomic source inputs; TP53 measured',
        'limitation':'Incomplete-panel clinical accuracy not separately validated by authors',
        'publication':'https://www.nature.com/articles/s41392-023-01358-y',
        'author_code':'https://github.com/difuSJTU/LymphPlex'}
    display(calls.groupby(['call','completion_invariant']).size().rename(cohort))
'''),M('''### Public example masking check

Rerun the author-supplied example matrix with full inputs and then mask the two BLYMFv2-missing genes, optionally also BCL6 or both BCL2/BCL6 rearrangements. Report agreement and per-label recall/precision against the full-input **native calls**, not against the unrelated label column in the example files. This is a reproducibility sensitivity check, not an independent clinical validation. TP53 priority contributes substantially to overall agreement; also show non-TP53 agreement and label-specific results. These external examples never define study labels or fit a new model.'''),C('''example=pd.read_csv(VENDOR/'LymphPlex/Matrix.txt',sep='\\t',dtype={'Sample':str}).set_index('Sample')
tp53=pd.read_csv(VENDOR/'LymphPlex/TP53.txt',sep='\\t',dtype={'Sample':str}).set_index('Sample')
assert example.index.equals(tp53.index)
example['TP53']=tp53.TP53
baseline=lymphplex_native(example,VENDOR/'LymphPlex',config['rscript'],OUT/'lymphplex_example/full')
masks={'genes':['DDX3X','ZFP36L1'],
       'genes_BCL6':['DDX3X','ZFP36L1','BCL6'],
       'genes_both_FISH':['DDX3X','ZFP36L1','BCL2','BCL6']}
summary=[];metrics=[]
for name,features in masks.items():
    masked=example.copy();masked[features]=np.nan
    calls=lymphplex_native(masked,VENDOR/'LymphPlex',config['rscript'],OUT/'lymphplex_example'/name)
    matches=calls.eq(baseline);non_tp53=baseline.ne('TP53')
    summary.append(dict(mask=name,n=len(calls),matches=int(matches.sum()),agreement=float(matches.mean()),
                        non_TP53_n=int(non_tp53.sum()),non_TP53_agreement=float(matches[non_tp53].mean())))
    pd.crosstab(baseline,calls,rownames=['full'],colnames=['masked']).to_csv(OUT/f'lymphplex_example_{name}_confusion.csv')
    for label in sorted(set(baseline)|set(calls)):
        true=baseline.eq(label);pred=calls.eq(label);tp=int((true&pred).sum())
        metrics.append(dict(mask=name,label=label,n_full=int(true.sum()),n_masked=int(pred.sum()),
            recall=tp/true.sum() if true.any() else np.nan,precision=tp/pred.sum() if pred.any() else np.nan))
summary=pd.DataFrame(summary);metrics=pd.DataFrame(metrics)
summary.to_csv(OUT/'lymphplex_example_agreement.csv',index=False)
metrics.to_csv(OUT/'lymphplex_example_label_metrics.csv',index=False)
display(summary);display(metrics)
'''),
M('''## 6. Before/after audit and authoritative replacement

Promotion includes only the explicitly eligible genomic cases. Cases with unavailable source inputs are listed in `eligibility.csv`. T29, T41 and T48 have their unsupported calls cleared to unknown, as explicitly specified in the configuration; their old calls remain in the audit backups. Other noneligible cases are retained. Absence from this source is not converted into a negative genotype. The current source hash is checked again and unrelated AnnData contents are verified unchanged. HMRN biological names and native LymphPlex labels are kept separate from missing-input and sensitivity annotations. Unavailable genomic source inputs remain unknown.'''),C('''# Explicit source-unavailability correction; never infer a negative genotype from no rows.
cleared=[]
for pid in config.get('clear_unavailable_discovery_calls',[]):
    if pid not in updates['Discovery'].index:
        assert pid in set(tables['Discovery'].patient_id)
        updates['Discovery'].loc[pid,['Lymphgen','DLBclass','HMRN','LymphPlex']]='unknown'
        cleared.append(pid)
for cohort,new in updates.items():
    new['genomic_classification_status']='classified from available source inputs'
updates['Discovery'].loc[cleared,'genomic_classification_status']='source inputs unavailable'
rows=[]
for cohort,new in updates.items():
    old=tables[cohort].copy()
    if 'patient_id' in old: old=old.set_index('patient_id',drop=False)
    assert not new.isna().any().any()
    new.to_csv(OUT/f'{cohort}_fresh_calls.csv')
    for col in new:
        rows.append(dict(cohort=cohort,classifier=col,n=len(new),changed=int((old[col].reindex(new.index).astype(str) if col in old else pd.Series('',index=new.index)).ne(new[col].astype(str)).sum())))
display(pd.DataFrame(rows))
# Replace accompanying DLBclass scores/method metadata so they cannot describe old calls.
auxiliary=dict(lymphplex_auxiliary)
for short,cohort in [('discovery','Discovery'),('validation','Validation')]:
    prefix='uns' if cohort=='Discovery' else 'uns/validation_cohort'
    scores=pd.read_csv(OUT/'dlbclass'/short/'fresh_results.tsv',sep='\\t',index_col=0)
    auxiliary[prefix+'/dlbclass_mean_scores']=scores[['C1','C2','C3','C4','C5','Confidence']]
    auxiliary[prefix+'/dlbclass_input_method']={
        'date':'2026-10-17','missing':'699-reference encoded-feature means',
        'coverage':'Discovery: recorded BLYMF200/BLYMFv2; validation: provisional BLYMFv2',
        'eligibility':'Explicit genomic source manifest; scores include only freshly classified cases',
        'fish':'Missing imputed, negative zero, positive three','regional_cnv':'Unmeasured and imputed',
        'mutation_encoding':'Selected synonymous=1, nonsynonymous=2',
        'reduction':'Author linear reduction, floating-point means preserved'}
auxiliary['uns/lymphgen_discovery_method']={
    'date':'2026-10-17','source':'Native LymphGen 2.0 rerun from panel-specific mutation/FISH inputs',
    'copy_number':'Unavailable; A53 not evaluated','coverage':'Recorded BLYMF200/BLYMFv2',
    'case_count':str(int(cases.cohort.eq('discovery').sum()))}

provenance={'config':config,'method_files':METHOD_FILES,'input_manifest':json.loads((OUT/'input_manifest.json').read_text()),
            'hmrn_reference':hmrn_meta,'cleared_unavailable_discovery_calls':cleared,'HMRN_partial_input':'Observed-feature Bernoulli likelihood; CN and incomplete SHM negatives marginalized',
            'LymphPlex_partial_input':'Native author GetCluster/PAM with numeric NA, observed-feature distances, TP53 override; no imputation; completion sensitivity does not gate labels',
            'limitations':'Pathogenic-selected variants; panel coverage is not sample-level callable-base coverage; validation BLYMFv2 provisional'}
(OUT/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
if UPDATE_ADATA:
    changes=promote_calls(SOURCE,updates,OUT/'promotion',audit_name='genomic',run_provenance=provenance,expected_source_sha256=SOURCE_HASH,auxiliary_updates=auxiliary)
    display(changes.groupby(['cohort','column']).changed.agg(['size','sum']))
else: print('Review-only run; UPDATE_ADATA is False.')
''')]
notebook('genomic_classification_audit.ipynb',gen)
