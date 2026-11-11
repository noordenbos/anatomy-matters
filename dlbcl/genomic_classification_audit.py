"""Panel-aware feature availability and explicit partial-input research adapters."""
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def hmrn_features(variants, cases, panel, reference_features, fish, canon):
    """Known events=1; complete measured negatives=0; unavailable/partial=NaN.

    _OR_del: mutation OR homozygous deletion; a mutation establishes positive,
    but mutation-negative with unavailable CN does not establish negative.
    _S: broader SHM ascertainment is not established by pathogenic-only export;
    known driver events are positive and unobserved events remain unavailable.
    """
    x=pd.DataFrame(np.nan,index=cases.patient_id,columns=reference_features)
    evidence=[]
    for case in cases.itertuples():
        measured=set(panel.loc[panel[case.panel],'approved_symbol'])
        v=variants[variants.patient_id.eq(case.patient_id)]
        for feature in reference_features:
            if feature.endswith('_amp'):
                reason='copy_number_unmeasured'; selected=v.iloc[:0];available=False
            else:
                gene=feature
                for suffix in ['_OR_del','_S','_265','_noncan','_646','_239']:
                    if gene.endswith(suffix): gene=gene[:-len(suffix)];break
                gene=canon(gene)
                selected=v[v.gene.map(canon).eq(gene)&v.effect.ne('Synon')]
                available=gene in measured
                reason='panel_measured' if available else 'outside_panel'
                if feature.startswith('MYD88_'):
                    selected=selected[selected.effect.eq('L265P') if feature.endswith('_265') else selected.effect.ne('L265P')]
                elif feature.startswith('EZH2_'):
                    hit=selected.aa_change.fillna('').str.contains(r'Y(?:641|646)',case=False,regex=True)
                    selected=selected[hit if feature.endswith('_646') else ~hit]
                elif feature.startswith('POU2F2_'):
                    hit=selected.aa_change.fillna('').str.contains('T239',case=False,regex=False)
                    selected=selected[hit if feature.endswith('_239') else ~hit]
                if feature.endswith('_OR_del'):
                    available=False;reason='negative_requires_homozygous_deletion_measurement'
                if feature.endswith('_S'):
                    available=False;reason='negative_requires_broader_SHM_ascertainment'
            if len(selected): value=1.;reason='observed_selected_mutation'
            else: value=0. if available else np.nan
            x.at[case.patient_id,feature]=value
            evidence.append(dict(patient_id=case.patient_id,feature=feature,value=value,reason=reason))
    return x,pd.DataFrame(evidence)


def bernoulli_posterior(x, centers, priors):
    """Exact observed-feature likelihood; unobserved features integrate to one."""
    if not x.columns.equals(centers.index): raise ValueError('HMRN feature order mismatch')
    observed=x.notna().to_numpy()
    values=x.fillna(0).to_numpy()
    if not np.isin(values,[0,1]).all(): raise ValueError('HMRN requires binary observed features')
    p=np.clip(centers.to_numpy(),np.finfo(float).eps,1-np.finfo(float).eps)
    logp=np.log(np.asarray(priors))[None,:]+values@np.log(p)+(observed-values)@np.log1p(-p)
    posterior=np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
    return pd.DataFrame(posterior,index=x.index,columns=centers.columns)


def lymphplex_features(variants,cases,panel,centers,fish,canon):
    x=pd.DataFrame(np.nan,index=cases.patient_id,columns=list(centers.columns)+['TP53'])
    for case in cases.itertuples():
        measured=set(panel.loc[panel[case.panel],'approved_symbol'])
        v=variants[variants.patient_id.eq(case.patient_id)&variants.effect.ne('Synon')]
        positive=set(v.gene.map(canon))
        for gene in x:
            if gene in {'BCL2','BCL6'}:
                x.at[case.patient_id,gene]=fish.loc[case.patient_id,gene.lower()+'_final']
            elif canon(gene) in positive: x.at[case.patient_id,gene]=1
            elif canon(gene) in measured: x.at[case.patient_id,gene]=0
    return x


def lymphplex_complete(x,centers,seeds):
    """Author seed override then fixed-medoid Euclidean assignment; TP53 last.

    Returns every tied nearest subtype rather than inventing a tie break. Tests
    compare the non-tied complete-input cases to the native R implementation.
    """
    x=x.astype(float)
    if x.isna().any(): raise ValueError('Complete binary values required')
    if not x.isin([0,1]).all(): raise ValueError('Nonbinary LymphPlex input')
    if x.TP53==1: return {'TP53'}
    seed_hits=[label for label,g in seeds.groupby('Cluster',sort=False) if x.reindex(g.Seed).eq(1).all()]
    if len(seed_hits)==1: return set(seed_hits)
    distance=centers.sub(x.reindex(centers.columns),axis=1).pow(2).sum(axis=1)
    return set(distance.index[np.isclose(distance,distance.min(),rtol=0,atol=1e-12)])


def lymphplex_partial(x,centers,seeds,max_missing=12):
    """Enumerate absent measurements, not a probability distribution over them."""
    rows=[];completions=[]
    for pid,row in x.iterrows():
        missing=row.index[row.isna()].tolist()
        if len(missing)>max_missing: raise ValueError(f'{pid}: too many missing features for exact enumeration')
        possible=set()
        for i,bits in enumerate(itertools.product([0.,1.],repeat=len(missing))):
            filled=row.copy();filled.loc[missing]=bits
            labels=lymphplex_complete(filled,centers,seeds);possible.update(labels)
            completions.append(dict(patient_id=pid,completion=i,possible_labels='|'.join(sorted(labels))))
        rows.append(dict(patient_id=pid,call=next(iter(possible)) if len(possible)==1 else 'Unassigned',
            status='invariant_across_missing_features' if len(possible)==1 else 'ambiguous_missing_features',
            possible_labels='|'.join(sorted(possible)),missing_features='|'.join(missing),n_completions=2**len(missing)))
    return pd.DataFrame(rows).set_index('patient_id'),pd.DataFrame(completions)


def lymphplex_observed(x, centers, seeds):
    """Independent check of native seed/PAM assignment using observed features.

    Return every nearest-center tie for parity checks; native R determines the
    actual label. TP53 must be measured. Missing inputs are never encoded as zero.
    """
    x=x.reindex([*centers.columns,'TP53']).astype(float)
    if not x.dropna().isin([0,1]).all(): raise ValueError('Nonbinary LymphPlex input')
    if pd.isna(x.TP53): raise ValueError('TP53 measurement required for priority rule')
    if x.TP53==1: return {'TP53'}
    seed_hits=[label for label,g in seeds.groupby('Cluster',sort=False) if x.reindex(g.Seed).eq(1).all()]
    if len(seed_hits)==1: return set(seed_hits)
    measured=x.reindex(centers.columns).dropna()
    if measured.empty: raise ValueError('No measured LymphPlex center features')
    # Native PAM rescales distances by the same missingness factor for every center.
    distance=centers.loc[:,measured.index].sub(measured,axis=1).pow(2).sum(axis=1)
    return set(distance.index[np.isclose(distance,distance.min(),rtol=0,atol=1e-12)])


def lymphplex_native(x, vendor, rscript, output):
    """Run author GetCluster/PAM on numeric NA inputs, with the TP53 override.

    The paper does not separately validate incomplete panels. Preserve feature
    coverage and completion sensitivity as annotations, not an abstention gate.
    """
    from .classification_audit import run
    vendor=Path(vendor);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    centers=pd.read_csv(vendor/'Centers.txt',sep='\t',index_col=0)
    seeds=pd.read_csv(vendor/'Seeds.txt',sep='\t')
    columns=[*centers.columns,'TP53']
    if not x.index.is_unique or not set(columns)<=set(x): raise ValueError('Require unique cases and explicit classifier feature columns')
    inputs=x.loc[:,columns].astype(float).copy()
    expected={str(pid):lymphplex_observed(row,centers,seeds) for pid,row in inputs.iterrows()}
    inputs.index=inputs.index.astype(str)
    if not inputs.index.is_unique: raise ValueError('Case identifiers collide after string conversion')
    inputs.rename_axis('Sample').to_csv(output/'inputs.tsv',sep='\t',na_rep='NA')
    script=Path(__file__).resolve().parents[1]/'tools/classification_audit/run_lymphplex.R'
    run([rscript,script,vendor,output/'inputs.tsv',output/'calls.tsv'],output/'native.log')
    calls=pd.read_csv(output/'calls.tsv',sep='\t',dtype={'Sample':str}).set_index('Sample')['call']
    if calls.index.tolist()!=inputs.index.tolist(): raise ValueError('Native output case identities differ')
    if any(call not in expected[pid] for pid,call in calls.items()): raise ValueError('Native PAM differs from independent observed-feature calculation')
    calls.index=x.index.copy()
    return calls
