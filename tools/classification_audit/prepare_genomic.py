"""Build fresh genomic classifier inputs from released selected events and availability."""
from pathlib import Path
import hashlib, json, sys
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
RELEASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RELEASE))
from dlbcl.classification_audit import load_audit_config
from dlbcl.lymphgen_portal import resolve_gene, fish_value, write_bundle
CONFIG,_=load_audit_config(RELEASE)
OUT=Path(CONFIG['output_root'])/'genomic'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path(CONFIG['source_adata'])
    with h5py.File(source,'r') as f:
        if 'uns/classification_inputs' not in f:
            raise ValueError('This audit requires the v1.1.0 AnnData containing uns["classification_inputs"]. Download the matching release bundle.')
        bundle=read_elem(f['uns/classification_inputs'])
    if bundle['schema']!='selected-genomic-inputs-v1':raise ValueError('Unsupported genomic-input schema')
    panel=bundle['panels']; variants=bundle['selected_variants']; cases=bundle['cases']
    hgnc=pd.read_csv(CONFIG['hgnc'],sep='\t',dtype=str)
    def gene(g):return resolve_gene(g,hgnc)
    panels={p:set(panel.loc[panel[p],'entrez_id']) for p in ['LYMFv1','BLYMF200','BLYMFv2']}
    panel.to_csv(OUT/'panel_coverage.tsv',sep='\t',index=False)
    refs=pd.read_csv(RELEASE/'config/dlbclass/feature_reference.csv',index_col=0)
    weights=pd.read_csv(RELEASE/'config/dlbclass/reduction_weights.csv',index_col=0)
    feature_ids={feature:gene(r.gene)[1] for feature,r in refs.iterrows() if r.type=='Mutation'}
    rows=variants.to_dict('records'); report={}
    for cohort in ['discovery','validation']:
        c=cases.loc[cases.cohort.eq(cohort)]
        if not c.patient_id.is_unique:raise ValueError('Duplicate case identifier in genomic inputs')
        pmap=c.set_index('patient_id').panel
        assert pmap.isin(panels).all()
        meta=bundle[cohort+'_fish'].rename(columns={l.lower()+'_final':('t'+l if cohort=='discovery' else l.lower()+'_fish') for l in ['MYC','BCL2','BCL6']})
        calls=pd.DataFrame(rows);calls=calls.loc[calls.cohort.eq(cohort)].drop_duplicates()
        ids=sorted(pmap.index,key=lambda x:int(x[1:]));samples=[]
        fish=pd.DataFrame(index=ids,columns=['myc_final','bcl2_final','bcl6_final'],dtype=float)
        for pid in ids:
            values={}
            for locus in ['MYC','BCL2','BCL6']:
                col='t'+locus if cohort=='discovery' else locus.lower()+'_fish'
                value=fish_value(meta.loc[pid,col]) if col in meta else 'NA'
                values[locus]=value;fish.loc[pid,locus.lower()+'_final']=np.nan if value=='NA' else int(value)
            samples.append({'Sample':pid,'Copy.Number':0,'BCL2.Translocation':values['BCL2'],'BCL6.Translocation':values['BCL6']})
        fish.to_csv(OUT/f'{cohort}_fish.csv')
        samples=pd.DataFrame(samples)
        for p in sorted(pmap.unique()):
            selected=set(pmap.index[pmap.eq(p)])
            m=calls.loc[calls.patient_id.isin(selected),['patient_id','entrez_id','effect','position']].copy()
            m.columns=['Sample','ENTREZ.ID','Type','Location']
            write_bundle(OUT/'inputs'/f'{cohort}_{p}',samples.loc[samples.Sample.isin(selected)],m,list(panels[p]))
        # Official feature encoding: silent=1, nonsilent=2, SV=3. Unmeasured
        # features receive original encoded-feature means; no regional CN inferred.
        gsm=pd.DataFrame(0.,index=refs.index,columns=ids)
        mask=pd.DataFrame(False,index=refs.index,columns=ids)
        for feature,r in refs.iterrows():
            if r.type=='Mutation':
                gid=feature_ids[feature]
                for pid in ids:mask.loc[feature,pid]=gid in panels[pmap[pid]]
                selected=calls.loc[calls.entrez_id.eq(gid)]
                if feature=='MYD88.L265P':selected=selected.loc[selected.effect.eq('L265P')]
                if feature=='MYD88.OTHER':selected=selected.loc[selected.effect.ne('L265P')]
                for pid,group in selected.groupby('patient_id'):
                    gsm.loc[feature,pid]=max(1 if x=='Synon' else 2 for x in group.effect)
                    assert mask.loc[feature,pid]
            elif r.type=='Rearrangement':
                status=fish[r.gene.lower()+'_final']
                mask.loc[feature]=status.notna();gsm.loc[feature]=status.fillna(0).to_numpy()*3
            else:assert r.type=='Regional CNV'
        corrected=gsm.where(mask,refs.reference_mean,axis=0)
        reduced=corrected.T @ weights.loc[corrected.index]
        folder=OUT/'dlbclass'/cohort;folder.mkdir(parents=True,exist_ok=True)
        corrected.to_csv(folder/'imputed_gsm.tsv',sep='\t');mask.to_csv(folder/'measured_mask.tsv',sep='\t')
        reduced.to_csv(folder/'reduced_inputs.tsv',sep='\t')
        report[cohort]={'cases':len(ids),'panels':pmap.value_counts().to_dict(),'selected_variant_rows':len(calls)}
    variants.to_csv(OUT/'selected_variants.tsv',sep='\t',index=False)
    cases.to_csv(OUT/'case_metadata.csv',index=False)
    report['sources']={'release_adata_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    report['canonical_assignments_modified']=False
    report['scope']=bundle['scope']
    (OUT/'input_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
