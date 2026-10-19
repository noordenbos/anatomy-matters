"""Panel design coverage from static reference requirements; no AnnData dependency."""
from pathlib import Path
import json
import pandas as pd


def load_requirements(path):
    payload=json.loads(Path(path).read_text())
    if payload.get('schema')!='declared-panel-features-v1':
        raise ValueError('Expected a declared-panel feature requirements file')
    return payload


def genomic_design_coverage(payload):
    details=pd.DataFrame(payload['genomic_features'])
    keys=['classifier','panel','fish','subtype']
    if details.duplicated(keys+['feature']).any():
        raise ValueError('Duplicate genomic feature memberships')
    rows=[]
    for key,q in details.groupby(keys,sort=False):
        row=dict(zip(keys,key),requested=len(q),delivered=int(q.delivered.sum()))
        for modality in ['mutation','sv','cn']:
            z=q[q.modality.eq(modality)]
            row[modality+'_requested']=len(z)
            row[modality+'_delivered']=int(z.delivered.sum())
            row['missing_'+modality]=len(z)-int(z.delivered.sum())
        rows.append(row)
    return pd.DataFrame(rows),details


def transcriptomic_design_coverage(payload):
    panel=payload['transcriptomic_panel']
    targets=set(panel['genes'])
    rows=[];details=[]
    for feature_set in payload['transcriptomic_reference_sets']:
        metadata={k:v for k,v in feature_set.items() if k not in ['genes','requested']}
        genes=set(feature_set['genes'])
        n=feature_set['requested'];covered=genes & targets
        if len(genes)>n:raise ValueError('Feature list exceeds stated requirement')
        rows.append(dict(metadata,panel=panel['name'],requested=n,
            on_panel=len(covered),delivered=len(covered),not_on_panel=n-len(covered),
            on_panel_missing_source=0,coverage_pct=100*len(covered)/n))
        for gene in sorted(genes):
            details.append(dict(metadata,panel=panel['name'],gene=gene,on_panel=gene in targets))
    return pd.DataFrame(rows),pd.DataFrame(details)


def kotlov_signature_coverage(payload):
    """Unweighted gene-set retention, separated from weighted pathway scoring."""
    targets=set(payload['transcriptomic_panel']['genes'])
    rows=[];details=[];unions={'All inputs':set(),'22 ssGSEA signatures':set(),'3 PROGENy pathways':set()}
    for entry in payload['kotlov_signatures']:
        genes=set(entry['genes']);hit=genes & targets
        method=entry['method'];name=entry['signature']
        rows.append(dict(signature=name,method=method,requested=len(genes),on_panel=len(hit),
                         missing=len(genes-hit),coverage_pct=100*len(hit)/len(genes)))
        unions['All inputs'].update(genes)
        unions['22 ssGSEA signatures' if method=='ssGSEA' else '3 PROGENy pathways'].update(genes)
        for gene in sorted(genes):
            details.append(dict(signature=name,method=method,gene=gene,on_panel=gene in targets,
                                coefficient=entry['coefficients'].get(gene)))
    summary=pd.DataFrame([dict(group=name,requested=len(genes),on_panel=len(genes & targets),
                              missing=len(genes-targets),coverage_pct=100*len(genes & targets)/len(genes))
                          for name,genes in unions.items()])
    return pd.DataFrame(rows),summary,pd.DataFrame(details)


def export_kotlov_gene_sets(payload, out):
    """Write the requested/covered ssGSEA GMT sets and weighted pathway inputs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    coverage,summary,details=kotlov_signature_coverage(payload)
    coverage.to_csv(out/'kotlov_signature_coverage.csv',index=False)
    summary.to_csv(out/'kotlov_feature_coverage_summary.csv',index=False)
    details.to_csv(out/'kotlov_signature_features.csv',index=False)
    for restricted,filename in [(False,'kotlov_reference_ssgsea.gmt'),(True,'kotlov_blym777_ssgsea.gmt')]:
        lines=[]
        for entry in payload['kotlov_signatures']:
            if entry['method']!='ssGSEA':continue
            genes=entry['genes']
            if restricted:genes=sorted(set(genes)&set(payload['transcriptomic_panel']['genes']))
            lines.append('\t'.join([entry['signature'],'Kotlov author ssGSEA signature']+genes))
        (out/filename).write_text('\n'.join(lines)+'\n')
    details[details.method.eq('PROGENy')].to_csv(out/'kotlov_progeny_coefficients.csv',index=False)


def plot_kotlov_signature_coverage(coverage, summary):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(3,1,figsize=(10,13),layout='constrained',
                           gridspec_kw={'height_ratios':[2,8,2]})
    groups=[(summary,'group','Unique gene unions (shared genes counted once)'),
            (coverage[coverage.method.eq('ssGSEA')],'signature','22 ssGSEA signatures'),
            (coverage[coverage.method.eq('PROGENy')],'signature','3 PROGENy pathways')]
    for ax,(q,label,title) in zip(axes,groups):
        q=q.reset_index(drop=True)
        ax.barh(q.index,100,color='#DFDFDF',height=.66)
        ax.barh(q.index,q.coverage_pct,color='#318582',height=.66)
        for i,r in enumerate(q.itertuples()):
            ax.text(102,i,f'{r.on_panel}/{r.requested} ({r.coverage_pct:.0f}%)',va='center',fontsize=9)
        ax.set_yticks(q.index,q[label].str.replace('_',' ',regex=False));ax.invert_yaxis()
        ax.set_xlim(0,135);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Reference genes on BLYM777 (%)')
        ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
        ax.spines[['top','right','left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Kotlov: BLYM777 coverage of the 25 functional inputs\nDeclared panel targets versus author signature/pathway genes',fontsize=13)
    fig.legend(handles=[Patch(color='#318582',label='On panel'),Patch(color='#DFDFDF',label='Missing from panel')],
               loc='outside lower center',ncol=2,frameon=False)
    return fig
