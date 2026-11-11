"""Oncoprint annotations and paired genetic summaries from canonical case tables."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .case_assignments import discovery_assignments, validation_assignments, patient_frame
from .classifier_colors import reference_palette
from .genomic_profiling import apply_genomic_profiling_exclusions

SPECIAL = {'Unavailable':'#D9D9D9', 'Unknown assignment':'#A6A6A6', 'Unclassified':'#737373'}


def display_assignment(value, classifier):
    if pd.isna(value) or str(value).strip().lower() in {'','nan','none','<na>','na'}:
        return 'Unavailable'
    label = str(value).strip()
    if label.lower() == 'unknown': return 'Unknown assignment'
    if label.lower() in {'unclassified','unassigned'} or (classifier=='DLBclass' and label=='0'):
        return 'Unclassified'
    colors = reference_palette(classifier)
    components = label.split('/')
    if not all(c in colors for c in components):
        raise ValueError(f'Unrecognized {classifier} assignment: {label}')
    return label


def component_colors(label, classifier):
    if label in SPECIAL:return [SPECIAL[label]]
    colors=reference_palette(classifier)
    return [colors[c] for c in label.split('/')]


def annotation_table(adata, patient_ids):
    """Strict ID join in caller order; retain raw labels beside display labels."""
    ids=pd.Index([str(p) for p in patient_ids],name='patient_id')
    if ids.has_duplicates:raise ValueError('Duplicate oncoprint patient IDs')
    cc=discovery_assignments(adata)
    if not ids.isin(cc.index).all():raise ValueError('Oncoprint patient absent from canonical discovery table')
    out=cc.reindex(ids)[['Lymphgen','DLBclass']].copy()
    out=out.rename(columns={'Lymphgen':'LymphGen_raw','DLBclass':'DLBclass_raw'})
    out['LymphGen']=out.LymphGen_raw.map(lambda x:display_assignment(x,'Lymphgen'))
    out['DLBclass']=out.DLBclass_raw.map(lambda x:display_assignment(x,'DLBclass'))
    return out


def export_oncoprint_annotations(adata, patient_ids, output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    frame=annotation_table(adata,patient_ids)
    annotations=output_dir/'oncoprint_classifier_annotations.csv';frame.to_csv(annotations)
    rows=[]
    for col,key in [('LymphGen','Lymphgen'),('DLBclass','DLBclass')]:
        for label in frame[col].unique():
            for i,color in enumerate(component_colors(label,key),1):
                rows.append({'classifier':col,'label':label,'component':i,'color':color})
    palette=output_dir/'oncoprint_classifier_colors.csv'
    pd.DataFrame(rows).to_csv(palette,index=False)
    return {'annotations':annotations,'classifier_colors':palette}


def cohort_pair_table(adata, cohort):
    if cohort=='discovery':
        cc=discovery_assignments(adata)
        events=pd.DataFrame(apply_genomic_profiling_exclusions(adata.uns['genomic_profiling'])['events'])
        available=set(events.patient_id.astype(str))
        availability_definition='At least one alteration row in embedded discovery genomic_profiling/events after existing exclusions; not independent assay-success evidence.'
    elif cohort=='validation':
        vc=adata.uns['validation_cohort'];cc=validation_assignments(vc)
        ngs=pd.DataFrame(vc['ngs_data']);available=set(ngs.patient_alias.dropna().astype(str))
        availability_definition='At least one row in embedded validation ngs_data; not necessarily a pathogenic variant or independent assay-success evidence.'
    else:raise ValueError('cohort must be discovery or validation')
    out=cc[['Lymphgen','DLBclass']].copy().rename(columns={'Lymphgen':'LymphGen_raw','DLBclass':'DLBclass_raw'})
    out.index.name='patient_id'
    out['has_embedded_genomic_data']=out.index.isin(available)
    for col,key in [('LymphGen','Lymphgen'),('DLBclass','DLBclass')]:
        out[col]=out[col+'_raw'].map(lambda x:display_assignment(x,key))
    out['both_outputs_available']=~out[['LymphGen','DLBclass']].isin(list(SPECIAL)).any(axis=1)
    out['lymphgen_composite']=out.LymphGen.str.contains('/',regex=False)
    out['cohort']=cohort
    info={'cohort':cohort,'n_all_cases':len(out),'n_with_embedded_genomic_data':int(out.has_embedded_genomic_data.sum()),
          'n_with_both_outputs_and_embedded_genomic_data':int((out.has_embedded_genomic_data & out.both_outputs_available).sum()),
          'n_with_output_but_no_embedded_genomic_data':int((~out.has_embedded_genomic_data & (~out.LymphGen.isin(list(SPECIAL))|~out.DLBclass.isin(list(SPECIAL)))).sum()),
          'availability_definition':availability_definition,
          'interpretation':'Descriptive comparison of supplied canonical labels, not classifier accuracy or biological absence of C2/A53. Other and composite LymphGen calls are retained.'}
    return out,info


def pair_counts(frame):
    counts=pd.crosstab(frame.LymphGen,frame.DLBclass,dropna=False)
    yorder=['MCD','BN2','N1','EZB','ST2','A53']
    yorder += sorted(x for x in frame.LymphGen.unique() if '/' in x)
    yorder += ['Other',*SPECIAL]
    yorder=list(dict.fromkeys(x for x in yorder if x in set(frame.LymphGen)))
    xorder=['C1','C2','C3','C4','C5']+[x for x in SPECIAL if x in set(frame.DLBclass)]
    return counts.reindex(index=yorder,columns=xorder,fill_value=0)


def plot_pair_counts(counts, info, output_stem):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams['svg.fonttype']='none'
    n=int(counts.to_numpy().sum())
    fig,ax=plt.subplots(figsize=(7.8,max(4.8,len(counts)*.46+2.2)))
    ax.imshow(counts.to_numpy(),cmap='Blues',vmin=0,vmax=max(1,counts.to_numpy().max()),aspect='auto')
    ax.set_xticks(range(len(counts.columns)),counts.columns,rotation=35,ha='right')
    ax.set_yticks(range(len(counts)),counts.index)
    ax.tick_params(axis='y',pad=40)
    for i,label in enumerate(counts.index):
        colors=component_colors(label,'Lymphgen')
        for k,color in enumerate(colors):
            ax.add_patch(Rectangle((-.06+k*.035/len(colors),i-.34),.035/len(colors),.68,
                transform=ax.get_yaxis_transform(),facecolor=color,edgecolor='white',linewidth=.3,clip_on=False))
    maximum=counts.to_numpy().max()
    for i in range(len(counts)):
        for j in range(len(counts.columns)):
            v=int(counts.iloc[i,j]);ax.text(j,i,str(v),ha='center',va='center',color='white' if v>maximum*.55 else '#202020',fontsize=11)
    ax.set_xlabel('DLBclass');ax.set_ylabel('LymphGen',labelpad=8)
    ax.set_title(f"{info['cohort'].capitalize()}: paired genetic classifier assignments\nEmbedded genomic data: n={n}; both outputs available: n={info['n_with_both_outputs_and_embedded_genomic_data']}",fontsize=11,pad=15)
    fig.text(.02,.015,'Counts retain residual and missing calls. Imputed C2 is not measured CN evidence; A53 requires CN input.',fontsize=7)
    fig.tight_layout(rect=[0,.055,1,1])
    output_stem=Path(output_stem)
    for suffix in ['.svg','.png','.pdf']:fig.savefig(output_stem.with_suffix(suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


def export_paired_summaries(adata, output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for cohort in ['discovery','validation']:
        frame,info=cohort_pair_table(adata,cohort)
        frame.to_csv(output_dir/f'{cohort}_all_case_assignments.csv')
        profiled=frame.loc[frame.has_embedded_genomic_data]
        counts=pair_counts(profiled);all_counts=pair_counts(frame)
        counts.to_csv(output_dir/f'{cohort}_paired_counts_genomic_data.csv')
        all_counts.to_csv(output_dir/f'{cohort}_paired_counts_all_cases.csv')
        counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).mul(100).to_csv(output_dir/f'{cohort}_paired_row_percentages.csv')
        info['n_LymphGen_composites_with_genomic_data']=int(profiled.lymphgen_composite.sum())
        info['genomic_subset_LymphGen_counts']=profiled.LymphGen.value_counts().to_dict()
        info['genomic_subset_DLBclass_counts']=profiled.DLBclass.value_counts().to_dict()
        assert int(counts.to_numpy().sum())==len(profiled)
        plot_pair_counts(counts,info,output_dir/f'{cohort}_lymphgen_dlbclass')
        summary.append(info)
    (output_dir/'paired_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
