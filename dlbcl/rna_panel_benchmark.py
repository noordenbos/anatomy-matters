"""Paired RNA feature-restriction metrics and publication-ready figures.

Full-input calls are a comparator, not biological ground truth. Unassigned calls
remain explicit; zero-denominator precision/recall is undefined, never zero.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS={'LymphoMAP':['FMAC','LN','TEX','Unassigned'],
        'EcoTyper':['LE'+str(i) for i in range(1,10)]+['Unassigned']}
COHORTS=['Schmitz','Morin','Li']

def pooled_plot_metrics(metrics):
    """Pool Li tumors for display; preserve controls and clinical groups in exports."""
    d=metrics[metrics.group.ne('Control')]
    d=d.groupby(['classifier','cohort','label'],sort=False)[['tp','full_n','panel_n']].sum().reset_index()
    for name,den in [('recall','full_n'),('precision','panel_n')]:
        d[name]=d.tp/d[den].replace(0,np.nan)
        bounds=[wilson(k,n) for k,n in zip(d.tp,d[den])]
        d[name+'_low']=[x[0] for x in bounds];d[name+'_high']=[x[1] for x in bounds]
    return d

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def summarize(pairs):
    required=['classifier','cohort','group','sample_id','full_label','panel_label']
    if pairs[required].isna().any().any():raise ValueError('Missing paired call: do not silently drop failed runs')
    if pairs.duplicated(['classifier','cohort','sample_id']).any():raise ValueError('Duplicate paired sample')
    metrics=[];overall=[];transitions=[]
    for (clf,cohort,group),d in pairs.groupby(['classifier','cohort','group'],sort=False):
        key=dict(classifier=clf,cohort=cohort,group=group)
        unknown=(set(d.full_label)|set(d.panel_label))-set(LABELS[clf])
        if unknown:raise ValueError(f'Unexpected {clf} labels: {unknown}')
        agreement=d.full_label.eq(d.panel_label)
        assigned=d.full_label.ne('Unassigned')
        k=int(agreement.sum());n=len(d);lo,hi=wilson(k,n)
        overall.append(dict(**key,n=n,matching_calls=k,agreement=k/n,agreement_low=lo,agreement_high=hi,
                            full_assigned=int(assigned.sum()),panel_assigned=int(d.panel_label.ne('Unassigned').sum()),
                            full_assigned_retained=int((agreement&assigned).sum()),
                            retention_among_full_assigned=float(agreement[assigned].mean()) if assigned.any() else np.nan))
        for label in LABELS[clf]:
            full=d.full_label.eq(label);panel=d.panel_label.eq(label)
            tp=int((full&panel).sum());nf=int(full.sum());npanel=int(panel.sum())
            row=dict(**key,label=label,tp=tp,full_n=nf,panel_n=npanel)
            for name,den in [('recall',nf),('precision',npanel)]:
                lo,hi=wilson(tp,den)
                row.update({name:tp/den if den else np.nan,name+'_low':lo,name+'_high':hi})
            metrics.append(row)
            for dest in LABELS[clf]:
                count=int((full&d.panel_label.eq(dest)).sum())
                transitions.append(dict(**key,full_label=label,panel_label=dest,count=count,
                                        row_n=nf,row_fraction=count/nf if nf else np.nan))
    return pd.DataFrame(metrics),pd.DataFrame(overall),pd.DataFrame(transitions)

def palettes(config):
    d=json.loads(Path(config).read_text())['classifiers']
    return {'LymphoMAP':d['lymphomap']['reference_colors'],
            'EcoTyper':{**d['ecotyper_ecotypes']['reference_colors'],'Unassigned':'#BBBBBB'}}

def plot_metrics(metrics,colors,cohorts=COHORTS):
    metrics=pooled_plot_metrics(metrics)
    fig,axs=plt.subplots(4,len(cohorts),figsize=(max(11,5.3*len(cohorts)),12),sharey=True,squeeze=False)
    for block,clf in enumerate(LABELS):
        labels=LABELS[clf]
        for col,cohort in enumerate(cohorts):
            d=metrics[(metrics.classifier==clf)&(metrics.cohort==cohort)].set_index('label').reindex(labels)
            for r,metric in enumerate(['recall','precision']):
                ax=axs[2*block+r,col];y=d[metric].to_numpy();x=np.arange(len(labels))
                good=np.isfinite(y)
                err=np.vstack([y-d[metric+'_low'].to_numpy(),d[metric+'_high'].to_numpy()-y])
                ax.bar(x[good],y[good],color='#274D70',width=.68,yerr=err[:,good],error_kw={'lw':.8,'capsize':2})
                for j in x[~good]:ax.text(j,.045,'NA',ha='center',fontsize=7,color='#777777')
                ax.set_xticks(x,['UA' if l=='Unassigned' else l for l in labels],fontsize=8)
                ax.set_xlim(-.6,len(labels)-.4)
                for tick,label in zip(ax.get_xticklabels(),labels):
                    tick.set_color(colors[clf][label]);tick.set_fontweight('bold')
                ax.set_ylim(0,1.08);ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
                ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
                if col==0:ax.set_ylabel(f'{clf}\n{metric.title()} vs full-input calls')
                if r==0:
                    n=int(d.full_n.sum())
                    suffix=' · discovery' if clf=='EcoTyper' and cohort=='Schmitz' else ' · training overlap' if clf=='LymphoMAP' and cohort=='Li' else ''
                    ax.set_title(f'{cohort} · n={n}'+suffix,loc='left',fontsize=11)
    platform='Microarray sensitivity' if list(cohorts)==['Chapuy'] else 'RNA-seq'
    fig.suptitle(f'{platform} restricted to BLYM777: full-input call retention',x=.07,ha='left',fontsize=14,y=.985)
    fig.text(.07,.945,'Paired samples and native classifier procedures; 797-target panel definition. Cohorts analysed separately.',fontsize=10)
    fig.text(.07,.025,'Recall: fraction of full-input subtype calls retained. Precision: fraction of panel calls matching that full-input subtype.\n'
             'Error bars: 95% Wilson intervals. UA = Unassigned; NA = zero denominator. Full-input calls are not biological ground truth.\n'
             'Schmitz is EcoTyper discovery data. Li includes 142 tumors with LymphoMAP training overlap; controls are separate.',fontsize=8)
    fig.subplots_adjust(left=.07,right=.985,top=.89,bottom=.13,hspace=.48,wspace=.15)
    return fig

def heatmap(ax,d,clf,title):
    labels=LABELS[clf]
    c=d.pivot(index='full_label',columns='panel_label',values='count').reindex(index=labels,columns=labels).fillna(0)
    f=c.div(c.sum(axis=1).replace(0,np.nan),axis=0)
    ax.imshow(f,cmap='Blues',vmin=0,vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value=f.iloc[i,j];count=int(c.iloc[i,j])
            txt='–' if not np.isfinite(value) else (f'{count}\n{value:.0%}' if count else '')
            ax.text(j,i,txt,ha='center',va='center',fontsize=7 if len(labels)<6 else 6,
                    color='white' if value>.55 else '#333333')
    display=['UA' if l=='Unassigned' else l for l in labels]
    ax.set_xticks(range(len(labels)),display,fontsize=8)
    ax.set_yticks(range(len(labels)),display,fontsize=8)
    ax.set_xlabel('BLYM777-restricted call');ax.set_ylabel('Full-input call')
    ax.set_title(title,fontsize=11)

def plot_directions(transitions,clf,cohorts=COHORTS):
    transitions=transitions[transitions.group.ne('Control')].groupby(['classifier','cohort','full_label','panel_label'],sort=False)['count'].sum().reset_index()
    fig,axs=plt.subplots(1,len(cohorts),figsize=(max(11,5.3*len(cohorts)),6),squeeze=False)
    for ax,cohort in zip(axs.ravel(),cohorts):
        d=transitions[(transitions.classifier==clf)&(transitions.cohort==cohort)]
        heatmap(ax,d,clf,cohort+(' · discovery' if clf=='EcoTyper' and cohort=='Schmitz' else ''))
    fig.suptitle(f'{clf}: direction of call changes after BLYM777 restriction',y=.97,fontsize=15)
    fig.text(.06,.05,'Cells show sample counts and percentages within each full-input category. UA = Unassigned. Empty cells = zero; dash = no reference cases.\n'
             'These are changes relative to full-input calls, not established biological errors.',fontsize=9)
    fig.subplots_adjust(left=.06,right=.99,top=.86,bottom=.19,wspace=.26)
    return fig

def plot_li(transitions):
    groups=['Newly diagnosed','Relapsed/refractory','Control']
    fig,axs=plt.subplots(2,3,figsize=(16,11))
    for i,clf in enumerate(LABELS):
        for j,group in enumerate(groups):
            d=transitions[(transitions.classifier==clf)&(transitions.cohort=='Li')&(transitions.group==group)]
            n=int(d['count'].sum())
            heatmap(axs[i,j],d,clf,f'{clf} · {group} · n={n}')
    fig.suptitle('Li: supplementary sensitivity within LymphoMAP training samples',y=.985,fontsize=15)
    fig.text(.06,.025,'Training overlap precludes interpreting LymphoMAP agreement as independent validation. Controls are displayed separately.\n'
             'Cells show counts and row percentages; UA = Unassigned. Published-label comparisons are exported separately.',fontsize=9)
    fig.subplots_adjust(left=.06,right=.985,top=.91,bottom=.13,hspace=.35,wspace=.26)
    return fig

def export_figures(data,output,config):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_csv(data/'paired_calls.csv',keep_default_na=False)
    metrics,overall,transitions=summarize(pairs)
    colors=palettes(config)
    figures=[('rna_panel_performance',plot_metrics(metrics,colors)),
             ('lymphomap_call_transitions',plot_directions(transitions,'LymphoMAP')),
             ('ecotyper_call_transitions',plot_directions(transitions,'EcoTyper')),
             ('li_training_set_sensitivity',plot_li(transitions))]
    if 'Chapuy' in set(pairs.cohort):
        figures.extend([('chapuy_microarray_performance',plot_metrics(metrics,colors,['Chapuy'])),
                        ('chapuy_lymphomap_transitions',plot_directions(transitions,'LymphoMAP',['Chapuy'])),
                        ('chapuy_ecotyper_transitions',plot_directions(transitions,'EcoTyper',['Chapuy']))])
    for name,fig in figures:
        for ext in ['png','pdf','svg']:fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    return metrics,overall,transitions
