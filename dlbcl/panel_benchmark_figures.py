"""Matched displays of classifier-specific reference benchmarks and transitions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dlbcl.classifier_colors import classifier_color

PANELS=['BLYMF200','BLYMFv2']
SUBTYPES={'LymphGen v2':['BN2','EZB','MCD','N1','ST2'], 'DLBclass':['C1','C2','C3','C4','C5']}
COLORS={'unavailable':'#8DB9C9','available':'#274D70'}


def plot_aligned(metrics):
    fig,axes=plt.subplots(4,2,figsize=(12,12.5),sharey=True)
    for block,classifier in enumerate(SUBTYPES):
        labels=SUBTYPES[classifier];x=np.arange(len(labels))
        for col,panel in enumerate(PANELS):
            for m,metric in enumerate(['recall','precision']):
                ax=axes[block*2+m,col]
                for i,fish in enumerate(['unavailable','available']):
                    d=metrics.loc[metrics.classifier.eq(classifier)&metrics.panel.eq(panel)&metrics.fish.eq(fish)].set_index('subtype').loc[labels]
                    y=d[metric].to_numpy()
                    errors=np.vstack([y-d[metric+'_low'],d[metric+'_high']-y])
                    ax.bar(x+(i-.5)*.34,y,.32,color=COLORS[fish],label='Panel mutations' if i==0 else 'Panel mutations + FISH',
                           yerr=errors,error_kw={'lw':.8,'capsize':2,'ecolor':'#555555'},zorder=3)
                ax.set_xticks(x,labels);ax.set_ylim(0,1.08)
                ax.set_yticks(np.arange(0,1.01,.25),['0%','25%','50%','75%','100%'])
                for tick,label in zip(ax.get_xticklabels(),labels):
                    tick.set_color(classifier_color('LymphGen' if block==0 else 'DLBclass',label));tick.set_fontweight('bold')
                ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
                if col==0:ax.set_ylabel(f'{classifier}\n{metric.title()} vs full-feature calls',fontsize=10)
                if m==0:ax.set_title(f'{panel} · {classifier}',loc='left',fontsize=11,fontweight='bold',pad=10)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper left',bbox_to_anchor=(.085,.925),ncol=2,frameon=False)
    fig.suptitle('Reproduction of full-feature calls under study-panel restrictions',x=.09,ha='left',y=.98,fontsize=14)
    fig.text(.09,.947,'LymphGen v2: 574 internal reference cases. DLBclass: 149 test cases. Reference cohorts differ.',fontsize=10)
    fig.text(.09,.025,'Recall: full-feature subtype calls retained. Precision: panel subtype calls matching full features. Error bars: 95% Wilson intervals.\n'
             'Both omit copy-number data. LymphGen uses BCL2/BCL6 FISH; DLBclass uses MYC/BCL2/BCL6 FISH and reference-mean imputation.\n'
             'LymphGen A53 is excluded; composite memberships contribute to each constituent subtype. These are not head-to-head accuracy estimates.',fontsize=8.5)
    fig.subplots_adjust(left=.09,right=.985,top=.87,bottom=.11,hspace=.43,wspace=.14)
    return fig


def summarize_transitions(pairs):
    """Keep exact composite labels in the source table; aggregate only for display."""
    keys=['classifier','panel','fish','full_label','panel_label']
    exact=pairs.groupby(keys,sort=False).size().rename('count').reset_index()
    exact['row_fraction']=exact['count']/exact.groupby(keys[:-1])['count'].transform('sum')
    grouped=pairs.copy()
    for col in ['full_label','panel_label']:
        grouped[col]=grouped[col].map(lambda x:'Composite' if '/' in x else x)
    grouped=grouped.groupby(keys,sort=False).size().rename('count').reset_index()
    grouped['row_fraction']=grouped['count']/grouped.groupby(keys[:-1])['count'].transform('sum')
    return exact,grouped


def plot_directions(grouped,classifier):
    labels=SUBTYPES[classifier]+(['Composite','Other'] if classifier=='LymphGen v2' else [])
    fig,axes=plt.subplots(2,2,figsize=(11,10))
    for col,panel in enumerate(PANELS):
        for row,fish in enumerate(['unavailable','available']):
            ax=axes[row,col]
            d=grouped.loc[grouped.classifier.eq(classifier)&grouped.panel.eq(panel)&grouped.fish.eq(fish)]
            counts=d.pivot(index='full_label',columns='panel_label',values='count').reindex(index=labels,columns=labels).fillna(0).astype(int)
            fractions=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0)
            ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value=fractions.iloc[i,j]
                    label=f'{counts.iloc[i,j]}\n{value:.0%}' if np.isfinite(value) else '–'
                    ax.text(j,i,label,ha='center',va='center',fontsize=7 if len(labels)>5 else 8,
                            color='white' if value>.55 else '#222222')
            ax.set_xticks(range(len(labels)),labels,rotation=40 if len(labels)>5 else 0,ha='right' if len(labels)>5 else 'center')
            ax.set_yticks(range(len(labels)),labels)
            ax.set_title(panel+(' · mutations' if row==0 else ' · mutations + FISH'),fontsize=10)
            ax.set_xlabel('Panel-restricted call')
            if col==0:ax.set_ylabel('Full-feature call')
    fig.suptitle(f'{classifier}: direction of changes from full-feature calls',fontsize=14,y=.98)
    footer='Cells show case counts and percentages within each full-feature category. Changes from full features do not necessarily represent errors.'
    if classifier=='LymphGen v2':footer+='\nComposite calls are pooled for display; Composite → Composite can hide constituent changes. Exact-label transitions are exported. A53 is excluded.'
    fig.text(.08,.025,footer,fontsize=8.5)
    fig.subplots_adjust(left=.09,right=.97,top=.92,bottom=.20,hspace=.48,wspace=.28)
    return fig


def export_aligned(data,output):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(data/'aligned_metrics.csv')
    pairs=pd.read_csv(data/'reference_pairs.csv')
    if pairs.duplicated(['classifier','panel','fish','reference_id']).any():
        raise ValueError('Duplicate reference case within configuration')
    exact,grouped=summarize_transitions(pairs)
    exact.to_csv(output/'exact_label_transitions.csv',index=False)
    grouped.to_csv(output/'display_label_transitions.csv',index=False)
    for name,fig in [('aligned_classifier_panel_performance',plot_aligned(metrics)),
                     ('lymphgen_directional_transitions',plot_directions(grouped,'LymphGen v2')),
                     ('dlbclass_directional_transitions',plot_directions(grouped,'DLBclass'))]:
        for ext in ['png','pdf','svg']:fig.savefig(output/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    return metrics,pairs,exact


def export_aligned_public(adata, output):
    """Recreate the aligned display from released external counts, without local CSVs."""
    from dlbcl.public_panel_confidence import metrics_from_counts
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=metrics_from_counts(adata.uns['genomic']['counts'],full='reference_n',panel='predicted_n')
    exact=adata.uns['genomic']['transitions'].copy()
    keys=['classifier','panel','fish','full_label','panel_label']
    if exact.duplicated(keys).any() or exact['count'].lt(0).any():
        raise ValueError('Invalid external-reference transition counts')
    exact['row_fraction']=exact['count']/exact.groupby(keys[:-1])['count'].transform('sum')
    grouped=exact.copy()
    for col in ['full_label','panel_label']:
        grouped[col]=grouped[col].map(lambda x:'Composite' if '/' in x else x)
    grouped=grouped.groupby(keys,sort=False)['count'].sum().reset_index()
    grouped['row_fraction']=grouped['count']/grouped.groupby(keys[:-1])['count'].transform('sum')
    metrics.to_csv(output/'aligned_metrics.csv',index=False)
    exact.to_csv(output/'exact_label_transitions.csv',index=False)
    grouped.to_csv(output/'display_label_transitions.csv',index=False)
    for name,fig in [('aligned_classifier_panel_performance',plot_aligned(metrics)),
                     ('lymphgen_directional_transitions',plot_directions(grouped,'LymphGen v2')),
                     ('dlbclass_directional_transitions',plot_directions(grouped,'DLBclass'))]:
        for ext in ['png','pdf','svg']:fig.savefig(output/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    return metrics,exact,grouped
