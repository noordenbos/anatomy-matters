"""Evaluate panel-restricted DLBclass predictions against fixed reference labels."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dlbcl.classifier_colors import classifier_color

LABELS = ['C1', 'C2', 'C3', 'C4', 'C5']
PANELS = ['LYMFv1', 'BLYMF200', 'BLYMFv2']


def proportion_interval(numerator, denominator):
    """Binomial proportion and 95% Wilson interval; undefined if denominator=0."""
    if denominator == 0:
        return np.nan, np.nan, np.nan
    if not 0 <= numerator <= denominator:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = numerator / denominator
    scale = 1 + z*z/denominator
    center = (p + z*z/(2*denominator))/scale
    half = z*math.sqrt(p*(1-p)/denominator + z*z/(4*denominator**2))/scale
    return p, max(0., center-half), min(1., center+half)


def evaluate(cases):
    """Return per-class metrics, overall metrics and all confusion counts."""
    if cases.duplicated(['configuration', 'patient_id']).any():
        raise ValueError('Duplicate case within configuration')
    if not cases.reference_label.isin(LABELS).all() or not cases.DLBclass.isin(LABELS).all():
        raise ValueError('Unknown reference or predicted label')
    full = cases.loc[cases.configuration.eq('Full_features')].set_index('patient_id')
    if len(full) == 0:
        raise ValueError('Full-feature baseline is required')
    metrics, overall, confusions = [], [], []
    for config, group in cases.groupby('configuration', sort=False):
        g = group.set_index('patient_id')
        if set(g.index) != set(full.index):
            raise ValueError('Configuration cases differ from the baseline')
        g = g.loc[full.index]
        if not g.reference_label.eq(full.reference_label).all():
            raise ValueError('Reference labels differ between configurations')
        info = {k: g.iloc[0][k] for k in ['configuration','panel','fish','mean_source']}
        correct = int(g.reference_label.eq(g.DLBclass).sum())
        accuracy, lo, hi = proportion_interval(correct, len(g))
        agreement = int(g.DLBclass.eq(full.DLBclass).sum())
        overall.append({**info, 'n_cases':len(g), 'n_correct':correct, 'accuracy':accuracy,
                        'accuracy_low':lo, 'accuracy_high':hi, 'agreement_with_full':agreement/len(g),
                        'n_changed_from_full':len(g)-agreement,
                        'C2_reference_n':int(g.reference_label.eq('C2').sum()),
                        'C2_predicted_n':int(g.DLBclass.eq('C2').sum())})
        for label in LABELS:
            truth, pred = g.reference_label.eq(label), g.DLBclass.eq(label)
            tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
            sensitivity, sl, sh = proportion_interval(tp, tp+fn)
            ppv, pl, ph = proportion_interval(tp, tp+fp)
            metrics.append({**info, 'subtype':label,'reference_n':int(truth.sum()),
                            'predicted_n':int(pred.sum()),'tp':tp,'fp':fp,'fn':fn,
                            'sensitivity':sensitivity,'sensitivity_low':sl,'sensitivity_high':sh,
                            'ppv':ppv,'ppv_low':pl,'ppv_high':ph})
            for predicted in LABELS:
                n = int((truth & g.DLBclass.eq(predicted)).sum())
                confusions.append({**info,'reference_label':label,'predicted_label':predicted,
                                   'count':n,'row_fraction':n/int(truth.sum()) if truth.any() else np.nan})
    return pd.DataFrame(metrics), pd.DataFrame(overall), pd.DataFrame(confusions)


def evaluate_against_full(cases):
    """Use the full-feature model's assignments as the comparator, not published labels."""
    baseline=cases.loc[cases.configuration.eq('Full_features')].set_index('patient_id').DLBclass
    if not baseline.index.is_unique:raise ValueError('Duplicate full-feature case')
    compared=cases.copy()
    compared['reference_label']=compared.patient_id.map(baseline)
    metrics,overall,confusions=evaluate(compared)
    metrics=metrics.rename(columns={'reference_n':'full_feature_n','sensitivity':'recall',
                                   'sensitivity_low':'recall_low','sensitivity_high':'recall_high',
                                   'ppv':'precision','ppv_low':'precision_low','ppv_high':'precision_high'})
    overall=overall.rename(columns={'accuracy':'agreement','accuracy_low':'agreement_low',
                                   'accuracy_high':'agreement_high','n_correct':'n_matching',
                                   'C2_reference_n':'C2_full_feature_n'})
    return metrics,overall,confusions


def plot_full_agreement(metrics):
    plotted=metrics.rename(columns={'full_feature_n':'reference_n','recall':'sensitivity',
                                   'recall_low':'sensitivity_low','recall_high':'sensitivity_high',
                                   'precision':'ppv','precision_low':'ppv_low','precision_high':'ppv_high'})
    fig=plot_metrics(plotted,panels=['BLYMF200','BLYMFv2'],show_baseline=False)
    fig._suptitle.set_text('How well do study panels reproduce full-feature DLBclass calls?')
    fig.texts[1].set_text('149 test cases; full-feature model assignments are the reference. Perfect agreement is 100%.')
    fig.texts[2].set_text('Unavailable features are imputed using the original 699-case reference means; all panel configurations omit copy-number data.\n'
                         'Recall: fraction of full-feature subtype calls retained. Precision: fraction of panel subtype calls matching full features. Error bars: 95% Wilson intervals.')
    fig.axes[0].set_ylabel('Recall vs full-feature calls')
    fig.axes[len(fig.axes)//2].set_ylabel('Precision vs full-feature calls')
    return fig


def compare_to_full(metrics, overall):
    """Express absolute metrics as differences and ratios to the same full-feature baseline."""
    base=metrics.loc[metrics.configuration.eq('Full_features'),['subtype','sensitivity','ppv']]
    relative=metrics.merge(base.rename(columns={m:m+'_full' for m in ['sensitivity','ppv']}),
                           on='subtype',validate='many_to_one')
    for metric in ['sensitivity','ppv']:
        relative[metric+'_change_pp']=100*(relative[metric]-relative[metric+'_full'])
        denominator=relative[metric+'_full'].where(relative[metric+'_full'].ne(0))
        relative[metric+'_percent_of_full']=100*relative[metric]/denominator
    baseline=overall.loc[overall.configuration.eq('Full_features'),'accuracy']
    if len(baseline)!=1:raise ValueError('Exactly one full-feature accuracy is required')
    overall_relative=overall.copy()
    overall_relative['accuracy_full']=baseline.iloc[0]
    overall_relative['accuracy_change_pp']=100*(overall_relative.accuracy-baseline.iloc[0])
    overall_relative['accuracy_percent_of_full']=100*overall_relative.accuracy/baseline.iloc[0] if baseline.iloc[0]!=0 else np.nan
    return relative,overall_relative


def plot_relative_to_full(relative, mean_source='reference699'):
    fig,axes=plt.subplots(2,3,figsize=(14,8.2),sharey=True)
    x=np.arange(5)
    for col,panel in enumerate(PANELS):
        for row,metric in enumerate(['sensitivity','ppv']):
            ax=axes[row,col]
            for i,(fish,color,title) in enumerate([
                ('unavailable','#8DB9C9','Panel mutations'),
                ('available','#274D70','Panel mutations + FISH')]):
                data=relative.loc[relative.panel.eq(panel)&relative.fish.eq(fish)&relative.mean_source.eq(mean_source)].set_index('subtype').loc[LABELS]
                values=data[metric+'_percent_of_full']
                bars=ax.bar(x+(i-.5)*.34,values,.32,color=color,label=title,zorder=3)
                ax.bar_label(bars,labels=[f'{v:.0f}' if np.isfinite(v) else 'NA' for v in values],padding=3,fontsize=8)
            ax.axhline(100,color='#111111',ls='--',lw=1.2,label='Full-feature baseline = 100%',zorder=2)
            ax.set_xticks(x,LABELS)
            for tick,label in zip(ax.get_xticklabels(),LABELS):
                tick.set_color(classifier_color('DLBclass',label));tick.set_fontweight('bold')
            ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100,125],['0%','25%','50%','75%','100%','125%'])
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_title(panel,fontsize=12,fontweight='bold',pad=12)
            if col==0:ax.set_ylabel(('Sensitivity' if row==0 else 'Positive predictive value')+'\n(% of full-feature value)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.91),ncol=3,frameon=False)
    fig.suptitle('DLBclass performance relative to full features',x=.07,ha='left',y=.98,fontsize=15)
    fig.text(.07,.928,'Each subtype is normalized to its own full-feature sensitivity or PPV; 100% means equal performance.',fontsize=10)
    fig.text(.07,.035,'149 test cases; original 699-case imputation means. Bars show point estimates: 100 × panel metric / full-feature metric.\n'
             'Metrics remain evaluated against published reference labels. This ratio is not case-level agreement; values above 100% indicate an increase.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,top=.80,bottom=.13,hspace=.30,wspace=.14)
    return fig


def plot_metrics(metrics, mean_source='reference699', panels=None, show_baseline=True):
    panels=PANELS if panels is None else panels
    fig, axes = plt.subplots(2, len(panels), figsize=(14 if len(panels)==3 else 11, 8.2), sharey=True, squeeze=False)
    baseline = metrics.loc[metrics.configuration.eq('Full_features')].set_index('subtype').loc[LABELS]
    x = np.arange(5)
    for col, panel in enumerate(panels):
        for row, metric in enumerate(['sensitivity','ppv']):
            ax = axes[row,col]
            for i, (fish, color, title) in enumerate([
                    ('unavailable','#8DB9C9','Panel mutations'),
                    ('available','#274D70','Panel mutations + FISH')]):
                d=metrics.loc[metrics.panel.eq(panel)&metrics.mean_source.eq(mean_source)&metrics.fish.eq(fish)].set_index('subtype').loc[LABELS]
                values=d[metric].to_numpy()
                err=np.vstack([values-d[metric+'_low'].to_numpy(),d[metric+'_high'].to_numpy()-values])
                ax.bar(x+(i-.5)*.34, values, .32, color=color, label=title,
                       yerr=err, error_kw={'lw':.8,'capsize':2,'ecolor':'#596168'}, zorder=3)
            if show_baseline:
                ax.plot(x,baseline[metric],color='#111111',ls='--',marker='o',ms=3,lw=1.2,label='Full-feature baseline',zorder=4)
            ax.set_xticks(x,LABELS)
            for tick,label in zip(ax.get_xticklabels(),LABELS):
                tick.set_color(classifier_color('DLBclass',label)); tick.set_fontweight('bold')
            ax.set_ylim(0,1.12);ax.set_yticks(np.arange(0,1.01,.2),[f'{v:.0%}' for v in np.arange(0,1.01,.2)])
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_title(panel,fontsize=12,fontweight='bold',pad=12)
            if col==0:ax.set_ylabel('Sensitivity' if row==0 else 'Positive predictive value')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.91),ncol=3,frameon=False)
    fig.suptitle('DLBclass performance after restricting features to study panels',x=.07,ha='left',y=.98,fontsize=15)
    fig.text(.07,.928,'149 author-designated test cases; comparison with published C1–C5 reference labels.',fontsize=10)
    mean_note='Original 699-case reference means' if mean_source=='reference699' else 'Training-only means (550 cases)'
    fig.text(.07,.035,f'{mean_note} impute unavailable features. All panel configurations omit regional copy-number data.\n'
             'FISH retains MYC, BCL2 and BCL6 rearrangements. Error bars: 95% Wilson intervals. Reference counts: C1 26, C2 44, C3 22, C4 22, C5 35.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.80,bottom=.13,hspace=.30,wspace=.14)
    return fig


def plot_confusions(confusions, mean_source='reference699'):
    fig,axes=plt.subplots(2,3,figsize=(12,8.5))
    for col,panel in enumerate(PANELS):
        for row,fish in enumerate(['unavailable','available']):
            ax=axes[row,col]
            data=confusions.loc[confusions.panel.eq(panel)&confusions.fish.eq(fish)&confusions.mean_source.eq(mean_source)]
            counts=data.pivot(index='reference_label',columns='predicted_label',values='count').loc[LABELS,LABELS]
            fractions=counts.div(counts.sum(axis=1),axis=0)
            ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
            for i in range(5):
                for j in range(5):
                    ax.text(j,i,f'{counts.iloc[i,j]}\n{fractions.iloc[i,j]:.0%}',ha='center',va='center',
                            fontsize=8,color='white' if fractions.iloc[i,j]>.55 else '#222222')
            ax.set_xticks(range(5),LABELS);ax.set_yticks(range(5),LABELS)
            ax.set_title(panel + (' · mutations' if row==0 else ' · mutations + FISH'),fontsize=10)
            ax.set_xlabel('Predicted subtype')
            if col==0:ax.set_ylabel('Reference subtype')
    fig.suptitle('Where panel-restricted assignments differ from reference labels',fontsize=14,y=.98)
    fig.text(.07,.025,'Cells show case counts and row percentages. All configurations impute unavailable regional copy-number features.\n'
             '149 test cases; primary analysis uses the original 699-case imputation reference.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,top=.91,bottom=.12,hspace=.34,wspace=.25)
    return fig


def analyze(data_dir, output_dir):
    data_dir,output_dir=Path(data_dir),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    design=pd.read_csv(data_dir/'case_design.csv')
    predictions=pd.read_csv(data_dir/'predictions.tsv',sep='\t')
    cases=design.merge(predictions,on='run_id',validate='one_to_one')
    if len(cases)!=len(design) or len(cases)!=len(predictions):
        raise ValueError('Missing or unmatched predictions')
    np.testing.assert_allclose(cases[LABELS].sum(axis=1),1,atol=1e-6)
    if not cases.DLBclass.eq(cases[LABELS].idxmax(axis=1)).all():
        raise ValueError('Assigned label does not match winning score')
    metrics,overall,confusions=evaluate(cases)
    full_metrics,full_overall,full_confusions=evaluate_against_full(cases)
    full_metrics.to_csv(output_dir/'full_feature_subtype_agreement.csv',index=False)
    full_overall.to_csv(output_dir/'full_feature_overall_agreement.csv',index=False)
    full_confusions.to_csv(output_dir/'full_feature_confusion_counts.csv',index=False)
    relative,overall_relative=compare_to_full(metrics,overall)
    relative.to_csv(output_dir/'baseline_relative_subtype_metrics.csv',index=False)
    overall_relative.to_csv(output_dir/'baseline_relative_overall_metrics.csv',index=False)
    for name,frame in [('case_predictions',cases),('subtype_metrics',metrics),('overall_metrics',overall),('confusion_counts',confusions)]:
        frame.to_csv(output_dir/f'{name}.csv',index=False)
    primary=metrics.loc[metrics.mean_source.eq('reference699')]
    secondary=metrics.loc[metrics.mean_source.eq('training550')]
    comparison=primary.merge(secondary,on=['panel','fish','subtype'],suffixes=('_reference699','_training550'),validate='one_to_one')
    for metric in ['sensitivity','ppv']:
        comparison[f'{metric}_change_training_minus_reference']=comparison[f'{metric}_training550']-comparison[f'{metric}_reference699']
    comparison.to_csv(output_dir/'imputation_reference_sensitivity.csv',index=False)
    for name,figure in [('dlbclass_panel_performance',plot_metrics(metrics)),
                        ('dlbclass_relative_to_full_features',plot_full_agreement(full_metrics)),
                        ('dlbclass_panel_confusions',plot_confusions(confusions)),
                        ('dlbclass_panel_performance_training_means',plot_metrics(metrics,'training550'))]:
        for ext in ['png','pdf','svg']:figure.savefig(output_dir/f'{name}.{ext}',dpi=200)
        plt.close(figure)
    return cases,metrics,overall,confusions
