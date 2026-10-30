"""Public external-reference benchmarks. No study AnnData or private-path fallback."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import anndata as ad

SCHEMA = 'external-panel-confidence-v1'
SOURCES = {
    'LymphGen': {
        'paper': 'Wright et al., Cancer Cell 2020, LymphGen',
        'citation': 'https://doi.org/10.1016/j.ccell.2020.03.015',
        'access': 'https://llmpp.nih.gov/lymphgen/index.php',
        'status': 'source_not_bundled',
        'note': 'Author-reference benchmark aggregates are included; the model/reference source object is not redistributed.'},
    'DLBclass': {
        'paper': 'Chapuy et al., Blood 2025, DLBclass',
        'citation': 'https://doi.org/10.1182/blood.2024025652',
        'access': 'https://github.com/getzlab/DLBCL-Classifier',
        'status': 'public_matrix_included',
        'note': 'Public encoded 699-case gene-sample matrix; original sequencing/MAF files are not included.'},
    'Schmitz': {
        'paper': 'Schmitz et al., NEJM 2018, Genetics and Pathogenesis of DLBCL',
        'citation': 'https://doi.org/10.1056/NEJMoa1801445',
        'access': 'https://gdc.cancer.gov/about-data/publications/DLBCL-2018',
        'status': 'source_not_bundled',
        'note': 'Harmonized expression matrix used in this benchmark is not redistributed; use the original access route.'},
    'Morin': {
        'paper': 'Ennishi et al., JCO 2019, Double-Hit Gene Expression Signature (BC Cancer / Morin cohort)',
        'citation': 'https://doi.org/10.1200/JCO.18.01583',
        'access': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8178179/',
        'status': 'source_not_bundled',
        'note': 'Expression matrix used in this benchmark is not redistributed; harmonized external cohort also described by Kotlov et al.'},
    'Li': {
        'paper': 'Li et al., Cancer Cell 2025, Large B cell lymphoma microenvironment archetype profiles',
        'citation': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12417684/',
        'access': 'https://github.com/Green-Lab-MDACC/LymphoMapR/tree/main/data',
        'status': 'public_upstream_not_bundled',
        'note': 'Author training data are publicly distributed by LymphoMapR; the converted benchmark matrix is not bundled.'},
    'Chapuy': {
        'paper': 'Chapuy et al., Nature Medicine 2018, Molecular subtypes of DLBCL',
        'citation': 'https://doi.org/10.1038/s41591-018-0016-8',
        'access': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98588',
        'status': 'public_upstream_not_bundled',
        'note': 'Public microarray deposit; exact preprocessing of the benchmark matrix is unresolved, so it is not bundled.'},
    'GSE178394': {
        'paper': 'Miyawaki and Sugio, GSE178394: NanoString profiling of newly diagnosed DLBCL; GEO lists no linked publication',
        'citation': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178394',
        'access': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178394',
        'status': 'public_matrix_included',
        'note': 'All 20 Lymph2Cx genes and 170 published assay labels included; classifier paper: https://doi.org/10.1182/blood-2013-11-536433'},
}


class SourceDataUnavailableError(FileNotFoundError):
    """A cited external source is absent; never substitute a private local file."""


def source_error(source):
    if source not in SOURCES:
        raise ValueError(f'Unknown external source: {source}')
    info = SOURCES[source]
    if info['status'] == 'source_not_bundled':
        reason = 'The source data cannot be released with this public AnnData bundle.'
    else:
        reason = 'The public source data are not present in this AnnData bundle.'
    return SourceDataUnavailableError(
        f'{source}: {reason} '
        f"Reference: {info['paper']}. {info['citation']} "
        f"Data access: {info['access']}. {info['note']}")


def load_public_adata(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Public external-reference AnnData missing: {path}. '
            'Obtain panel_call_confidence.public.h5ad from the accompanying release. '
            'Do not substitute study AnnData. Source papers: '
            + '; '.join(v['citation'] for v in SOURCES.values()))
    # Check the public schema before deserializing any payload, even if a user
    # points this loader at the study AnnData by mistake.
    with h5py.File(path, 'r') as handle:
        if 'uns/public_panel_confidence_schema' not in handle:
            raise ValueError('Not a public external-reference AnnData. Study AnnData is not accepted.')
        schema = handle['uns/public_panel_confidence_schema'][()]
        if isinstance(schema, bytes):
            schema = schema.decode()
        if schema != SCHEMA:
            raise ValueError('Unsupported public reference schema')
        allowed = {'public_panel_confidence_schema', 'genomic', 'transcriptomic',
                   'source_data', 'sources_json', 'provenance_json', 'palettes_json', 'lymph2cx_model_json'}
        if set(handle['uns']) - allowed:
            raise ValueError('Unexpected payload in public reference AnnData')
    result = ad.read_h5ad(path)
    if result.shape != (0, 0) or result.layers or result.obsm or result.raw is not None:
        raise ValueError('Public benchmark container must not contain study observations, expression or embeddings')
    if set(result.uns.get('source_data', {})) - {'DLBclass','GSE178394','GSE178394_labels'}:
        raise ValueError('Unregistered source matrix in public AnnData')
    for section in ['genomic', 'transcriptomic']:
        for value in result.uns.get(section, {}).values():
            if isinstance(value, pd.DataFrame):
                if set(value.columns) & {'patient_id','sample_id','query_id','Location','OS','DSS'}:
                    raise ValueError('Benchmark sections must contain aggregate external results only')
                if 'cohort' in value and not set(value.cohort) <= {'Schmitz','Morin','Li','Chapuy','GSE178394'}:
                    raise ValueError('Unexpected cohort in external reference results')
    return result


def table(adata, section, name, source):
    value = adata.uns.get(section, {}).get(name)
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise source_error(source)
    return value.copy()


def require_source(adata, source):
    """Read a public source only when explicitly present; raise a cited error otherwise."""
    if source not in SOURCES:
        raise ValueError(f'Unknown external source: {source}')
    value = adata.uns.get('source_data', {}).get(source)
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise source_error(source)
    return value.copy()


def wilson(k, n):
    if not n:
        return np.nan, np.nan
    p = k/n; z = 1.959963984540054; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half, center+half


def metrics_from_counts(frame, full='full_n', panel='panel_n'):
    frame = frame.copy()
    if (frame[['tp',full,panel]] < 0).any().any() or (frame.tp > frame[full]).any() or (frame.tp > frame[panel]).any():
        raise ValueError('Invalid benchmark counts')
    for metric, denominator in [('recall', full), ('precision', panel)]:
        frame[metric] = frame.tp / frame[denominator].replace(0, np.nan)
        bounds = np.array([wilson(k,n) for k,n in zip(frame.tp, frame[denominator])])
        frame[metric+'_low'], frame[metric+'_high'] = bounds[:,0], bounds[:,1]
    return frame


def label_colors(config_path):
    config=json.loads(Path(config_path).read_text())['classifiers']
    mapping={'LymphGen v2':'lymphgen','DLBclass':'dlbclass','LymphoMAP':'lymphomap',
             'EcoTyper':'ecotyper_ecotypes','Lymph2Cx':'lymph2cx_coo'}
    colors={name:dict(config[key]['reference_colors']) for name,key in mapping.items()}
    colors['Lymph2Cx']['UNC']=colors['Lymph2Cx']['Unclassified']
    colors['EcoTyper']['Unassigned']='#BBBBBB'
    colors['Ciavarella']={name:config['ciavarella']['reference_colors'][key]
                          for name,key in [('High','Cluster 1'),('Intermediate','Cluster 2'),('Low','Cluster 3')]}
    return colors


def plot_bars(metrics, classifiers, cohorts, *, genomic=False, title='Panel call retention', label_colors=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2*len(classifiers), len(cohorts), figsize=(max(10,5.2*len(cohorts)), 3.1*2*len(classifiers)),
                             squeeze=False, layout='constrained')
    for block, classifier in enumerate(classifiers):
        for col, cohort in enumerate(cohorts):
            frame = metrics[metrics.classifier.eq(classifier) & metrics['panel' if genomic else 'cohort'].eq(cohort)]
            label_col = 'subtype' if genomic else 'label'
            labels = list(dict.fromkeys(frame[label_col]))
            x = np.arange(len(labels))
            for row, metric in enumerate(['recall','precision']):
                ax = axes[2*block+row,col]
                scenarios = ['unavailable','available'] if genomic else [None]
                for i, scenario in enumerate(scenarios):
                    d = frame[frame.fish.eq(scenario)] if genomic else frame
                    d = d.set_index(label_col).reindex(labels)
                    y = d[metric].to_numpy(); good = np.isfinite(y)
                    pos = x+(i-.5)*.36 if genomic else x
                    errors = np.maximum(0,np.vstack([y-d[metric+'_low'],d[metric+'_high']-y]))
                    ax.bar(pos[good],y[good],.34 if genomic else .65,
                           color=['#8DB9C9','#274D70'][i] if genomic else '#274D70',
                           yerr=errors[:,good],error_kw={'lw':.7,'capsize':2},
                           label=('Mutations only' if i==0 else 'Mutations + FISH') if genomic else None)
                    for j in x[~good]:
                        ax.text(pos[j],.03,'NA',ha='center',fontsize=8)
                ax.set(xticks=x, xticklabels=labels, ylim=(0,1.08),
                       yticks=[0,.25,.5,.75,1], yticklabels=['0%','25%','50%','75%','100%'],
                       title=f'{classifier} · {cohort}' if row==0 else '', ylabel=metric.title())
                ax.tick_params(axis='x', labelsize=8)
                for tick,label in zip(ax.get_xticklabels(),labels):
                    if label_colors and label in label_colors.get(classifier,{}):
                        tick.set_color(label_colors[classifier][label]);tick.set_fontweight('bold')
                ax.spines[['right','top']].set_visible(False)
                if genomic and block==0 and row==0 and col==0:
                    ax.legend(fontsize=8,loc='lower left')
    fig.suptitle(title+'\nRecall and precision against full-input calls; error bars: descriptive 95% Wilson intervals',fontsize=12)
    return fig


def plot_transitions(transitions, classifier, cohorts, *, genomic=False):
    import matplotlib.pyplot as plt
    scenarios = ['unavailable','available'] if genomic else [None]
    fig,axes = plt.subplots(len(scenarios),len(cohorts),figsize=(5.5*len(cohorts),5*len(scenarios)),
                            squeeze=False,layout='constrained')
    for row, scenario in enumerate(scenarios):
        for col, cohort in enumerate(cohorts):
            d = transitions[transitions.classifier.eq(classifier) & transitions['panel' if genomic else 'cohort'].eq(cohort)]
            if genomic: d = d[d.fish.eq(scenario)]
            d = d.groupby(['full_label','panel_label'],sort=False)['count'].sum().reset_index()
            labels = sorted(set(d.full_label)|set(d.panel_label))
            c = d.pivot(index='full_label',columns='panel_label',values='count').reindex(index=labels,columns=labels).fillna(0)
            frac = c.div(c.sum(axis=1).replace(0,np.nan),axis=0)
            ax=axes[row,col]; ax.imshow(frac,cmap='Blues',vmin=0,vmax=1)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value=frac.iloc[i,j]
                    label='–' if not np.isfinite(value) else (f'{int(c.iloc[i,j])}\n{value:.0%}' if c.iloc[i,j] else '')
                    ax.text(j,i,label,ha='center',va='center',fontsize=7 if len(labels)<8 else 6,
                            color='white' if value>.55 else '#222222')
            ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
                   xlabel='Panel-restricted call',ylabel='Full-input call',
                   title=cohort+(' · '+('mutations + FISH' if scenario=='available' else 'mutations only') if genomic else ''))
            ax.tick_params(axis='both',labelsize=8);ax.tick_params(axis='x',rotation=45)
    fig.suptitle(f'{classifier}: direction of changes\nCounts and row percentages; changes are not established biological errors',fontsize=12)
    return fig


def lymph2cx_external(adata, lower, upper, seed=20261016):
    """Recompute the complete fixed-weight model from public NanoString counts."""
    from sklearn.model_selection import StratifiedKFold
    from .lymph2cx_research import score_positive_counts
    from .lymph2cx_proxy_benchmark import fit_thresholds, call_from_thresholds, LABELS
    if not np.isfinite([lower,upper]).all() or lower>=upper:
        raise ValueError('Finite, ordered Lymph2Cx boundaries required')
    counts=require_source(adata,'GSE178394')
    refs=adata.uns['source_data'].get('GSE178394_labels')
    if not isinstance(refs,pd.DataFrame): raise source_error('GSE178394')
    assert counts.index.is_unique and refs.index.is_unique and set(counts.index)==set(refs.index)
    refs=refs.reindex(counts.index).reference_label
    model=json.loads(adata.uns['lymph2cx_model_json'])
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        scores=score_positive_counts(counts,model)
    assert np.isfinite(scores).all()
    frame=pd.DataFrame({'reference_label':refs,'score':scores,
                        'selected_call':call_from_thresholds(scores,lower,upper)})
    frame['held_out_call']='';frame['fold']=0;folds=[]
    for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(frame,refs),1):
        lo,hi=fit_thresholds(scores.iloc[train],refs.iloc[train])
        frame.loc[frame.index[test],'held_out_call']=call_from_thresholds(scores.iloc[test],lo,hi)
        frame.loc[frame.index[test],'fold']=fold
        folds.append(dict(fold=fold,training_n=len(train),test_n=len(test),lower=lo,upper=hi))
    metrics=[]
    for label in LABELS:
        ref=frame.reference_label.eq(label);pred=frame.held_out_call.eq(label)
        metrics.append(dict(label=label,tp=int((ref&pred).sum()),full_n=int(ref.sum()),panel_n=int(pred.sum())))
    return frame,metrics_from_counts(pd.DataFrame(metrics)),pd.DataFrame(folds)


def plot_lymph2cx(frame, metrics, lower, upper, *, colors=None):
    import matplotlib.pyplot as plt
    labels=['GCB','UNC','ABC']
    colors=[colors[label] for label in labels] if colors else ['#4477AA','#999999','#CC6677']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for i,label in enumerate(labels):
        values=frame.loc[frame.reference_label.eq(label),'score']
        axes[0].scatter(i+np.random.default_rng(i).uniform(-.17,.17,len(values)),values,s=15,alpha=.65,color=colors[i])
    for border in [lower,upper]:
        axes[0].axhline(border,linestyle='--',color='#785028')
        axes[0].text(.98,border,f'{border:.2f}',transform=axes[0].get_yaxis_transform(),ha='right',va='bottom')
    axes[0].set(xticks=range(3),xticklabels=labels,ylabel='15-predictor score',title='Selected fixed boundaries')
    matrix=pd.crosstab(frame.reference_label,frame.held_out_call).reindex(index=labels,columns=labels,fill_value=0)
    fractions=matrix.div(matrix.sum(axis=1),axis=0)
    axes[1].imshow(fractions,cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            axes[1].text(j,i,f'{matrix.iloc[i,j]}\n{fractions.iloc[i,j]:.1%}',ha='center',va='center',
                         color='white' if fractions.iloc[i,j]>.55 else 'black')
    axes[1].set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,
                xlabel='Held-out reconstructed call',ylabel='Reported assay call',title='Five-fold held-out agreement')
    m=metrics.set_index('label').loc[labels];x=np.arange(3)
    for offset,key,color in [(-.18,'recall','#4477AA'),(.18,'precision','#EE9944')]:
        bars=axes[2].bar(x+offset,m[key]*100,.36,label=key.title(),color=color)
        axes[2].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    axes[2].set(xticks=x,xticklabels=labels,ylim=(0,112),ylabel='Percent',title='Held-out recall and precision')
    axes[2].legend(loc='lower right')
    fig.suptitle('Lymph2Cx reconstruction · external NanoString GSE178394 (n=170)\nSelected borders affect the left plot only; held-out borders are fitted within training folds',fontsize=12)
    return fig


def plot_feature_coverage(frame, classifiers, *, genomic=False, title='Requested versus delivered features'):
    """Coverage of explicit feature sets, with measured inputs distinct from imputation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    subsets = [frame[frame.classifier.eq(c)].copy() for c in classifiers]
    if any(q.empty for q in subsets):
        raise ValueError('Every requested classifier must have a feature inventory')
    heights = [max(2, .38*len(q)+.8) for q in subsets]
    fig, axes = plt.subplots(len(classifiers), 1, figsize=(11, sum(heights)+1),
                             gridspec_kw={'height_ratios':heights}, squeeze=False, layout='constrained')
    colors = ['#318582', '#E7A34B', '#DDDDDD']
    for ax, classifier, q in zip(axes[:,0], classifiers, subsets):
        q = q.reset_index(drop=True)
        if genomic:
            labels = [f"{r.panel} {'+ FISH' if r.fish=='available' else '(mutation only)'}" for r in q.itertuples()]
        else:
            labels = [r.cohort + (' · reselected' if r.feature_basis=='Panel-reselected genes' else
                      ' · original' if classifier=='LymphoMAP' else '') for r in q.itertuples()]
        left = np.zeros(len(q))
        for column, color in zip(['delivered','on_panel_missing_source','not_on_panel'], colors):
            width = 100*q[column].to_numpy()/q.requested.to_numpy()
            ax.barh(np.arange(len(q)), width, left=left, color=color, height=.65)
            left += width
        for i,r in enumerate(q.itertuples()):
            ax.text(102, i, f'{r.delivered:,} / {r.requested:,}', va='center', fontsize=10)
        ax.set_yticks(np.arange(len(q)),labels)
        ax.invert_yaxis()
        ax.set_xlim(0,129)
        ax.set_xticks([0,25,50,75,100],['0','25','50','75','100'])
        ax.set_title(classifier,loc='left',fontsize=12,fontweight='bold')
        ax.set_xlabel('Requested features (%)')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    legend = [Patch(color=colors[0],label='Delivered'),
              Patch(color=colors[2],label='Unavailable in panel scenario')]
    if not genomic:
        legend.insert(1,Patch(color=colors[1],label='On panel, absent from source matrix'))
    fig.legend(handles=legend,loc='outside lower center',ncol=1 if genomic else 3,frameon=False,fontsize=9)
    fig.suptitle(title+'\nCounts at right: delivered / requested',fontsize=14)
    return fig


def subtype_coverage_table(frame, *, genomic=False, design=False):
    """Compact delivered/requested values, preserving zero structural coverage."""
    frame=frame.copy()
    def ratio(delivered, requested):
        return ['—' if not n else f'{int(k)}/{int(n)}' for k,n in zip(delivered,requested)]
    frame['All features']=ratio(frame.delivered,frame.requested)
    if genomic:
        for key,label in [('mutation','Mutation'),('sv','Rearrangement / SV'),('cn','Copy-number dependent')]:
            frame[label]=ratio(frame[key+'_delivered'],frame[key+'_requested'])
        return frame[['panel','subtype','All features','Mutation','Rearrangement / SV','Copy-number dependent']]
    if design:
        frame['Panel coverage']=ratio(frame.on_panel,frame.requested)
        return frame[['panel','subtype','feature_basis','Panel coverage','not_on_panel']]
    frame['Panel targets']=ratio(frame.on_panel,frame.requested)
    return frame[['cohort','subtype','feature_basis','All features','Panel targets','on_panel_missing_source']]


def plot_subtype_feature_coverage(frame, classifier, *, genomic=False, fish='unavailable', design=False):
    """One classifier: All + every subtype, with distinct feature-set semantics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    q=frame[frame.classifier.eq(classifier)].copy()
    if genomic:
        q=q[q.fish.eq(fish)]
        facets=[p for p in ['BLYMF200','BLYMFv2'] if p in set(q.panel)]
        facet_key='panel'
    elif design:
        facets=list(dict.fromkeys(q.panel))
        facet_key='panel'
    else:
        facets=[c for c in ['Schmitz','Morin','Li','Chapuy','GSE178394'] if c in set(q.cohort)]
        facet_key='cohort'
    if not facets:
        raise ValueError('No feature coverage for '+classifier)
    paired=classifier=='LymphoMAP'
    ncols=min(2,len(facets))
    nrows=int(np.ceil(len(facets)/ncols))
    nlabels=max(q[q[facet_key].eq(f)].subtype.nunique() for f in facets)
    height=max(3.2,nlabels*(.85 if paired else .43)+1.7)
    fig,axes=plt.subplots(nrows,ncols,figsize=(8 if ncols==1 else 14,nrows*height+1),squeeze=False,layout='constrained')
    colors=dict(delivered='#318582',original='#497BA5',absent='#E5A34A',missing='#DEDEDE',sv='#C76A79',cn='#8C839C',fish='#527CB6')
    for ax,facet in zip(axes.flat,facets):
        z=q[q[facet_key].eq(facet)]
        labels=['All']+sorted(set(z.subtype)-{'All'})
        y=[];ticks=[];ticknames=[];ordered=[]
        for i,label in enumerate(labels):
            item=z[z.subtype.eq(label)]
            if paired:
                for offset,basis in [(-.2,'Panel-reselected genes'),(.2,'Full-reference genes')]:
                    row=item[item.feature_basis.eq(basis)]
                    if len(row)!=1: raise ValueError('Missing/duplicate LymphoMAP feature set')
                    ordered.append(row.iloc[0]);y.append(i+offset)
            else:
                if len(item)!=1: raise ValueError('Duplicate subtype coverage')
                ordered.append(item.iloc[0]);y.append(i)
            ticks.append(i);ticknames.append('All active inputs' if genomic and label=='All' else label)
        for yy,r in zip(y,ordered):
            n=r.requested
            if genomic:
                components=[(r.mutation_delivered,colors['delivered'],None),
                            (r.sv_delivered,colors['fish'],None),
                            (r.missing_mutation,colors['missing'],None),
                            (r.missing_sv,colors['sv'],'///'),
                            (r.missing_cn,colors['cn'],'...')]
            else:
                measured=colors['original'] if paired and r.feature_basis=='Full-reference genes' else colors['delivered']
                components=[(r.delivered,measured,None),(r.on_panel_missing_source,colors['absent'],None),(r.not_on_panel,colors['missing'],None)]
            left=0
            for count,color,hatch in components:
                width=100*count/n
                ax.barh(yy,width,left=left,color=color,height=.31 if paired else .65,hatch=hatch,
                        edgecolor='white' if hatch else color,linewidth=.4)
                left+=width
            ax.text(102,yy,f'{int(r.delivered):,}/{n:,}',va='center',fontsize=9)
        title=facet
        if genomic: title+=' + separate FISH' if fish=='available' else ' · mutation panel alone'
        elif design:title+=' · declared panel targets'
        elif facet=='Chapuy':title+=' · microarray'
        elif facet=='GSE178394':title+=' · NanoString'
        ax.set_title(title,fontsize=11,loc='left')
        ax.set_yticks(ticks,ticknames);ax.invert_yaxis();ax.set_xlim(0,139)
        ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Requested features (%)')
        ax.spines[['top','right','left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
    for ax in list(axes.flat)[len(facets):]:ax.set_visible(False)
    if genomic:
        legend=[Patch(color=colors['delivered'],label='Delivered mutation input'),
                Patch(color=colors['missing'],label='Missing mutation input'),
                Patch(facecolor=colors['sv'],hatch='///',edgecolor='white',label='Missing rearrangement / SV'),
                Patch(facecolor=colors['cn'],hatch='...',edgecolor='white',label='Missing copy-number input')]
        if fish=='available':legend.insert(1,Patch(color=colors['fish'],label='Separate FISH input'))
        subtitle='Panel alone supplies no rearrangement/SV or copy-number inputs' if fish=='unavailable' else 'FISH adds rearrangement inputs; copy-number inputs remain unavailable'
    else:
        legend=[Patch(color=colors['delivered'],label='Panel-selected requirement met' if paired else 'On panel' if design else 'Delivered'),
                Patch(color=colors['missing'],label='Missing from panel'),
                Patch(color=colors['absent'],label='On panel, absent from reference matrix')]
        if design:legend=[p for p in legend if p.get_label() != 'On panel, absent from reference matrix']
        if paired:legend.insert(1,Patch(color=colors['original'],label='Authors’ full-input selection retained'))
        subtitle='Upper bar: panel-selected requirement; lower bar: authors’ full-input selection' if paired else 'Declared panel genes versus author-model reference requirements' if design else 'Unique genes overall and within each stated subtype feature set'
    fig.legend(handles=legend,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    counts_label='on panel / requested' if design else 'delivered / requested'
    fig.suptitle(f'{classifier}: overall and per-subtype feature coverage\n{subtitle}\nCounts: {counts_label}',fontsize=13)
    return fig
