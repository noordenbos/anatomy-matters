"""Empirical RNA-seq proxy experiment; distinct from the full Lymph2Cx assay."""
import numpy as np
import pandas as pd

LABELS = ['GCB', 'UNC', 'ABC']


def proxy_score(log_expression, model, use_isy1_proxy=True):
    """Apply fixed patent coefficients to 14 measured predictors.

    RAB29's normalized score contribution is omitted, not imputed as zero RNA.
    ISY1-RAB43 supplies ISY1 only for this experiment; the comparator uses the
    other four housekeeping genes. Input is samples x genes, log2(TPM+1).
    """
    features = pd.DataFrame(model['features']).set_index('gene')
    predictors = features[(features.coefficient != 0) & features.index.to_series().ne('RAB29')].index
    housekeeping = features[features.normalization_weight > 0].index.tolist()
    housekeeping.remove('ISY1')
    if use_isy1_proxy:
        housekeeping.append('ISY1-RAB43')
    required = list(predictors) + housekeeping
    if not log_expression.columns.is_unique or not set(required) <= set(log_expression.columns):
        raise ValueError('Duplicate or missing proxy-experiment features')
    x = log_expression[required].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite expression')
    norm = log_expression[housekeeping].mean(axis=1)
    score = log_expression[predictors].sub(norm, axis=0) @ features.loc[predictors,'coefficient']
    return score.rename('score')


def call_from_thresholds(scores, lower, upper):
    return np.where(np.asarray(scores) < lower, 'GCB', np.where(np.asarray(scores) < upper, 'UNC', 'ABC'))


def fit_thresholds(scores, labels):
    """Fit two ordered cutpoints by training-only three-class macro recall.

    Search all gaps between distinct scores, including empty end/intermediate
    classes. Ties select the lowest lower cutpoint then lowest upper cutpoint.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    if set(labels) != set(LABELS) or not np.isfinite(scores).all():
        raise ValueError('All three classes and finite training scores required')
    unique = np.unique(scores)
    thresholds = np.r_[-np.inf, unique[:-1] + np.diff(unique)/2, np.inf]
    counts = np.array([[(scores[labels==label] < t).sum() for label in LABELS] for t in thresholds])
    totals = np.array([(labels==label).sum() for label in LABELS])
    fraction = counts / totals
    objective = (fraction[:,0,None] + fraction[None,:,1] - fraction[:,1,None] + 1-fraction[None,:,2])/3
    objective[np.tril_indices(len(thresholds), -1)] = -np.inf
    lo, hi = np.unravel_index(np.argmax(objective), objective.shape)
    return float(thresholds[lo]), float(thresholds[hi])


def plot_proxy_results(scores, calls, metrics):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    colors = {'GCB':'#4477AA', 'UNC':'#999999', 'ABC':'#CC6677'}
    ax = axes[0]
    for i, label in enumerate(LABELS):
        values = scores.loc[scores.reference_label.eq(label), 'isy1_readthrough_proxy']
        ax.scatter(np.full(len(values),i)+np.random.default_rng(17+i).uniform(-.18,.18,len(values)), values,
                   s=10, alpha=.5, color=colors[label])
        ax.plot([i-.25,i+.25], [values.median()]*2, color='black')
    ax.set(xticks=range(3), xticklabels=LABELS, ylabel='Fixed-weight 14-predictor score', title='Scores by supplied Lymph2Cx label')
    primary = calls[calls.variant.eq('isy1_readthrough_proxy') & calls.evaluation.eq('five_fold_out_of_fold')]
    counts = pd.crosstab(primary.reference_label, primary.predicted_label).reindex(index=LABELS,columns=LABELS,fill_value=0)
    percent = counts.div(counts.sum(axis=1),axis=0)*100
    ax=axes[1]
    ax.imshow(percent, vmin=0,vmax=100,cmap='Blues')
    for i in range(3):
        for j in range(3):
            value=percent.iloc[i,j]
            ax.text(j,i,f'{value:.0f}%\n(n={counts.iloc[i,j]})',ha='center',va='center',color='white' if value>55 else 'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
           xlabel='Held-out prediction',ylabel='Supplied Lymph2Cx label',title='Proxy: five-fold cross-validation')
    ax=axes[2]
    m=metrics[metrics.variant.eq('isy1_readthrough_proxy') & metrics.evaluation.eq('five_fold_out_of_fold')].set_index('label').reindex(LABELS)
    x=np.arange(3)
    ax.bar(x-.18,m.recall*100,.36,label='Recall',color='#4477AA')
    ax.bar(x+.18,m.precision*100,.36,label='Precision',color='#EE9944')
    ax.set(xticks=x,xticklabels=LABELS,ylim=(0,105),ylabel='Percent',title='Held-out label agreement')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False)
    fig.suptitle('Morin exploratory reconstruction: ISY1-RAB43 proxy; RAB29 contribution omitted',fontsize=12)
    return fig
