"""Public-patent Lymph2Cx linear score, not a validated portal replacement.

The 20 coefficients and normalization are recoverable. Portal calibration,
zero-count handling, exact quality control and probability output are not yet
verified. This module deliberately emits scores, not ABC/GCB clinical labels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def linear_score(log_expression, model, log_calibration=None):
    """Score samples x canonical genes on an explicitly supplied log2 scale.

    Optional log_calibration is a gene-indexed Series of log2(gold/reference).
    No pseudocount, hidden imputation or inferred calibration is applied.
    """
    if isinstance(model, (str, Path)):
        model = json.loads(Path(model).read_text())
    features = pd.DataFrame(model['features']).set_index('gene')
    missing = set(features.index) - set(log_expression.columns)
    if missing:
        raise ValueError('Missing Lymph2Cx input genes: ' + ', '.join(sorted(missing)))
    if not log_expression.columns.is_unique:
        raise ValueError('Duplicate expression genes')
    x = np.ascontiguousarray(log_expression.loc[:, features.index].to_numpy(dtype=float))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite Lymph2Cx expression')
    normalization = x @ features.normalization_weight.to_numpy()
    normalized = x - normalization[:,None]
    if log_calibration is not None:
        correction = log_calibration.reindex(features.index)
        if not np.isfinite(correction.to_numpy()).all():
            raise ValueError('Incomplete reference calibration')
        normalized = normalized + correction.to_numpy()[None,:]
    return pd.Series(normalized @ features.coefficient.to_numpy(), index=log_expression.index, name='research_linear_score')


def score_positive_counts(counts, model, log_calibration=None):
    if not np.isfinite(counts.to_numpy()).all() or (counts <= 0).any().any():
        raise ValueError('Strictly positive counts required; portal zero-count handling is unverified')
    return linear_score(np.log2(counts), model, log_calibration)
