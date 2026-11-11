from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
from dlbcl.lymph2cx_research import linear_score, score_positive_counts


def test_housekeeping_normalization_and_panel_identity():
    model=json.loads((Path(__file__).resolve().parents[1]/'config/lymph2cx_patent_research.json').read_text())
    genes=[r['gene'] for r in model['features']]
    counts=pd.DataFrame(np.arange(1,41,dtype=float).reshape(2,20)+20,columns=genes)
    a=score_positive_counts(counts,model)
    b=score_positive_counts(counts*8,model)
    assert np.allclose(a,b,atol=1e-10)
    expanded=counts.assign(IRRELEVANT_GENE=[100000.,1.])
    assert np.array_equal(score_positive_counts(expanded,model).to_numpy(),a.to_numpy())
    assert np.allclose(linear_score(np.log2(counts),model),a)
    with unittest.TestCase().assertRaises(ValueError):
        score_positive_counts(counts.drop(columns=genes[0]),model)
    counts.iloc[0,0]=0
    with unittest.TestCase().assertRaises(ValueError):
        score_positive_counts(counts,model)


if __name__=='__main__':
    test_housekeeping_normalization_and_panel_identity()
    print('Lymph2Cx research-score normalization, feature invariance and missing/zero-input checks passed')
