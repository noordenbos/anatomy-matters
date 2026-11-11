import numpy as np
import pandas as pd
import unittest
from dlbcl.rna_panel_benchmark import summarize

def example():
    return pd.DataFrame(dict(classifier=['LymphoMAP']*4,cohort=['test']*4,group=['test']*4,
                             sample_id=['a','b','c','d'],full_label=['LN','LN','TEX','Unassigned'],
                             panel_label=['LN','TEX','Unassigned','LN']))

def test_abstentions_and_empty_denominators():
    m,o,t=summarize(example());m=m.set_index('label')
    assert m.loc['LN','recall']==.5 and m.loc['LN','precision']==.5
    assert m.loc['TEX','recall']==0
    assert np.isnan(m.loc['FMAC','recall']) and np.isnan(m.loc['FMAC','precision'])
    assert o.iloc[0].agreement==.25
    assert o.iloc[0].full_assigned==3 and o.iloc[0].panel_assigned==3
    assert t['count'].sum()==4
    assert t[t.full_label=='LN'].row_fraction.sum()==1

def test_missing_or_duplicate_pairs_cannot_disappear():
    d=example();d.loc[0,'panel_label']=None
    with unittest.TestCase().assertRaises(ValueError):summarize(d)
    with unittest.TestCase().assertRaises(ValueError):summarize(pd.concat([example(),example().iloc[:1]]))

if __name__=='__main__':
    test_abstentions_and_empty_denominators()
    test_missing_or_duplicate_pairs_cannot_disappear()
    print('2 paired-metric checks passed')
