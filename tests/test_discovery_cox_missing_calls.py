"""Regression for unavailable genomic calls entering S4F as a Cox subtype."""
import ast
import json
from pathlib import Path
import unittest
import pandas as pd


class DiscoveryCoxMissingCallsTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/supplemental_fig4.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if 'def _design_matrix' in ''.join(c['source']))
        nodes = [node for node in ast.parse(source).body
                 if (isinstance(node, ast.FunctionDef) and node.name in {'_design_matrix', '_categorical_levels'})
                 or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CATEGORICAL_EXCLUDE_LEVELS' for t in node.targets))]
        self.scope = {'pd': pd, 'TIME_COL': 'time', 'CONTINUOUS_COVARIATES': {'Age'}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<S4F helpers>', 'exec'), self.scope)

    def test_excludes_unavailable_calls_before_encoding_and_retains_reference(self):
        labels = ['C4', 'C1', 'unknown', ' Unclassified ', '0', None]
        frame = pd.DataFrame({'time': range(1, 7), 'event': [0, 1]*3,
                              'DLBclass': pd.Categorical(labels, categories=['C4','C1','unknown',' Unclassified ','0']),
                              'Age': range(50, 56), 'COO': pd.Categorical(['GCB','ABC']*3)})
        before = frame.copy(deep=True)
        matrix = self.scope['_design_matrix'](frame, 'DLBclass', 'event')
        self.assertEqual(list(matrix.index), [0, 1])
        self.assertEqual(list(matrix.columns), ['time', 'event', 'DLBclass_C1'])
        self.assertEqual(matrix.DLBclass_C1.tolist(), [False, True])
        self.scope['surv'] = frame
        self.assertEqual(self.scope['_categorical_levels']('DLBclass', 'C4'), ['C4', 'C1'])
        self.assertEqual(len(self.scope['_design_matrix'](frame, 'Age', 'event')), 6)
        self.assertEqual(len(self.scope['_design_matrix'](frame, 'COO', 'event')), 6)
        pd.testing.assert_frame_equal(frame, before)
