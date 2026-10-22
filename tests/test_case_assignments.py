"""Canonical labels must win, preserve case alignment, and survive bundle cleanup."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import anndata as ad
import numpy as np
import pandas as pd

from dlbcl.case_assignments import (
    apply_archetype_assignments, clean_assignment_storage,
    discovery_archetype_dict, validation_analysis_frame,
)
from dlbcl.migrate_case_assignments import migrate_bundle


def fixture():
    cc = pd.DataFrame({'patient_id':['T1','T2'], 'tumorimmune_archetype_id':[1,3],
        'tumorimmune_archetype':['low immune','complex immune'], 'DLBclass':['C1','C5'],
        'COO_NanoString':['ABC','GCB']})
    vc = pd.DataFrame({'tumorimmune_archetype_id':[1,3],
        'tumorimmune_archetype':['low immune','complex immune'], 'DLBclass':['C1','C5'],
        'COO_NanoString':['ABC','GCB']}, index=pd.Index(['V1','V2'],name='patient_id'))
    meta = pd.DataFrame({'disease_type':['Brain','Bone'], 'age':[50,60],
        'COO_NanoString':['WRONG','WRONG']}, index=pd.Index(['V1','V2'],name='patient_alias'))
    pred = meta.copy();pred['pred_tumorimmune_archetype_id']=[2,2]
    pred['pred_tumorimmune_archetype']=['cytotoxic predominant']*2
    pred['prob_1']=[.7,.1];pred['prob_2']=[.2,.2];pred['prob_3']=[.1,.7];pred['max_prob']=[.7,.7]
    return {'case_classifications':cc,
        'case_clinical':pd.DataFrame({'COO_NanoString':['ABC','GCB'],'Age':[50,60]},index=['T1','T2']),
        'tumorimmune_archetype_id':{'T1':2,'T2':2},
        'validation_cohort':{'meta':meta,'case_classification_validation':vc,'predictions':pred}}


class CaseAssignmentTests(unittest.TestCase):
    def test_canonical_wins_over_conflicting_legacy_copies_and_reordered_ids(self):
        uns=fixture();uns['validation_cohort']['case_classification_validation']=uns['validation_cohort']['case_classification_validation'].iloc[::-1]
        frame=validation_analysis_frame(uns['validation_cohort'])
        self.assertEqual(frame.pred_tumorimmune_archetype_id.tolist(),[1,3])
        self.assertEqual(frame.COO_NanoString.tolist(),['ABC','GCB'])
        self.assertEqual(discovery_archetype_dict(SimpleNamespace(uns=uns)),{'T1':1,'T2':3})

    def test_missing_canonical_has_no_fallback(self):
        uns=fixture();del uns['validation_cohort']['case_classification_validation']
        with self.assertRaises(KeyError):validation_analysis_frame(uns['validation_cohort'])
        del uns['case_classifications']
        with self.assertRaises(KeyError):discovery_archetype_dict(SimpleNamespace(uns=uns))

    def test_duplicate_and_missing_patient_ids_fail(self):
        for bad in ['duplicate','missing']:
            vc=fixture()['validation_cohort']
            if bad=='duplicate':vc['case_classification_validation'].index=['V1','V1']
            else:vc['case_classification_validation']=vc['case_classification_validation'].iloc[:1]
            with self.assertRaises(ValueError):validation_analysis_frame(vc)

    def test_cleanup_preserves_canonical_tables_scores_and_clinical_data(self):
        original=fixture();out=clean_assignment_storage(original)
        pd.testing.assert_frame_equal(original['case_classifications'],out['case_classifications'])
        pd.testing.assert_frame_equal(original['validation_cohort']['case_classification_validation'],out['validation_cohort']['case_classification_validation'])
        self.assertNotIn('tumorimmune_archetype_id',out)
        self.assertNotIn('predictions',out['validation_cohort'])
        self.assertNotIn('COO_NanoString',out['validation_cohort']['meta'])
        self.assertEqual(out['validation_cohort']['prediction_scores'].shape,(2,4))
        self.assertEqual(out['case_clinical'].Age.tolist(),[50,60])
        self.assertIn('predictions',original['validation_cohort'])

    def test_explicit_update_changes_only_canonical_labels_and_refreshes_views(self):
        a=SimpleNamespace(uns=fixture())
        candidate=pd.DataFrame({'pred_tumorimmune_archetype_id':[2,1]},index=['V2','V1'])
        apply_archetype_assignments(a,candidate,cohort='validation')
        out=validation_analysis_frame(a.uns['validation_cohort'])
        self.assertEqual(out.pred_tumorimmune_archetype_id.tolist(),[1,2])
        self.assertEqual(out.DLBclass.tolist(),['C1','C5'])
        self.assertNotIn('predictions',a.uns['validation_cohort'])
        before=copy.deepcopy(a.uns)
        with self.assertRaises(ValueError):apply_archetype_assignments(a,candidate.iloc[:1],cohort='validation')
        pd.testing.assert_frame_equal(before['validation_cohort']['case_classification_validation'],a.uns['validation_cohort']['case_classification_validation'])

    def test_discovery_update_keeps_patient_column_and_other_labels(self):
        a=SimpleNamespace(uns=fixture());c=pd.DataFrame({'tumorimmune_archetype_id':[2,1]},index=['T2','T1'])
        apply_archetype_assignments(a,c,cohort='discovery')
        self.assertEqual(a.uns['case_classifications'].patient_id.tolist(),['T1','T2'])
        self.assertEqual(discovery_archetype_dict(a),{'T1':1,'T2':2})
        self.assertEqual(a.uns['case_classifications'].DLBclass.tolist(),['C1','C5'])
        self.assertNotIn('tumorimmune_archetype_id',a.uns)

    def test_migration_is_backed_up_atomic_idempotent_and_preserves_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bundle.h5ad';a=ad.AnnData(X=np.arange(6).reshape(2,3),uns=fixture())
            a.write_h5ad(path)
            preview=migrate_bundle(path);self.assertTrue(preview['removed_validation_keys'])
            original_bytes=path.read_bytes()
            result=migrate_bundle(path,apply=True)
            self.assertEqual(Path(result['backup']).read_bytes(),original_bytes)
            b=ad.read_h5ad(path)
            np.testing.assert_array_equal(a.X,b.X)
            pd.testing.assert_frame_equal(a.uns['case_classifications'],b.uns['case_classifications'])
            self.assertEqual(migrate_bundle(path,apply=True)['status'],'already clean')

if __name__=='__main__':unittest.main()
