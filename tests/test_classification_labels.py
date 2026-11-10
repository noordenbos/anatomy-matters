import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dlbcl.classification_labels import ingest_ciavarella_labels, ingest_ciavarella_table
from dlbcl.case_assignments import discovery_assignments
from dlbcl.rna_clustering_benchmark import ward_partition


class ClassificationLabelsTests(unittest.TestCase):
    def test_ward_names_high_and_low_expression_directly(self):
        x=pd.DataFrame([[10,11],[10.1,11.2],[5,5],[5.1,5.2],[0,1],[0.1,1.2]],index=list('abcdef'))
        expected=pd.Series(['Hot','Hot','Intermediate','Intermediate','Cold','Cold'],index=x.index)
        for order in [x.index,x.index[::-1]]:
            observed=pd.Series(ward_partition(x.loc[order]),index=order)
            pd.testing.assert_series_equal(observed.reindex(x.index),expected)

    def test_legacy_external_labels_are_normalized_only_at_ingestion(self):
        source=pd.Series(['High','Intermediate','Low',None,'unknown'],index=list('abcde'))
        result=ingest_ciavarella_labels(source)
        self.assertEqual(result.loc[['a','b','c']].tolist(),['Hot','Intermediate','Cold'])
        self.assertIsNone(result.loc['d'])
        self.assertEqual(source.loc['a'],'High')
        data=pd.DataFrame({'classifier':['Ciavarella','Other method','Ciavarella'],'label':['High','High','All'],'tp':[5,7,10]})
        result=ingest_ciavarella_table(data)
        self.assertEqual(result.label.tolist(),['Hot','High','All'])
        np.testing.assert_array_equal(result.tp,data.tp)

    def test_canonical_read_does_not_silently_remap_legacy_calls(self):
        adata=SimpleNamespace(uns={'case_classifications':pd.DataFrame({'patient_id':['T1'],'Ciav_Cluster':['High']})})
        with self.assertRaisesRegex(ValueError,'Noncanonical Ciavarella'):
            discovery_assignments(adata)


    def test_obsolete_archetype_is_rejected_at_canonical_read(self):
        adata=SimpleNamespace(uns={'case_classifications':pd.DataFrame({'patient_id':['T1'],'tumorimmune_archetype':['complex immune']})})
        with self.assertRaisesRegex(ValueError,'Noncanonical spatial archetype'):
            discovery_assignments(adata)


if __name__=='__main__':unittest.main()
