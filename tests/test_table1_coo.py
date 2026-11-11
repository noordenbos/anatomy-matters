"""Regression coverage for valid UNC calls and genuinely missing COO results."""

import unittest

import pandas as pd

from dlbcl.validation_clinical_table import (
    COO_UNCLASSIFIED_ROW,
    build_clinical_table_dataframe,
    build_combined_clinical_table,
    combined_coo_missing_footnote,
    normalize_coo_label,
    validation_coo_missing_footnote,
)


class TestTable1CellOfOrigin(unittest.TestCase):
    def table(self, calls):
        return build_clinical_table_dataframe(pd.DataFrame({
            "disease_type": ["Nodal"] * len(calls),
            "COO_NanoString": calls,
        }))

    def test_unc_aliases_count_as_results_in_both_cohorts(self):
        table = self.table(["ABC", "GCB", "UNC", "Intermediate", " Unclassified "])
        rows = table.set_index("Characteristic")
        self.assertEqual(rows.at[COO_UNCLASSIFIED_ROW, "Nodal"], "3")
        self.assertEqual(rows.at["  ABC", "Nodal"], "1")
        self.assertEqual(rows.at["  GCB", "Nodal"], "1")
        self.assertIsNone(validation_coo_missing_footnote(table.attrs["group_stats"]))
        self.assertIsNone(combined_coo_missing_footnote(
            discovery_table=table, group_stats=table.attrs["group_stats"]))
        # Legacy discovery tables align with the canonical row, rather than duplicating it.
        legacy = table.copy()
        legacy["Characteristic"] = legacy["Characteristic"].replace(
            {COO_UNCLASSIFIED_ROW: "  Intermediate"})
        combined = build_combined_clinical_table(legacy, table).set_index("Characteristic")
        self.assertEqual(combined.at[COO_UNCLASSIFIED_ROW, "d_Nodal"], "3")
        self.assertEqual(combined.at[COO_UNCLASSIFIED_ROW, "v_Nodal"], "3")
        self.assertNotIn("  Intermediate", combined.index)

    def test_missing_results_are_not_counted_as_unc(self):
        table = self.table(["UNC", "Unknown", None, pd.NA, ""])
        self.assertEqual(table.set_index("Characteristic").at[COO_UNCLASSIFIED_ROW, "Nodal"], "1")
        note = validation_coo_missing_footnote(table.attrs["group_stats"])
        self.assertIn("patients with missing cell-of-origin results", note)
        self.assertIn("4", note)

    def test_unrecognized_labels_do_not_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized cell-of-origin"):
            normalize_coo_label("misspelled")


if __name__ == "__main__":
    unittest.main()
