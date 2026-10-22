"""Presentation changes must preserve classifications and plotted numbers."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlbcl.display_labels import (
    classifier_display_label, display_text, finalize_figure_labels,
)


class DisplayLabelTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_classifier_context_preserves_ciavarella_and_missingness(self):
        self.assertEqual(classifier_display_label("COO_NanoString", "Intermediate"), "Unclassified")
        self.assertEqual(classifier_display_label("Cell of Origin", "Intermediate"), "Unclassified")
        self.assertEqual(classifier_display_label("Ciav_Cluster", "Intermediate"), "Intermediate")
        self.assertEqual(classifier_display_label("COO_NanoString", "NA"), "NA")

    def test_confidence_intervals_are_not_archetype_abbreviations(self):
        self.assertEqual(display_text("HR (95% CI)"), "HR (95% CI)")
        self.assertEqual(display_text("low immune__vs__complex immune"), "low immune__vs__diverse immune")
        self.assertEqual(display_text("Complex immune vs rest (95% CI)"), "Diverse immune vs rest (95% CI)")

    def test_rendered_labels_survive_redraw_without_changing_data(self):
        source = pd.DataFrame({"complex immune": [1.0, 2.0], "low immune": [3.0, 4.0]})
        before = source.copy(deep=True)
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1], source.sum().values, label="complex immune")
        ax.set_xticks([0, 1], source.columns)
        ax.set_title("Complex immune (95% CI)")
        ax.legend()
        heights = [bar.get_height() for bar in bars]
        finalize_figure_labels(fig)
        finalize_figure_labels(fig)
        fig.canvas.draw()
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["diverse immune", "low immune"])
        self.assertEqual(ax.get_title(), "Diverse immune (95% CI)")
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "diverse immune")
        self.assertEqual(heights, [bar.get_height() for bar in bars])
        pd.testing.assert_frame_equal(source, before)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.svg"
            fig.savefig(p)
            self.assertNotIn("complex immune", p.read_text().lower())

    def test_numeric_axes_are_unchanged(self):
        fig, ax = plt.subplots()
        ax.plot([1000, 2000], [0.01, 0.03])
        fig.canvas.draw()
        original = ax.xaxis.get_major_formatter()
        before = ax.get_xlim()
        finalize_figure_labels(fig)
        self.assertIs(ax.xaxis.get_major_formatter(), original)
        np.testing.assert_equal(ax.get_xlim(), before)

    def test_coo_plot_only_changes_text(self):
        fig, ax = plt.subplots()
        ax.scatter([0], [1], label="Intermediate")
        ax.legend()
        finalize_figure_labels(fig, classifier="coo")
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "Unclassified")
        np.testing.assert_equal(ax.collections[0].get_offsets(), [[0, 1]])

    def test_figure5_registry_matches_approved_sequence(self):
        from dlbcl.figure_artifacts import FIGURE_ARTIFACTS
        from dlbcl.figure_registry import SUPPLEMENTARY_TABLES
        for artifact, table, letter in [
            ("5E_os_forest", "5E_os_forest", "E"),
            ("5F_val_loc_arch_heatmaps", "5F_loc_arch_counts", "F"),
            ("5G_cox_forest", "5G_mv_forest", "G"),
            ("5H_event_grid", "5H_event_grid", "H"),
            ("5I_symmetric_lrt", "5I_sym_lrt", "I"),
            ("5J_ll_partition", "5J_partition", "J"),
        ]:
            self.assertEqual(FIGURE_ARTIFACTS[artifact].manuscript, f"Fig 5{letter}")
            self.assertEqual(SUPPLEMENTARY_TABLES[table].manuscript, f"Fig 5{letter}")


if __name__ == "__main__":
    unittest.main()
