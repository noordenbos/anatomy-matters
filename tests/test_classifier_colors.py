"""Guard against order-dependent or silently misassigned classifier colors."""

import json
import unittest
import warnings

from dlbcl.classifier_colors import (
    CONFIG_PATH, PaletteFallbackWarning, classifier_color, classifier_palette,
    missing_color, reference_palette,
)


class ClassifierColorTests(unittest.TestCase):
    def test_order_and_subset_do_not_change_colors(self):
        expected = {"C1": "#803E98", "C2": "#00A4D1", "C5": "#DD2929"}
        self.assertEqual(classifier_palette("DLBclass", ["C5", "C2", "C1"]), expected)
        self.assertEqual(classifier_color("DLBclass", "C2"), expected["C2"])

    def test_same_classifier_across_column_and_figure_names(self):
        for name in ["lymphomap", "Li 2025\nLymphoMAP"]:
            self.assertEqual(classifier_color(name, "FMAC"), "#377EB8")
        for name in ["COO_NanoString_update", "coo", "Cell of Origin"]:
            self.assertEqual(classifier_color(name, "Intermediate"), "#006412")
        self.assertEqual(classifier_color("Ciav_Cluster", "Intermediate"), "#3753A3")

    def test_missing_is_not_a_classifier_residual_category(self):
        for value in [None, float("nan"), "NA", "unknown", " UNKNOWN "]:
            self.assertEqual(classifier_color("Lymphgen", value), missing_color())
        self.assertNotEqual(classifier_color("Lymphgen", "Other"), missing_color())
        self.assertNotEqual(classifier_color("HMRN", "NEC"), missing_color())
        self.assertNotEqual(classifier_color("DLBclass", "0"), missing_color())

    def test_author_confirmed_hmrn_mapping_uses_original_subtype_colors(self):
        expected = {
            "C1": ("NEC", "#BF0023"),
            "C2": ("SOCS1/SGK1", "#12497F"),
            "C3": ("TET2/SGK1", "#3AA933"),
            "C4": ("MYD88", "#5B2063"),
            "C5": ("BCL2", "#F28C00"),
            "C6": ("NOTCH2", "#E4007D"),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", PaletteFallbackWarning)
            for code, (label, color) in expected.items():
                self.assertEqual(classifier_color("HMRN", code), color)
                self.assertEqual(classifier_color("HMRN", label), color)

    def test_composite_fallback_is_not_claimed_as_original(self):
        self.assertNotIn("BN2/MCD", reference_palette("Lymphgen"))
        with self.assertWarns(PaletteFallbackWarning):
            classifier_color("Lymphgen", "BN2/MCD")

    def test_unknown_labels_fail_instead_of_becoming_missing(self):
        with self.assertRaisesRegex(ValueError, "Unconfigured labels"):
            classifier_color("Lymphgen", "MCD_typo")
        with self.assertRaises(KeyError):
            classifier_palette("not_a_classifier")

    def test_returned_palette_cannot_mutate_shared_configuration(self):
        palette = classifier_palette("DLBclass")
        palette["C1"] = "#000000"
        self.assertEqual(classifier_color("DLBclass", "C1"), "#803E98")

    def test_every_alias_targets_an_explicit_color(self):
        for key, spec in json.loads(CONFIG_PATH.read_text())["classifiers"].items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PaletteFallbackWarning)
                for alias, target in spec["label_aliases"].items():
                    self.assertEqual(classifier_color(key, alias), classifier_color(key, target))

    def test_heatmap_and_pie_use_same_colors_with_different_category_order(self):
        import pandas as pd
        from dlbcl.archetype_heatmap import make_categorical_color_series
        from dlbcl.location_pie_patchwork import _palette_for_column
        values = pd.Series(["GCB", "Intermediate", "ABC", None], name="COO_NanoString_update")
        strip, legend = make_categorical_color_series(values)
        pie, _ = _palette_for_column("COO_NanoString", "Cell of Origin", values)
        for i, label in enumerate(values[:3]):
            self.assertEqual(strip.iloc[i], pie[label])
            self.assertEqual(legend[label], pie[label])
        self.assertEqual(strip.iloc[3], missing_color())

    def test_notebook_classifier_panels_request_named_config_palettes(self):
        root = CONFIG_PATH.parent.parent
        for name, indices in [("fig2", [8, 10]), ("supplemental_fig3", [15, 17])]:
            nb = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
            for index in indices:
                source = "".join(nb["cells"][index]["source"])
                self.assertIn("palette=classifier_palette(", source)
                self.assertNotIn("palette=[", source)


if __name__ == "__main__":
    unittest.main()
