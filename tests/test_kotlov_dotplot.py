import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from dlbcl.integration_figures import plot_association_enrichment_dotplot_by_classifier


class KotlovDotplotTests(unittest.TestCase):
    def test_absent_gc_has_grey_placeholders_without_annotation(self):
        frame = pd.DataFrame({
            'KotlovSig': ['Depleted', 'Inflammatory', 'Mesenchymal'] * 8,
            'Location': ['bone', 'nodal', 'PCNS', 'testis'] * 6,
            'tumorimmune_archetype': ['low immune', 'complex immune'] * 12,
            'genomic_tested': True,
        })
        for stratifier in ['location', 'archetype']:
            with self.subTest(stratifier=stratifier), tempfile.TemporaryDirectory() as tmp, patch('dlbcl.integration_figures._save_show_close') as save:
                plot_association_enrichment_dotplot_by_classifier(
                    frame, stratifier=stratifier, out_svg=Path(tmp)/'test.svg', title='Test', show=False)
                fig=save.call_args.args[0]
                ax=next(a for a in fig.axes if a.get_ylabel()=='Kotlov LME')
                labels=[t.get_text() for t in ax.get_yticklabels()]
                gc_row=labels.index('GC-like')
                self.assertNotIn('No assigned cases', [t.get_text() for t in ax.texts])
                gc_dots = [(x, y) for collection in ax.collections for x,y in collection.get_offsets() if y==gc_row]
                self.assertEqual(len(gc_dots), frame['Location' if stratifier=='location' else 'tumorimmune_archetype'].nunique())
                plt.close(fig)
