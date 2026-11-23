"""Exercise the frozen production profile and its overlapping instance GT."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simcalc_zebrafish_pipeline'))
from calibrated_simcalc import generate


class GenerationTests(unittest.TestCase):
    def test_frozen_fish4_export_and_complete_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'generated'
            path = generate(ROOT / 'profiles/fish4_plane16', output, count=1, seed=7101)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data['synthetic_images'].shape, (1, 1728, 288))
                self.assertEqual(data['synthetic_images'].dtype, np.dtype('uint16'))
                labels = data['synthetic_labels'][0]
                tissue = data['synthetic_tissue_masks'][0]
                self.assertFalse(np.any((labels > 0) & ~tissue))
                self.assertEqual(int(data['synthetic_seeds'][0]), 7101)
                metadata = json.loads(str(data['metadata_json']))
            footprint = output / metadata['full_instance_footprints'][0]
            with np.load(footprint, allow_pickle=False) as data:
                self.assertEqual(len(data['offsets']), 2420)
                self.assertTrue(np.all(np.diff(data['offsets'].astype(np.int64)) > 0))
                union = np.zeros(labels.size, dtype=bool)
                union[data['pixel_index']] = True
                np.testing.assert_array_equal(union.reshape(labels.shape), labels > 0)
            with self.assertRaises(FileExistsError):
                generate(ROOT / 'profiles/fish4_plane16', output, count=1)


if __name__ == '__main__':
    unittest.main()
