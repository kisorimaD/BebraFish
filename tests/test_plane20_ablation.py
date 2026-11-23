"""Ablation contract: thinning must not change which image regions are sampled."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from training.plane20.data import CropSource


class PairedCropTests(unittest.TestCase):
    def test_sparse_targets_keep_full_density_crop_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(5)
            full = rng.uniform([140, 140], [240, 240], (150, 2))
            image = rng.integers(100, 500, (400, 400), dtype=np.uint16)
            for name, centers in [('normal', full), ('half', full[::2])]:
                (root/name/'train').mkdir(parents=True)
                np.savez_compressed(root/name/'train/scene_00000.npz', image=image,
                    centers=centers, sampling_centers=full,
                    major_axis_px=np.full(len(centers), 8.),
                    minor_axis_px=np.full(len(centers), 6.),
                    orientation_deg=np.zeros(len(centers)))
            normal = CropSource(root/'normal', 'train', 42, use_precomputed=False)
            half = CropSource(root/'half', 'train', 42, use_precomputed=False)
            for index in range(50):
                a, sa, oa = normal.crop(index)
                b, sb, ob = half.crop(index)
                self.assertEqual(oa, ob)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(len(sa['centers']), 2*len(sb['centers']))
                np.testing.assert_array_equal(sb['centers'], sa['centers'][::2])


if __name__ == '__main__':
    unittest.main()
