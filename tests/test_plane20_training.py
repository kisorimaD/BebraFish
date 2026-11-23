"""Contract tests for full-ellipse supervision, independent of TensorFlow."""
import unittest
import tempfile
from pathlib import Path
import hashlib
import json
from unittest.mock import patch
import numpy as np
from training.plane20.data import targets, normalize, CropSource
from training.plane20.precompute import precompute


class TargetTests(unittest.TestCase):
    def scene(self, centers, angles=None):
        return dict(centers=np.asarray(centers, float),
            major_axis_px=np.full(len(centers), 10.),
            minor_axis_px=np.full(len(centers), 8.),
            orientation_deg=np.zeros(len(centers)) if angles is None else np.array(angles))

    def test_overlap_does_not_shorten_rays(self):
        p, d = targets(self.scene([[64., 64.], [64., 69.]]), (0, 0))
        self.assertEqual(p[64, 64, 0], 1)
        # The right-hand soma overlaps, but the first soma still extends 5px.
        np.testing.assert_allclose(d[64, 64, [0, 8, 16, 24]], [5, 4, 5, 4], atol=1e-6)

    def test_rotated_subpixel_boundaries(self):
        scene = self.scene([[64.2, 64.3]], [31.])
        _, d = targets(scene, (0, 0))
        theta = np.arange(32)*2*np.pi/32
        endpoints = np.array([64, 64])[:, None]+np.array([np.sin(theta), np.cos(theta)])*d[64, 64, :-1]
        dy, dx = endpoints-scene['centers'][0, :, None]
        c, s = np.cos(np.radians(31)), np.sin(np.radians(31))
        rho2 = ((dx*c+dy*s)/5)**2+((-dx*s+dy*c)/4)**2
        np.testing.assert_allclose(rho2, 1, atol=2e-6)

    def test_crop_does_not_truncate_targets(self):
        scene = self.scene([[100., 100.]])
        p, d = targets(scene, (36, 36))
        q, e = targets(self.scene([[64., 64.]]), (0, 0))
        np.testing.assert_array_equal(p, q)
        np.testing.assert_array_equal(d, e)
        self.assertTrue((p[:16] == -1).all())
        self.assertTrue((d[:16, :, -1] == 0).all())

    def test_empty_crop(self):
        p, d = targets(self.scene([[500., 500.]]), (0, 0))
        self.assertTrue((p[16:-16, 16:-16] == 0).all())
        self.assertTrue((d == 0).all())
        self.assertTrue(np.isfinite(normalize(np.full((128, 128), 100))).all())


class CachedTargetTests(unittest.TestCase):
    def test_paired_renders_share_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'train').mkdir()
            image = np.arange(128*128, dtype=np.uint16).reshape(128, 128)
            structured = image[::-1].copy()
            scene = dict(image=image, image_structured=structured, centers=np.array([[64., 64.]]),
                         major_axis_px=np.array([12.]), minor_axis_px=np.array([10.]),
                         orientation_deg=np.array([0.]))
            np.savez(directory/'train/scene_00000.npz', **scene)
            source = CropSource(directory, 'train', 42, use_precomputed=False)
            baseline, baseline_targets = source.sample(0)
            variant, variant_targets = source.sample(1)
            np.testing.assert_array_equal(baseline[..., 0], normalize(image))
            np.testing.assert_array_equal(variant[..., 0], normalize(structured))
            for left, right in zip(baseline_targets, variant_targets):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(len(source.cache), 1)

    def test_cache_preserves_sampling_targets_and_seams(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'train').mkdir()
            rng = np.random.default_rng(91)
            centers = rng.uniform([0, 0], [260, 280], size=(140, 2)).astype(np.float32)
            centers[:4] = [[127.25, 127.75], [130.5, 130.25], [256.1, 256.5], [1.25, 1.5]]
            scene = dict(image=rng.integers(90, 300, (260, 280), dtype=np.uint16),
                centers=centers, major_axis_px=rng.uniform(8, 18, 140),
                minor_axis_px=rng.uniform(4, 7, 140), orientation_deg=rng.uniform(-90, 90, 140))
            path = directory/'train/scene_00000.npz'
            np.savez_compressed(path, **scene)
            row = dict(path='train/scene_00000.npz', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (directory/'manifest.jsonl').write_text(json.dumps(row)+'\n')
            (directory/'COMPLETE').touch()
            precompute(directory, workers=1)
            # Idempotent precompute can reuse completed scenes.
            precompute(directory, workers=1)
            base = directory/'target_cache/train/scene_00000'
            prob_map = np.load(base/'prob.npy', mmap_mode='r')
            dist_map = np.load(base/'dist.npy', mmap_mode='r')
            for origin in [(0, 0), (62, 73), (110, 119), (128, 152), (132, 0)]:
                y, x = origin
                p, d = targets(scene, origin, halo=0)
                np.testing.assert_array_equal(prob_map[y:y+128, x:x+128], p[..., 0])
                np.testing.assert_array_equal(dist_map[y:y+128, x:x+128], d[..., :-1])
            old = CropSource(directory, 'train', 42, use_precomputed=False)
            fast = CropSource(directory, 'train', 42, use_precomputed=True)
            for index in [0, 1, 4, 5, 10, 99, 128, 10000]:
                a, b = old.sample(index), fast.sample(index)
                np.testing.assert_array_equal(a[0], b[0])
                for expected, actual in zip(a[1], b[1]):
                    np.testing.assert_array_equal(expected, actual)
            expected_repeat = old.sample(99)
            with patch('training.plane20.data.targets', side_effect=AssertionError('recomputed targets')):
                # Also exercise repeated reads: halo masking must not mutate cached maps.
                for _ in range(2):
                    actual = fast.sample(99)
                    for expected, found in zip(expected_repeat[1], actual[1]):
                        np.testing.assert_array_equal(expected, found)
            with self.assertRaises(ValueError):
                CropSource(directory, 'train', 42, rays=16, use_precomputed=True)
            (directory/'target_cache/COMPLETE').unlink()
            with self.assertRaises(ValueError):
                CropSource(directory, 'train', 42, use_precomputed=True)


if __name__ == '__main__':
    unittest.main()
