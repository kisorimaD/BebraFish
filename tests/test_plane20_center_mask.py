"""Contracts for overlapping independent masks and deterministic shared crops."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from training.plane20.center_data import roi_masks, center_targets, batch_samples, ROI_SIZE, InstanceSource, roi_anchor


class IndependentMaskTests(unittest.TestCase):
    def scene(self):
        return dict(centers=np.array([[64.2, 64.3], [64.4, 68.1]]),
            major_axis_px=np.array([10., 9.]), minor_axis_px=np.array([8., 7.]),
            orientation_deg=np.array([20., -15.]))

    def test_each_overlapping_cell_keeps_its_full_mask(self):
        scene = self.scene();points = scene['centers']
        masks = roi_masks(scene, (0, 0), np.array([0, 1]), points)
        full = []
        for i, point in enumerate(points):
            anchor = np.floor(point+.5).astype(int);yy, xx = np.mgrid[:128, :128]
            theta = np.radians(scene['orientation_deg'][i]);c, s = np.cos(theta), np.sin(theta)
            dy, dx = yy-point[0], xx-point[1]
            expected = ((dx*c+dy*s)/(scene['major_axis_px'][i]/2))**2+((-dx*s+dy*c)/(scene['minor_axis_px'][i]/2))**2 <= 1
            half = ROI_SIZE//2
            np.testing.assert_array_equal(masks[i, ..., 0], expected[anchor[0]-half:anchor[0]+half, anchor[1]-half:anchor[1]+half])
            full.append(expected)
        self.assertGreater(np.sum(full[0]&full[1]), 0)
        self.assertGreater(masks[0].sum()+masks[1].sum(), np.sum(full[0]|full[1]))

    def test_jitter_moves_roi_without_moving_gt_geometry(self):
        scene = self.scene();point = scene['centers'][0]+[1., -1.]
        original = roi_masks(scene, (0, 0), np.array([0]), scene['centers'][:1])[0, ..., 0]
        shifted = roi_masks(scene, (0, 0), np.array([0]), point[None])[0, ..., 0]
        np.testing.assert_array_equal(shifted[:-1, 1:], original[1:, :-1])

    def test_roi_anchor_matches_float32_decoder_near_half_pixel(self):
        point = np.array([[64.4999999, 64.4999999]])
        np.testing.assert_array_equal(roi_anchor(point), [[65, 65]])
        scene = self.scene()
        a = roi_masks(scene, (0, 0), np.array([0]), point)
        b = roi_masks(scene, (0, 0), np.array([0]), point.astype(np.float32))
        np.testing.assert_array_equal(a, b)

    def test_centers_offsets_and_halo(self):
        centers = np.array([[64.2, 64.3], [64.4, 68.1], [15.9, 32.]])
        heat, valid, selected, pixels, offsets = center_targets(centers)
        np.testing.assert_array_equal(selected, [0, 1])
        np.testing.assert_array_equal(pixels, [[64, 64], [64, 68]])
        np.testing.assert_allclose(offsets, [[.2, .3], [.4, .1]], atol=1e-6)
        self.assertEqual(heat[64, 64, 0], 1)
        self.assertEqual(valid[16, 32, 0], 0)
        self.assertEqual(valid[:16].sum(), 0)

    def test_ambiguous_same_pixel_centers_fail_explicitly(self):
        with self.assertRaises(ValueError):
            center_targets(np.array([[64.1, 64.1], [64.2, 64.2]]))

    def test_empty_batch_has_zero_weight_mask(self):
        heat, valid, selected, pixels, offsets = center_targets(np.empty((0, 2)))
        row = dict(images=np.zeros((128, 128, 1), np.float32), heatmap=heat, valid_region=valid,
                   offset_pixels=pixels, offset_targets=offsets, points=np.empty((0, 2), np.float32),
                   masks=np.zeros((0, ROI_SIZE, ROI_SIZE, 1), np.float32))
        data = batch_samples([row])
        self.assertEqual(data['offset_indices'].shape, (0, 3))
        self.assertEqual(data['mask_valid'].sum(), 0)
        self.assertEqual(data['masks'].shape, (1, ROI_SIZE, ROI_SIZE, 1))
        padded = batch_samples([row], mask_slots=8)
        self.assertEqual(padded['masks'].shape, (8, ROI_SIZE, ROI_SIZE, 1))
        self.assertEqual(padded['mask_valid'].sum(), 0)

    def test_cached_scene_targets_match_direct_rasterization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp);(directory/'train').mkdir()
            y, x = np.meshgrid(np.arange(16, 241, 16), np.arange(16, 241, 16), indexing='ij')
            centers = np.column_stack((y.ravel(), x.ravel())).astype(np.float32)+.2
            rng = np.random.default_rng(7);n = len(centers)
            scene = dict(image=rng.integers(90, 300, (256, 256), dtype=np.uint16), centers=centers,
                major_axis_px=rng.uniform(14, 18, n), minor_axis_px=rng.uniform(7, 10, n),
                orientation_deg=rng.uniform(-90, 90, n))
            np.savez_compressed(directory/'train/scene_00000.npz', **scene)
            source = InstanceSource(directory, 'train', 42, max_masks=16, jitter=.35)
            for index in [0, 1, 99, 10000]:
                row = source.sample(index)
                image, loaded, origin = source.source.crop(index)
                heat, valid, ids, pixels, offsets = center_targets(loaded['centers']-origin)
                np.testing.assert_array_equal(row['heatmap']*valid, heat*valid)
                np.testing.assert_array_equal(row['offset_pixels'], pixels)
                np.testing.assert_array_equal(row['offset_targets'], offsets)
                choice = np.random.default_rng(np.random.SeedSequence([42, index, 197]))
                if len(ids) > 16:
                    ids = choice.choice(ids, 16, replace=False)
                points = loaded['centers'][ids]-origin
                points += np.clip(choice.normal(0, .35, points.shape), -.75, .75)
                points = points.astype(np.float32)
                expected = roi_masks(loaded, origin, ids, points)
                np.testing.assert_array_equal(row['masks'], expected)


if __name__ == '__main__':
    unittest.main()
