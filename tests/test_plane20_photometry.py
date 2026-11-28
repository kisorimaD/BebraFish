"""Regression checks for static brightness-dependent noise."""
import unittest

import numpy as np

from training.plane20.photometry import (blend_reference_background, reference_background_weights,
                                         render_intensity_noise, spatial_noise, temporal_noise)


class StaticPhotometryTests(unittest.TestCase):
    def test_multiscale_seam_preserves_pixels_and_softens_brightness_step(self):
        support = np.zeros((48, 96), dtype=bool)
        support[:, :32] = True
        render = np.full(support.shape, 100., dtype=np.float32)
        reference = np.full_like(render, 160)
        reference[:, ::2] += 8
        weights = reference_background_weights(support)
        output = blend_reference_background(render, reference, weights)
        exact = (weights[0] == 1) & (weights[1] == 1)
        self.assertTrue(exact.any())
        np.testing.assert_array_equal(output[support], render[support])
        np.testing.assert_array_equal(output[exact], reference[exact])
        self.assertLess(np.max(np.abs(np.diff(output[:, 30:37], axis=1))), 6)

    def test_variance_tracks_the_render_inside_and_reference_outside(self):
        background = np.full((256, 384), 100.)
        envelope = np.full_like(background, 125.)
        cells = np.zeros_like(background)
        cells[:, :128] = 50.
        cells[:, 128:256] = 100.
        mask = cells > 0
        curve = dict(levels_adu=[100, 125, 150, 200], sigma_adu=[1, 2, 4, 8])
        frame = render_intensity_noise(background, cells, envelope, mask, curve, 12)
        noise = frame - background - cells
        np.testing.assert_allclose([band.std() for band in np.split(noise, 3, axis=1)],
                                   [4, 8, 2], rtol=.04)
        np.testing.assert_array_equal(
            frame, render_intensity_noise(background, cells, envelope, mask, curve, 12))

    def test_spatial_noise_accepts_a_constant_image(self):
        curve = spatial_noise(np.full((20, 20), 100, dtype=np.uint16))
        self.assertEqual(curve['levels_adu'], [100.])
        self.assertEqual(curve['sigma_adu'], [0.])

    def test_temporal_noise_uses_triplets_and_scales_to_median(self):
        class Page:
            def __init__(self, image):
                self.image = image

            def asarray(self):
                return self.image

        rng = np.random.default_rng(17)
        pages = [Page(100 + 2*i + rng.normal(0, 8, (32, 32))) for i in range(36)]
        curve = temporal_noise(pages, (32, 32), median_frames=36)
        self.assertAlmostEqual(np.median(curve['raw_frame_sigma_adu']), 8, delta=1)
        np.testing.assert_allclose(curve['sigma_adu'],
                                   np.asarray(curve['raw_frame_sigma_adu'])*np.sqrt(np.pi/72))
        self.assertIsNone(temporal_noise(pages[:1], (32, 32), median_frames=1))
