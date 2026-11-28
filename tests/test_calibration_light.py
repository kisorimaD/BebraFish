"""Checks for relighting a crop of an already grown scene."""

import json
import unittest

import numpy as np

from calibration_lab.light import LightPreview
from training.plane20.generate_median import render_somas


class LightPreviewTests(unittest.TestCase):
    def test_saved_growth_can_relight_crops_that_exclude_neighboring_centers(self):
        shape = (48, 48)
        reference = np.full(shape, 120, dtype=np.uint16)
        allowed = np.ones(shape, dtype=bool)
        centers = np.array([[24.2, 13.3], [23.7, 20.2], [23.2, 27.1]])
        geometry = dict(major_axis_px=np.full(3, 10.), minor_axis_px=np.full(3, 8.),
                        orientation_deg=np.zeros(3))
        config = dict(grow_somas=True, peak=.8, width=.22, amplitude=1.)
        settings = dict(amplitude_cv=0.)
        seed = 12
        grown = render_somas(allowed, centers, geometry, config, settings, seed,
                             np.ones(3), np.zeros((3, 1, 2)))
        profile = dict(normalizer=1., neuron_support=allowed, median_config=json.dumps(config),
                       settings_json=json.dumps(dict(selected=settings)))
        scene = dict(image=reference, centers=centers, seed=seed, **geometry,
                     **{key: grown[key] for key in ('growth_reach', 'instance_offsets', 'instance_pixel_index')})
        preview = LightPreview(reference, profile, scene, (centers, *geometry.values()), [], 'fixed')
        changed = config | {'width': .5}
        # The center at x=13 lies outside the padded crop starting at x=14,
        # but its broad light still contributes inside the crop.
        crop = (20, 22, 8, 6)
        full = preview.light((0, 0, *shape), changed)
        actual = preview.light(crop, changed)
        np.testing.assert_allclose(actual, full[20:28, 22:28], rtol=1e-6, atol=1e-6)
        result = preview.render(dict(crop=list(crop), settings={'width': .5}))
        self.assertNotEqual(result['images'][1], result['images'][2])
