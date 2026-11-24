"""Volume contracts across acquisition grids and on-disk representations."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
import tifffile

from simcalc_zebrafish_pipeline.generate_volume import write_stack
from simcalc_zebrafish_pipeline.volume import (
    CELL_DTYPE,
    PROFILES,
    Acquisition,
    Profile,
    Scene,
    TissueMaps,
    render_stack,
    sample_scene,
)


def read_stack(path):
    with np.load(path / 'instances.npz', allow_pickle=False) as data:
        gt = dict(data)
    image = tifffile.imread(path / 'image.ome.tif').reshape(gt['shape_zyx'])
    return gt, image, np.load(path / 'clean_adu.npy', allow_pickle=False)


def masks_at(gt, k):
    _, height, width = gt['shape_zyx']
    plane_area = height * width
    masks = {}
    for cell_id, start, stop in zip(
            gt['cells']['cell_id'], gt['instance_offsets'][:-1], gt['instance_offsets'][1:]):
        z, yx = np.divmod(gt['instance_voxel_index'][start:stop], plane_area)
        masks[cell_id] = yx[z == k]
    return masks


class VolumeTest(unittest.TestCase):
    def test_common_depths_after_disk_roundtrip(self):
        profile = PROFILES['demo']
        scene = sample_scene(profile, 42)
        np.testing.assert_array_equal(scene.cells, sample_scene(profile, 42).cells)
        with TemporaryDirectory() as tmp:
            coarse_path, fine_path = Path(tmp) / 'coarse', Path(tmp) / 'fine'
            for acquisition, path in ((Acquisition(5), coarse_path), (Acquisition(9, 0., 5.), fine_path)):
                write_stack(path, scene, acquisition, render_stack(scene, acquisition))
            coarse, image, clean = read_stack(coarse_path)
            fine, fine_image, fine_clean = read_stack(fine_path)
            with tifffile.TiffFile(coarse_path / 'image.ome.tif') as tif:
                pixels = ElementTree.fromstring(tif.ome_metadata).find('.//{*}Pixels')
                self.assertEqual(tif.series[0].axes, 'ZYX')
                self.assertEqual([float(pixels.attrib['PhysicalSize' + axis]) for axis in 'ZYX'],
                                 [10., .576, .576])
                self.assertEqual([float(plane.attrib['PositionZ']) for plane in pixels.findall('{*}Plane')],
                                 [0., 10., 20., 30., 40.])
        self.assertEqual(tuple(coarse['shape_zyx']), (5, 128, 128))
        self.assertEqual(tuple(fine['shape_zyx']), (9, 128, 128))
        self.assertEqual(image.dtype, np.uint16)
        self.assertEqual(json.loads(str(coarse['metadata_json']))['seed'], 42)
        np.testing.assert_array_equal(coarse['cells'], fine['cells'])
        np.testing.assert_array_equal(coarse['z_positions_um'], fine['z_positions_um'][::2])
        np.testing.assert_array_equal(image, fine_image[::2])
        np.testing.assert_allclose(clean, fine_clean[::2], atol=1e-6, rtol=0)
        np.testing.assert_allclose(coarse['cell_peak_adu'], fine['cell_peak_adu'][:, ::2], atol=1e-6, rtol=0)
        self.assertGreater(len(coarse['instance_voxel_index']), len(np.unique(coarse['instance_voxel_index'])))
        for k in range(5):
            first, second = masks_at(coarse, k), masks_at(fine, 2 * k)
            self.assertEqual(first.keys(), second.keys())
            for cell_id in first:
                np.testing.assert_array_equal(first[cell_id], second[cell_id])

    def test_ellipsoid_section_and_psf(self):
        profile = Profile(shape_yx=(32, 32), pixel_yx_um=(.5, .5), depth_range_um=(0., 10.))
        cells = np.zeros(2, dtype=CELL_DTYPE)
        cells['cell_id'] = [1001, 1002]
        cells['center_zyx_um'] = [(1., 8., 8.), (-1., 8., 8.)]
        cells['semiaxes_zyx_um'] = (2., 2., 3.)
        cells['shell_peak_radius'], cells['shell_width'], cells['amplitude_adu'] = .78, .12, 150.
        scene = Scene(profile, 1, cells, np.zeros(7))
        acquisition = Acquisition(3, 0., 2.)
        rendered = render_stack(scene, acquisition)
        y, x = np.ogrid[:32, :32]
        expected = np.flatnonzero((((y + .5) * .5 - 8) / 2)**2
                                  + (((x + .5) * .5 - 8) / 3)**2 <= .75)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'volume'
            write_stack(path, scene, acquisition, rendered)
            gt, _, _ = read_stack(path)
            for cell_id in (1001, 1002):
                np.testing.assert_array_equal(masks_at(gt, 0)[cell_id], expected)
            np.testing.assert_array_equal(masks_at(gt, 1)[1001], expected)
            self.assertEqual(len(masks_at(gt, 2)[1001]), 0)
            self.assertGreater(gt['cell_peak_adu'][0, 2], 0.)
            self.assertAlmostEqual(len(expected) * .25, np.pi * 2 * 3 * .75, delta=1.3)
            singleton = Acquisition(1, 4., 50.)
            single = render_stack(scene, singleton)
            self.assertEqual(single.clean_adu.shape, (1, 32, 32))
            self.assertEqual(single.instance_voxel_index.size, 0)
            write_stack(Path(tmp) / 'single', scene, singleton, single)
            single_gt, image, _ = read_stack(Path(tmp) / 'single')
            self.assertEqual(image.shape, (1, 32, 32))
            np.testing.assert_array_equal(single_gt['z_positions_um'], [4.])

    def test_invalid_acquisition_and_shape(self):
        scene = sample_scene(PROFILES['demo'], 2)
        for acquisition in (Acquisition(0), Acquisition(5, 0., 0.), Acquisition(1, float('nan')),
                            Acquisition(5, 0., 11.)):
            with self.assertRaises(ValueError):
                render_stack(scene, acquisition)
        with self.assertRaisesRegex(ValueError, 'shape_yx'):
            sample_scene(Profile(shape_yx=(0, 13)), 2)

    def test_decimal_depths_with_shifted_start(self):
        np.testing.assert_array_equal(Acquisition(3, .3, .3).depths(),
                                      Acquisition(10, 0., .1).depths()[3::3])

    def test_median_texture_and_background_on_common_depths(self):
        profile = replace(PROFILES['demo'], shape_yx=(32, 32), texture_band_rms_adu=5.,
                          background_floor_adu=95., median_style='structured', depth_margin_um=27.,
                          emission_model='projected')
        depths = Acquisition(5).depths()
        support = np.ones((len(depths), *profile.shape_yx), dtype=np.float32)
        maps = TissueMaps(depths, support, support, support * 130.)
        scene = sample_scene(profile, 42, tissue_maps=maps)
        coarse = render_stack(scene, Acquisition(5))
        fine = render_stack(scene, Acquisition(9, 0., 5.))
        np.testing.assert_array_equal(coarse.clean_adu, fine.clean_adu[::2])
        np.testing.assert_array_equal(coarse.cell_peak_adu, fine.cell_peak_adu[:, ::2])
        self.assertTrue(np.all(coarse.clean_adu >= profile.background_floor_adu))
        ellipsoid = render_stack(replace(scene, profile=replace(profile, emission_model='ellipsoid')),
                                 Acquisition(5))
        np.testing.assert_array_equal(coarse.instance_offsets, ellipsoid.instance_offsets)
        np.testing.assert_array_equal(coarse.instance_voxel_index, ellipsoid.instance_voxel_index)
        self.assertGreater(float(np.std(coarse.clean_adu - ellipsoid.clean_adu)), 1.)
        no_texture = render_stack(replace(scene, texture=None), Acquisition(5))
        self.assertGreater(float(np.std(coarse.clean_adu - no_texture.clean_adu)), 1.)
