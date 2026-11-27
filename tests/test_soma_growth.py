"""Contact geometry and repeatable reslicing for the experimental mask generator."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from simcalc_zebrafish_pipeline.generate_growth_field import render_tiled
from simcalc_zebrafish_pipeline.growth_photometry import GrowthOptics, render_growth
from simcalc_zebrafish_pipeline.soma_growth import (
    GrowthConfig, Masks, Somata, grow, grow_to_shared_boundaries, load_growth, rasterize, save_growth,
)
from simcalc_zebrafish_pipeline.soma_id_matching import fit_id_map, remap_ids


class SomaGrowthTests(unittest.TestCase):
    def catalog(self, centers):
        n = len(centers)
        return Somata(np.asarray(centers), np.full(n, 220.), np.full(n, 1.22),
                      np.full(n, .85), np.zeros((n, 3)), np.full(n, .65),
                      np.arange(17, 17 + n, dtype=np.uint32))

    def test_single_soma_retains_the_free_shape(self):
        catalog = self.catalog([[6., 6., 6.]])
        growth = grow(catalog, (12., 12., 12.), deform=True, config=GrowthConfig(steps=3))
        relative = growth.vertices_zyx_um[:, 0] - catalog.centers_zyx_um[0]
        np.testing.assert_allclose(relative, growth.scale[:, None, None] * relative[-1])
        np.testing.assert_allclose(growth.volumes_um3[:, 0], 220. * growth.scale**3)
        np.testing.assert_array_equal(growth.centers_zyx_um,
                                      np.broadcast_to(catalog.centers_zyx_um, growth.centers_zyx_um.shape))

    def test_contact_flattens_facing_side_and_preserves_ids_when_resliced(self):
        catalog = self.catalog([[6., 6., 5.], [6., 6., 10.2]])
        tissue = (12., 12., 16.)
        config = GrowthConfig(steps=4)
        rigid = grow(catalog, tissue, deform=False, config=config)
        shaped = grow(catalog, tissue, deform=True, config=config)
        np.testing.assert_array_equal(rigid.centers_zyx_um, shaped.centers_zyx_um)
        np.testing.assert_allclose(shaped.volumes_um3[-1], catalog.target_volume_um3, rtol=.002)
        self.assertTrue(np.all((shaped.vertices_zyx_um >= 0) & (shaped.vertices_zyx_um <= tissue)))
        final = shaped.vertices_zyx_um[-1]
        self.assertGreaterEqual(final[1, :, 2].min() - final[0, :, 2].max(), config.clearance_um - 1e-8)
        self.assertLess(final[0, :, 2].max(), rigid.vertices_zyx_um[-1, 0, :, 2].max() - .4)
        self.assertLess(final[0, :, 2].min(), rigid.vertices_zyx_um[-1, 0, :, 2].min())
        z = np.arange(1., 12., .5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scene.npz'
            save_growth(path, shaped)
            saved = load_growth(path)
            fine = rasterize(saved, z, (.5, .5))
            coarse = rasterize(saved, z[::2], (.5, .5))
        np.testing.assert_array_equal(fine.cell_id, [17, 18])
        np.testing.assert_array_equal(coarse.cell_id, fine.cell_id)
        area = np.prod(fine.shape_zyx[1:])
        for i in range(len(fine.cell_id)):
            zf, yx = np.divmod(fine.instance_voxel_index[fine.instance_offsets[i]:fine.instance_offsets[i+1]], area)
            shared = zf % 2 == 0
            np.testing.assert_array_equal(
                coarse.instance_voxel_index[coarse.instance_offsets[i]:coarse.instance_offsets[i+1]],
                zf[shared] // 2 * area + yx[shared],
            )
        self.assertTrue(np.all(fine.occupancy() <= 1))
        self.assertTrue(np.any(rasterize(rigid, z, (.5, .5)).occupancy() > 1))

    def test_tissue_wall_can_deform_a_soma_with_fixed_center(self):
        catalog = self.catalog([[6., 6., 3.]])
        config = GrowthConfig(steps=4, max_shift_um=0.)
        growth = grow(catalog, (12., 12., 12.), deform=True, config=config)
        np.testing.assert_array_equal(growth.centers_zyx_um[-1], catalog.centers_zyx_um)
        self.assertAlmostEqual(growth.vertices_zyx_um[-1, 0, :, 2].min(), 0., places=9)
        np.testing.assert_allclose(growth.volumes_um3[-1], catalog.target_volume_um3, rtol=.002)

    def test_growth_to_shared_boundaries_closes_the_gap_without_overlap(self):
        catalog = self.catalog([[6., 6., 5.], [6., 6., 10.8]])
        config = GrowthConfig(steps=2, max_shift_um=0., subdivisions=1)
        base = grow(catalog, (12., 12., 16.), deform=True, config=config)
        filled = grow_to_shared_boundaries(base, [1.5, 1.5])
        np.testing.assert_array_equal(filled.centers_zyx_um[-1], base.centers_zyx_um[-1])
        np.testing.assert_array_equal(filled.somata.cell_id, base.somata.cell_id)
        self.assertTrue(np.all(np.diff(filled.volumes_um3, axis=0) > 0))
        final = filled.vertices_zyx_um[-1]
        gap = final[1, :, 2].min() - final[0, :, 2].max()
        self.assertGreaterEqual(gap, .008 - 1e-9)
        self.assertLess(gap, .02)
        self.assertLessEqual(rasterize(filled, [5., 6., 7.], (.3, .3)).occupancy().max(), 1)

    def test_fluorescence_beyond_mask_and_common_depths(self):
        catalog = self.catalog([[6., 6., 6.]])
        growth = grow(catalog, (12., 12., 12.), deform=True, config=GrowthConfig(steps=3))
        depths = np.array([2., 4., 6., 8., 10.])
        fine, masks = render_growth(growth, depths)
        coarse, _ = render_growth(growth, depths[::2])
        np.testing.assert_array_equal(coarse, fine[::2])
        self.assertFalse(masks.occupancy()[-1].any())
        self.assertGreater(fine[-1].max(), 120.)

    def test_tiled_fluorescence_matches_a_whole_field(self):
        catalog = self.catalog([[6., 6., 7.], [6., 6., 12.]])
        growth = grow(catalog, (12., 12., 20.), deform=True, config=GrowthConfig(steps=3))
        depths = np.array([4., 6., 8.])
        pixel = (.5, .5)
        optics = GrowthOptics(internal_pitch_um=.25)
        whole, _ = render_growth(growth, depths, pixel, optics=optics)
        tiled = render_tiled(growth, depths, pixel, optics, tile_size=10)
        np.testing.assert_allclose(tiled, whole, rtol=0., atol=2e-5)

    def test_count_matching_reassigns_touching_regions_without_creating_gaps(self):
        ids = np.array([11, 12, 21, 22, 23])
        masks = Masks(ids, (2, 2, 3), np.array([5., 15.]), (.315, .315),
                      np.array([0, 2, 5, 6, 8, 11]),
                      np.array([0, 3, 1, 2, 4, 6, 7, 10, 8, 9, 11]))
        owners = fit_id_map(masks, [1, 2])
        result = remap_ids(masks, ids, owners)
        np.testing.assert_array_equal(result.occupancy(), masks.occupancy())
        planes = result.instance_voxel_index // 6
        per_voxel = np.repeat(result.cell_id, np.diff(result.instance_offsets))
        self.assertEqual([len(np.unique(per_voxel[planes == i])) for i in range(2)], [1, 2])

    def test_sparse_cell_removal_preserves_dense_pixels(self):
        ids = np.array([11, 12, 13])
        masks = Masks(ids, (1, 2, 3), np.array([5.]), (.315, .315),
                      np.array([0, 2, 4, 6]), np.array([0, 3, 1, 4, 2, 5]))
        reference = np.array([[[1, 1, 0], [1, 1, 0]]])
        dense = reference > 0
        owners = fit_id_map(masks, [2], reference=reference, dense=dense, reduce_excess=1.)
        np.testing.assert_array_equal(owners, [11, 12, 0])
        result = remap_ids(masks, ids, owners)
        np.testing.assert_array_equal(result.cell_id, [11, 12])
        np.testing.assert_array_equal(result.occupancy() > 0, dense)
