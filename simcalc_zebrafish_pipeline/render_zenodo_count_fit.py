"""Render closed-gap Zenodo masks before and after matching the reference counts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from multiprocessing import get_context
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree
import tifffile

from .close_contact_gaps import close_dense_gaps, load_gap_fill
from .generate_growth_field import _tile_scene
from .generate_volume import write_ome
from .generate_zenodo import _calibrate, _comparisons, read_reference
from .growth_photometry import GrowthOptics, sample_fluorescence
from .soma_growth import Masks, load_growth, rasterize, save_masks
from .soma_id_matching import fit_id_map, remap_ids
from .volume import Acquisition
from .zenodo_contrast import (
    label_image, match_reference_contrast, refine_local_neighbor_contrast, refine_reference_texture,
)


def _init_tiles(*state):
    global _TILE_STATE
    _TILE_STATE = state


def _render_tile(bounds):
    (scene, lower_xy, upper_xy, pixel, support, depths, fixed_z, optics, gain,
     owner_ids, group_ids, group_gain, render_unmatched) = _TILE_STATE
    core, end, start, stop = bounds
    lo, hi = start * pixel, stop * pixel
    cells = np.all((upper_xy >= lo) & (lower_xy <= hi), axis=1)
    if not cells.any():
        return core, end, None
    tile = _tile_scene(scene, lo, hi, cells)
    pitch = optics.internal_pitch_um
    masks = rasterize(tile, fixed_z, (pitch, pitch))
    masks = close_dense_gaps(tile, masks, support, yx_origin_um=lo)
    matched = remap_ids(masks, scene.somata.cell_id, owner_ids)
    in_tile = group_gain[np.searchsorted(group_ids, matched.cell_id)]
    camera_shape = tuple(stop - start)
    section = (slice(None), *(slice(a, b) for a, b in zip(core-start, end-start)))
    variants = [(matched, in_tile)]
    if render_unmatched:
        variants.insert(0, (masks, gain[cells]))
    frames = [
        sample_fluorescence(m, depths, pixel, camera_shape, optics, g)[section].copy()
        for m, g in variants
    ]
    return core, end, frames


def _render(scene, support, depths, pixel, optics, gain, owner_ids, workers, *, render_unmatched=True):
    group_ids, inverse = np.unique(owner_ids, return_inverse=True)
    volumes = scene.volumes_um3[-1]
    group_gain = (np.bincount(inverse, weights=gain * volumes) / np.bincount(inverse, weights=volumes))
    pixel = np.asarray(pixel)
    pitch = optics.internal_pitch_um
    if not np.allclose(pixel / pitch, np.rint(pixel / pitch)):
        raise ValueError('Camera pixels must be integer multiples of the internal optical grid')
    nz = int(np.floor(scene.tissue_zyx_um[0] / pitch + .5))
    fixed_z = (np.arange(nz) + .5) * pitch
    verts = scene.vertices_zyx_um[-1, :, :, 1:]
    state = (scene, verts.min(axis=1), verts.max(axis=1), pixel, support, depths, fixed_z, optics,
             gain, owner_ids, group_ids, group_gain, render_unmatched)
    shape = np.rint(scene.tissue_zyx_um[1:] / pixel).astype(int)
    halo_um = (support.max_distance_um + 4 * max(optics.psf_sigma_zyx_um[1:])
               + 4 * optics.shell_sigma_um + optics.shell_inset_um + 3 * pitch)
    halo = np.ceil(halo_um / pixel).astype(int)
    tasks = []
    tile_size = 192
    for y in range(0, shape[0], tile_size):
        for x in range(0, shape[1], tile_size):
            core = np.array([y, x])
            end = np.minimum(core + tile_size, shape)
            tasks.append((core, end, np.maximum(core - halo, 0), np.minimum(end + halo, shape)))
    stack = np.full((1 + int(render_unmatched), len(depths), *shape), optics.background_adu, dtype=np.float32)

    def collect(results):
        for index, (core, end, frames) in enumerate(results, 1):
            if frames is not None:
                region = tuple(slice(a, b) for a, b in zip(core, end))
                stack[(slice(None), slice(None), *region)] = frames
            if index % 5 == 0 or index == len(tasks):
                print(f'Rendered {index}/{len(tasks)} tiles', flush=True)

    if workers == 1:
        _init_tiles(*state)
        collect(map(_render_tile, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'),
                                 initializer=_init_tiles, initargs=state) as pool:
            collect(pool.map(_render_tile, tasks))
    return stack


def run(source, output, workers, *, refine=False, compare_to=None):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if workers < 1:
        raise ValueError('Expected at least one render worker')
    previous = None if compare_to is None else tifffile.imread(compare_to / 'image.ome.tif')
    input_meta = json.loads((source / 'summary.json').read_text())
    scene_path = Path(input_meta['scene'])
    reference_path = Path(input_meta['source_reference'])
    source_settings = json.loads((scene_path.parent / 'summary.json').read_text())
    real, reference, pixel, _, _, borders = read_reference(reference_path)
    acquisition = Acquisition(**source_settings['acquisition'])
    depths = acquisition.depths()
    support_path = source / 'dense_gap_fill.npz'
    support = load_gap_fill(support_path)
    with np.load(source / 'instances.npz', allow_pickle=False) as stored:
        masks = Masks(stored['cell_id'], tuple(stored['shape_zyx']), stored['z_positions_um'],
                      tuple(stored['pixel_yx_um']), stored['instance_offsets'], stored['instance_voxel_index'])
    if (masks.shape_zyx != real.shape or not np.array_equal(masks.z_positions_um, depths)
            or not np.array_equal(masks.pixel_yx_um, pixel)):
        raise ValueError('Reference and closed-gap masks must share the acquisition grid')
    scene = load_growth(scene_path)
    if previous is not None and previous.shape != real.shape:
        raise ValueError('The comparison render must use the same acquisition grid')
    if not np.array_equal(scene.somata.cell_id, masks.cell_id):
        raise ValueError('Saved mesh catalog must match the input mask IDs')
    with np.load(scene_path.parent / 'emission.npz', allow_pickle=False) as emission:
        if not np.array_equal(emission['cell_id'], scene.somata.cell_id):
            raise ValueError('Saved illumination gains must match the soma IDs')
        gain = emission['gain']
    targets = [np.unique(p[p > 0]).size for p in reference]
    dense = support.density_zyx >= support.min_density
    owner_ids = fit_id_map(masks, targets, reference=reference, dense=dense, reduce_excess=float(refine))
    matched = remap_ids(masks, masks.cell_id, owner_ids)
    label_stacks = [label_image(m.shape_zyx, m.instance_voxel_index, m.instance_offsets, m.cell_id)
                    for m in (masks, matched)]
    counts = [[np.unique(p[p > 0]).size for p in labels] for labels in label_stacks]
    protected = (label_stacks[0] > 0) & (dense if refine else True)
    if counts[1] != targets or np.any(label_stacks[1][protected] == 0):
        raise ValueError('The matching must reach all target counts and preserve protected pixels')
    print(f'Visible somata before: {counts[0]}. Matched to Cellpose: {counts[1]}', flush=True)
    optics = GrowthOptics(**source_settings['optics'])
    if refine:
        optics = replace(optics, psf_sigma_zyx_um=(optics.psf_sigma_zyx_um[0], .39, .39))
        centers = scene.centers_zyx_um[-1]
        _, near = cKDTree(centers).query(centers, k=min(len(centers), 11))
        log_gain = np.log(gain)
        local = np.mean(log_gain[near].reshape(len(gain), -1), axis=1)
        gain = np.exp(local + .55 * (log_gain - local)).astype(np.float32)
    render_unmatched = previous is None
    uncalibrated = _render(scene, support, depths, pixel, optics, gain, owner_ids, workers,
                          render_unmatched=render_unmatched)
    frames = []
    texture = None
    for clean, labels in zip(uncalibrated, label_stacks if render_unmatched else label_stacks[1:]):
        image, _ = _calibrate(real, reference, borders, clean, np.random.default_rng(42))
        image = match_reference_contrast(real, reference, image, labels, pixel, borders)
        if refine:
            image, texture = refine_reference_texture(real, reference, image, labels, pixel, borders, dense)
            image = refine_local_neighbor_contrast(real, reference, image, labels, pixel, borders, dense)
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        if render_unmatched:
            write_ome(pending / 'closed_gaps.ome.tif', frames[0], acquisition, pixel)
        write_ome(pending / 'image.ome.tif', frames[-1], acquisition, pixel)
        save_masks(pending / 'instances.npz', matched)
        np.savez_compressed(pending / 'soma_id_map.npz', source_cell_id=masks.cell_id, owner_cell_id=owner_ids)
        copyfile(support_path, pending / 'dense_gap_fill.npz')
        crop = _comparisons(pending, real, frames[-1], reference, depths,
                            before=frames[0] if render_unmatched else previous,
                            before_label='Щели закрыты' if render_unmatched else 'Прежний рендер',
                            after_label='Мягче и мельче' if refine else 'Число сом совпадает')
        report = dict(scene=str(scene_path), source=str(source), reference=str(reference_path),
                      acquisition=asdict(acquisition), pixel_yx_um=pixel, optics=asdict(optics),
                      reference_counts=targets, closed_gap_counts=counts[0], matched_counts=counts[1],
                      occupied_fraction=np.mean(label_stacks[1] > 0, axis=(1, 2)).tolist(),
                      reference_occupied_fraction=np.mean(reference > 0, axis=(1, 2)).tolist(),
                      removed_sparse_somas=int(np.count_nonzero(owner_ids == 0)), texture=texture,
                      local_neighbor_contrast=refine,
                      comparison_render=str(compare_to) if compare_to is not None else None,
                      crop_yx_size=crop)
        (pending / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--refine-appearance', action='store_true',
                        help='Reduce excess sparse coverage and fit the optical and neighbor contrast')
    parser.add_argument('--compare-to', type=Path, help='Compare against an existing render of the same acquisition')
    args = parser.parse_args()
    try:
        run(args.source, args.output, args.workers, refine=args.refine_appearance, compare_to=args.compare_to)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
