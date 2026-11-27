"""Generate a 3D synthetic acquisition fitted to Figure 5l,m HDF5 data."""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import re
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
import tifffile

from .generate_growth_field import render_tiled
from .generate_volume import write_ome
from .growth_photometry import GrowthOptics
from .soma_growth import (
    GrowthConfig, Somata, fill_shared_boundaries, grow, grow_to_shared_boundaries,
    load_growth, rasterize, save_growth, save_masks,
)
from .volume import Acquisition
from .zenodo_contrast import label_image, match_reference_contrast


def read_reference(path):
    import h5py

    images, labels, positions, pitches, borders = [], [], [], [], []
    with h5py.File(path, 'r') as source:
        planes = sorted(k for k in source if re.fullmatch(r'repeat00_tile000_z\d+_950nm', k))
        step = float(source.attrs['z_scan_abs_dz'])
        for name in planes:
            plane = source[name]['preprocessed_data/fish00']
            image = plane['imaging_data_channel0_time_averaged']
            segmentation = plane['cellpose_segmentation']
            ids = segmentation['masks'][:]
            names = segmentation['unit_names'][:]
            images.append(image[:])
            # The export encodes the original background as 9999.
            labels.append(np.where(np.isin(ids, names), ids, 0))
            pitches.append((float(image.attrs['dy']), float(image.attrs['dx'])))
            positions.append(float(image.attrs['position_z']))
            borders.append(int(image.attrs['motion_correction_border_distance']))
    if not images or not np.isfinite(step) or step <= 0:
        raise ValueError('Expected imaging planes and a positive nominal Z step')
    if len(set(pitches)) != 1 or pitches[0][0] != pitches[0][1]:
        raise ValueError('Expected a common square camera pixel size')
    images, labels = np.stack(images), np.stack(labels)
    if images.shape != labels.shape or images.ndim != 3:
        raise ValueError('A mask must share the native image grid of its plane')
    return images, labels, pitches[0], step, np.asarray(positions), borders


def _shape_samples(labels):
    y, x = np.nonzero(labels)
    ids = labels[y, x]
    count = np.bincount(ids)
    keep = np.flatnonzero(count > 0)
    quantiles = np.percentile(count[keep], [5, 95])
    keep = keep[(count[keep] >= quantiles[0]) & (count[keep] <= quantiles[1])]
    area = count[keep]
    cy, cx = [np.bincount(ids, weights=coord)[keep] / area for coord in (y, x)]
    yy = np.bincount(ids, weights=y*y)[keep] / area - cy*cy
    xx = np.bincount(ids, weights=x*x)[keep] / area - cx*cx
    xy = np.bincount(ids, weights=x*y)[keep] / area - cy*cx
    eigenvalues = np.linalg.eigvalsh(np.stack((yy, xy, xy, xx), axis=-1).reshape(-1, 2, 2))
    aspect = np.sqrt(eigenvalues[:, 1] / np.maximum(.25, eigenvalues[:, 0]))
    return np.sqrt(4 * area / np.pi), np.clip(aspect, 1., 2.2)


def _seed_somata(labels, pixel, acquisition, rng, density_scale):
    dy, dx = pixel
    step = acquisition.z_step_um
    tissue = np.r_[len(labels) * step, np.array(labels.shape[1:]) * pixel]
    points, diameters, aspects = [], [], []
    for plane, depth in zip(labels, acquisition.depths()):
        occupancy = plane != 0
        probability = gaussian_filter(occupancy.astype(np.float32), 17.)
        probability[probability < .055] = 0.
        probability[:10] = probability[-10:] = 0.
        probability[:, :10] = probability[:, -10:] = 0.
        cell_count = np.unique(plane[occupancy]).size
        samples = rng.choice(plane.size, int(3.3 * cell_count * density_scale),
                             p=probability.ravel().astype(float) / probability.sum(dtype=float))
        y, x = np.unravel_index(samples, plane.shape)
        z = depth + rng.uniform(-step / 2, step / 2, len(samples))
        points.append(np.column_stack((z, (y + rng.random(len(y))) * dy, (x + rng.random(len(x))) * dx)))
        diam, aspect = _shape_samples(plane)
        drawn = rng.integers(len(diam), size=len(samples))
        # A random ellipsoid section has on average two thirds of the equatorial area.
        diameters.append(diam[drawn] * dx * np.sqrt(1.5))
        aspects.append(aspect[drawn])
    points = np.concatenate(points)
    diameter, aspect = np.concatenate(diameters), np.concatenate(aspects)
    axial_ratio = .88
    axes = np.column_stack((diameter * axial_ratio, diameter / np.sqrt(aspect),
                            diameter * np.sqrt(aspect))) / 2
    order = rng.permutation(len(points))
    wall = np.minimum(points, tissue - points).min(axis=1)
    allowed = wall > 1.07 * axes.max(axis=1)
    neighbors = cKDTree(points).query_ball_point(points, 2.05)
    kept = []
    for i in order:
        if allowed[i]:
            kept.append(i)
            allowed[neighbors[i]] = False
    if not kept:
        raise ValueError('No somata fit inside the scene')
    n = len(kept)
    angles = np.column_stack((rng.normal(0, .15, (n, 2)), rng.uniform(0, np.pi, n)))
    return Somata(points[kept], 4*np.pi/3 * np.prod(axes[kept], axis=1), aspect[kept],
                  np.full(n, axial_ratio), angles, np.full(n, .65), np.arange(1, n+1)), tissue


def _calibrate(real, labels, borders, fluorescence, rng):
    image, statistics = [], []
    for reference, segmentation, border, emission in zip(real, labels, borders, fluorescence):
        support = gaussian_filter((segmentation > 0).astype(np.float32), 17.) > .16
        reference = reference.astype(np.float32)
        envelope = gaussian_filter(reference, 24.)
        real_band = reference - envelope
        simulated_band = emission - gaussian_filter(emission, 24.)
        real_rms = np.std(real_band[support])
        synthetic_rms = np.std(simulated_band[support])
        if synthetic_rms <= 0:
            raise ValueError('Cannot calibrate an empty fluorescence plane')
        real_detail = reference - gaussian_filter(reference, 2.5)
        synthetic_detail = emission - gaussian_filter(emission, 2.5)
        real_detail_rms = np.std(real_detail[support])
        synthetic_detail_rms = np.std(synthetic_detail[support])
        gain = real_rms / synthetic_rms
        if synthetic_detail_rms > 0:
            gain = min(gain, real_detail_rms / synthetic_detail_rms)
        rendered = envelope + gain * simulated_band
        highpass = reference - gaussian_filter(reference, 1.)
        noise = .85 * np.std(highpass[support])
        rendered += gaussian_filter(rng.normal(0., noise, reference.shape), .55)
        if border:
            valid = np.zeros(reference.shape, dtype=bool)
            valid[border:-border, border:-border] = True
            rendered[~valid] = np.median(reference[~valid])
        image.append(rendered)
        rendered_band = rendered - gaussian_filter(rendered, 24.)
        statistics.append({'optical_gain_adu': float(gain), 'reference_band_rms_adu': float(real_rms),
                           'synthetic_band_rms_adu': float(np.std(rendered_band[support]))})
    return np.rint(np.clip(image, 0, 65535)).astype(np.uint16), statistics


def _comparisons(output, reference, synthetic, labels, depths, *, before=None,
                 before_label='До коррекции', after_label='После коррекции'):
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(reference, [5, 99.5])
    n = len(depths)
    size = min(128, *reference.shape[1:])
    coverage = gaussian_filter(np.mean(labels > 0, axis=0), size/3)
    y, x = np.unravel_index(np.argmax(coverage), coverage.shape)
    y, x = np.clip([y-size//2, x-size//2], 0, np.array(coverage.shape)-size)
    crop = np.s_[y:y+size, x:x+size]
    columns = [('Zenodo', reference), ('Синтетика', synthetic)]
    if before is not None:
        columns = [('Zenodo', reference), (before_label, before), (after_label, synthetic)]
    for filename, region in [('comparison_full.png', np.s_[:,:]), ('comparison_crops.png', crop)]:
        fig, axes = plt.subplots(n, len(columns), figsize=(4*len(columns), 3.5*n),
                                 squeeze=False, layout='constrained')
        for i, (row, depth) in enumerate(zip(axes, depths)):
            for ax, (name, frames) in zip(row, columns):
                ax.imshow(frames[i][region], cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
                ax.set_title(f'{name} · Z = {depth:g} мкм')
                ax.set_axis_off()
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
    return [int(y), int(x), int(size)]


def _local_fill_map(labels):
    return gaussian_filter((labels > 0).astype(np.float32), (0., 14., 14.))


def _local_fill_at(density, centers, pixel, step):
    coordinates = centers / np.r_[step, pixel] - .5
    return map_coordinates(density, coordinates.T, order=1, mode='nearest')


def _growth_radius_limit(local_fill):
    return np.interp(local_fill, [0., .30, .56, .85, 1.], [.76, .88, 1.15, 2.2, 2.2])


def _fill_dense_regions(scene, labels, pixel, step):
    density = _local_fill_map(labels)
    local_fill = _local_fill_at(density, scene.centers_zyx_um[-1], pixel, step)
    return grow_to_shared_boundaries(scene, _growth_radius_limit(local_fill))


def _replace_large_somata(scene, labels, pixel, step, split_largest_fraction=.25):
    centers = scene.centers_zyx_um[-1]
    volumes = scene.volumes_um3[-1]
    density = _local_fill_map(labels)
    local_fill = _local_fill_at(density, centers, pixel, step)
    eligible = local_fill >= .65
    if not eligible.any():
        return scene, 0
    cutoff = np.quantile(volumes[eligible], 1. - split_largest_fraction)
    large = np.flatnonzero(eligible & (volumes >= cutoff))
    radial = scene.vertices_zyx_um[-1, large] - centers[large, None, :]
    xy = radial[:, :, 1:]
    covariance = xy.transpose(0, 2, 1) @ xy
    major_axis = np.linalg.eigh(covariance)[1][:, :, -1]
    along = np.einsum('nvc,nc->nv', xy, major_axis)
    axial_penalty = .6 * np.abs(radial[:, :, 0])
    first = centers[large] + .52 * radial[np.arange(len(large)), (along - axial_penalty).argmax(axis=1)]
    second = centers[large] + .52 * radial[np.arange(len(large)), (along + axial_penalty).argmin(axis=1)]

    # Both seeds lie in the original mesh, preferably near the same axial level.
    neighbors = cKDTree(centers)
    keep = np.linalg.norm(first - second, axis=1) >= 1.8
    for point in (first, second):
        distance, index = neighbors.query(point, k=2)
        other_distance = np.where(index[:, 0] == large, distance[:, 1], distance[:, 0])
        keep &= other_distance > .95
    large, first, second = large[keep], first[keep], second[keep]
    old = scene.somata
    initial = old.centers_zyx_um.copy()
    initial[large] = first
    updated = centers.copy()
    updated[large] = first
    updated = np.concatenate((updated, second))
    volume, aspect = old.target_volume_um3.copy(), old.aspect_yx.copy()
    volume[large] *= .56
    aspect[large] = 1. + .7 * (aspect[large] - 1.)
    children = np.arange(int(old.cell_id.max()) + 1, int(old.cell_id.max()) + 1 + len(large))
    somata = Somata(np.concatenate((initial, second)), np.r_[volume, volume[large]],
                    np.r_[aspect, aspect[large]], np.r_[old.axial_ratio, old.axial_ratio[large]],
                    np.concatenate((old.angles_xyz_rad, old.angles_xyz_rad[large])),
                    np.r_[old.stiffness, old.stiffness[large]], np.r_[old.cell_id, children])
    local_fill = _local_fill_at(density, updated, pixel, step)
    result = fill_shared_boundaries(somata, scene.tissue_zyx_um, updated, _growth_radius_limit(local_fill),
                                    smoothing=.12)
    return result, len(large)


def run(reference, output, seed, density_scale, *, fill_contacts=False, base_scene=None, smaller_somata=False,
        split_largest_fraction=.25):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not np.isfinite(density_scale) or density_scale <= 0:
        raise ValueError('Density scale must be finite and positive')
    if smaller_somata and not fill_contacts:
        raise ValueError('The smaller somata variant requires --fill-contacts')
    if not 0. < split_largest_fraction <= 1.:
        raise ValueError('The fraction of large somata to repack must be in (0, 1]')
    real, labels, pixel, step, positions, borders = read_reference(reference)
    acquisition = Acquisition(len(real), step/2, step)
    depths = acquisition.depths()
    rng = np.random.default_rng(seed)
    somata, tissue = _seed_somata(labels, pixel, acquisition, rng, density_scale)
    volume_scale = 1.24
    somata = replace(somata, target_volume_um3=somata.target_volume_um3 * volume_scale)
    config = GrowthConfig(steps=4, relaxation_steps=5, subdivisions=1,
                          mobility=.5, max_shift_um=1.1, clearance_um=.06)
    if base_scene is None:
        print(f'Growing {len(somata.cell_id):,} somata', flush=True)
        scene = grow(somata, tissue, deform=True, config=config)
    else:
        scene = load_growth(base_scene)
        same_catalog = all(np.array_equal(getattr(scene.somata, field.name), getattr(somata, field.name))
                           for field in fields(Somata))
        if not same_catalog or not np.array_equal(scene.tissue_zyx_um, tissue):
            raise ValueError('Base scene must come from the same reference, seed and density')
    if fill_contacts:
        print('Growing to shared boundaries in dense tissue', flush=True)
        scene = _fill_dense_regions(scene, labels, pixel, step)
    split_count = 0
    if smaller_somata:
        scene, split_count = _replace_large_somata(scene, labels, pixel, step, split_largest_fraction)
        print(f'Repacked {split_count:,} large somata with an additional seed', flush=True)
    somata = scene.somata
    masks = rasterize(scene, depths, pixel)
    print('Rendering the Z stack with optical blur', flush=True)
    gains = np.clip(rng.lognormal(-.16, .48, len(somata.cell_id)), .28, 3.05).astype(np.float32)
    optics = GrowthOptics(internal_pitch_um=pixel[0], psf_sigma_zyx_um=(.70, .29, .29),
                          shell_inset_um=.14, shell_sigma_um=.18, amplitude_adu=1., background_adu=0.,
                          interior_fraction=.21)
    if smaller_somata:
        optics = replace(optics, shell_sigma_um=.14, interior_fraction=.46)
    fluorescence = render_tiled(scene, depths, pixel, optics, cell_gain=gains)
    image, statistics = _calibrate(real, labels, borders, fluorescence, rng)
    before_contrast = None
    if smaller_somata:
        before_contrast = image
        synthetic_labels = label_image(masks.shape_zyx, masks.instance_voxel_index, masks.instance_offsets,
                                       masks.cell_id)
        image = match_reference_contrast(real, labels, image, synthetic_labels, pixel, borders)
        for stats in statistics:
            stats['synthetic_band_rms_before_contrast_adu'] = stats.pop('synthetic_band_rms_adu')
    occupied = masks.occupancy()
    plane_numbers = masks.instance_voxel_index // np.prod(masks.shape_zyx[1:])
    voxel_ids = np.repeat(masks.cell_id, np.diff(masks.instance_offsets))
    for i, stats in enumerate(statistics):
        stats.update(reference_count=int(np.unique(labels[i][labels[i] > 0]).size),
                     synthetic_count=int(np.unique(voxel_ids[plane_numbers == i]).size),
                     reference_coverage=float(np.mean(labels[i] > 0)),
                     synthetic_coverage=float(np.mean(occupied[i] > 0)),
                     overlapping_pixels=int(np.count_nonzero(occupied[i] > 1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        save_growth(pending / 'scene.npz', scene)
        save_masks(pending / 'instances.npz', masks)
        write_ome(pending / 'image.ome.tif', image, acquisition, pixel)
        np.savez_compressed(pending / 'emission.npz', cell_id=somata.cell_id, gain=gains)
        crop = _comparisons(pending, real, image, labels, depths, before=before_contrast)
        report = dict(reference=str(reference), seed=seed, density_scale=density_scale,
                      source_stage_z_um=positions.tolist(), acquisition=asdict(acquisition),
                      pixel_yx_um=pixel, shape_zyx=list(image.shape), cells_3d=len(somata.cell_id),
                      growth=asdict(config), target_volume_scale=volume_scale,
                      fill_contacts=fill_contacts, base_scene=str(base_scene) if base_scene is not None else None,
                      smaller_somata=smaller_somata, replaced_large_somata=split_count,
                      split_largest_fraction=split_largest_fraction if smaller_somata else None,
                      contrast_calibration=smaller_somata,
                      mesh_vertices_per_soma=scene.vertices_zyx_um.shape[-2],
                      optics=asdict(optics), crop_yx_size=crop,
                      background='Source reference lowpass, sigma 24 native pixels',
                      per_plane=statistics)
        (pending / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
    print(json.dumps(report, indent=2))


def recalibrate_render(reference, source, output):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    saved = json.loads((source / 'summary.json').read_text())
    if saved.get('contrast_calibration'):
        raise ValueError('This render already has local contrast calibration')
    if Path(saved['reference']).resolve() != reference.resolve():
        raise ValueError('Use the reference HDF5 for this render')
    real, labels, pixel, _, _, borders = read_reference(reference)
    acquisition = Acquisition(**saved['acquisition'])
    depths = acquisition.depths()
    original = tifffile.imread(source / 'image.ome.tif')
    masks_path = source / 'instances.npz'
    with np.load(masks_path, allow_pickle=False) as masks:
        if not np.array_equal(masks['z_positions_um'], depths) or not np.array_equal(masks['pixel_yx_um'], pixel):
            raise ValueError('Saved masks do not match the acquisition grid')
        synthetic_labels = label_image(masks['shape_zyx'], masks['instance_voxel_index'],
                                       masks['instance_offsets'], masks['cell_id'])
    image = match_reference_contrast(real, labels, original, synthetic_labels, pixel, borders)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        write_ome(pending / 'image.ome.tif', image, acquisition, pixel)
        copyfile(masks_path, pending / 'instances.npz')
        crop = _comparisons(pending, real, image, labels, depths, before=original)
        report = dict(reference=str(reference), source_render=str(source), scene=str(source / 'scene.npz'),
                      acquisition=asdict(acquisition), pixel_yx_um=pixel, shape_zyx=list(image.shape),
                      crop_yx_size=crop, contrast_calibration=True)
        (pending / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--density-scale', type=float, default=1.)
    parser.add_argument('--fill-contacts', action='store_true', help='Grow to shared boundaries in dense tissue')
    parser.add_argument('--smaller-somata', action='store_true',
                        help='Repack the largest dense cells with additional seeds and thinner, filled shells')
    parser.add_argument('--split-largest-fraction', type=float, default=.25,
                        help='Fraction of large dense cells considered for extra seeds with --smaller-somata')
    parser.add_argument('--base-scene', type=Path, help='Reuse the relaxed centers of a saved scene for the same catalog')
    parser.add_argument('--match-render', type=Path, help='Correct the contrast of a saved Zenodo render')
    args = parser.parse_args()
    try:
        if args.match_render is not None:
            recalibrate_render(args.reference, args.match_render, args.output)
        else:
            run(args.reference, args.output, args.seed, args.density_scale,
                fill_contacts=args.fill_contacts, base_scene=args.base_scene, smaller_somata=args.smaller_somata,
                split_largest_fraction=args.split_largest_fraction)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
