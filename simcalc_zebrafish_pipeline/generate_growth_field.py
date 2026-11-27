"""Grow and render soma meshes over the full Plane20 TIFF field."""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import numpy as np

from .generate_volume import write_ome
from .growth_photometry import GrowthOptics, render_growth
from .median_photometry import median_background
from .soma_growth import Growth, GrowthConfig, Somata, grow, rasterize, save_growth, save_masks
from .volume import Acquisition, sample_scene
from .volume_reference import read_reference_profile


SLAB_DEPTH_UM = 18.432
PLANE20_Z_UM = SLAB_DEPTH_UM / 2
MODES = ('shift_only', 'deform')


def _prepare(reference, seed, density_per_um3):
    source, maps, _ = read_reference_profile(reference)
    floor = source.background_floor_adu
    dy, dx = source.pixel_yx_um
    if dy != dx:
        raise ValueError('The full-field optical grid currently requires equal TIFF XY pitches')
    tissue = np.array([SLAB_DEPTH_UM, source.shape_yx[0] * dy, source.shape_yx[1] * dx])
    reference_offset = 20. - PLANE20_Z_UM
    # Leave seed margins above and below the optical slab.
    source = replace(source, depth_range_um=(reference_offset + 3.4, reference_offset + tissue[0] - 3.4),
                     depth_margin_um=.1, diameter_xy_um=5.2, diameter_z_um=4.8,
                     density_per_um3=density_per_um3, min_center_distance_um=2.3, texture_band_rms_adu=0.)
    catalog = sample_scene(source, seed, tissue_maps=maps).cells
    centers = catalog['center_zyx_um'] - [reference_offset, 0., 0.]
    axes = catalog['semiaxes_zyx_um']
    radius = 1.06 * axes.max(axis=1)
    keep = np.all(np.minimum(centers, tissue - centers) > radius[:, None], axis=1)
    if not keep.any():
        raise ValueError('No somata fit inside the optical slab')
    centers, axes, catalog = centers[keep], axes[keep], catalog[keep]
    axial_ratio = axes[:, 0] / np.sqrt(axes[:, 1] * axes[:, 2])
    aspect = axes[:, 2] / axes[:, 1]
    angles = np.column_stack((np.zeros((len(centers), 2)), catalog['angle_rad']))
    somata = Somata(centers, 4 * np.pi / 3 * np.prod(axes, axis=1), aspect, axial_ratio, angles,
                    np.full(len(centers), .65), catalog['cell_id'])
    return somata, tissue, (dy, dx), floor


def _tile_scene(scene, lo, hi, selected):
    shift = np.r_[0., lo]
    source = scene.somata
    somata = Somata(*(getattr(source, field.name)[selected] for field in fields(source)))
    somata = replace(somata, centers_zyx_um=somata.centers_zyx_um - shift)
    return Growth(somata, np.r_[scene.tissue_zyx_um[0], hi-lo], scene.faces, scene.scale[-1:],
                  scene.centers_zyx_um[-1:, selected] - shift,
                  scene.vertices_zyx_um[-1:, selected] - shift, scene.volumes_um3[-1:, selected])


def render_tiled(scene, depths, pixel, optics, *, tile_size=192, cell_gain=None):
    """Rasterize optical tiles with halos and stitch their disjoint camera regions."""
    lo = scene.vertices_zyx_um[-1, :, :, 1:].min(axis=1)
    hi = scene.vertices_zyx_um[-1, :, :, 1:].max(axis=1)
    pitch = np.asarray(pixel)
    if not np.allclose(pitch / optics.internal_pitch_um, np.rint(pitch / optics.internal_pitch_um)):
        raise ValueError('Camera pixels must be integer multiples of the optical sampling pitch')
    height, width = np.rint(scene.tissue_zyx_um[1:] / pitch).astype(int)
    if cell_gain is not None:
        cell_gain = np.asarray(cell_gain)
        if cell_gain.shape != scene.somata.cell_id.shape:
            raise ValueError('Expected one emission gain per soma')
    halo_um = (4 * max(optics.psf_sigma_zyx_um[1:]) + 4 * optics.shell_sigma_um
               + optics.shell_inset_um + 3 * optics.internal_pitch_um)
    halo = np.ceil(halo_um / pitch).astype(int)
    signal = np.full((len(depths), height, width), optics.background_adu, dtype=np.float32)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            core = np.array([y, x])
            end = np.minimum(core + tile_size, (height, width))
            start = np.maximum(core - halo, 0)
            stop = np.minimum(end + halo, (height, width))
            lower, upper = start * pitch, stop * pitch
            selected = np.all((hi >= lower) & (lo <= upper), axis=1)
            if selected.any():
                tile = _tile_scene(scene, lower, upper, selected)
                tile_gain = None if cell_gain is None else cell_gain[selected]
                image, _ = render_growth(tile, depths, pixel, optics=optics, cell_gain=tile_gain)
                source = tuple(slice(a, b) for a, b in zip(core - start, end - start))
                target = tuple(slice(a, b) for a, b in zip(core, end))
                signal[(slice(None), *target)] = image[(slice(None), *source)]
    return signal


def _write_geometry(folder, scene, depths, pixel):
    folder.mkdir()
    save_growth(folder / 'scene.npz', scene)
    masks = rasterize(scene, depths, pixel)
    save_masks(folder / 'instances.npz', masks)
    occupied = masks.occupancy()
    summary = {
        'plane_intersecting_somas': (np.diff(masks.instance_offsets) > 0).sum().item(),
        'covered_fraction': float(np.mean(occupied > 0)),
        'overlapping_pixels': int(np.count_nonzero(occupied > 1)),
        'volume_median_abs_error_pct': float(100 * np.median(np.abs(
            scene.volumes_um3[-1] / scene.somata.target_volume_um3 - 1))),
    }
    return summary


def _preview(output, panels, real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(real, [2, 99])
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 9), layout='constrained')
    for ax, (title, signal) in zip(axes, panels):
        im = ax.imshow(signal.T, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        ax.set_title(title, fontsize=12, loc='left')
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=.8, label='ADU')
    fig.savefig(output / 'plane20_full.png', dpi=145)
    plt.close(fig)


def run(reference, output, seed, *, density_per_um3=.008, geometry_only=False, target_volume_scale=1.):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not np.isfinite(target_volume_scale) or target_volume_scale <= 0:
        raise ValueError('Target volume scale must be finite and positive')
    somata, tissue, pixel, floor = _prepare(reference, seed, density_per_um3)
    somata = replace(somata, target_volume_um3=somata.target_volume_um3 * target_volume_scale)
    config = GrowthConfig(steps=5, subdivisions=1, relaxation_steps=5)
    optics = GrowthOptics(internal_pitch_um=pixel[0] / 2, shell_sigma_um=.24, amplitude_adu=45.)
    if not geometry_only:
        with np.load(reference, allow_pickle=False) as source:
            envelope = source['background_adu'][2]
            real = source['median_adu'][2]
    output.mkdir(parents=True)
    data = {'seed': seed, 'n_somas_in_slab': len(somata.cell_id), 'tissue_zyx_um': tissue.tolist(),
            'plane20_scene_z_um': PLANE20_Z_UM, 'reference_z_um': 20.,
            'density_per_um3': density_per_um3, 'target_volume_scale': target_volume_scale,
            'config': asdict(config), 'modes': {}}
    panels = [] if geometry_only else [('Реальный Plane20 · временная медиана', real)]
    labels = dict(zip(MODES, ('Рост, только сдвиг центров', 'Рост со сдвигом и деформацией')))
    acquisition = Acquisition(1, PLANE20_Z_UM)
    depths = acquisition.depths()
    for mode in ('deform',) if geometry_only else MODES:
        print(f'{mode}: рост {len(somata.cell_id)} сом в общем объёме', flush=True)
        growth = grow(somata, tissue, deform=mode == 'deform', config=config)
        folder = output / mode
        data['modes'][mode] = _write_geometry(folder, growth, depths, pixel)
        print(mode, data['modes'][mode], flush=True)
        if not geometry_only:
            print(f'{mode}: растровка свечения полного поля', flush=True)
            foreground = render_tiled(growth, depths, pixel, optics) - optics.background_adu
            signal = foreground + median_background(envelope, foreground[0], floor)
            image = np.rint(np.clip(signal, 0, 65535)).astype(np.uint16)
            write_ome(folder / 'image.ome.tif', image, acquisition, pixel)
            np.save(folder / 'clean_adu.npy', signal, allow_pickle=False)
            panels.append((labels[mode], signal[0]))
    if not geometry_only:
        data['optics'] = asdict(optics)
        _preview(output, panels, real)
    (output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--density-per-um3', default=.008, type=float)
    parser.add_argument('--target-volume-scale', default=1., type=float,
                        help='Multiply target soma volumes after sampling the seed catalog')
    parser.add_argument('--geometry-only', action='store_true', help='Save grown meshes and masks for the median renderer')
    args = parser.parse_args()
    try:
        run(args.reference, args.output, args.seed, density_per_um3=args.density_per_um3,
            geometry_only=args.geometry_only, target_volume_scale=args.target_volume_scale)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
