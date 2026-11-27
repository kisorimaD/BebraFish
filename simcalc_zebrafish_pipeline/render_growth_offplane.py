"""Render Plane20-style sections with a constant tissue background and off-plane soma light."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter

from .generate_growth_field import _tile_scene, render_tiled
from .generate_volume import write_ome
from .growth_photometry import GrowthOptics, render_growth
from .render_growth_plane20 import _plane20_light
from .soma_growth import load_growth, rasterize, save_masks
from .volume import Acquisition


def _calibration_patch(tissue, occupied):
    """Choose a well-populated crop for matching the two optical models."""
    h, w = tissue.shape
    size = min(h, w, 128)
    ys = np.unique(np.r_[np.arange(0, h-size+1, size), h-size])
    xs = np.unique(np.r_[np.arange(0, w-size+1, size), w-size])
    origin = np.array(max(
        ((y, x) for y in ys for x in xs),
        key=lambda yx: occupied[yx[0]:yx[0]+size, yx[1]:yx[1]+size].sum(),
    ))
    return origin, origin + size


def _contrast(image, tissue):
    band = image - gaussian_filter(image, 3.)
    return float(np.percentile(band[tissue], 95))


def _match_gain(growth, intersecting, masks, direct, tissue, optics):
    start, stop = _calibration_patch(tissue, masks.occupancy()[0] > 0)
    dy, dx = masks.pixel_yx_um
    lo = np.maximum(start - 16, 0)
    hi = np.minimum(stop + 16, tissue.shape)
    lo_um, hi_um = lo * [dy, dx], hi * [dy, dx]
    verts = growth.vertices_zyx_um[-1, :, :, 1:]
    selected = intersecting & np.all((verts.max(axis=1) >= lo_um) & (verts.min(axis=1) <= hi_um), axis=1)
    tile = _tile_scene(growth, lo_um, hi_um, selected)
    image, _ = render_growth(tile, masks.z_positions_um, masks.pixel_yx_um, optics=optics)
    core_tile = tuple(slice(a, b) for a, b in zip(start-lo, stop-lo))
    core_field = tuple(slice(a, b) for a, b in zip(start, stop))
    target = _contrast(direct[core_field], tissue[core_field])
    simulated = _contrast(image[0][core_tile], tissue[core_field])
    if target <= 0 or simulated <= 0:
        raise ValueError('Need nonzero in-plane contrast to match the 2D and 3D light scales')
    return target / simulated, start.tolist(), stop.tolist()


def run(scene_path, profile_path, output, depth, pixel, seed, amplitude_scale):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    growth = load_growth(scene_path)
    acquisition = Acquisition(1, depth)
    pixel_yx = (pixel, pixel)
    masks = rasterize(growth, acquisition.depths(), pixel_yx)
    with np.load(profile_path, allow_pickle=False) as saved:
        profile = dict(saved)
    config = json.loads(str(profile['median_config']))
    tissue = profile['tissue'].astype(bool)
    if tissue.shape != masks.shape_zyx[1:] or not tissue.any():
        raise ValueError('The Plane20 profile must cover the same nonempty grid as the masks')
    direct, _ = _plane20_light(growth, masks, profile, config, seed, -1, amplitude_scale)
    verts = growth.vertices_zyx_um[-1, :, :, 0]
    intersecting = (verts.min(axis=1) <= depth) & (verts.max(axis=1) >= depth)
    optics = GrowthOptics(internal_pitch_um=pixel / 2, background_adu=0.)
    scale, fit_lo, fit_hi = _match_gain(growth, intersecting, masks, direct, tissue, optics)
    optics = replace(optics, amplitude_adu=optics.amplitude_adu * scale)
    print(f'Рендер вне сечения: {(~intersecting).sum()} сом, амплитуда {optics.amplitude_adu:.2f} ADU',
          flush=True)
    outside = _tile_scene(growth, np.zeros(2), growth.tissue_zyx_um[1:], ~intersecting)
    off_plane = render_tiled(outside, masks.z_positions_um, pixel_yx, optics)[0]
    # Use tissue support only; bright structures in the real image cannot enter the background.
    background = 99. + 21. * gaussian_filter(tissue.astype(np.float32), 2.)
    control = background + direct
    image = control + off_plane
    report = {
        'scene': str(scene_path), 'median_profile': str(profile_path), 'z_um': depth, 'seed': seed,
        'pixel_yx_um': pixel_yx, 'amplitude_scale': amplitude_scale, 'optics': asdict(optics),
        'background_adu': {'outside_tissue': 99., 'inside_tissue': 120.},
        'gain_match': {'crop_bounds_yx': [fit_lo, fit_hi], 'method': 'synthetic in-plane positive band P95'},
        'mask_count': int(np.count_nonzero(np.diff(masks.instance_offsets))),
        'mean_off_plane_adu_in_tissue': float(off_plane[tissue].mean()),
        'mean_in_plane_adu_in_tissue': float(direct[tissue].mean()),
        'off_plane_fraction_of_cell_signal': float(
            off_plane[tissue].sum(dtype=np.float64) / (direct + off_plane)[tissue].sum(dtype=np.float64)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        for name, frame in (('image', image), ('in_plane_only', control)):
            write_ome(pending / f'{name}.ome.tif', np.rint(np.clip(frame[None], 0, 65535)).astype(np.uint16),
                      acquisition, pixel_yx)
            np.save(pending / f'{name}_adu.npy', frame, allow_pickle=False)
        np.save(pending / 'off_plane_adu.npy', off_plane, allow_pickle=False)
        save_masks(pending / 'instances.npz', masks)
        (pending / 'render.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--z-um', type=float, required=True)
    parser.add_argument('--pixel-um', type=float, default=.576)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amplitude-scale', type=float, default=.65)
    args = parser.parse_args()
    try:
        run(args.scene, args.profile, args.output, args.z_um, args.pixel_um, args.seed, args.amplitude_scale)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
