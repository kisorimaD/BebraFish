"""Render grown soma sections with the Plane20 median photometry."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import tifffile

from .median_photometry import cell_amplitudes, median_background, median_cell_light, smooth_gain_field
from .soma_growth import load_growth, rasterize, save_masks
from .soma_shell_model import angular_log_modulation, gaussian_shell, modulate_shell, sample_angular_coefficients


ANGLES = np.linspace(0., 2 * np.pi, 257)
RAYS = np.column_stack((np.sin(ANGLES), np.cos(ANGLES)))


def section_segments(vertices, faces, depth_um, pixel_yx_um):
    """Intersect the triangular mesh with a plane, returning YX pixel coordinates."""
    corners = vertices[faces]
    following = np.roll(corners, -1, axis=1)
    crosses = (corners[..., 0] <= depth_um) != (following[..., 0] <= depth_um)
    start, stop = corners[crosses], following[crosses]
    weight = (depth_um - start[:, 0]) / (stop[:, 0] - start[:, 0])
    points = start[:, 1:] + weight[:, None] * (stop[:, 1:] - start[:, 1:])
    return points.reshape(-1, 2, 2) / pixel_yx_um - .5


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def outline_radii(segments, center):
    """Distance to the first contour crossing along each of the fixed angular rays."""
    start = segments[:, 0] - center
    edge = segments[:, 1] - segments[:, 0]
    denominator = _cross(RAYS[:, None, :], edge)
    with np.errstate(divide='ignore', invalid='ignore'):
        ray = _cross(start, edge) / denominator
        segment = _cross(start, RAYS[:, None, :]) / denominator
    return np.min(np.where((ray > 1e-8) & (segment >= -1e-9) & (segment <= 1 + 1e-9),
                           ray, np.inf), axis=1, initial=np.inf)


def _shell_patch(segments, mask_yx, angular, peak, width):
    center = mask_yx.mean(axis=0)
    radii = outline_radii(segments, center)
    if not np.isfinite(radii).all() and len(segments):
        # A raster centroid can fall outside a deep concavity. Use the nearest
        # occupied pixel as the origin of the radial coordinates in that case.
        center = mask_yx[np.argmin(np.sum((mask_yx - center)**2, axis=1))]
        radii = outline_radii(segments, center)
    if not np.isfinite(radii).all():
        # At a tangency a mask can cover a single sample with no closed section.
        valid = np.isfinite(radii)
        radii = (np.interp(ANGLES, ANGLES[valid], radii[valid], period=2*np.pi) if valid.any()
                 else np.full_like(ANGLES, np.sqrt(len(mask_yx) / np.pi)))
    radii[-1] = radii[0]
    outline = radii[:, None] * RAYS
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(outline, axis=0), axis=1))]
    radius = int(np.ceil(radii.max() * (peak + 4*width) + 3*.6)) + 2
    origin = np.floor(center).astype(int) - radius
    axis = np.arange(2*radius + 1)
    y, x = np.ix_(axis + origin[0] - center[0], axis + origin[1] - center[1])
    phi = np.mod(np.arctan2(y, x), 2*np.pi)
    rho = np.hypot(y, x) / np.interp(phi, ANGLES, radii)
    phase = np.interp(phi, ANGLES, arc) * (2*np.pi / arc[-1])
    light = gaussian_shell(rho, peak, width)
    light = modulate_shell(light, angular_log_modulation(phase, angular))
    return origin, gaussian_filter(light, .6, mode='constant').astype(np.float32)


def _plane20_light(growth, masks, profile, config, seed, stage, amplitude_scale):
    if len(masks.z_positions_um) != 1:
        raise ValueError('The Plane20 median renderer takes one geometric section at a time')
    if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
        raise ValueError('Amplitude scale must be finite and nonnegative')
    shape = masks.shape_zyx[1:]
    allowed = profile['neuron_support'].astype(bool)
    if allowed.shape != shape:
        raise ValueError('The Plane20 profile must cover the same grid as the masks')
    _, _, _, spatial, brightness, angular_seed, focus, texture = np.random.default_rng(seed).integers(
        2**32, size=8)
    centers = growth.centers_zyx_um[stage, :, 1:] / masks.pixel_yx_um - .5
    gain_field = smooth_gain_field(shape, 27.93, spatial)
    gains = cell_amplitudes(gain_field, centers, .55, .35, brightness)
    angular = .5 * sample_angular_coefficients(len(centers), angular_seed)
    mixture = 1/(1+np.exp(-1.5*smooth_gain_field(shape, 40, focus)))
    gains *= .36 / (config['width'] * config['peak'])
    signal = np.zeros(shape, dtype=np.float32)
    vertices = growth.vertices_zyx_um[stage]
    depth = masks.z_positions_um[0]
    for i, (start, stop) in enumerate(zip(masks.instance_offsets[:-1], masks.instance_offsets[1:])):
        if start == stop:
            continue
        yx = np.column_stack(np.unravel_index(masks.instance_voxel_index[start:stop], shape))
        segments = section_segments(vertices[i], growth.faces, depth, masks.pixel_yx_um)
        origin, patch = _shell_patch(segments, yx, angular[i], config['peak'], config['width'])
        lo, hi = np.maximum(origin, 0), np.minimum(origin + patch.shape, shape)
        target = tuple(slice(a, b) for a, b in zip(lo, hi))
        local = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, origin))
        signal[target] += patch[local] * gains[i]
    light = median_cell_light(signal, mixture, distance_transform_edt(allowed),
                              profile['normalizer'], config['amplitude'] * amplitude_scale)
    return light, texture


def render_median_plane(growth, masks, profile, seed, *, stage=-1, amplitude_scale=1.):
    """Keep the growth masks and apply the legacy 2D shell and compositing model."""
    from training.plane20.generate_median import structured_render

    tissue = profile['tissue'].astype(bool)
    envelope = profile['median_envelope']
    if any(a.shape != masks.shape_zyx[1:] for a in (tissue, envelope)) or not tissue.any():
        raise ValueError('The median profile must cover the same nonempty tissue grid as the masks')
    config = json.loads(str(profile['median_config']))
    light, texture = _plane20_light(growth, masks, profile, config, seed, stage, amplitude_scale)
    background = median_background(envelope, light, config['density_floor'])
    return background+light, structured_render(background, light, tissue, config, texture)


def run(scene_path, profile_path, output, depth_um, pixel_um, seed, stage, amplitude_scale=1.):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    with np.load(profile_path, allow_pickle=False) as saved:
        profile = dict(saved)
    growth = load_growth(scene_path)
    masks = rasterize(growth, [depth_um], (pixel_um, pixel_um), stage=stage)
    baseline, structured = render_median_plane(growth, masks, profile, seed, stage=stage,
                                               amplitude_scale=amplitude_scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        for name, image in (('baseline', baseline), ('structured', structured)):
            tifffile.imwrite(pending / f'{name}.tif', np.rint(np.clip(image, 0, 65535)).astype(np.uint16),
                             photometric='minisblack')
        save_masks(pending / 'instances.npz', masks)
        metadata = {
            'scene': str(scene_path), 'median_profile': str(profile_path), 'stage': stage % len(growth.scale),
            'scene_depth_um': depth_um, 'pixel_yx_um': list(masks.pixel_yx_um), 'seed': seed,
            'amplitude_scale': amplitude_scale,
            'n_projected_somas': int(np.count_nonzero(np.diff(masks.instance_offsets))),
            'photometry': 'Plane20 median baseline and structured; shell radius follows the grown contour',
        }
        (pending / 'render.json').write_text(json.dumps(metadata, indent=2) + '\n')
        pending.rename(output)
    print(f'{masks.shape_zyx[1:]} YX, {metadata["n_projected_somas"]} masks → {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--z-um', type=float, required=True)
    parser.add_argument('--pixel-um', type=float, default=.576)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', type=int, default=-1)
    parser.add_argument('--amplitude-scale', type=float, default=1.,
                        help='Scale soma light before Plane20 median compositing')
    args = parser.parse_args()
    try:
        run(args.scene, args.profile, args.output, args.z_um, args.pixel_um, args.seed, args.stage,
            args.amplitude_scale)
    except (ValueError, OSError, IndexError) as exc:
        parser.error(str(exc))
