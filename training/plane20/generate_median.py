"""Generate paired baseline and structured-light median-style scenes."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
import tifffile

from .generate import ROOT, add_generation_arguments, generate_dataset, profile_arrays
from .center_data import ROI_SIZE, ellipse_bounds
from compare_packing import relaxed_centers
from empirical_morphology import build_empirical_scene
from soma_shell_model import sample_morphology, sample_angular_coefficients, shell_patch
from try_correlated_cell_gain import cell_amplitudes, smooth_gain_field
from try_heterogeneous_scene import effective_blur

AXES = ('major_axis_px', 'minor_axis_px', 'orientation_deg')


def export_median_profile(path):
    arrays = profile_arrays()
    pilot = ROOT/'outputs/plane20_median_morphology/manual_median'
    geom = json.loads((pilot/'inner_peak'/'fit_report.json').read_text())
    light = json.loads((pilot/'arc_brightness'/'report.json').read_text())
    patch = json.loads((pilot/'localized_texture'/'report.json').read_text())
    texture = patch['background_texture_fit']
    median = tifffile.imread(ROOT/'outputs/plane20_median_resolution/local.tiff')
    arrays['median_envelope'] = gaussian_filter(median, 12).astype(np.float32)
    arrays['median_config'] = np.array(json.dumps(dict(
        size_scale=geom['size_scale'], aspect_scale=geom['aspect_scale'],
        count=geom['fitted_cell_count'], density_floor=float(np.percentile(arrays['background'], 2)),
        width=geom['best']['width'], peak=geom['best']['peak'],
        amplitude=geom['best']['gain']*light['gain'],
        subdued_origin=patch['crop_origins_yx'][1], subdued_gain=patch['local_shell_gain_second_crop'],
        texture_sigma=texture['texture_sigma_px'], texture_band_rms=texture['background_band_rms_adu'],
        texture_shell_gain=texture['shell_gain'])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def pack(centers, geom, distance, target, rng):
    # Keep full ellipses inside the support and within the decoder's ROI, even
    # at the largest anchor displacement (.75 jitter + .5 nearest rounding).
    yi, xi = np.rint(centers).astype(int).T
    radii = geom['major_axis_px']/2
    bounds = ellipse_bounds(geom, np.arange(len(centers)))
    fits_image = np.all((centers-bounds >= 0) & (centers+bounds <= np.asarray(distance.shape)-1), axis=1)
    valid = (distance[yi, xi] > radii+.75) & (bounds.max(axis=1) < ROI_SIZE/2-1.25) & fits_image
    diameter = np.sqrt(geom['major_axis_px']*geom['minor_axis_px'])
    neighbors = cKDTree(centers).query_ball_point(centers, .55*np.median(diameter))
    keep = []
    for index in rng.permutation(len(centers)):
        if valid[index]:
            keep.append(index)
            valid[neighbors[index]] = False
            if len(keep) == target:
                break
    return np.sort(keep)


def fade_region(shape, origin):
    coords = np.ogrid[:shape[0], :shape[1]]
    dy, dx = [np.maximum(0, np.maximum(start-axis, axis-(start+63)))
              for axis, start in zip(coords, origin)]
    return (.5+.5*np.cos(np.pi*np.minimum(np.hypot(dy, dx)/24, 1))).astype(np.float32)


def structured_render(background, cells, tissue, config, seed):
    rng = np.random.default_rng(seed)
    # Random locations at the scale of several somas, with no copied pixels.
    placement = smooth_gain_field(cells.shape, 35, rng.integers(2**32))
    threshold = np.percentile(placement[tissue], 65)
    amount = np.clip((placement-threshold)/1.2, 0, 1)
    # Fade gradually at the perimeter while allowing diffuse light between somas.
    amount *= np.clip(gaussian_filter(tissue.astype(np.float32), 9)*1.5, 0, 1)
    field = gaussian_filter(rng.standard_normal(cells.shape, dtype=np.float32), config['texture_sigma'])
    field /= max(float(field.std()), 1e-6)
    field += .24*np.maximum(field-.55, 0)**2
    field -= gaussian_filter(field, 9)
    broad = gaussian_filter(field, 3)-gaussian_filter(field, 7)
    scale = config['texture_band_rms']/max(float(broad[tissue].std()), 1e-6)
    texture = field*scale*amount*rng.uniform(.72, 1.08)
    subdued = fade_region(cells.shape, config['subdued_origin'])
    fraction_removed = 1-(1-(1-config['subdued_gain'])*subdued)*(1-(1-config['texture_shell_gain'])*amount)
    removed = (cells*fraction_removed).astype(np.float32)
    diffuse = np.maximum(background+gaussian_filter(removed, 8)+texture, config['density_floor'])
    return diffuse+cells-removed


def generate_pair(job, profile_path):
    split, index, seed, output = job
    with np.load(profile_path) as source:
        profile = {key: source[key] for key in source.files}
    config = json.loads(str(profile['median_config']))
    settings = json.loads(str(profile['settings_json']))['selected'] | dict(
        ring_fraction=1., ring_center_fill=0., blur_px=.6)
    rng = np.random.default_rng(seed)
    packing, morphology, thinning, spatial, brightness, angular_seed, focus, texture = rng.integers(2**32, size=8)
    tissue = profile['tissue'].astype(bool)
    allowed = profile['neuron_support'].astype(bool)
    distance = distance_transform_edt(allowed)
    support = distance_transform_edt(tissue) > 1.5*settings['sigma_px']
    centers = relaxed_centers(support, profile['density'], 6830, packing)
    geom = sample_morphology(json.loads(str(profile['morphology_json'])), len(centers), morphology)
    shape_gain = np.sqrt(config['aspect_scale'])
    geom['major_axis_px'] *= config['size_scale']*shape_gain
    geom['minor_axis_px'] *= config['size_scale']/shape_gain
    geom['minor_axis_px'] = np.minimum(geom['major_axis_px'], geom['minor_axis_px'])
    keep = pack(centers, geom, distance, config['count'], np.random.default_rng(thinning))
    centers, geom = centers[keep], {key: geom[key][keep] for key in AXES}
    field = smooth_gain_field(tissue.shape, 27.93, spatial)
    amplitudes = cell_amplitudes(field, centers, .55, .35, brightness)
    angular = .5*sample_angular_coefficients(len(centers), angular_seed)
    mixture = 1/(1+np.exp(-1.5*smooth_gain_field(tissue.shape, 40, focus)))
    ones = np.ones(tissue.shape, np.float32)
    scene = build_empirical_scene(dict(tissue=allowed, tissue_field=ones, cell_gain=ones),
        settings, centers, None, seed, geometry=geom, angular_coefficients=angular,
        amplitudes=amplitudes*.36/(config['width']*config['peak']),
        patch_renderer=partial(shell_patch, width=config['width'], peak_radius=config['peak']))
    cells = (effective_blur(scene['signal'], .2, .8, mixture)*profile['normalizer']
             *np.clip(distance/2, 0, 1)*config['amplitude']).astype(np.float32)
    background = np.maximum(profile['median_envelope']-gaussian_filter(cells, 12), config['density_floor'])
    image = np.rint(np.clip(background+cells, 0, 65535)).astype(np.uint16)
    image_structured = np.rint(np.clip(structured_render(background,cells,tissue,config,texture),0,65535)).astype(np.uint16)
    relative = Path(split)/f'scene_{index:05d}.npz'
    path = Path(output)/relative
    np.savez_compressed(path, image=image, image_structured=image_structured, centers=centers, **geom,
        instance_offsets=scene['instance_offsets'], instance_pixel_index=scene['instance_pixel_index'],
        seed=np.uint64(seed))
    return dict(split=split, index=index, seed=seed, cells=len(centers), path=relative.as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), renders=['baseline','structured'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(parser)
    args = parser.parse_args()
    if args.export_profile:
        export_median_profile(args.export_profile)
        return
    if args.profile is None or args.output is None:
        parser.error('--profile and --output are required')
    if min(args.train, args.val, args.test, args.workers) < 1:
        parser.error('Scene counts and workers must be positive')
    generate_dataset(args, generate_pair, 'Median-style baseline and structured render share full instance targets. Geometry and texture seeds are independent across scenes; each pair belongs to one split.')


if __name__ == '__main__':
    main()
