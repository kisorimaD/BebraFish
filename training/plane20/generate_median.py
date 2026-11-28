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
from .photometry import (BACKGROUND_BLEND, blend_reference_background, reference_background_weights,
                         render_intensity_noise, spatial_noise)
from .soma_growth import GROWTH_LIMIT, grow_somas, light_reaches, radial_reaches
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
    arrays['reference_background'] = np.where(arrays['neuron_support'], 0, median)
    arrays['median_envelope'] = gaussian_filter(median, 12).astype(np.float32)
    arrays['median_config'] = np.array(json.dumps(dict(
        size_scale=geom['size_scale'], aspect_scale=geom['aspect_scale'],
        count=geom['fitted_cell_count'], density_floor=float(np.percentile(arrays['background'], 2)),
        width=geom['best']['width'], peak=geom['best']['peak'],
        amplitude=geom['best']['gain']*light['gain'],
        subdued_origin=patch['crop_origins_yx'][1], subdued_gain=patch['local_shell_gain_second_crop'],
        texture_sigma=texture['texture_sigma_px'], texture_band_rms=texture['background_band_rms_adu'],
        texture_shell_gain=texture['shell_gain'], intensity_noise=spatial_noise(median))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def pack(centers, geom, distance, target, rng, *, spacing=None, roi_size=ROI_SIZE, extent=1.):
    # Keep full ellipses inside the support and within the decoder's ROI, even
    # at the largest anchor displacement (.75 jitter + .5 nearest rounding).
    yi, xi = np.rint(centers).astype(int).T
    radii = geom['major_axis_px']/2
    bounds = ellipse_bounds(geom, np.arange(len(centers)))
    fits_image = np.all((centers-bounds >= 0) & (centers+bounds <= np.asarray(distance.shape)-1), axis=1)
    valid = (distance[yi, xi] > radii+.75) & fits_image
    if roi_size is not None:
        valid &= extent*bounds.max(axis=1) < roi_size/2-1.25
    diameter = np.sqrt(geom['major_axis_px']*geom['minor_axis_px'])
    spacing = .55*np.median(diameter) if spacing is None else spacing
    if extent > 1.:
        spacing = max(spacing, np.sqrt(2))  # A distinct rounded seed for each front.
    neighbors = cKDTree(centers).query_ball_point(centers, spacing)
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
    subdued = fade_region(cells.shape, config['subdued_origin']) if 'subdued_origin' in config else 0.
    fraction_removed = 1-(1-(1-config.get('subdued_gain', 1.))*subdued)*(1-(1-config['texture_shell_gain'])*amount)
    removed = (cells*fraction_removed).astype(np.float32)
    diffuse = np.maximum(background+gaussian_filter(removed, 8)+texture, config['density_floor'])
    return diffuse+cells-removed


def light_fields(shape, count, config, seed):
    """Reuse the packed scene's random draws when adjusting its photometry."""
    _, _, _, spatial, brightness, angular, focus, _ = np.random.default_rng(seed).integers(2**32, size=8)
    field = smooth_gain_field(shape, config.get('gain_length', 27.93), spatial)
    coefficients = sample_angular_coefficients(count, angular)
    mixture = 1/(1+np.exp(-1.5*smooth_gain_field(shape, 40, focus)))
    return field, brightness, coefficients, mixture


def light_strength(field, centers, config, brightness):
    amplitudes = cell_amplitudes(field, centers, config.get('gain_field_sd', .55),
                                 config.get('gain_jitter_sd', .35), brightness)
    return amplitudes*config.get('shell_flux', .36)/(config['width']*config['peak'])


def render_somas(allowed, centers, geom, config, settings, seed, amplitudes, coefficients, *, growth=None):
    settings = settings | dict(ring_fraction=1., ring_center_fill=config.get('core_fill', 0.),
                               blur_px=config.get('blur_px', .6))
    renderer = partial(shell_patch, width=config['width'], peak_radius=config['peak'])
    metadata = {}
    if growth is None and config.get('grow_somas'):
        owners = grow_somas(allowed, centers, geom)
        growth = owners, radial_reaches(owners, centers, geom)
    if growth is not None:
        owners, reaches = growth
        scale = light_reaches(reaches)
        index_by_center = {tuple(center): index for index, center in enumerate(centers)}

        def grown_patch(center, *args, **kwargs):
            index = index_by_center[tuple(center)]
            origin, kernel, _ = renderer(center, *args, radial_scale=scale[index], **kwargs)
            start = np.maximum(origin, 0)
            end = np.minimum(origin+kernel.shape, allowed.shape)
            region = tuple(slice(a, b) for a, b in zip(start, end))
            local = tuple(slice(a-o, b-o) for a, b, o in zip(start, end, origin))
            mask = np.zeros(kernel.shape, dtype=bool)
            mask[local] = owners[region] == index
            return origin, kernel, mask

        patch_renderer = grown_patch
        metadata = dict(growth_reach=reaches, growth_limit=np.float32(GROWTH_LIMIT))
    else:
        patch_renderer = renderer
    ones = np.ones(allowed.shape, np.float32)
    return build_empirical_scene(dict(tissue=allowed, tissue_field=ones, cell_gain=ones),
        settings, centers, None, seed, geometry=geom,
        angular_coefficients=config.get('angular_sd', .5)*coefficients, amplitudes=amplitudes,
        patch_renderer=patch_renderer) | metadata


def generate_pair(job, profile_path, *, fixed_centers=None, grow=None, fit_mask_roi=False):
    split, index, seed, output = job
    with np.load(profile_path) as source:
        profile = {key: source[key] for key in source.files}
    config = json.loads(str(profile['median_config']))
    if grow is not None:
        config['grow_somas'] = grow
    settings = json.loads(str(profile['settings_json']))['selected']
    rng = np.random.default_rng(seed)
    packing, morphology, thinning, _, _, _, _, _ = rng.integers(2**32, size=8)
    tissue = profile['tissue'].astype(bool)
    allowed = profile['neuron_support'].astype(bool)
    if fixed_centers is None:
        distance = distance_transform_edt(allowed)
        support = distance_transform_edt(tissue) > 1.5*settings['sigma_px']
        centers = relaxed_centers(support, profile['density'], config.get('candidate_count', 6830), packing,
                                  iterations=config.get('packing_iterations', 35))
        edge_gain = np.clip(distance/2, 0, 1)
    else:
        centers = np.asarray(fixed_centers, dtype=float)
        if (centers.ndim != 2 or centers.shape[1] != 2 or not 0 < len(centers) < 65535
                or not np.isfinite(centers).all()
                or np.any((centers < 0) | (centers > np.asarray(tissue.shape)-1))):
            raise ValueError('Expected a nonempty array of in-frame (y, x) pixel centers.')
        # Prescribed centers can be near edges or outside the fitted tissue.
        allowed = np.ones_like(tissue)
        edge_gain = 1.
    geom = sample_morphology(json.loads(str(profile['morphology_json'])), len(centers), morphology)
    shape_gain = np.sqrt(config['aspect_scale'])
    geom['major_axis_px'] *= config['size_scale']*shape_gain
    geom['minor_axis_px'] *= config['size_scale']/shape_gain
    geom['minor_axis_px'] = np.minimum(geom['major_axis_px'], geom['minor_axis_px'])
    if fixed_centers is None:
        roi_size = config.get('roi_size', ROI_SIZE)
        if fit_mask_roi and (roi_size is None or roi_size > ROI_SIZE):
            roi_size = ROI_SIZE
        keep = pack(centers, geom, distance, config['count'], np.random.default_rng(thinning),
                    spacing=config.get('spacing'), roi_size=roi_size,
                    extent=GROWTH_LIMIT if config.get('grow_somas') else 1.)
        if not len(keep):
            raise ValueError('По оценённой геометрии в ткани не помещается ни одна полная сома.')
        centers, geom = centers[keep], {key: geom[key][keep] for key in AXES}
    placement = 'packed' if fixed_centers is None else 'fixed'
    return render_pair(job, profile, config, settings, centers, geom, tissue, allowed, edge_gain,
                       placement=placement)


def relight_pair(job, profile_path, layout, placement):
    """Render a saved scene again using its original geometry and seed."""
    with np.load(profile_path) as source:
        profile = dict(source)
    config = json.loads(str(profile['median_config']))
    settings = json.loads(str(profile['settings_json']))['selected']
    tissue = profile['tissue'].astype(bool)
    allowed = profile['neuron_support'].astype(bool) if placement == 'packed' else np.ones_like(tissue)
    edge_gain = np.clip(distance_transform_edt(allowed)/2, 0, 1) if placement == 'packed' else 1.
    centers = layout['centers']
    geom = {key: layout[key] for key in AXES}
    return render_pair(job, profile, config, settings, centers, geom, tissue, allowed, edge_gain,
                       placement=placement, layout=layout)


def render_pair(job, profile, config, settings, centers, geom, tissue, allowed, edge_gain, *,
                placement, layout=None):
    split, index, seed, output = job
    texture = np.random.default_rng(seed).integers(2**32, size=8)[-1]
    growth = None
    object_mask = None
    if layout is not None:
        pixels = layout['instance_pixel_index']
        object_mask = np.zeros(tissue.shape, dtype=bool)
        object_mask.ravel()[pixels] = True
        if 'growth_reach' in layout:
            owners = np.full(tissue.shape, -1, dtype=np.int32)
            owners.ravel()[pixels] = np.repeat(np.arange(len(centers)), np.diff(layout['instance_offsets']))
            growth = owners, layout['growth_reach']
        config = config | {'grow_somas': False}
    field, brightness, coefficients, mixture = light_fields(tissue.shape, len(centers), config, seed)
    amplitudes = light_strength(field, centers, config, brightness)
    scene = render_somas(allowed, centers, geom, config, settings, seed, amplitudes, coefficients, growth=growth)
    cells = (effective_blur(scene['signal'], .2, .8, mixture)*profile['normalizer']
             *edge_gain*config['amplitude']).astype(np.float32)
    envelope = profile['median_envelope']
    background = np.maximum(envelope-gaussian_filter(cells, 12), config['density_floor'])
    clean = background + cells
    curve = config.get('intensity_noise')
    if curve is None:
        structured = structured_render(background, cells, tissue, config, texture)
        method = 'legacy_diffuse'
    else:
        if object_mask is None:
            object_mask = scene['labels'] > 0
        structured = render_intensity_noise(background, cells, envelope, object_mask, curve, texture)
        method = curve['method']
    background_metadata = {}
    if 'reference_background' in profile:
        support = profile['neuron_support']
        if placement == 'fixed':
            support = support | (object_mask if object_mask is not None else scene['labels'] > 0)
        weights = reference_background_weights(support)
        reference = profile['reference_background']
        clean = blend_reference_background(clean, reference, weights)
        structured = blend_reference_background(structured, reference, weights)
        background_metadata['background_blend'] = BACKGROUND_BLEND
    image = np.rint(np.clip(clean, 0, 65535)).astype(np.uint16)
    image_structured = np.rint(np.clip(structured, 0, 65535)).astype(np.uint16)
    relative = Path(split)/f'scene_{index:05d}.npz'
    path = Path(output)/relative
    masks = scene if layout is None else layout
    growth = {key: masks[key] for key in ('growth_reach', 'growth_limit') if key in masks}
    np.savez_compressed(path, image=image, image_structured=image_structured, centers=centers, **geom,
        instance_offsets=masks['instance_offsets'], instance_pixel_index=masks['instance_pixel_index'],
        seed=np.uint64(seed), **growth, **background_metadata)
    return dict(split=split, index=index, seed=seed, cells=len(centers), path=relative.as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), renders=['baseline','structured'],
                structured_method=method, soma_shape='bounded_growth' if growth else 'ellipse',
                placement=placement, **background_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(parser)
    parser.add_argument('--grow-somas', action='store_true', default=None,
                        help='Grow bounded soma masks that stop at contacts.')
    parser.add_argument('--fit-mask-roi', action='store_true',
                        help='Require each instance to fit the center-mask training ROI.')
    args = parser.parse_args()
    if args.export_profile:
        export_median_profile(args.export_profile)
        return
    if args.profile is None or args.output is None:
        parser.error('--profile and --output are required')
    if min(args.train, args.val, args.test, args.workers) < 1:
        parser.error('Scene counts and workers must be positive')
    generate_dataset(args, partial(generate_pair, grow=args.grow_somas, fit_mask_roi=args.fit_mask_roi), 'Median-style baseline and structured render share full instance targets. Geometry and texture seeds are independent across scenes; each pair belongs to one split.')


if __name__ == '__main__':
    main()
