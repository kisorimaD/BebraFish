"""Fit cellular, shared and diffuse dynamics separately for the saved Plane20 scene."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import tifffile

from calibrate_flash_dynamics import (BLOCK, TILE, calibrated_baselines, finish_gif, flash_metrics,
                                     grain_metrics, make_gif, mask_matrix, median_center, raw_signal,
                                     render_flash_frame, tile_mean)
from optical_layers import add_optical_noise, halo_light
from plot_render_components import display_values, prepare_render
from render_recorded_bank import HERE, PROFILE

SOURCE = HERE / 'movie_foreground_conservative'
PREVIOUS = HERE / 'movie_coordinated_flashes'
OUTPUT = HERE / 'movie_decoupled_dynamics'
DISTANCE_EDGES = (0, 10, 20, 35, 50, 75, 100, 150, 200)
PARAMETERS = ('positive_soma_gain', 'peak_compression', 'common_soma_gain', 'regional_soma_gain',
              'diffuse_gain', 'static_gain', 'static_offset_adu')


def spatial_pairs(centers, rng):
    distance = pdist(centers)
    left, right = np.triu_indices(len(centers), k=1)
    samples = []
    for lo, hi in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        available = np.flatnonzero((distance >= lo) & (distance < hi))
        samples.append(rng.choice(available, min(len(available), 300), replace=False))
    return [(left[indices], right[indices]) for indices in samples]


def training_statistics(tiles, rois, pixels, pairs, blocks, low, high, photometry_selection=None):
    tile_delta = median_center(tiles)
    roi_delta = median_center(rois)
    relative = roi_delta / np.median(rois, axis=0)
    amplitude = np.quantile(rois, .95, axis=0) - np.median(rois, axis=0)
    flash = np.quantile(tile_delta, .9, axis=1)
    synchronous = np.quantile(relative, .99, axis=1)
    # With equal weights for windows, one unusually bright burst cannot set all the gains.
    typical_flash = np.median([np.quantile(flash[block], .9) for block in blocks])
    typical_synchrony = np.median([np.quantile(synchronous[block], .9) for block in blocks])
    residual = relative - np.median(relative, axis=1, keepdims=True)
    centered = residual - residual.mean(axis=0)
    centered /= np.maximum(np.linalg.norm(centered, axis=0), 1e-6)
    correlation = [np.median(np.sum(centered[:, a] * centered[:, b], axis=0)) if len(a) else np.nan
                   for a, b in pairs]
    if photometry_selection is not None:
        pixels = pixels[photometry_selection]
    shown = pixels.clip(low, high)
    return np.r_[np.quantile(amplitude, [.25, .5, .75, .9]), np.quantile(flash, [.75, .9, .97]),
                 typical_flash, np.quantile(synchronous, [.8, .95]), typical_synchrony, correlation,
                 shown.mean(), shown.std(), np.quantile(shown, [.5, .9, .98]),
                 100*np.mean(pixels <= low), 100*np.mean(pixels >= high)]


def comparison_metrics(projections, low, high):
    tiles, rois, left, right = (np.stack(x) for x in zip(*projections))
    result = flash_metrics(tiles, rois, left, low, high)
    result['grain_by_brightness'] = grain_metrics(left, right, low, high)
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rng = np.random.default_rng(327)
    state = prepare_render(SOURCE)
    meta = state['meta']
    before = json.loads((PREVIOUS / 'validation.json').read_text())
    with np.load(PREVIOUS / 'activity.npz', allow_pickle=False) as loaded:
        activity = dict(loaded)
    with np.load(SOURCE / meta['scene'], allow_pickle=False) as loaded:
        scene = dict(loaded)
    with np.load(SOURCE / meta.get('profile', PROFILE), allow_pickle=False) as loaded:
        tissue = loaded['tissue'].astype(bool)
    bank_path = SOURCE / meta['bank']
    plane = meta.get('reference_plane', 20)
    with np.load(bank_path, allow_pickle=False) as bank:
        select = bank['plane'] == plane
        ids = bank['mask_id'][select]
        ref_centers = bank['centers_yx'][select]
    ref_rois = raw_signal(bank_path.parent.parent / 'per_plane', plane, ids, np.ones(len(ids), dtype=bool))
    starts = np.asarray(before['training_source_blocks'])
    frames = np.arange(len(activity['dff']))
    train = np.isin(frames // BLOCK * BLOCK, starts)
    held = np.array(before['heldout_reference_frames'])
    assert train[activity['donor_frames']].all() and train[activity['diffuse_sample_frames']].all()
    assert not train[held].any()
    fit = (starts[:, None] + np.arange(1, BLOCK, 3)).ravel()
    fit_blocks = np.split(np.arange(len(fit)), len(starts))
    valid_tiles = tile_mean(tissue) > .7
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coordinates = ((y+.5)/TILE-.5, (x+.5)/TILE-.5)
    support = gaussian_filter(tissue.astype(np.float32), 12)
    masks = mask_matrix(scene)
    low, high = meta['display_black_adu'], meta['display_white_adu']
    pixels = rng.choice(np.flatnonzero(x.ravel() < tissue.shape[1]-1), 6000, replace=False)
    fit_pixels = slice(0, 1000)

    sampled = activity['dff']
    common = np.median(sampled, axis=1, keepdims=True)
    local = sampled - common
    positive = np.maximum(local, 0)
    event_strength = positive.mean(axis=1)
    typical_strength = max(np.median(event_strength[train]), 1e-6)
    excess_fraction = np.maximum(1 - typical_strength / np.maximum(event_strength, 1e-6), 0)
    regional = np.empty_like(sampled)
    coarse = activity['diffuse_tile_changes']
    soma_coords = ((scene['centers'].T + .5) / TILE - .5).astype(np.float32)
    for i, tile in enumerate(coarse):
        regional[i] = map_coordinates(tile - np.mean(tile[valid_tiles]), soma_coords,
                                      order=1, mode='nearest') / 150
    regional -= np.median(regional[train], axis=0)

    def diffuse(index):
        return map_coordinates(coarse[index], coordinates, order=1, mode='nearest') * support

    def signal(modulation):
        activity_image = (modulation @ state['maps']).reshape(tissue.shape)
        halo = halo_light(activity_image)
        return activity_image + halo, halo

    def project(image, rois=None):
        flat = image.ravel()
        if rois is None:
            rois = masks @ flat
        return tile_mean(image)[valid_tiles], rois, flat[pixels], flat[pixels + 1]

    def fit_projection(image):
        tiles, rois, left, _ = project(image)
        return tiles, rois, left[fit_pixels]

    prior_baselines = calibrated_baselines(state, before['static_gain'], before['static_offset_adu'])

    def prior(index):
        total, halo = signal(activity['render_modulation'][index])
        return render_flash_frame(state, index, total, halo, before['diffuse_gain']*diffuse(index),
                                  prior_baselines)[0]

    with (tifffile.TiffFile(meta['real_source']) as recording,
          ThreadPoolExecutor(max_workers=3) as workers):
        fit_real = [project(recording.pages[int(i)].asarray(), ref_rois[i]) for i in fit]
        fit_prior = list(workers.map(lambda i: project(prior(i)), fit))
        real_grain = comparison_metrics(fit_real, low, high)['grain_by_brightness']
        prior_grain = comparison_metrics(fit_prior, low, high)['grain_by_brightness']
        correction = {'intensity_adu': [], 'scale': []}
        for real, generated in zip(real_grain, prior_grain):
            if min(real['pairs'], generated['pairs']) < 20:
                continue
            correction['intensity_adu'].append(float(np.mean(real['range_adu'])))
            correction['scale'].append(float(np.clip(real['robust_sd_adu']/generated['robust_sd_adu'], .65, 1.15)))
        if not correction['scale']:
            raise RuntimeError('Too few pixel pairs for grain calibration.')

        config = meta['detector_noise']
        base = state['base']
        noise_correction = np.interp(base, correction['intensity_adu'], correction['scale'])

        def bases(i):
            noise, halo_noise = add_optical_noise(base, state['halo_static'], config['intensity_adu'],
                                                 config['sigma_adu'], state['seeds'][i], state['halo_seeds'][i])
            images = (signal(local[i] - positive[i])[0], signal(positive[i])[0],
                      signal(np.broadcast_to(common[i], local[i].shape))[0], signal(regional[i])[0],
                      diffuse(i), (noise+halo_noise)*noise_correction)
            return [fit_projection(image) for image in images]

        print('Measuring training projections.', flush=True)
        cached = list(workers.map(bases, fit))
    components = [tuple(np.stack([frame[k][j] for frame in cached]) for j in range(3)) for k in range(6)]
    del cached
    baseline = fit_projection(state['base'])
    real_tile, real_roi, real_pixel, _ = (np.stack(x) for x in zip(*fit_real))
    reference = training_statistics(real_tile, real_roi, real_pixel[:, fit_pixels],
                                    spatial_pairs(ref_centers, rng), fit_blocks, low, high)
    synthetic_pairs = spatial_pairs(scene['centers'], rng)
    tolerances = np.r_[1.5, 1.5, 2, 3, 1.2, 2, 3, 1.4, .015, .025, .018,
                       .1, .09, .07, .07, .06, .06, .06, .06, .45, .75, .8, 2, 3, 1, .35]

    def predict(parameters):
        gain, compression, shared, region, broad, tone, offset = parameters
        gains = (gain * (1 - compression*excess_fraction[fit]))[:, None]
        return tuple(150 + tone*(base - 150) + offset + negative + gains*positive + shared*common
                     + region*regional + broad*diffuse + noise
                     for base, negative, positive, common, regional, diffuse, noise in zip(baseline, *components))

    def loss(parameters):
        measured = training_statistics(*predict(parameters), synthetic_pairs, fit_blocks, low, high)
        return np.mean(((measured-reference)/tolerances)**2)

    initial = [3, .5, 1, 1, .35, before['static_gain'], before['static_offset_adu']]
    print('Fitting gains on training blocks.', flush=True)
    fitted = minimize(loss, initial, method='Powell',
                      bounds=[(.1, 9), (0, 1), (0, 6), (0, 5), (0, 3), (.8, 1.2), (-12, 12)],
                      options={'maxiter': 14, 'xtol': .025, 'ftol': .008})
    gain, compression, shared, region, broad, tone, offset = fitted.x
    print('Fitted', dict(zip(PARAMETERS, fitted.x)), 'training loss', fitted.fun, flush=True)
    baselines = calibrated_baselines(state, tone, offset)
    modulation = np.maximum(local - positive + (gain * (1-compression*excess_fraction[:, None]))*positive
                            + shared*common + region*regional, -1).astype(np.float32)

    def render(index):
        total, halo = signal(modulation[index])
        return render_flash_frame(state, index, total, halo, broad*diffuse(index), baselines, correction)[0]

    display_frames = before['reference_gif_frames'][:180]
    width, height = meta['width'], meta['panel_height']
    transpose = meta.get('transpose_for_display', True)
    with tempfile.TemporaryDirectory(prefix='.recalibration-', dir=HERE) as temp:
        stage = Path(temp) / OUTPUT.name
        stage.mkdir()
        gif = stage / 'real_vs_synthetic.gif'
        writer = make_gif(gif, (width, height*2))
        measurements = {name: [] for name in ('real', 'old', 'new')}
        show = set(display_frames)
        check = set(held)
        indices = sorted(show | check)
        print('Rendering comparison and measuring held-out frames.', flush=True)
        try:
            with (tifffile.TiffFile(meta['real_source']) as recording,
                  ThreadPoolExecutor(max_workers=3) as workers):
                for index, synthetic in zip(indices, workers.map(render, indices)):
                    real = recording.pages[index].asarray()
                    if index in check:
                        measurements['real'].append(project(real, ref_rois[index]))
                        measurements['old'].append(project(prior(index)))
                        measurements['new'].append(project(synthetic))
                    if index in show:
                        panels = [display_values(image, low, high, transpose) for image in (real, synthetic)]
                        writer.stdin.write(np.concatenate(panels).tobytes())
            finish_gif(writer, gif, len(display_frames))
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait()
        outputs = dict(zip(PARAMETERS, map(float, fitted.x)))
        outputs.update(training_source_blocks=starts.tolist(), training_frames=fit.tolist(),
                       heldout_reference_frames=held.tolist(), reference_gif_frames=display_frames,
                       training_objective=float(fitted.fun), initial_training_objective=float(loss(initial)),
                       grain_correction=correction,
                       heldout={name: comparison_metrics(rows, low, high) for name, rows in measurements.items()})
        earlier_fit_blocks = np.array(meta['photometry']['fitted_frames']) // BLOCK
        stricter = ~np.isin(held // BLOCK, earlier_fit_blocks)
        outputs['stricter_reference_frames'] = held[stricter].tolist()
        outputs['stricter'] = {name: comparison_metrics([row for row, keep in zip(rows, stricter) if keep], low, high)
                               for name, rows in measurements.items()}
        activity['render_modulation'] = modulation
        np.savez_compressed(stage / 'activity.npz', **activity)
        (stage / 'validation.json').write_text(json.dumps(outputs, indent=2, allow_nan=False) + '\n')
        stage.rename(OUTPUT)
    print('Saved', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
