"""Calibrate a coordinated activity sample in image units and render it without motion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.special import expit
import tifffile

from calibrate_flash_dynamics import (BLOCK, TILE, calibrated_baselines, dark_weights, finish_gif,
                                     flash_metrics, grain_metrics, make_gif, mask_matrix,
                                     median_center, prepare_flash_replay, raw_signal, read_reference, render_flash_frame,
                                     save_temporal_median, tile_mean)
from optical_layers import add_optical_noise, halo_light
from paper_figures.make_validation_figure import correlation_by_distance, measure as measure_saved_report, plot
from plot_render_components import display_values, prepare_render, render_layers
from recalibrate_plane20 import PARAMETERS, spatial_pairs, training_statistics
from sample_plane20_activity import serializable


def background_windows(annuli, observed, sampled, train, valid_tiles, seed):
    """Draw training-only background windows with comparable population activity."""
    rng = np.random.default_rng(seed)
    length = len(sampled)
    starts = np.flatnonzero(train & (np.arange(length) % BLOCK == 0))
    starts = starts[starts+BLOCK <= length]
    reference_activity = np.maximum(median_center(observed)/np.median(observed, axis=0), 0).mean(axis=1)
    synthetic_activity = np.maximum(sampled, 0).mean(axis=1)
    source_level = np.array([reference_activity[i:i+BLOCK].mean() for i in starts])
    targets = np.arange(0, length, BLOCK)
    synthetic_level = np.array([synthetic_activity[i:i+BLOCK].mean() for i in targets])
    sorted_starts = starts[np.argsort(source_level)]
    ranks = np.argsort(np.argsort(synthetic_level, kind='stable'), kind='stable')
    percentiles = np.clip((ranks+.5)/len(ranks)+rng.uniform(-.1, .1, len(ranks)), 0, .999)
    selected = sorted_starts[(percentiles*len(sorted_starts)).astype(int)]
    background = np.empty_like(annuli)
    y, x = np.indices(annuli.shape[1:], dtype=np.float32)
    for start, source in zip(targets, selected):
        count = min(length-start, BLOCK)
        coords = (y+rng.uniform(-.6, .6), x+rng.uniform(-.6, .6))
        background[start:start+count] = [map_coordinates(frame, coords, order=1, mode='nearest')
                                         for frame in annuli[source:source+count]]
        if start:
            blend = np.arange(1, min(3, count)+1, dtype=np.float32)[:, None, None]/4
            join = slice(start, start+len(blend))
            background[join] = blend*background[join]+(1-blend)*background[start-1]
    background -= np.mean(background[train], axis=0)
    return background, selected.tolist()


def projections(rows):
    return tuple(np.stack(column) for column in zip(*rows))


def activity_components(sampled, coarse, valid_tiles, centers):
    common = np.median(sampled, axis=1, keepdims=True)
    local = sampled-common
    positive = np.maximum(local, 0)
    events = positive.mean(axis=1)
    compression = np.maximum(1-np.median(events)/np.maximum(events, 1e-6), 0)
    soma_coords = ((centers.T+.5)/TILE-.5).astype(np.float32)
    regional = np.array([map_coordinates(tile-tile[valid_tiles].mean(), soma_coords,
                                         order=1, mode='nearest')/150 for tile in coarse])
    return local, positive, compression, common, median_center(regional)


def apply_activity_calibration(components, gain, peak_compression, shared, region):
    local, positive, compression, common, regional = components
    return np.maximum(local-positive+gain*(1-peak_compression*compression[:, None])*positive
                      +shared*common+region*regional, -1).astype(np.float32)


def evaluate(rows, held, centers, edges, low, high):
    tiles, rois, left, right = projections(rows)
    grain = grain_metrics(left[held], right[held], low, high, edges=edges)
    held_metrics = flash_metrics(tiles[held], rois[held], left[held], low, high)
    full_metrics = flash_metrics(tiles, rois, left, low, high)
    relative = median_center(rois)/np.median(rois, axis=0)
    plot_values = dict(metrics=held_metrics, grain=grain,
                       amplitude=np.quantile(rois, .95, axis=0)-np.median(rois, axis=0),
                       correlation=correlation_by_distance(relative, centers),
                       bright_tiles=100*np.mean(median_center(tiles) >= 10, axis=1),
                       bright_rois=100*np.mean(relative > .2, axis=1))
    report = dict(heldout_photometry=held_metrics, grain_by_brightness=grain, full_recording=full_metrics,
                  median_residual_pearson_r=serializable({'r': plot_values['correlation']})['r'])
    return plot_values, report


def restore_saved_report(source, saved, output, figures, diagnostic_seconds):
    config, meta, render = prepare_flash_replay(source, saved)
    low, high = meta['display_black_adu'], meta['display_white_adu']
    transpose = meta.get('transpose_for_display', True)
    indices = config['reference_gif_frames']
    diagnostic_count = min(3*diagnostic_seconds, len(indices))
    output.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.saved-render-') as tmp:
        stage = Path(tmp)/output.name
        stage.mkdir()
        movie = saved/'real_vs_synthetic.gif'
        layers = stage/'signal_and_background.gif'
        writer = make_gif(layers, (meta['width'], 4*meta['panel_height']))
        try:
            with Image.open(movie) as gif, tifffile.TiffFile(meta['real_source']) as real:
                if gif.n_frames != len(indices):
                    raise ValueError('The saved GIF and validation indices have different lengths.')
                check = sorted(set(range(diagnostic_count)) | {len(indices)//2, len(indices)-1})
                for number in check:
                    index = indices[number]
                    synthetic, soma, background = render(index)
                    np.testing.assert_allclose(soma+background, synthetic, atol=.002)
                    expected = np.concatenate([display_values(im, low, high, transpose) for im in
                                               (real.pages[index].asarray(), synthetic)])
                    gif.seek(number)
                    decoded = np.asarray(gif.convert('L'))
                    error = np.abs(decoded.astype(np.int16)-expected).max()
                    if error > 1:
                        raise ValueError(f'The saved and replayed frames differ by {error} display levels.')
                    if number < diagnostic_count:
                        parts = [display_values(im, 0, high, transpose) for im in (soma, background)]
                        writer.stdin.write(np.concatenate([decoded, *parts]).tobytes())
            finish_gif(writer, layers, diagnostic_count)
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait()
        save_temporal_median(layers)
        shutil.copyfile(movie, stage/movie.name)
        shutil.copyfile(saved/'activity.npz', stage/'activity.npz')
        with movie.open('rb') as file:
            config['gif_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
        config['saved_render'] = str(saved.resolve())
        config['optical_model'] = str(source.resolve())
        (stage/'validation.json').write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
        values, _, _ = measure_saved_report(recalibrated=True, previous=source, current=saved)
        plot(values, figures/'render_validation', dict(real='Plane 20', old='Earlier synthetic', new='Restored synthetic'))
        stage.rename(output)
    print(f'Restored the saved GIF and verified {len(check)} replayed frames.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--diagnostic-seconds', type=int, default=15)
    parser.add_argument('--saved-report', type=Path, help='Verify and restore the saved Plane20 render recipe.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory.')
    if args.saved_report is not None:
        restore_saved_report(args.source, args.saved_report, args.output, args.figures, args.diagnostic_seconds)
        return
    rng = np.random.default_rng(args.seed)
    state = prepare_render(args.source)
    meta = state['meta']
    base = state['base']
    sampled = state['modulation']
    length = len(sampled)
    time = np.arange(length)
    train = (time//BLOCK) % 2 == 0
    held = ~train
    fit = time[time % 5 == 2]
    fit_blocks = [np.flatnonzero(fit//BLOCK == block) for block in np.unique(fit//BLOCK)]
    fit_optics = train[fit]
    bank_path = Path(meta['bank'])
    source_dir = bank_path.parent.parent/'per_plane'
    plane = meta['reference_plane']
    with np.load(bank_path, allow_pickle=False) as arrays:
        bank = dict(arrays)
    reference = bank['plane'] == plane
    ids = bank['mask_id'][reference]
    reference_centers = bank['centers_yx'][reference]
    observed = raw_signal(source_dir, plane, ids, np.ones(len(ids), dtype=bool))
    with np.load(meta['profile'], allow_pickle=False) as arrays:
        tissue = arrays['tissue'].astype(bool)
    with np.load(meta['scene'], allow_pickle=False) as arrays:
        scene = dict(arrays)
    with np.load(args.source/'activity.npz', allow_pickle=False) as arrays:
        dff, columns = arrays['dff'], arrays['bank_columns']
    donor_scope = meta.get('donor_scope', 'other-planes')
    if donor_scope != 'all-planes' and np.any(bank['plane'][columns] == plane):
        raise ValueError('The activity sample includes donors from the reference plane in single-plane mode.')
    valid_tiles = tile_mean(tissue) > .7
    if not np.any(valid_tiles):
        valid_tiles = tile_mean(tissue) > 0
    if not np.any(valid_tiles):
        raise ValueError('No tissue lies within the tile grid for intensity calibration.')
    annuli = read_reference(meta, source_dir, bank, reference, np.flatnonzero(train), tissue.shape)
    coarse, diffuse_sources = background_windows(annuli, observed, sampled, train, valid_tiles, args.seed)
    assert train[np.array(diffuse_sources)].all()
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coords = ((y+.5)/TILE-.5, (x+.5)/TILE-.5)
    support = gaussian_filter(tissue.astype(np.float32), 12)
    masks = mask_matrix(scene)
    low, high = meta['display_black_adu'], meta['display_white_adu']
    candidates = np.flatnonzero(x.ravel() < tissue.shape[1]-1)
    pixels = rng.choice(candidates, min(4500, len(candidates)), replace=False)
    fit_pixels = slice(0, 1400)

    def diffuse(i):
        return map_coordinates(coarse[i], coords, order=1, mode='nearest')*support

    def signal(values):
        core = (values @ state['maps']).reshape(tissue.shape)
        halo = halo_light(core)
        return core+halo, halo

    def measure(image, rois=None):
        flat = image.ravel()
        if rois is None:
            rois = masks @ flat
        return tile_mean(image)[valid_tiles], rois, flat[pixels], flat[pixels+1]

    def project(image):
        tiles, rois, left, _ = measure(image)
        return tiles, rois, left[fit_pixels]

    activity_parts = activity_components(sampled, coarse, valid_tiles, scene['centers'])
    local, positive, compression, common, regional = activity_parts

    with (tifffile.TiffFile(meta['real_source']) as real,
          ThreadPoolExecutor(max_workers=3) as workers):
        ref_rows = [measure(real.pages[int(i)].asarray(), observed[i]) for i in fit]
        noisy = list(workers.map(lambda i: measure(render_layers(state, i)[0]), fit[fit_optics]))
        _, _, left, right = projections(ref_rows)
        _, _, old_left, old_right = projections(noisy)
        brightness = np.median(left[fit_optics], axis=0)
        edges = np.unique(np.quantile(brightness, [.15, .36, .57, .76, .91, .99])).tolist()
        ref_grain = grain_metrics(left[fit_optics], right[fit_optics], low, high, edges=edges)
        base_grain = grain_metrics(old_left, old_right, low, high, edges=edges)
        correction = dict(intensity_adu=[], scale=[])
        for target, prior in zip(ref_grain, base_grain):
            if min(target['pairs'], prior['pairs']) >= 20 and prior['robust_sd_adu'] > 0:
                correction['intensity_adu'].append(float(np.mean(target['range_adu'])))
                correction['scale'].append(float(np.clip(target['robust_sd_adu']/prior['robust_sd_adu'], .55, 1.35)))
        if not correction['scale']:
            correction = None
        noise_config = meta['detector_noise']
        noise_factor = (np.interp(base, correction['intensity_adu'], correction['scale']) if correction else 1.)

        def basis(i):
            noise, halo_noise = add_optical_noise(base, state['halo_static'], noise_config['intensity_adu'],
                                                 noise_config['sigma_adu'], state['seeds'][i], state['halo_seeds'][i])
            return tuple(map(project, (signal(local[i]-positive[i])[0], signal(positive[i])[0],
                                       signal(np.broadcast_to(common[i], local[i].shape))[0], signal(regional[i])[0],
                                       diffuse(i), (noise+halo_noise)*noise_factor)))

        print('Measuring optical and activity projections.', flush=True)
        projected = list(workers.map(basis, fit))
    components = [projections([frame[k] for frame in projected]) for k in range(6)]
    del projected
    ref_tiles, ref_rois, ref_pixels, _ = projections(ref_rows)
    real_pairs = spatial_pairs(reference_centers, rng)
    synthetic_pairs = spatial_pairs(scene['centers'], rng)

    def statistics(tiles, rois, px, pairs):
        core = training_statistics(tiles, rois, px, pairs, fit_blocks, low, high, fit_optics)
        extent = np.quantile(median_center(tiles), .9, axis=1)
        relative = median_center(rois)/np.median(rois, axis=0)
        occupancy = 100*expit((relative-.2)/.012).mean(axis=1)
        return np.r_[core, 100*expit((extent-10)/1.1).mean(), np.quantile(occupancy, [.95,.99])]

    target = statistics(ref_tiles, ref_rois, ref_pixels[:, fit_pixels], real_pairs)
    tolerances = np.r_[1.4, 1.4, 2, 3, 1.2, 2, 3, 1.4, .016, .027, .018,
                       .095, .085, .065, .065, .06, .06, .06, .06, .65, .85, 1., 2., 3., 1.2, .45, 3., .8, 2.5]
    baseline, dark = project(base), project(dark_weights(state))

    def predict(parameters):
        gain, peak_compression, shared, region, broad, tone, offset, dark_offset = parameters
        gains = (gain*(1-peak_compression*compression[fit]))[:, None]
        return tuple(150+tone*(original-150)+offset+dark_offset*weight+negative+gains*positive
                     +shared*common+region*regional+broad*diffuse+noise
                     for original, weight, negative, positive, common, regional, diffuse, noise
                     in zip(baseline, dark, *components))

    def objective(parameters):
        estimate = statistics(*predict(parameters), synthetic_pairs)
        errors = ((estimate-target)/tolerances)**2
        return np.nanmean(errors)

    initial = [4, .35, 1, .5, .35, 1, 0, 0]
    bounds = [(.1, 10), (0, 1), (0, 7), (0, 5), (0, 3.), (.8, 1.2), (-14, 14), (-15, 15)]
    print('Fitting image brightness, local activity and diffuse events.', flush=True)
    fitted = minimize(objective, initial, method='Powell', bounds=bounds,
                      options={'maxiter': 14, 'xtol': .025, 'ftol': .008})
    coefficients = tuple(map(float, fitted.x))
    gain, peak_compression, shared, region, broad, tone, offset, dark_offset = coefficients
    config = dict(zip((*PARAMETERS, 'dark_offset_adu'), coefficients))
    print('Optical parameters', config, flush=True)
    baselines = calibrated_baselines(state, tone, offset, dark_offset)
    modulation = apply_activity_calibration(activity_parts, gain, peak_compression, shared, region)

    def render(i):
        total, halo = signal(modulation[i])
        return render_flash_frame(state, i, total, halo, broad*diffuse(i), baselines, correction)

    shown = meta['displayed_source_frames'][:180]
    diagnostic = shown[:args.diagnostic_seconds*3]
    shown, diagnostic = set(shown), set(diagnostic)
    width, height = meta['width'], meta['panel_height']
    transpose = meta['transpose_for_display']
    args.figures.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.calibrated-render-') as tmp:
        stage = Path(tmp)/args.output.name
        stage.mkdir()
        gif, layers = stage/'real_vs_synthetic.gif', stage/'signal_and_background.gif'
        writers = [make_gif(gif, (width, 2*height)), make_gif(layers, (width, 4*height))]
        results = dict(real=[], new=[])
        print(f'Rendering and measuring all {length} frames.', flush=True)
        try:
            with (tifffile.TiffFile(meta['real_source']) as real,
                  ThreadPoolExecutor(max_workers=3) as workers):
                for i, (synthetic, soma, background) in enumerate(workers.map(render, time)):
                    observed_frame = real.pages[i].asarray()
                    results['real'].append(measure(observed_frame, observed[i]))
                    results['new'].append(measure(synthetic))
                    if i in shown:
                        panels = [display_values(im, low, high, transpose) for im in (observed_frame, synthetic)]
                        writers[0].stdin.write(np.concatenate(panels).tobytes())
                        if i in diagnostic:
                            np.testing.assert_allclose(soma+background, synthetic, atol=.002)
                            panels.extend(display_values(im, 0, high, transpose) for im in (soma, background))
                            writers[1].stdin.write(np.concatenate(panels).tobytes())
            finish_gif(writers[0], gif, len(shown))
            finish_gif(writers[1], layers, len(diagnostic))
        finally:
            for writer in writers:
                if writer.poll() is None:
                    writer.kill()
                    writer.wait()
        save_temporal_median(layers)
        figure_data, comparisons = {}, {}
        for key, centers in (('real', reference_centers), ('new', scene['centers'])):
            figure_data[key], comparisons[key] = evaluate(results[key], held, centers, edges, low, high)
        plot(figure_data, args.figures/'render_validation', dict(real=f'Plane {plane}', new='Synthetic'))
        config.update(grain_correction=correction, display_range_adu=[low, high],
                      donor_scope=donor_scope,
                      photometry_fit_frames=fit[fit_optics].tolist(),
                      heldout_photometry_frames=np.flatnonzero(held).tolist(),
                      matched_dynamics_frames=length, diffuse_source_blocks=diffuse_sources,
                      reference_gif_frames=sorted(shown), training_objective=float(fitted.fun),
                      activity_sample=meta['activity_sample'], comparison=comparisons)
        (stage/'validation.json').write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
        np.savez_compressed(stage/'activity.npz', dff=dff, bank_columns=columns,
                            render_modulation=modulation, diffuse_tile_changes=coarse)
        stage.rename(args.output)
    print(f'Saved the calibrated render to {args.output}', flush=True)


if __name__ == '__main__':
    main()
