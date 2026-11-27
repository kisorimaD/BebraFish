"""Render other activity draws with the exact optics fitted to the reference draw."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import tifffile

from calibrate_flash_dynamics import (TILE, calibrated_baselines, finish_gif, make_gif, mask_matrix,
                                     raw_signal, read_reference, render_flash_frame, tile_mean)
from optical_layers import halo_light
from plot_render_components import display_values, prepare_render
from render_calibrated_dataset import (activity_components, apply_activity_calibration,
                                       background_windows, evaluate)
from render_recorded_bank import HERE


def fingerprint(filename):
    with filename.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def table_row(label, metrics, reference):
    photometry = metrics['heldout_photometry']
    dynamics = metrics['full_recording']
    grain = []
    for a, b in zip(metrics['grain_by_brightness'], reference['grain_by_brightness']):
        first, second = a['robust_sd_adu'], b['robust_sd_adu']
        if first is not None and second is not None:
            grain.append(abs(first-second))
    return dict(variant=label, mean_adu=photometry['displayed_mean_adu'], sd_adu=photometry['displayed_std_adu'],
                intensity_p98_adu=photometry['displayed_quantiles_adu'][-2],
                grain_mean_absolute_error_adu=float(np.mean(grain)) if grain else None,
                black_clipped_pct=photometry['pct_clipped_black'], white_clipped_pct=photometry['pct_clipped_white'],
                soma_median_amplitude_adu=dynamics['roi_median_p95_minus_p50_adu'],
                wide_flash_frames_pct=dynamics['pct_frames_with_wide_flash'],
                bright_soma_pct_p95=dynamics['roi_simultaneous_up_20pct_p95'],
                nearest_soma_residual_r=metrics['median_residual_pearson_r'][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE/'one_pass_plane20_original_masks',
                        help='Completed calibrated run, or its planes/plane_N directory.')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive.')
    run = args.run.resolve()
    source = run/'optical_model'
    archived = run/'report'
    calibration_path = archived/'validation.json'
    config = json.loads(calibration_path.read_text())
    reference_report = config['comparison']['real']
    fitted_sample = Path(config['activity_sample'])
    samples = run/'activity_variants_5'
    if not samples.is_dir():
        samples = run/'activity_resamples_coordinated'
    paths = sorted(samples.glob('activity_seed_*.npz'),
                   key=lambda path: int(path.stem.rsplit('_', 1)[-1]))
    if not paths or fitted_sample.name not in {path.name for path in paths}:
        raise ValueError('The activity directory must include the calibrated draw.')
    root = run/'video_variants'
    root.mkdir(exist_ok=True)
    calibration_sha = fingerprint(calibration_path)
    optics_sha = fingerprint(source/'generation.json')
    state = prepare_render(source)
    meta = state['meta']
    low, high = config['display_range_adu']
    time = np.arange(config['matched_dynamics_frames'])
    held = np.isin(time, config['heldout_photometry_frames'])
    train = ~held
    np.testing.assert_array_equal([low, high], [meta['display_black_adu'], meta['display_white_adu']])
    if np.any(held[config['photometry_fit_frames']]):
        raise ValueError('The calibration file has overlapping photometry splits.')
    bank_path = Path(meta['bank'])
    with np.load(meta['scene'], allow_pickle=False) as arrays:
        scene = dict(arrays)
    with np.load(meta['profile'], allow_pickle=False) as arrays:
        tissue = arrays['tissue'].astype(bool)
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_plane = bank['plane']
        reference = bank_plane == meta['reference_plane']
        ids = bank['mask_id'][reference]
        per_plane = bank_path.parent.parent/'per_plane'
        observed = raw_signal(per_plane, meta['reference_plane'], ids, np.ones(len(ids), dtype=bool))
        annuli = read_reference(meta, per_plane, bank, reference, np.flatnonzero(train), tissue.shape)
    valid_tiles = tile_mean(tissue) > .7
    if not valid_tiles.any():
        valid_tiles = tile_mean(tissue) > 0
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coords = ((y+.5)/TILE-.5, (x+.5)/TILE-.5)
    support = gaussian_filter(tissue.astype(np.float32), 12)
    masks = mask_matrix(scene)
    pool = np.flatnonzero(x.ravel() < tissue.shape[1]-1)
    pixels = np.random.default_rng(meta['seed']).choice(pool, min(4500, len(pool)), replace=False)
    edges = [reference_report['grain_by_brightness'][0]['range_adu'][0],
             *(row['range_adu'][1] for row in reference_report['grain_by_brightness'])]
    baselines = calibrated_baselines(state, config['static_gain'], config['static_offset_adu'],
                                     config.get('dark_offset_adu', 0.))
    gain_args = [config[name] for name in ('positive_soma_gain', 'peak_compression',
                                          'common_soma_gain', 'regional_soma_gain')]
    indices = config['reference_gif_frames']
    previews = {indices[i] for i in (0, len(indices)//2, len(indices)-1)}
    transpose = meta.get('transpose_for_display', True)
    with tifffile.TiffFile(meta['real_source']) as recording:
        observed_panels = {i: display_values(recording.pages[i].asarray(), low, high, transpose) for i in indices}

    def prepare_activity(path):
        with np.load(path, allow_pickle=False) as arrays:
            dff = arrays['dff']
            np.testing.assert_array_equal(arrays['centers_yx'], scene['centers'])
            sample_scope = str(arrays['donor_scope']) if 'donor_scope' in arrays else 'other-planes'
            donor_scope = meta.get('donor_scope', 'other-planes')
            if sample_scope != donor_scope:
                raise ValueError(f'The sample uses a different donor scope from the fitted sample: {path}')
            if donor_scope == 'other-planes' and np.any(bank_plane[arrays['bank_columns']] == meta['reference_plane']):
                raise ValueError('The reference plane appears in the synthetic donor set.')
        if dff.shape != (len(time), len(scene['centers'])) or not np.isfinite(dff).all():
            raise ValueError(f'The sample does not match the saved scene and timeline: {path}')
        light = np.maximum(1+dff, 0)
        sampled = (light/np.median(light, axis=0)-1).astype(np.float32)
        coarse, blocks = background_windows(annuli, observed, sampled, train, valid_tiles, meta['seed'])
        if not np.all(train[blocks]):
            raise ValueError('The diffuse background sampled an evaluation block.')
        parts = activity_components(sampled, coarse, valid_tiles, scene['centers'])
        modulation = apply_activity_calibration(parts, *gain_args)
        return coarse, blocks, modulation, dff

    source_copy = next(path for path in paths if path.name == fitted_sample.name)
    coarse, blocks, modulation, dff = prepare_activity(source_copy)
    with np.load(archived/'activity.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(dff, arrays['dff'])
        np.testing.assert_array_equal(coarse, arrays['diffuse_tile_changes'])
        np.testing.assert_allclose(modulation, arrays['render_modulation'], rtol=3e-6, atol=1e-7)
        assert blocks == config['diffuse_source_blocks']
    print(f'Verified that the frozen recipe reproduces the original {source_copy.stem}.', flush=True)

    def render_variant(path, directory):
        provenance = dict(activity_sample=str(path), activity_sha256=fingerprint(path),
                          optical_model=str(source/'generation.json'), optical_model_sha256=optics_sha,
                          calibration=str(calibration_path), calibration_sha256=calibration_sha)
        if (directory/'validation.json').exists():
            stored = json.loads((directory/'validation.json').read_text())
            if stored['inputs'] != provenance or not (directory/'real_vs_synthetic.gif').is_file():
                raise ValueError(f'Existing output no longer matches its inputs: {directory}')
            return stored['comparison']
        selected, blocks, modulation, _ = prepare_activity(path)

        def render(index):
            broad = map_coordinates(selected[index], coords, order=1, mode='nearest')*support
            signal = (modulation[index] @ state['maps']).reshape(tissue.shape)
            halo = halo_light(signal)
            synthetic, _, _ = render_flash_frame(state, index, signal+halo, halo,
                                                 config['diffuse_gain']*broad, baselines,
                                                 config.get('grain_correction'))
            flat = synthetic.ravel()
            panel = display_values(synthetic, low, high, transpose) if index in observed_panels else None
            return (tile_mean(synthetic)[valid_tiles], masks@flat, flat[pixels], flat[pixels+1]), panel

        with tempfile.TemporaryDirectory(prefix='.video-validation-',dir=root) as temporary:
            stage = Path(temporary)/directory.name
            stage.mkdir()
            movie = stage/'real_vs_synthetic.gif'
            writer = make_gif(movie,(meta['width'],2*meta['panel_height']))
            rows, check = [], {}
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as workers:
                    for i,(projected, panel) in zip(time,workers.map(render,time)):
                        rows.append(projected)
                        if panel is not None:
                            image = np.concatenate((observed_panels[i],panel))
                            writer.stdin.write(image.tobytes())
                            if i in previews:
                                check[i] = image
                finish_gif(writer,movie,len(indices))
            finally:
                if writer.poll() is None:
                    writer.kill()
                    writer.wait()
            with Image.open(movie) as gif:
                for index,image in check.items():
                    gif.seek(indices.index(index))
                    error = np.abs(np.asarray(gif.convert('L')).astype(np.int16)-image).max()
                    if error > 1:
                        raise ValueError(f'Preview differs at frame {index} by {error} display levels.')
            _, metrics = evaluate(rows, held, scene['centers'], edges, low, high)
            record = dict(inputs=provenance, comparison=metrics, diffuse_source_blocks=blocks,
                          same_detector_noise_seeds=True, photometry_frames=int(held.sum()),
                          measured_frames=len(rows), gif_frames=len(indices), playback_fps=3)
            (stage/'validation.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            stage.rename(directory)
        return metrics

    results = [(f"Plane{meta['reference_plane']}", reference_report)]
    for path in paths:
        label = path.stem.removeprefix('activity_')
        if path == source_copy:
            results.append((label, config['comparison']['new']))
            continue
        print(f'Checking {label} with frozen optics.',flush=True)
        metrics = render_variant(path, root/label)
        results.append((label, metrics))
        print(label, table_row(label, metrics, reference_report),flush=True)

    rows = [table_row(label, metrics, reference_report) for label, metrics in results]
    with tempfile.TemporaryDirectory(prefix='.metrics-',dir=root) as temporary:
        output = Path(temporary)/'comparison.csv'
        with output.open('w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows({k:round(v,5) if isinstance(v,float) else v for k,v in row.items()} for row in rows)
        output.replace(root/output.name)
    print(f'Finished: {root}',flush=True)


if __name__ == '__main__':
    main()
