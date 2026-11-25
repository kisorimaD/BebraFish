"""Render the existing Plane20 scene with measured Convert_Crop fluorescence traces."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import imageio_ffmpeg
import numpy as np
import tifffile

from coordinate_traces import (
    check_coordination, check_groups, coordinate_traces, make_groups, measure_group_target, roll_traces,
)
from glare import redistribute_glare
from reference_photometry import fit_brightness, noise_curve, validate_comparison_gif
from spatial_amplitude import plane20_reference, plot_amplitude, spatial_assignment
from spatial_baseline import calibrate_baseline, plot_baseline
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / 'plane20_synthetic' / 'plane20_synthetic'
sys.path.insert(0, str(PACKAGE))

from optical_layers import add_optical_noise, halo_light, optical_layers
from soma_maps import soma_maps
from training.plane20.dataset import PROFILE
from training.plane20.photometry import correlated_noise


BANK = HERE.parent / 'convert_crop_activity' / 'foreground_conservative' / 'trace_bank.npz'
SCENE = PACKAGE / 'smoke_median/train/scene_00000.npz'
REFERENCE_TIFF = HERE.parents[1] / '20259815_Soma_GCaMP8f_Convert_Crop/original/Plane_Aligned_20.tiff'
REFERENCE_MEDIAN = BANK.parents[1] / 'per_plane/plane_20/median.tiff'
GIF_FPS = 3
MAX_GIF_SECONDS = 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--scene', type=Path, default=SCENE)
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--reference-tiff', type=Path, default=REFERENCE_TIFF)
    parser.add_argument('--reference-median', type=Path, default=REFERENCE_MEDIAN)
    parser.add_argument('--reference-plane', type=int, default=20)
    parser.add_argument('--source-fps', type=float, help='Acquisition rate. If omitted, displays at the legacy 12.5 Hz time scale.')
    parser.add_argument('--seed', type=int, default=20261125)
    parser.add_argument('--independent', action='store_true',
                        help='Use the original common timeline and random placement.')
    parser.add_argument('--no-detector-noise', action='store_true',
                        help='Render without the Plane20 detector noise layer.')
    parser.add_argument('--noise-scale', type=float, default=1.,
                        help='Scale the measured Plane20 detector noise profile.')
    parser.add_argument('--glare-reduction', type=float, default=.45,
                        help='Fraction of excess illumination removed from the cell-free background.')
    parser.add_argument('--legacy-optics', action='store_true',
                        help='Use the previous composite background without soma-generated halos.')
    parser.add_argument('--uniform-amplitudes', action='store_true',
                        help='Place sampled activity traces without using the local amplitude field.')
    parser.add_argument('--original-baseline', action='store_true',
                        help='Keep the uncalibrated synthetic scene brightness.')
    parser.add_argument('--pairwise-only', action='store_true',
                        help='Match median correlations without explicit groups.')
    args = parser.parse_args()
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0:
        parser.error('--noise-scale must be nonnegative and finite.')
    if not 0 <= args.glare_reduction <= 1:
        parser.error('--glare-reduction must be between zero and one.')
    if args.output.exists():
        parser.error('Choose a new output directory.')
    if args.source_fps is not None and (not np.isfinite(args.source_fps) or args.source_fps < GIF_FPS):
        parser.error('--source-fps must be finite and at least 3.')

    with np.load(args.scene, allow_pickle=False) as source:
        scene = dict(source)
    with np.load(args.profile, allow_pickle=False) as source:
        profile = dict(source)
    cells = len(scene['centers'])
    bank_path = args.bank.resolve()
    bank_summary = json.loads((bank_path.parent / 'summary.json').read_text())
    spatial_amplitude = None
    with np.load(bank_path, allow_pickle=False) as source:
        bank = source['dff']
        if not np.isfinite(bank).all() or bank.shape[1] < cells:
            raise ValueError('The bank must contain one finite trace per cell.')
        available = np.flatnonzero(source['plane'] != args.reference_plane)
        if len(available) < cells:
            raise ValueError('The donor planes must contain at least one accepted trace per synthetic soma.')
        # Preserve the historical random seed order for the original Plane20 benchmark.
        pool = bank.shape[1] if args.bank.resolve() == BANK and not args.uniform_amplitudes else available
        columns = np.random.default_rng(args.seed).choice(pool, cells, replace=False)
        if not args.uniform_amplitudes:
            columns, expected, donor_amplitudes, assigned_map, spatial_amplitude = spatial_assignment(
                source, columns, scene['centers'], args.seed + 4, args.reference_plane)
            print('Spatial amplitude validation:', spatial_amplitude, flush=True)
        reference_centers = source['centers_yx'][plane20_reference(source, args.reference_plane)]
        measured = bank[:, columns]
        planes, mask_ids = source['plane'][columns], source['mask_id'][columns]
        donor_centers = source['centers_yx'][columns]
    assert len(set(zip(planes, mask_ids))) == cells
    frames = len(measured)
    offsets = np.zeros(cells, dtype=int)
    groups = None
    coordination = None
    if not args.independent:
        reference = json.loads((bank_path.parent / 'coordination.json').read_text())
        prominence = reference['peak_prominence_dff']
        targets = dict(peak_prominence=prominence, near_correlation=reference['by_distance'][0]['correlation_residual'],
                       mid_correlation=reference['by_distance'][1]['correlation_residual'],
                       near_peak_coincidence=reference['near_peak_sensitivity']['coincidence'])
        if not args.pairwise_only:
            groups = make_groups(scene['centers'], args.seed + 3)
            group_target, group_reference = measure_group_target(bank_path, groups, prominence)
            targets.update(near_p90=reference['by_distance'][0]['p90_correlation_residual'],
                           mid_p90=reference['by_distance'][1]['p90_correlation_residual'],
                           group_threshold=reference['by_distance'][1]['p90_correlation_residual'],
                           group_triple_frames=group_target)
        offsets = coordinate_traces(measured, scene['centers'], args.seed + 1, targets, groups)
        coordinated = roll_traces(measured, offsets)
        first_check = check_coordination(coordinated, scene['centers'], args.seed, prominence)
        # The phase search uses unadjusted pair correlations. Correct its targets
        # for the common signal seen in the rendered scene and evaluate again.
        search_targets = targets.copy()
        for band, key in enumerate(('near_correlation', 'mid_correlation')):
            search_targets[key] += first_check[band]['correlation'] - first_check[band]['correlation_residual']
        if groups is not None:
            for band, key in enumerate(('near_p90', 'mid_p90')):
                search_targets[key] += first_check[band]['correlation'] - first_check[band]['correlation_residual']
        adjusted_offsets = coordinate_traces(measured, scene['centers'], args.seed + 1, search_targets, groups)
        adjusted = roll_traces(measured, adjusted_offsets)
        adjusted_check = check_coordination(adjusted, scene['centers'], args.seed, prominence)

        def target_error(metrics):
            observed = [metrics[0]['correlation_residual'], metrics[1]['correlation_residual'],
                        metrics[0]['coincidence']]
            expected = [targets[key] for key in ('near_correlation', 'mid_correlation', 'near_peak_coincidence')]
            if groups is not None:
                observed.extend(metrics[band]['p90_correlation_residual'] for band in range(2))
                expected.extend(targets[key] for key in ('near_p90', 'mid_p90'))
            return np.linalg.norm(np.array(observed) - expected)

        selected_check = first_check
        if target_error(adjusted_check) < target_error(first_check):
            offsets, coordinated, selected_check = adjusted_offsets, adjusted, adjusted_check
        np.testing.assert_array_equal(roll_traces(coordinated, -offsets), measured)
        coordination = dict(targets=targets,
                            previous=check_coordination(measured, scene['centers'], args.seed, prominence),
                            first_pass=first_check, coordinated=selected_check)
        if groups is not None:
            coordination['groups'] = dict(reference_by_group_size=group_reference,
                                          original_timeline=check_groups(measured, groups, prominence),
                                          coordinated=check_groups(coordinated, groups, prominence))
        measured = coordinated
        print('Closest-pair validation:', coordination['coordinated'][0], flush=True)
    fps = args.source_fps or 12.5
    # The static scene represents a temporal median. Scale each soma around
    # that median and floor background-corrected negative light at zero.
    relative_light = np.maximum(1 + measured, 0)
    medians = np.median(relative_light, axis=0)
    if np.any(medians <= 0):
        raise ValueError('A selected donor has nonpositive median fluorescence.')
    modulation = (relative_light / medians - 1).astype(np.float32)
    np.testing.assert_allclose(np.median(modulation, axis=0), 0, atol=1e-6)
    print(f'Assigned {cells} unique measured traces, {frames} frames each', flush=True)

    maps = soma_maps(scene, profile)
    base = scene['image'].astype(np.float32)
    noise_seed = args.seed + 2
    noise_rng = np.random.default_rng(noise_seed)
    noise_fraction = 0.0 if args.no_detector_noise else args.noise_scale
    sample = (modulation[::20] @ maps).reshape(-1, *base.shape) + base
    reference_image = tifffile.imread(args.reference_median)
    legacy_profile = args.profile.resolve() == PROFILE.resolve()
    display_reference = sample if legacy_profile else reference_image
    black = float(np.percentile(base if legacy_profile else reference_image, 1))
    white = float(max(black + 1, np.percentile(display_reference, 99.8)))
    noise_levels, noise_estimates = noise_curve(args.reference_tiff, adaptive=not legacy_profile)
    if not np.isfinite(sample).all() or not np.any(np.ptp(sample, axis=0) > 0):
        raise ValueError('The rendered samples must be finite and time-varying.')
    spatial_baseline = None
    optical = None
    halo_static = None
    if not args.legacy_optics:
        maps, layers, optical = optical_layers(scene, profile, reference_image, maps, 1 - args.glare_reduction)
        base = sum(layers)
        halo_static = layers[-1]
        print('Optical layer verification:', optical, flush=True)
    elif not args.original_baseline:
        base, baseline_maps, spatial_baseline = calibrate_baseline(
            base, reference_image, scene['centers'], reference_centers, maps)
        print('Spatial baseline verification:', spatial_baseline, flush=True)
    glare = None
    with np.load(args.reference_median.parent / 'masks/instances.npz', allow_pickle=False) as roi:
        reference_mask_pixels = roi['instance_pixel_index']
    tissue = profile['tissue'].astype(bool)
    if args.legacy_optics and args.glare_reduction:
        base, glare = redistribute_glare(base, reference_image, reference_mask_pixels,
                                         tissue, maps, args.glare_reduction, white)
        print('Glare redistribution:', glare, flush=True)
    base, photometry = fit_brightness(base, maps, modulation, args.reference_tiff, noise_levels,
                                     noise_estimates, noise_fraction, black, white,
                                     halo_filter=halo_light if optical is not None else None,
                                     adaptive=not legacy_profile)
    print('Brightness held-out validation:', photometry, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        output = Path(temporary) / args.output.name
        output.mkdir()
        movie = output / 'plane20_recorded_activity.gif'
        gif_frames = min(MAX_GIF_SECONDS * GIF_FPS, int(frames * GIF_FPS / fps))
        source_indices = ((np.arange(gif_frames) + .5) * fps / GIF_FPS).astype(int)
        # Display the long axis horizontally in both panels.
        transpose = base.shape[0] >= base.shape[1]
        panel_width, panel_height = base.shape if transpose else base.shape[::-1]
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pixel_format', 'gray',
                   '-video_size', f'{panel_width}x{2 * panel_height}',
                   '-framerate', str(GIF_FPS), '-i', 'pipe:0',
                   '-filter_complex_threads', '1', '-filter_complex',
                   'split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none',
                   '-loop', '0', str(movie)]
        noise_seeds = noise_rng.integers(2**32, size=frames)
        halo_seeds = noise_rng.integers(2**32, size=frames)

        def display(frame):
            oriented = frame.T if transpose else frame
            scaled = np.rint((oriented.astype(np.float32) - black) * (255 / (white - black)))
            return np.clip(scaled, 0, 255).astype(np.uint8)

        with tifffile.TiffFile(args.reference_tiff) as real:
            if real.pages[0].shape != base.shape or len(real.pages) <= source_indices[-1]:
                raise ValueError('The real Plane20 recording does not match the rendered frames.')
            with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
                with writer.stdin:
                    for indices in np.array_split(source_indices, max(1, (gif_frames + 15) // 16)):
                        activity = (modulation[indices] @ maps).reshape(-1, *base.shape)
                        halo_activity = halo_light(activity) if optical is not None else np.zeros_like(activity)
                        block = base + activity + halo_activity
                        for source_index, frame, halo_delta in zip(indices, block, halo_activity):
                            if noise_fraction:
                                if optical is None:
                                    noise, _ = correlated_noise(base.shape, noise_seeds[source_index])
                                    noise_sigma = noise_fraction * np.interp(frame, noise_levels, noise_estimates)
                                    frame = frame + noise_sigma * noise
                                else:
                                    background_noise, halo_noise = add_optical_noise(
                                        frame, halo_static + halo_delta, noise_levels, noise_fraction * noise_estimates,
                                        noise_seeds[source_index], halo_seeds[source_index])
                                    frame = frame + background_noise + halo_noise
                            reference_frame = real.pages[int(source_index)].asarray()
                            stacked = np.concatenate((display(reference_frame), display(frame)), axis=0)
                            writer.stdin.write(stacked.tobytes())
                if writer.wait():
                    raise RuntimeError('GIF encoding failed.')
        duration_ms, gif_validation = validate_comparison_gif(
            movie, source_indices, panel_width, panel_height, black, white, adaptive=not legacy_profile)
        assert abs(duration_ms - gif_frames * 1000 / GIF_FPS) <= 20
        assert duration_ms <= MAX_GIF_SECONDS * 1000
        if spatial_amplitude is not None:
            plot_amplitude(output / 'spatial_amplitude.png', scene, expected, assigned_map, args.reference_plane)
        if spatial_baseline is not None:
            plot_baseline(output / 'spatial_baseline.png', scene, baseline_maps)
        np.savez_compressed(output / 'activity.npz', dff=measured,
                            bank_columns=columns, shift_frames=offsets)

        with (output / 'assignments.csv').open('w', newline='') as file:
            table = csv.writer(file)
            table.writerow(['scene_cell', 'bank_column', 'plane', 'source_mask_id',
                            'source_y', 'source_x', 'scene_y', 'scene_x', 'circular_shift_frames', 'group_id',
                            'local_expected_amplitude', 'donor_amplitude'])
            for cell in range(cells):
                table.writerow([cell, columns[cell], planes[cell], mask_ids[cell],
                                *donor_centers[cell], *scene['centers'][cell], offsets[cell],
                                groups[cell] if groups is not None else -1,
                                *([expected[cell], donor_amplitudes[cell]] if spatial_amplitude is not None else [None, None])])
        metadata = dict(bank=str(bank_path), input_signal=bank_summary.get('relative_signal', 'ΔF/F'),
                        scene=str(args.scene.resolve()), profile=str(args.profile.resolve()),
                        reference_median=str(args.reference_median.resolve()), reference_plane=args.reference_plane,
                        seed=args.seed, activity_frames=frames,
                        playback_fps=GIF_FPS, gif_frames=gif_frames, gif_duration_ms=duration_ms,
                        reference_playback_fps=fps, source_fps=args.source_fps,
                        width=panel_width, height=2 * panel_height, panel_height=panel_height,
                        panels=['real', 'synthetic'], real_source=str(args.reference_tiff.resolve()),
                        transpose_for_display=transpose,
                        displayed_source_frames=source_indices.tolist(),
                        source_planes=sorted(set(map(int, planes))), unique_donors=cells,
                        light_floor_count=int(np.count_nonzero(measured < -1)),
                        display_black_adu=black, display_white_adu=white,
                        coordination=coordination, spatial_amplitude=spatial_amplitude,
                        spatial_baseline=spatial_baseline, glare=glare, optical_layers=optical, photometry=photometry,
                        gif_validation=gif_validation,
                        temporal_transform='Full traces are circularly shifted by the offsets in '
                        'assignments.csv. GIF selects frames from those traces without interpolation. '
                        'Relative light max(0, 1 + input signal) divided by each trace median.',
                        detector_noise=dict(intensity_adu=noise_levels.tolist(),
                                            sigma_adu=(noise_fraction * noise_estimates).tolist(),
                                            seed=noise_seed, scale=noise_fraction))
        (output / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
        output.rename(args.output)
    print(f'Verified {gif_frames} GIF frames, {duration_ms / 1000:.1f} seconds: {args.output}', flush=True)


if __name__ == '__main__':
    main()
