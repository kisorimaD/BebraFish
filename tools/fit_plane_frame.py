"""Exploratory single-frame SimCalc adaptation; does not alter frozen calibration."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (gaussian_filter, binary_fill_holes, binary_opening,
                           binary_erosion, label)
from scipy.optimize import nnls
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simcalc_zebrafish_pipeline'))
from calibrated_simcalc import render


def band_variances(image, mask):
    return np.array([np.var((gaussian_filter(image, a) - gaussian_filter(image, b))[mask])
                     for a, b in [(0, 1), (1, 2), (2, 4), (4, 8)]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'data/Plane_Aligned_20.tiff')
    p.add_argument('--frame', type=int, default=500, help='Zero-based TIFF page index')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/plane20_trial')
    p.add_argument('--seed', type=int, default=20261113)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'profiles/fish4_plane16/calibration.json').read_text())
    with np.load(ROOT/'profiles/fish4_plane16/profile.npz') as z:
        old_area = int(z['tissue'].sum())
    with tifffile.TiffFile(args.input) as tif:
        n = len(tif.pages)
        if not 0 <= args.frame < n:
            raise ValueError(f'Frame must be in [0, {n-1}]')
        frame = tif.pages[args.frame].asarray().astype(np.float32)
        start = min(max(args.frame-8, 0), max(n-16, 0))
        burst = np.stack([tif.pages[i].asarray() for i in range(start, min(start+16, n))]).astype(np.float32)
    if len(burst) < 4:
        raise ValueError('At least four frames are required for exploratory noise estimation')
    mean = np.mean(burst, axis=0)
    coarse = gaussian_filter(frame, 12)
    # The darkest 10% of the temporal mean serves as effective camera background.
    dark = mean <= np.percentile(mean, 10)
    background = float(np.median(mean[dark]))
    tissue = binary_fill_holes(binary_opening(coarse > background+20, iterations=2))
    components, _ = label(tissue)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    tissue = sizes[components] > max(1000, sizes.max()*.01)
    interior = binary_erosion(tissue, iterations=8)
    # Adjacent-frame differences include calcium changes and residual registration.
    differences = (burst[1::2] - burst[::2]) / np.sqrt(2.)
    hp_differences = differences - gaussian_filter(differences, (0, 1.2, 1.2))
    impulse = np.zeros((33, 33)); impulse[16, 16] = 1
    white_response = np.sum((impulse-gaussian_filter(impulse, 1.2))**2)
    variance = np.mean(hp_differences**2, axis=0) / white_response
    signal = np.maximum(mean-background, 0)
    edges = np.unique(np.percentile(signal, np.linspace(0, 99, 16)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (signal >= lo) & (signal < hi)
        if m.sum() > 100:
            bins.append([float(np.mean(signal[m])), float(np.mean(variance[m])), int(m.sum())])
    bins = np.array(bins)
    design = np.stack([np.ones(len(bins)), bins[:, 0], bins[:, 0]**2], axis=1)
    scales = np.linalg.norm(design, axis=0)
    coefficients = nnls(design/scales, bins[:, 1])[0]/scales
    floor, gain, excess = map(float, coefficients)
    hp = frame-gaussian_filter(frame, 4)
    noise_hp = differences-gaussian_filter(differences, (0, 4, 4))
    target_contrast = float(np.sqrt(max(np.var(hp[interior])-np.mean(noise_hp[:, interior]**2), 1)))
    diffuse = np.maximum(coarse-background, 0)
    density = np.maximum(gaussian_filter(diffuse, 8), 1)**1.8 * tissue
    density /= np.mean(density[tissue])
    profile = dict(tissue=tissue, tissue_field=gaussian_filter(tissue.astype(np.float32), 1),
                   background=np.full(frame.shape, background, dtype=np.float32),
                   density=density.astype(np.float32), diffuse_target=diffuse,
                   temporal_gains=np.array([1.]), compensation_sigma_px=np.array(12.),
                   cell_gain=np.clip(diffuse/max(np.mean(diffuse[tissue]), 1), .3, 2))
    count = int(round(config['selected']['cell_count'] * tissue.sum()/old_area))
    settings = config['selected'] | dict(cell_count=count, cell_contrast_std_adu=target_contrast,
                shot_gain_adu=gain, read_variance_adu2=floor, excess_variance_fraction=excess)
    noise_bands = np.mean([band_variances(d, interior) for d in differences], axis=0)
    observed_bands = band_variances(frame, interior)
    structural_bands = np.maximum(observed_bands-noise_bands, 1e-6)
    candidates = []
    for sigma in [1.5, 2., 2.5, 3.214579315651832, 4.]:
        for ring in [.5, 1.]:
            trial = settings | dict(sigma_px=sigma, ring_fraction=ring)
            result = render(profile, trial, args.seed, temporal_gain=1)
            bands = band_variances(result['clean'], interior)
            loss = float(np.sqrt(np.mean(np.log(np.maximum(bands, 1e-6)/structural_bands)**2)))
            candidates.append(dict(sigma_px=sigma, ring_fraction=ring, loss=loss))
            print(candidates[-1], flush=True)
    best = min(candidates, key=lambda c: c['loss'])
    settings.update({k: best[k] for k in ['sigma_px', 'ring_fraction']})
    result = render(profile, settings, args.seed, temporal_gain=1)
    # A second random layout checks sensitivity to the placement seed.
    other = render(profile, settings, args.seed+100, temporal_gain=1)
    other_loss = float(np.sqrt(np.mean(np.log(np.maximum(band_variances(other['clean'], interior), 1e-6)/structural_bands)**2)))
    report = dict(input=str(args.input.resolve()), frame_index=args.frame, frame_number=args.frame+1,
        frame_count=n, shape_yx=list(frame.shape), dtype='uint16',
        previous_raw_shape_yx=config['source_recording']['raw_shape_yx'],
        previous_model_shape_yx=config['source_recording']['shape_yx'],
        same_pixel_dimensions=sorted(frame.shape) == sorted(config['source_recording']['raw_shape_yx']),
        pixel_size_um=None, noise_burst_indices=list(range(start, start+len(burst))),
        background_adu=background, tissue_pixels=int(tissue.sum()), selected=settings,
        noise_bins_signal_variance_count=bins.tolist(), noise_coefficients_floor_gain_excess=coefficients.tolist(),
        observed_band_variances=observed_bands.tolist(), difference_band_variances=noise_bands.tolist(),
        structure_band_variances=structural_bands.tolist(), candidates=candidates,
        fit_log_band_rmse=best['loss'], second_seed_log_band_rmse=other_loss,
        limitations=['Exploratory fit on one frame, no held-out recording validation.',
            'Cell count scaled from old tissue-area density; not an observed neuron count.',
            'Noise is an effective adjacent-frame estimate, including calcium and registration effects; resampling correlations are not modeled.',
            'Coarse fluorescence field is reused at sigma=12 pixels; cell positions are random, not detected neurons.',
            'No old raw TIFF is available: dimension comparison uses saved calibration.',
            'No physical pixel size in new TIFF; morphology is fitted in pixels.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'profile.npz', **profile)
    np.savez_compressed(args.output/'frame_fit.npz', real_frame=frame.astype(np.uint16),
        synthetic_image=result['image'], synthetic_clean=result['clean'], synthetic_labels=result['labels'],
        tissue_mask=tissue, centers_yx=result['centers'], seed=np.array(args.seed),
        instance_offsets=result['instance_offsets'], instance_pixel_index=result['instance_pixel_index'],
        shape_yx=np.array(frame.shape))
    tifffile.imwrite(args.output/'synthetic.tiff', result['image'])
    tifffile.imwrite(args.output/'synthetic_labels.tiff', result['labels'])
    # A common ADU display scale is used for all real/synthetic panels.
    vmin, vmax = np.percentile(frame, [1, 99.5])
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    for ax, im, title in zip(axs, [frame, result['image']],
            [f'Real Plane 20 | frame {args.frame} (zero-based) | {frame.shape[0]} x {frame.shape[1]} px (Y,X)',
             f'SimCalc trial | sigma={best["sigma_px"]:.2f} px | rings={best["ring_fraction"]:.0%} | N={count} (prior)']):
        shown = ax.imshow(im.T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_xlabel('Original Y (px)'); ax.set_ylabel('Original X (px)')
    fig.colorbar(shown, ax=axs, label='ADU', shrink=.8)
    fig.savefig(args.output/'comparison.png', dpi=160); plt.close(fig)
    # Choose a bright 160x160 region using only the smoothed real frame.
    score = gaussian_filter(diffuse, 25)
    score[:80] = score[-80:] = 0; score[:, :80] = score[:, -80:] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    roi = (slice(cy-80, cy+80), slice(cx-80, cx+80))
    report['crop_yx'] = [int(cy-80), int(cy+80), int(cx-80), int(cx+80)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, im, title in zip(axs, [frame, result['image'], result['labels']],
                            ['Real: 160 x 160 px', 'SimCalc: same display scale', 'Synthetic soma labels (not real GT)']):
        ax.imshow(im[roi].T, cmap='nipy_spectral' if im is result['labels'] else 'gray',
                  **({} if im is result['labels'] else dict(vmin=vmin, vmax=vmax)), interpolation='nearest')
        ax.set_title(title); ax.set_axis_off()
    fig.savefig(args.output/'crops.png', dpi=170); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for im, title in [(frame, 'Real'), (result['image'], 'SimCalc')]:
        axs[0].hist(im[tissue], bins=np.linspace(70, 600, 150), density=True, histtype='step', label=title)
    axs[0].set(xlabel='ADU', ylabel='Density', title='Tissue intensity distribution'); axs[0].legend()
    axs[1].plot(bins[:, 0], bins[:, 1], 'o', label='Adjacent-frame estimate')
    axs[1].plot(bins[:, 0], design@coefficients, label='Effective detector fit')
    axs[1].set(xlabel='Signal above background (ADU)', ylabel='Variance (ADU squared)', title='Noise fit'); axs[1].legend()
    fig.savefig(args.output/'diagnostics.png', dpi=160); plt.close(fig)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(shape=frame.shape, background=background, count=count, best=best,
                          noise=coefficients.tolist(), second_seed_loss=other_loss)), flush=True)


if __name__ == '__main__':
    main()
