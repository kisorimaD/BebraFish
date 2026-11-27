"""Compare the saved Plane20 render with the recording using aggregate statistics."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import pdist
import tifffile

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent
sys.path.insert(0, str(ROOT))

from calibrate_flash_dynamics import (BLOCK, TILE, calibrated_baselines, flash_metrics,
                                     grain_metrics, mask_matrix, median_center, raw_signal,
                                     render_flash_frame, tile_mean)
from optical_layers import halo_light
from plot_render_components import prepare_render, render_layers
from reference_photometry import QUANTILES
from render_recorded_bank import PROFILE


STYLES = {
    'real': dict(color='#222222', lw=1.45, zorder=3),
    'old': dict(color='#83888D', lw=1.25, ls=(0, (3, 2)), zorder=2),
    'new': dict(color='#00788C', lw=1.55, zorder=4),
}
LABELS = {'real': 'Recording', 'old': 'Earlier synthetic', 'new': 'Final synthetic'}
DISTANCE_EDGES = np.array([0, 10, 20, 35, 50, 75, 100, 150, 200])


def measure(recalibrated=False, previous=None, current=None):
    if previous is None:
        previous = ROOT / 'movie_foreground_conservative'
    coordinated = ROOT / 'movie_coordinated_flashes'
    if current is None:
        current = ROOT / 'movie_decoupled_dynamics' if recalibrated else coordinated
    state = prepare_render(previous)
    meta = state['meta']
    validation = json.loads((current / 'validation.json').read_text())
    frame_ids = np.array(validation['heldout_reference_frames'])
    blocks = frame_ids // BLOCK * BLOCK
    assert not np.isin(blocks, validation['training_source_blocks']).any()
    # Also exclude every 50-frame block used by the earlier intensity fit.
    fitted_blocks = np.unique(np.array(meta['photometry']['fitted_frames']) // BLOCK * BLOCK)
    evaluate = ~np.isin(blocks, fitted_blocks)

    with np.load(current / 'activity.npz', allow_pickle=False) as saved:
        columns = saved['bank_columns']
    with np.load(previous / meta.get('profile', PROFILE), allow_pickle=False) as saved:
        tissue = saved['tissue'].astype(bool)
    with np.load(previous / meta['scene'], allow_pickle=False) as saved:
        scene = dict(saved)
    bank_path = previous / meta['bank']
    plane = meta.get('reference_plane', 20)
    with np.load(bank_path, allow_pickle=False) as bank:
        reference = bank['plane'] == plane
        assert np.all(bank['plane'][columns] != plane)
        roi_ids = bank['mask_id'][reference]
        reference_centers = bank['centers_yx'][reference]
    centers = dict(real=reference_centers, old=scene['centers'], new=scene['centers'])
    source_dir = bank_path.parent.parent / 'per_plane'
    reference_rois = raw_signal(source_dir, plane, roi_ids, np.ones(len(roi_ids), dtype=bool))
    masks = mask_matrix(scene)
    valid_tiles = tile_mean(tissue) > .7
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coords = ((y + .5) / TILE - .5, (x + .5) / TILE - .5)
    support = gaussian_filter(tissue.astype(np.float32), 12)
    # Both members of each horizontal pair are inside the same image row.
    pixel_pool = np.flatnonzero(x.ravel() < tissue.shape[1] - 1)
    pixels = np.random.default_rng(42).choice(pixel_pool, 14000, replace=False)

    def project(image, rois):
        flat = image.ravel()
        return tile_mean(image)[valid_tiles], rois, flat[pixels], flat[pixels + 1]

    def flash_renderer(directory, config):
        with np.load(directory / 'activity.npz', allow_pickle=False) as saved:
            modulation = saved['render_modulation']
            diffuse_tiles = saved['diffuse_tile_changes']
        baselines = calibrated_baselines(state, config['static_gain'], config['static_offset_adu'])

        def render_frame(index):
            broad = map_coordinates(diffuse_tiles[index], coords, order=1, mode='nearest')
            broad *= support * config['diffuse_gain']
            signal = (modulation[index] @ state['maps']).reshape(tissue.shape)
            halo = halo_light(signal)
            return render_flash_frame(state, index, signal + halo, halo, broad, baselines,
                                      config.get('grain_correction'))[0]

        return render_frame

    if recalibrated:
        earlier_render = flash_renderer(coordinated, json.loads((coordinated / 'validation.json').read_text()))
    else:
        def earlier_render(index):
            return render_layers(state, index)[0]
    final_render = flash_renderer(current, validation)

    def render(index):
        earlier = earlier_render(index)
        final = final_render(index)
        return project(earlier, masks @ earlier.ravel()), project(final, masks @ final.ravel())

    rows = {key: [] for key in STYLES}
    with (tifffile.TiffFile(meta['real_source']) as recording,
          ThreadPoolExecutor(max_workers=3) as workers):
        for index, (earlier, final) in zip(frame_ids, workers.map(render, frame_ids)):
            rows['old'].append(earlier)
            rows['new'].append(final)
            rows['real'].append(project(recording.pages[int(index)].asarray(), reference_rois[index]))

    results = {}
    low, high = meta['display_black_adu'], meta['display_white_adu']
    for key, frames in rows.items():
        tiles, rois, left, right = (np.stack(column) for column in zip(*frames))
        replayed = flash_metrics(tiles, rois, left, low, high)
        archived = validation['heldout'][key]
        # The original temporal checks use all these frame indices and do not depend on pixel sampling.
        for name in ('roi_median_p95_minus_p50_adu', 'roi_simultaneous_up_20pct_p95',
                     'tile_delta_p99_adu', 'pct_frames_with_wide_flash'):
            np.testing.assert_allclose(replayed[name], archived[name], atol=.005, err_msg=f'{key}: {name}')

        tiles, rois, left, right = (array[evaluate] for array in (tiles, rois, left, right))
        relative = median_center(rois) / np.median(rois, axis=0)
        results[key] = dict(
            metrics=flash_metrics(tiles, rois, left, low, high),
            grain=grain_metrics(left, right, low, high),
            amplitude=np.quantile(rois, .95, axis=0) - np.median(rois, axis=0),
            correlation=correlation_by_distance(relative, centers[key]),
            bright_tiles=100 * np.mean(median_center(tiles) >= 10, axis=1),
            bright_rois=100 * np.mean(relative > .2, axis=1),
        )
    return results, frame_ids[evaluate], np.unique(blocks[evaluate])


def correlation_by_distance(relative, centers):
    residual = relative - np.median(relative, axis=1, keepdims=True)
    usable = np.std(residual, axis=0) > 0
    distances = pdist(centers[usable])
    correlations = 1 - pdist(residual[:, usable].T, metric='correlation')
    bins = np.searchsorted(DISTANCE_EDGES, distances, side='right') - 1
    return np.array([np.median(correlations[bins == i]) if np.any(bins == i) else np.nan
                     for i in range(len(DISTANCE_EDGES) - 1)])


def draw_exceedance(ax, values, xmax, style):
    x, counts = np.unique(values, return_counts=True)
    share = 100 * np.cumsum(counts[::-1])[::-1] / len(values)
    ax.step(np.r_[0, x, xmax], np.r_[100, share, 0], where='pre', **style)


def plot(results, target, labels):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'axes.linewidth': .65,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42,
    })
    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.9))
    figure.subplots_adjust(left=.098, right=.987, bottom=.105, top=.855, wspace=.49, hspace=.68)
    figure.legend([Line2D([], [], **STYLES[key]) for key in labels], labels.values(),
                  loc='upper center', bbox_to_anchor=(.5, .995), frameon=False,
                  ncol=len(labels), handlelength=2.5, columnspacing=2)

    panels = (
        ('Pixel intensity', 'Quantile (%)', 'Synthetic − real (ADU)'),
        ('Image grain', 'Local intensity (ADU)', 'Grain (ADU)'),
        ('Somatic amplitude', 'P95 − median (ADU)', 'Somata at or below (%)'),
        ('Spatial coactivity', 'Soma separation (px)', 'Median residual Pearson r'),
        ('Extent of brightening', 'Brightened tissue tiles (%)', 'Frames at or above (%)'),
        ('Simultaneous soma rises', 'Brightened somata (%)', 'Frames at or above (%)'),
    )
    for label, ax, (title, xlabel, ylabel) in zip('ABCDEF', axes.flat, panels):
        ax.text(-.17, 1.07, label, transform=ax.transAxes, fontweight='bold', fontsize=10)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5E5E5', lw=.6)

    a, b, c, d, e, f = axes.flat
    a.axhline(0, color=STYLES['real']['color'], lw=.8, zorder=1)
    a.set(xlim=(0, 101), xticks=(0, 25, 50, 75, 100))
    # A log-like scale keeps differences in the tails visible and retains zero.
    c.set_xscale('symlog', linthresh=5)
    xmax = max(5.1, 1.1 * max(data['amplitude'].max() for data in results.values()))
    amplitude_ticks = [value for value in (0, 5, 10, 20, 50, 100, 200, 500) if value < xmax]
    c.set(xlim=(0, xmax), xticks=amplitude_ticks, xticklabels=list(map(str, amplitude_ticks)),
          ylim=(0, 100), yticks=(0, 25, 50, 75, 100))
    c.minorticks_off()
    d.axhline(0, color='#C8C8C8', lw=.65)
    d.set(xlim=(0, 200), xticks=(0, 50, 100, 150, 200))
    e.axvline(10, color='#B2B2B2', lw=.75, ls=':')
    maxima = {}
    for ax, metric in ((e, 'bright_tiles'), (f, 'bright_rois')):
        end = max(data[metric].max() for data in results.values())
        maxima[metric] = 5 * (int(end / 5) + 1)
        ax.set_yscale('symlog', linthresh=1, linscale=.55)
        ax.set(xlim=(0, maxima[metric]), ylim=(0, 105),
               yticks=(0, 1, 5, 20, 100), yticklabels=('0', '1', '5', '20', '100'))

    real_quantiles = np.array(results['real']['metrics']['displayed_quantiles_adu'])
    for key in labels:
        style = STYLES[key]
        data = results[key]
        quantiles = np.array(data['metrics']['displayed_quantiles_adu'])
        if key != 'real':
            a.plot(QUANTILES * 100, quantiles - real_quantiles, marker='o', ms=2.8, **style)
        bins = [np.mean(row['range_adu']) for row in data['grain']]
        b.plot(bins, [row['robust_sd_adu'] for row in data['grain']], marker='o', ms=2.8, **style)
        x, counts = np.unique(data['amplitude'], return_counts=True)
        c.step(np.r_[0, x], np.r_[0, 100 * np.cumsum(counts) / counts.sum()], where='post', **style)
        d.plot((DISTANCE_EDGES[:-1] + DISTANCE_EDGES[1:]) / 2,
               data['correlation'], marker='o', ms=2.8, **style)
        draw_exceedance(e, data['bright_tiles'], maxima['bright_tiles'], style)
        draw_exceedance(f, data['bright_rois'], maxima['bright_rois'], style)

    b.set_ylim(bottom=0)
    if target is None:
        return figure
    figure.savefig(target.with_suffix('.pdf'))
    figure.savefig(target.with_suffix('.png'), dpi=350)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recalibrated', action='store_true', help='Compare the old and recalibrated flash renders.')
    args = parser.parse_args()
    labels = LABELS
    target = OUTPUT / 'plane20_validation'
    if args.recalibrated:
        labels = dict(real='Recording', old='Previous synthetic', new='Recalibrated synthetic')
        target = OUTPUT / 'plane20_validation_recalibrated'
    results, frames, blocks = measure(args.recalibrated)
    plot(results, target, labels)
    print(f'Verified replay on the original validation frames. Plotted {len(frames)} frames from blocks {blocks}.')
    for key, result in results.items():
        print(labels[key], json.dumps(result['metrics']))
        print('Grain:', [None if not row['pairs'] else round(row['robust_sd_adu'], 2) for row in result['grain']])
        print('Spatial correlation:', np.round(result['correlation'], 3))
