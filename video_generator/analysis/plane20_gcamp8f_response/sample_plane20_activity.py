"""Sample distinct trace sets calibrated against a reference plane."""

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from calibrate_flash_dynamics import raw_signal, stabilize_background
from recalibrate_plane20 import DISTANCE_EDGES, spatial_pairs
from render_recorded_bank import BANK, HERE, SCENE
from spatial_amplitude import amplitude, weights

LAGS = np.array([1, 2, 4, 8, 16, 32])
AMPLITUDE_QUANTILES = [.1, .25, .5, .75, .9]
EVENT_THRESHOLDS = np.array([.05, .10, .20])
EVENT_QUANTILES = [.5, .9, .95, .99]
WINDOW_LENGTHS = (50, 100, 125, 200)


def stabilized_bank(path):
    with np.load(path, allow_pickle=False) as arrays:
        bank = dict(arrays)
    traces = bank['dff']
    frames = np.arange(len(traces))
    source = path.parent.parent / 'per_plane'
    for plane in np.unique(bank['plane']):
        columns = bank['plane'] == plane
        raw = raw_signal(source, plane, bank['mask_id'][columns], bank['full_mask'][columns])
        traces[:, columns], _ = stabilize_background(raw, traces[:, columns], frames)
    return bank


def autocorrelation(traces, lags):
    centered = traces - traces.mean(axis=0)
    square = np.maximum(np.sum(centered**2, axis=0), 1e-9)
    return np.array([np.sum(centered[:-lag] * centered[lag:], axis=0) / square for lag in lags])


def donor_features(traces):
    return np.column_stack((np.log(np.maximum(amplitude(traces), .001)) / .18,
                            autocorrelation(traces, [2, 8]).T / .13,
                            np.log1p(100 * np.mean(traces < -.2, axis=0)) / .20))


def assign_donors(features, donors, reference, probabilities, groups, bank, localized, rng):
    positions = (probabilities < rng.random((len(probabilities), 1))).sum(axis=1)
    targets = features[reference[positions]]
    pool = rng.choice(donors, max(len(targets), int(.7*len(donors))), replace=False)
    tree = cKDTree(features[pool])
    closest = tree.query(targets, k=32)[1]
    used = np.zeros(len(pool), dtype=bool)
    assignment = np.empty(len(targets), dtype=int)
    donor_centers = bank['centers_yx'][pool]
    donor_planes = bank['plane'][pool]
    for group in rng.permutation(np.unique(groups)):
        neighbors = np.empty(0, dtype=int)
        members = rng.permutation(np.flatnonzero(groups == group))
        for soma in members:
            available = closest[soma][~used[closest[soma]]]
            local = neighbors[~used[neighbors]]
            error = np.sum((features[pool[local]] - targets[soma])**2, axis=1)
            local = local[np.argsort(error)[:3]]
            if len(local) and error.min() <= 2.3**2:
                available = local
            if len(available) == 0:
                free = np.flatnonzero(~used)
                distance = np.sum((features[pool[free]] - targets[soma])**2, axis=1)
                available = free[np.argsort(distance)[:3]]
            column = rng.choice(available[:3])
            used[column] = True
            assignment[soma] = pool[column]
            if localized and soma == members[0]:
                squared_distance = np.sum((donor_centers-donor_centers[column])**2, axis=1)
                neighbors = np.flatnonzero((donor_planes == donor_planes[column]) & (squared_distance <= 60**2))
    return assignment


def sample_windows(traces, groups, window_length, coupling, seed, shared_schedule=None):
    rng = np.random.default_rng(seed)
    frames, cells = traces.shape
    complete = frames // window_length * window_length
    starts = np.arange(0, complete, window_length)
    group_count = groups.max() + 1
    common = rng.permutation(starts) if shared_schedule is None else shared_schedule
    independent = np.column_stack([rng.permutation(starts) for _ in range(group_count)])
    shared = rng.random(independent.shape) < coupling
    planned = np.where(shared, common[:, None], independent)
    if complete < frames:
        planned = np.vstack((planned, np.full(group_count, complete)))
        common = np.r_[common, complete]
        starts = np.r_[starts, complete]
    delay = rng.integers(-2, 3, size=planned.shape)
    time = np.arange(frames)
    window = time // window_length
    position = time % window_length
    initial = planned[window[:, None], groups]
    available = np.minimum(window_length, frames-initial)
    indices = initial + np.clip(position[:, None] + delay[window[:, None], groups], 0, available-1)
    previous = indices.copy()
    blend = np.ones(frames, np.float32)
    for start in starts[1:]:
        stop = min(start+3, frames)
        previous[start:stop] = indices[start-1]
        blend[start:stop] = np.arange(1, stop-start+1) / 4
    columns = np.arange(cells)
    sampled = blend[:, None]*traces[indices, columns] + (1-blend[:, None])*traces[previous, columns]
    sampled -= np.median(sampled, axis=0)
    return sampled, indices.astype(np.int32), previous.astype(np.int32), blend, common


def statistics(traces, pairs):
    amp = amplitude(traces)
    acf = np.median(autocorrelation(traces, LAGS), axis=1)
    population = 100 * np.array([np.mean(traces > threshold, axis=1) for threshold in EVENT_THRESHOLDS])
    residual = traces - np.median(traces, axis=1, keepdims=True)
    residual -= np.mean(residual, axis=0)
    residual /= np.maximum(np.linalg.norm(residual, axis=0), 1e-6)
    correlation = np.array([np.median(np.sum(residual[:, a]*residual[:, b], axis=0)) if len(a) else np.nan
                            for a, b in pairs])
    features = dict(amplitude=np.quantile(amp, AMPLITUDE_QUANTILES), autocorrelation=acf,
                    coactivity=np.quantile(population, EVENT_QUANTILES, axis=1).T,
                    correlation=correlation, negative=float(100*np.mean(traces < -.2)))
    return features, amp, population[-1]


def loss(candidate, target):
    # Balance four comparisons so a good marginal amplitude cannot cancel excessive synchrony.
    comparable = np.isfinite(candidate['correlation']) & np.isfinite(target['correlation'])
    difference = candidate['correlation'][comparable] - target['correlation'][comparable]
    errors = [np.mean(((candidate['amplitude']-target['amplitude'])/.006)**2),
              np.mean(((candidate['autocorrelation']-target['autocorrelation'])/.055)**2),
              np.mean((difference/.05)**2) if comparable.any() else 0.,
              np.mean((np.log1p(candidate['coactivity']/.3)-np.log1p(target['coactivity']/.3))**2)/.2**2,
              ((candidate['negative']-target['negative'])/.15)**2]
    return float(np.mean(errors))


def draw_figure(path, real, samples, before=(), reference_label='Plane20'):
    plt.rcParams.update({'font.size': 8.5, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True})
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), layout='constrained')
    a, b, c, d = axes.flat
    for ax, letter, title, xlabel, ylabel in zip(
            axes.flat, 'ABCD', ('Somatic amplitude', 'Temporal autocorrelation', 'Simultaneous rises', 'Spatial coactivity'),
            ('P95 − median ΔF/F (%)', 'Lag (frames)', 'Somata above 20% ΔF/F (%)', 'Soma separation (px)'),
            ('Somata at or below (%)', 'Median autocorrelation', 'Frames at or above (%)', 'Median residual Pearson r')):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.text(-.12, 1.05, letter, transform=ax.transAxes, fontweight='bold')
        ax.grid(axis='y', color='#e5e5e5', lw=.6)
    styles = [dict(color='#202020', label=reference_label, lw=1.6, zorder=5)]
    for i in range(len(before)):
        styles.append(dict(color='#93999e', label='Before coordination' if i == 0 else '_nolegend_',
                           alpha=.65, lw=1.0, linestyle='--'))
    for i in range(len(samples)):
        styles.append(dict(color='#00788c', label='Resampled activity' if i == 0 else '_nolegend_',
                           alpha=.75, lw=1.15, linestyle=('-', '--', ':')[i % 3]))
    edges = np.asarray(DISTANCE_EDGES)
    midpoint = (edges[1:]+edges[:-1])/2
    series = [real, *before, *samples]
    maximum = max(max(result[2]) for result in series)
    for (features, amp, population), style in zip(series, styles):
        x, count = np.unique(100*amp, return_counts=True)
        a.step(np.r_[0, x], np.r_[0, 100*count.cumsum()/count.sum()], where='post', **style)
        b.plot(LAGS, features['autocorrelation'], marker='o', ms=2.5, **style)
        x, count = np.unique(population, return_counts=True)
        c.step(np.r_[0, x, maximum+1], np.r_[100, 100*count[::-1].cumsum()[::-1]/count.sum(), 0],
               where='pre', **style)
        d.plot(midpoint, features['correlation'], **style)
    a.set_xscale('symlog', linthresh=3)
    a.set_xticks([0, 3, 5, 10, 20, 50], ['0', '3', '5', '10', '20', '50'])
    a.minorticks_off()
    b.set_xscale('log', base=2)
    b.set_xticks(LAGS, [str(lag) for lag in LAGS])
    c.set_yscale('symlog', linthresh=1)
    c.set_yticks([0, 1, 5, 20, 100], ['0', '1', '5', '20', '100'])
    a.legend(frameon=False, loc='upper left', handlelength=2)
    if path is None:
        return fig
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=350)
    plt.close(fig)


def serializable(features):
    return {name: np.where(np.isfinite(value), value, None).tolist() for name, value in features.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'activity_resamples')
    parser.add_argument('--seeds', nargs='+', type=int, default=[27, 63, 105])
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--scene', type=Path, default=SCENE)
    parser.add_argument('--reference-plane', type=int, default=20)
    parser.add_argument('--all-plane-donors', action='store_true',
                        help='Use all recorded planes as donors and share the global window order across target planes.')
    parser.add_argument('--no-figure', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Choose a new output directory and distinct nonnegative seeds.')
    bank = stabilized_bank(args.bank)
    with np.load(args.scene, allow_pickle=False) as scene:
        centers = scene['centers']
    traces = bank['dff']
    window_lengths = tuple(length for length in WINDOW_LENGTHS if 2*length <= len(traces))
    reference = np.flatnonzero(bank['plane'] == args.reference_plane)
    donors = (np.arange(len(bank['plane'])) if args.all_plane_donors else
              np.flatnonzero(bank['plane'] != args.reference_plane))
    if not window_lengths or not len(reference) or len(donors) < len(centers):
        raise ValueError('Need at least 100 frames, reference traces and enough unique donors for the scene.')
    if args.all_plane_donors:
        window_lengths = (window_lengths[-1],)
    scope = 'all-planes' if args.all_plane_donors else 'other-planes'
    donor_score = donor_features(traces)
    spatial_weights = weights(centers, bank['centers_yx'][reference])
    spatial_weights += 8 / len(reference)
    probabilities = np.cumsum(spatial_weights / spatial_weights.sum(axis=1, keepdims=True), axis=1)
    probabilities[:, -1] = 1
    _, groups = np.unique((centers//48).astype(int), axis=0, return_inverse=True)
    _, neighborhoods = np.unique((centers//24).astype(int), axis=0, return_inverse=True)
    reference_pairs = spatial_pairs(bank['centers_yx'][reference], np.random.default_rng(193))
    sample_pairs = spatial_pairs(centers, np.random.default_rng(194))
    real = statistics(traces[:, reference], reference_pairs)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.activity-resamples-') as tmp:
        stage = Path(tmp)/output.name
        stage.mkdir()
        summaries, samples = [], []
        for seed in args.seeds:
            rng = np.random.default_rng([seed, args.reference_plane] if args.all_plane_donors else seed)
            shared_schedule = None
            if args.all_plane_donors:
                window = window_lengths[0]
                starts = np.arange(0, len(traces)//window*window, window)
                shared_schedule = np.random.default_rng(seed).permutation(starts)
            candidate_seeds = rng.integers(2**32, size=2)
            best = None
            for localized in (False, True):
                columns = assign_donors(donor_score, donors, reference, probabilities,
                                        neighborhoods, bank, localized, rng)
                assigned = traces[:, columns]
                for window_length in window_lengths:
                    for coupling in (.35, .65, .85, 1.):
                        for candidate_seed in candidate_seeds:
                            result = sample_windows(assigned, groups, window_length, coupling, candidate_seed,
                                                    shared_schedule)
                            summary = statistics(result[0], sample_pairs)
                            score = loss(summary[0], real[0])
                            if best is None or score < best[0]:
                                best = score, result, summary, columns, coupling, window_length, localized
            score, result, summary, columns, coupling, window_length, localized = best
            sampled, indices, previous, blend, common = result
            assert np.isfinite(sampled).all() and len(np.unique(columns)) == len(centers)
            assert np.isin(columns, donors).all()
            np.savez_compressed(stage/f'activity_seed_{seed}.npz', dff=sampled, bank_columns=columns,
                                donor_plane=bank['plane'][columns], donor_mask_id=bank['mask_id'][columns],
                                centers_yx=centers, group_id=groups, donor_frames=indices,
                                previous_donor_frames=previous, blend_weight=blend, common_source_blocks=common,
                                donor_scope=scope)
            summaries.append(dict(seed=seed, score=score, shared_window_probability=coupling,
                                  window_frames=window_length, local_donor_assignment=localized,
                                  statistics=serializable(summary[0])))
            samples.append(summary)
            print(f'Seed {seed}: windows {window_length}, coupling {coupling:.2f}, score {score:.3f}, '
                  f'median amplitude {100*summary[0]["amplitude"][2]:.2f}%.', flush=True)
        report = dict(frames=len(traces), reference_plane=args.reference_plane, bank=str(args.bank.resolve()),
                      reference_rois=len(reference), synthetic_somas=len(centers),
                      donor_scope=scope, shared_window_order=args.all_plane_donors,
                      amplitude_quantiles=AMPLITUDE_QUANTILES, lags_frames=LAGS.tolist(),
                      event_thresholds_dff=EVENT_THRESHOLDS.tolist(), event_quantiles=EVENT_QUANTILES,
                      distance_edges_px=DISTANCE_EDGES, real=serializable(real[0]), samples=summaries)
        (stage/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not args.no_figure:
            draw_figure(stage/'activity_comparison', real, samples, reference_label=f'Plane {args.reference_plane}')
        stage.rename(output)
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
