"""Measure spatial and temporal coordination of the extracted fluorescence traces."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
EDGES = [0, 15, 30, 60, 120, 240, np.inf]
LAGS = np.arange(-8, 9)


def summarize(traces, positions, rng, peak_smoothing=0, prominence=.30):
    n, frames = traces.shape
    pair = np.triu_indices(n, 1)
    distances = pdist(positions)
    bands = np.digitize(distances, EDGES[1:-1])
    shifted_index = (np.arange(frames)[None, :] + rng.integers(frames, size=(n, 1))) % frames
    z = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
    common = np.median(z, axis=0)
    residual = z - (z @ common / (common @ common))[:, None] * common
    raw = np.corrcoef(traces)[pair]
    corrected = np.corrcoef(residual)[pair]
    peaks = np.zeros_like(traces, dtype=np.float32)
    peak_traces = gaussian_filter1d(traces, peak_smoothing, axis=1) if peak_smoothing else traces
    for i, trace in enumerate(peak_traces):
        peaks[i, find_peaks(trace, prominence=prominence, distance=3)[0]] = 1
    counts = peaks.sum(axis=1)
    total = counts[pair[0]] + counts[pair[1]]

    def coincidence(events):
        windows = maximum_filter1d(events, size=5, axis=1, mode='constant')
        hits = events @ windows.T
        return (hits + hits.T)[pair]

    observed = coincidence(peaks)
    shifted = coincidence(np.take_along_axis(peaks, shifted_index, axis=1))
    records = []
    for band in range(len(EDGES) - 1):
        mask = bands == band
        if not mask.any():
            continue
        records.append(dict(band=band, pairs=int(mask.sum()), correlation=float(np.median(raw[mask])),
                            correlation_residual=float(np.median(corrected[mask])),
                            p90_correlation_residual=float(np.quantile(corrected[mask], .9)),
                            coincidence=float(observed[mask].sum() / total[mask].sum()),
                            coincidence_shifted=float(shifted[mask].sum() / total[mask].sum())))
    near = distances < 30
    a, b = pair[0][near], pair[1][near]
    lag_hits, lag_null = [], []
    shuffled_peaks = np.take_along_axis(peaks, shifted_index, axis=1)
    for lag in LAGS:
        start, end = max(0, -lag), min(frames, frames - lag)
        first, second = slice(start, end), slice(start + lag, end + lag)
        for source, destination in ((peaks, lag_hits), (shuffled_peaks, lag_null)):
            forward = source[:, first] @ source[:, second].T
            destination.append(float((forward[a, b] + forward[b, a]).sum() / total[near].sum()))
    return records, lag_hits, lag_null


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=HERE / 'foreground_conservative' / 'trace_bank.npz')
    parser.add_argument('--peak-prominence', type=float)
    args = parser.parse_args()
    if args.peak_prominence is None:
        summary = json.loads((args.bank.parent / 'summary.json').read_text())
        args.peak_prominence = summary.get('peak_prominence', .30)
    rng = np.random.default_rng(20261125)
    rows, hits, null, smoothed_near = [], [], [], []
    sensitivity_rng = np.random.default_rng(20261125)
    with np.load(args.bank) as bank:
        planes = bank['plane']
        traces = bank['dff']
        centers = bank['centers_yx']
    for plane in sorted(set(planes)):
        mask = planes == plane
        records, observed, shuffled = summarize(traces[:, mask].T, centers[mask], rng, prominence=args.peak_prominence)
        rows.extend(dict(plane=int(plane), **record) for record in records)
        hits.append(observed)
        null.append(shuffled)
        smoothed, _, _ = summarize(traces[:, mask].T, centers[mask], sensitivity_rng, peak_smoothing=.8,
                                   prominence=args.peak_prominence)
        smoothed_near.append(smoothed[0])
    labels = ['0–15', '15–30', '30–60', '60–120', '120–240', '240+']
    metrics = ['correlation', 'correlation_residual', 'coincidence', 'coincidence_shifted']
    summary_metrics = metrics + ['p90_correlation_residual']
    result = []
    for band, label in enumerate(labels):
        group = [row for row in rows if row['band'] == band]
        result.append(dict(distance_px=label, pairs=sum(row['pairs'] for row in group),
                           planes=len(group), **{key: float(np.median([row[key] for row in group]))
                                                 for key in summary_metrics}))
    report = dict(aggregation='Median across planes, with each plane weighted equally',
                  peak_prominence_dff=args.peak_prominence, coincidence_tolerance_frames=2,
                  distances='ROI center distance in native pixels, within the same plane',
                  by_distance=result, by_plane=rows, lag_frames=LAGS.tolist(),
                  near_peak_sensitivity_sigma_frames=.8,
                  near_peak_sensitivity={key: float(np.median([row[key] for row in smoothed_near]))
                                         for key in metrics[2:]},
                  near_lag_observed=np.median(hits, axis=0).tolist(),
                  near_lag_shifted=np.median(null, axis=0).tolist())
    (args.bank.parent / 'coordination.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for ax, keys, legend, title in [
        (axes[0], metrics[:2], ['Original', 'Plane-wide signal removed'], 'Trace correlation'),
        (axes[1], metrics[2:], ['Original', 'Circularly shifted'], 'Peak coincidence (±2 frames)'),
    ]:
        for key, name in zip(keys, legend):
            ax.plot(labels, [item[key] for item in result], 'o-', label=name)
        ax.set(title=title, xlabel='Distance between ROI centers (px)')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)
    axes[2].plot(LAGS, np.median(hits, axis=0), 'o-', label='Original')
    axes[2].plot(LAGS, np.median(null, axis=0), 'o-', label='Circularly shifted')
    axes[2].set(title='Peak lag for ROI pairs <30 px', xlabel='Lag (frames)', ylabel='Coincidence fraction')
    axes[2].legend(fontsize=8)
    fig.savefig(args.bank.parent / 'coordination.png', dpi=160)


if __name__ == '__main__':
    main()
