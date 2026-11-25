"""Assign circular offsets to measured traces using synthetic soma adjacency."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


def peak_trains(traces, prominence=.30):
    peaks = np.zeros_like(traces, dtype=np.float32)
    for i, trace in enumerate(gaussian_filter1d(traces, .8, axis=1)):
        indices, _ = find_peaks(trace, prominence=prominence, distance=3)
        peaks[i, indices] = 1
    return peaks


def roll_traces(traces, offsets):
    indices = (np.arange(len(traces))[:, None] - offsets) % len(traces)
    return np.take_along_axis(traces, indices, axis=0)


def make_groups(centers, seed):
    rng = np.random.default_rng(seed)
    tree = cKDTree(centers)
    groups = np.full(len(centers), -1, dtype=int)
    group_id = 0
    for cell in rng.permutation(len(centers)):
        if groups[cell] >= 0:
            continue
        neighbors = np.asarray(tree.query_ball_point(centers[cell], 14))
        neighbors = neighbors[(neighbors != cell) & (groups[neighbors] < 0)]
        if len(neighbors) < 2:
            continue
        count = min(len(neighbors), rng.integers(2, 5))
        groups[np.r_[cell, rng.choice(neighbors, count, replace=False)]] = group_id
        group_id += 1
        if np.count_nonzero(groups >= 0) >= len(groups) * .25:
            break
    return groups


def measure_group_target(bank_path, synthetic_groups, prominence=.30):
    with np.load(bank_path, allow_pickle=False) as bank:
        traces, planes, centers = bank['dff'], bank['plane'], bank['centers_yx']
    counts_by_size = {size: [] for size in (3, 4, 5)}
    for plane in np.unique(planes):
        selected = planes == plane
        peaks = peak_trains(traces[:, selected].T, prominence)
        activity = maximum_filter1d(peaks, 3, axis=1, mode='constant')
        for seed in range(12):
            labels = make_groups(centers[selected], seed)
            for group in range(labels.max() + 1):
                cells = labels == group
                counts_by_size[cells.sum()].append(int(np.count_nonzero(activity[cells].sum(axis=0) >= 3)))
    means = {size: float(np.mean(counts)) for size, counts in counts_by_size.items()}
    sizes = np.bincount(synthetic_groups[synthetic_groups >= 0])
    return float(np.mean([means[size] for size in sizes])), {
        size: dict(mean_frames=means[size], sampled_groups=len(counts))
        for size, counts in counts_by_size.items()
    }


def coordinate_traces(measured, centers, seed, targets, groups=None):
    """Match measured near-neighbor statistics by changing only trace phases."""
    rng = np.random.default_rng(seed)
    frames, cells = measured.shape
    pairs = cKDTree(centers).query_pairs(30, output_type='ndarray')
    left, right = pairs.T
    near = np.linalg.norm(centers[left] - centers[right], axis=1) < 15
    traces = measured.T
    centered = traces - traces.mean(axis=1, keepdims=True)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    peaks = peak_trains(traces, targets.get('peak_prominence', .30))
    windows = maximum_filter1d(peaks, 5, axis=1, mode='wrap')
    peak_fft, window_fft = np.fft.rfft(peaks, axis=1), np.fft.rfft(windows, axis=1)

    # Tables hold each pair's statistics for every possible relative phase.
    above = np.empty((len(left), frames), dtype=bool)
    hits = np.zeros((len(left), frames), dtype=np.int16)
    thresholds = np.where(near, targets['near_correlation'], targets['mid_correlation'])
    if groups is not None:
        in_group = (groups[left] >= 0) & (groups[left] == groups[right])
        high = np.empty_like(above)
        group_high = np.empty_like(above)
        high_thresholds = np.where(near, targets['near_p90'], targets['mid_p90'])
    for start in range(0, len(left), 256):
        sl = slice(start, start + 256)
        a, b = left[sl], right[sl]
        corr = np.fft.irfft(spectrum[a].conj() * spectrum[b], n=frames, axis=1)
        above[sl] = corr > thresholds[sl, None]
        if groups is not None:
            high[sl] = corr > high_thresholds[sl, None]
            group_high[sl] = (corr > targets['group_threshold']) & in_group[sl, None]
        hit = np.fft.irfft(peak_fft[a].conj() * window_fft[b] +
                           window_fft[a].conj() * peak_fft[b], n=frames, axis=1)
        hits[sl] = np.rint(hit).astype(np.int16) * near[sl, None]
    test_lag = 17
    test_corr = centered[left[0]] @ np.roll(centered[right[0]], -test_lag)
    assert above[0, test_lag] == (test_corr > thresholds[0])

    nearby_edges = [[] for _ in range(cells)]
    for edge, (a, b) in enumerate(pairs):
        nearby_edges[a].append(edge)
        nearby_edges[b].append(edge)
    offsets = rng.integers(frames, size=cells)
    phase = (offsets[left] - offsets[right]) % frames
    pair_id = np.arange(len(left))
    denominators = np.array([near.sum(), (~near).sum(),
                             (peaks[left[near]].sum() + peaks[right[near]].sum())])
    initial_above = above[pair_id, phase]
    totals = np.array([initial_above[near].sum(), initial_above[~near].sum(),
                       hits[pair_id, phase].sum()])
    desired = np.array([.5, .5, targets['near_peak_coincidence']])
    weights = np.array([1., 1., 3.])
    if groups is not None:
        initial_high = high[pair_id, phase]
        active = maximum_filter1d(peaks, 3, axis=1, mode='wrap').astype(bool)
        members = [np.flatnonzero(groups == group) for group in range(groups.max() + 1)]
        current_activity = roll_traces(active.T, offsets).T.copy()
        triple_frames = sum(np.count_nonzero(current_activity[cells].sum(axis=0) >= 3)
                            for cells in members)
        denominators = np.r_[denominators, near.sum(), (~near).sum(), in_group.sum(),
                             len(members) * 20]
        totals = np.r_[totals, initial_high[near].sum(), initial_high[~near].sum(),
                       group_high[pair_id, phase].sum(), triple_frames]
        desired = np.r_[desired, .10, .10, .55, targets['group_triple_frames'] / 20]
        weights = np.r_[weights, 2., 2., .30, .20]
        timeline = np.arange(frames)
    for _ in range(6 if groups is not None else 4):
        for cell in rng.permutation(cells):
            edges = np.asarray(nearby_edges[cell], dtype=int)
            if not len(edges):
                continue
            choices = np.r_[offsets[cell], rng.integers(frames, size=40)]
            is_left = left[edges] == cell
            other = np.where(is_left, right[edges], left[edges])
            lag = ((choices[:, None] - offsets[other]) * np.where(is_left, 1, -1)) % frames
            matched = above[edges, lag]
            is_near = near[edges]
            contribution = np.column_stack((matched[:, is_near].sum(axis=1),
                                            matched[:, ~is_near].sum(axis=1),
                                            hits[edges, lag].sum(axis=1)))
            if groups is not None:
                strong = high[edges, lag]
                triple_contribution = np.zeros(len(choices), dtype=int)
                if groups[cell] >= 0:
                    peers = members[groups[cell]]
                    peers = peers[peers != cell]
                    two_active = current_activity[peers].sum(axis=0) == 2
                    proposed_activity = active[cell, (timeline[None, :] - choices[:, None]) % frames]
                    triple_contribution = proposed_activity[:, two_active].sum(axis=1)
                contribution = np.column_stack((contribution, strong[:, is_near].sum(axis=1),
                                                strong[:, ~is_near].sum(axis=1),
                                                group_high[edges, lag].sum(axis=1),
                                                triple_contribution))
            proposals = totals + contribution - contribution[0]
            loss = np.sum(weights * (proposals / denominators - desired)**2, axis=1)
            best = np.argmin(loss)
            offsets[cell] = choices[best]
            if groups is not None and groups[cell] >= 0:
                current_activity[cell] = proposed_activity[best]
            totals = proposals[best]
        if np.max(np.abs(totals / denominators - desired)) < .002:
            break
    return offsets


def check_coordination(measured, centers, seed, prominence=.30):
    rng = np.random.default_rng(seed)
    permutation = rng.integers(len(measured), size=measured.shape[1])
    traces = measured.T
    z = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
    common = np.median(z, axis=0)
    residual = z - (z @ common / (common @ common))[:, None] * common
    raw = np.corrcoef(z)
    adjusted = np.corrcoef(residual)
    peaks = peak_trains(traces, prominence)
    shifted = roll_traces(peaks.T, permutation).T

    def hits(events):
        nearby = maximum_filter1d(events, 5, axis=1, mode='constant')
        overlap = events @ nearby.T
        return overlap + overlap.T

    overlap, null = hits(peaks), hits(shifted)
    counts = peaks.sum(axis=1)
    pair_a, pair_b = np.triu_indices(len(centers), 1)
    distances = np.linalg.norm(centers[pair_a] - centers[pair_b], axis=1)
    rows = []
    for minimum, maximum in [(0, 15), (15, 30), (30, 60), (60, np.inf)]:
        valid = (distances >= minimum) & (distances < maximum)
        a, b = pair_a[valid], pair_b[valid]
        total = (counts[a] + counts[b]).sum()
        rows.append(dict(min_px=minimum, max_px=maximum if np.isfinite(maximum) else None,
                         pairs=len(a), correlation=float(np.median(raw[a, b])),
                         correlation_residual=float(np.median(adjusted[a, b])),
                         p90_correlation_residual=float(np.quantile(adjusted[a, b], .9)),
                         coincidence=float(overlap[a, b].sum()/total),
                         coincidence_shifted=float(null[a, b].sum()/total)))
    return rows


def check_groups(measured, groups, prominence=.30):
    peaks = peak_trains(measured.T, prominence)
    active = maximum_filter1d(peaks, 3, axis=1, mode='constant')
    pair_correlations = []
    triple_frames = []
    for group in np.unique(groups[groups >= 0]):
        cells = np.flatnonzero(groups == group)
        pair_correlations.extend(np.corrcoef(measured[:, cells].T)[np.triu_indices(len(cells), 1)])
        triple_frames.append(np.count_nonzero(active[cells].sum(axis=0) >= 3))
    return dict(groups=len(triple_frames), member_cells=int(np.count_nonzero(groups >= 0)),
                median_pair_correlation=float(np.median(pair_correlations)),
                mean_frames_with_three_peaks=float(np.mean(triple_frames)))
