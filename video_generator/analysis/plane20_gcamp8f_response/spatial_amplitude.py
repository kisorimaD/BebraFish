"""Match measured donor amplitudes to a smooth field estimated from Plane20."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

BANDWIDTH_PX = 35
GLOBAL_WEIGHT = 8


def amplitude(traces):
    return np.quantile(traces, .95, axis=0) - np.median(traces, axis=0)


def weights(query, reference):
    return np.exp(-.5 * (cdist(query, reference) / BANDWIDTH_PX)**2)


def predict(local_weights, log_amplitude):
    return (local_weights @ log_amplitude + GLOBAL_WEIGHT * np.median(log_amplitude)) / (
        local_weights.sum(axis=1) + GLOBAL_WEIGHT)


def sample_targets(local, log_amplitude, rng):
    # Common contribution gives a fallback for positions with little local data.
    probabilities = local + GLOBAL_WEIGHT / len(log_amplitude)
    cdf = np.cumsum(probabilities, axis=1)
    indices = np.sum(cdf < (rng.uniform(size=len(local)) * cdf[:, -1])[:, None], axis=1)
    return log_amplitude[indices]


def match_unique(targets, candidates, candidate_amplitudes):
    donor_order = np.argsort(candidate_amplitudes, kind='stable')
    sorted_amplitude = candidate_amplitudes[donor_order]
    target_order = np.argsort(targets, kind='stable')
    values = targets[target_order]
    result = np.searchsorted(sorted_amplitude, values).clip(max=len(sorted_amplitude) - 1)
    previous = np.maximum(result - 1, 0)
    closer = abs(sorted_amplitude[previous] - values) < abs(sorted_amplitude[result] - values)
    result[closer] = previous[closer]
    # Reserve one unique donor per target, including in densely populated tails.
    for i in range(1, len(result)):
        result[i] = max(result[i], result[i - 1] + 1)
    result[-1] = min(result[-1], len(sorted_amplitude) - 1)
    for i in range(len(result) - 2, -1, -1):
        result[i] = min(result[i], result[i + 1] - 1)
    assert result[0] >= 0 and np.all(np.diff(result) > 0)
    assigned = np.empty(len(targets), dtype=int)
    assigned[target_order] = donor_order[result]
    return candidates[assigned], candidate_amplitudes[assigned]


def plane20_reference(bank, reference_plane=20):
    plane = bank['plane'] == reference_plane
    if {'local_background', 'wide_background', 'full_mask'} <= set(bank.files):
        clean = plane & bank['local_background'] & ~bank['wide_background'] & ~bank['full_mask']
        if clean.sum() >= 100:
            plane = clean
    return plane


def spatial_assignment(bank, donor_columns, scene_centers, seed, reference_plane=20):
    plane = plane20_reference(bank, reference_plane)
    centers = bank['centers_yx'][plane]
    if len(centers) < 100:
        raise ValueError('Spatial amplitude estimation requires at least 100 Plane20 ROI.')
    activity = bank['dff']
    observed = amplitude(activity[:, plane])
    log_amplitude = np.log(np.maximum(observed, .001))
    donors = amplitude(activity[:, donor_columns])
    region = np.floor(centers / 48).astype(int)
    folds = (region[:, 0] + 2 * region[:, 1]) % 5
    validation, constant = np.empty(len(centers)), np.empty(len(centers))
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        validation[test] = predict(weights(centers[test], centers[train]), log_amplitude[train])
        constant[test] = np.median(log_amplitude[train])

    rmse_global = float(np.sqrt(np.mean((constant - log_amplitude)**2)))
    rmse_model = float(np.sqrt(np.mean((validation - log_amplitude)**2)))
    use_spatial_field = rmse_model < rmse_global
    local = weights(scene_centers, centers) if use_spatial_field else np.zeros((len(scene_centers), len(centers)))
    expected = predict(local, log_amplitude)
    targets = np.exp(sample_targets(local, log_amplitude, np.random.default_rng(seed)))
    candidates = np.flatnonzero(bank['plane'] != reference_plane)
    if len(candidates) < len(scene_centers):
        raise ValueError('Amplitude matching requires one donor per soma from outside the reference plane.')
    columns, assigned = match_unique(targets, candidates, amplitude(activity[:, candidates]))
    scene_weights = weights(scene_centers, scene_centers)
    assigned_map = np.exp(predict(scene_weights, np.log(np.maximum(assigned, .001))))
    expected_map = np.exp(expected)
    error = float(np.sqrt(np.mean((assigned_map - expected_map)**2)))

    stats = dict(reference_plane=reference_plane, reference_rois=len(centers), bandwidth_px=BANDWIDTH_PX,
                 spatial_field_used=use_spatial_field,
                 spatial_holdout_block_px=48, heldout_spearman=float(spearmanr(validation, log_amplitude).statistic),
                 heldout_log_rmse_model=rmse_model, heldout_log_rmse_global=rmse_global,
                 amplitude_definition='P95 minus temporal median of the donor relative fluorescence',
                 spatial_expected_p10_p90=np.quantile(expected_map, [.1, .9]).tolist(),
                 spatial_assigned_p10_p90=np.quantile(assigned_map, [.1, .9]).tolist(),
                 matched_amplitude_rmse=float(np.sqrt(np.mean((targets - assigned)**2))),
                 map_rmse=error, maps_median_amplitude=[float(np.median(m)) for m in (expected_map, assigned_map)],
                 reference_plane_excluded_from_donors=True,
                 scene_rank_r_before=float(spearmanr(expected, donors).statistic) if use_spatial_field else None,
                 scene_rank_r_after=float(spearmanr(expected, assigned).statistic) if use_spatial_field else None)
    return columns, expected_map, assigned, assigned_map, stats


def plot_amplitude(path, scene, expected, assigned_map, reference_plane=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    centers = scene['centers']
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), layout='constrained')
    low, high = 100 * np.quantile(np.r_[expected, assigned_map], [.05, .95])
    for ax, values, title in zip(axes, (expected, assigned_map),
                                 (f'Real plane {reference_plane} · local amplitude', 'Synthetic · local amplitude')):
        ax.imshow(scene['image'].T, cmap='gray', alpha=.65)
        overlay = ax.scatter(centers[:, 0], centers[:, 1], c=100*values, s=7,
                             cmap='magma', vmin=low, vmax=high, linewidths=0)
        ax.set_title(title, loc='left')
        ax.axis('off')
    fig.colorbar(overlay, ax=axes, label='Fluorescence change (%)')
    fig.savefig(path, dpi=155)
    plt.close(fig)
