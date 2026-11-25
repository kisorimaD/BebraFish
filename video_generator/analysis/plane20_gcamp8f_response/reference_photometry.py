"""Fit global brightness statistics and intensity-dependent noise to a real recording."""
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import tifffile

QUANTILES = np.array([.02, .05, .10, .25, .50, .75, .90, .95, .98, .995])
TONE_ANCHORS = np.array([70, 95, 110, 130, 155, 185, 220, 265, 320, 410, 520], dtype=float)


def noise_curve(source, *, adaptive=False):
    rng = np.random.default_rng(51)
    with tifffile.TiffFile(source) as recording:
        height, width = recording.pages[0].shape
        y, x = rng.integers(height, size=35000), rng.integers(width, size=35000)
        windows = []
        for block in range(0, len(recording.pages) // 50, 4):
            start = block * 50 + 9
            windows.append(np.array([recording.pages[i].asarray()[y, x]
                                     for i in range(start, start + 32)], dtype=np.float32))
    median = np.median(np.concatenate(windows), axis=0)
    detail = np.concatenate([frames[1:-1] - .5 * (frames[:-2] + frames[2:]) for frames in windows])
    sigma = 1.4826 * np.median(abs(detail - np.median(detail, axis=0)), axis=0) / np.sqrt(1.5)
    levels, estimates = [], []
    edges = np.unique(np.quantile(median, np.linspace(0, 1, 17))) if adaptive else np.arange(95, 436, 20)
    for lo, hi in zip(edges, edges[1:]):
        selected = (median >= lo) & (median < hi)
        if selected.sum() >= 30:
            levels.append(float(np.median(median[selected])))
            estimates.append(float(np.median(sigma[selected])))
    if not levels:
        return np.array([float(np.median(median))]), np.array([float(np.median(sigma))])
    return np.array(levels), np.maximum.accumulate(estimates)


def brightness_statistics(values, black, white):
    output = np.clip(values, black, white)
    return dict(mean_adu=float(output.mean()), std_adu=float(output.std()),
                quantiles_adu=np.quantile(output, QUANTILES).tolist(),
                clipped_black_fraction=float(np.mean(values <= black)),
                clipped_white_fraction=float(np.mean(values >= white)))


def fit_brightness(base, maps, modulation, source, levels, sigma, noise_fraction, black, white,
                   halo_filter=None, adaptive=False):
    """Apply one brightness mapping to the static scene; activity keeps its ADU scale."""
    rng = np.random.default_rng(264)
    pixels = rng.choice(base.size, min(10000, base.size), replace=False)
    nominal = base.ravel()[pixels]
    local_maps = maps[:, pixels]
    with tifffile.TiffFile(source) as recording:
        frame_ids = np.arange(len(recording.pages))
        train_pool = frame_ids[(frame_ids // 50) % 2 == 0]
        test_pool = frame_ids[(frame_ids // 50) % 2 == 1]
        train = train_pool[np.linspace(0, len(train_pool) - 1, 30, dtype=int)]
        test = test_pool[np.linspace(0, len(test_pool) - 1, 30, dtype=int)]

        def sample(indices):
            actual = np.stack([recording.pages[i].asarray().ravel()[pixels] for i in indices])
            if halo_filter is None:
                activity = modulation[indices] @ local_maps
            else:
                full = (modulation[indices] @ maps).reshape(-1, *base.shape)
                activity = (full + halo_filter(full)).reshape(len(indices), -1)[:, pixels]
            noise = rng.normal(size=activity.shape)
            return actual, activity, noise

        train_reference, train_activity, train_noise = sample(train)
        test_reference, test_activity, test_noise = sample(test)

    def simulate(mapped, activity, noise):
        clean = mapped + activity
        return clean + noise * noise_fraction * np.interp(clean, levels, sigma)

    input_anchors = np.linspace(black - .15 * (white - black), white + .25 * (white - black),
                                len(TONE_ANCHORS)) if adaptive else TONE_ANCHORS
    pivot = float(np.median(train_reference)) if adaptive else 150.

    def mapped_brightness(values, anchors):
        return values + np.interp(values, input_anchors, anchors - input_anchors)

    # Work in the visible range, keeping the same display for both halves.
    objective = brightness_statistics(train_reference, black, white)
    target = np.array(objective['quantiles_adu'])
    anchors = input_anchors.copy()
    for _ in range(5):
        current_values = simulate(mapped_brightness(nominal, anchors), train_activity, train_noise)
        measured = np.quantile(current_values, QUANTILES)
        adjustment = np.interp(anchors, measured, target - np.clip(measured, black, white))
        anchors += .65 * adjustment
        anchors = np.maximum.accumulate(anchors - .1 * input_anchors) + .1 * input_anchors

    mapped = mapped_brightness(nominal, anchors)

    def residual(parameters):
        offset, gain = parameters
        values = simulate(pivot + (mapped - pivot) * gain + offset, train_activity, train_noise)
        visible = np.clip(values, black, white)
        return [visible.mean() - objective['mean_adu'], visible.std() - objective['std_adu']]

    offset_limit = max(15., .06 * (white - black)) if adaptive else 15.
    final_offset, final_gain = least_squares(
        residual, [0., 1.], bounds=([-offset_limit, .8], [offset_limit, 1.2])).x
    adjusted = (pivot + (mapped_brightness(base, anchors) - pivot) * final_gain + final_offset).astype(np.float32)
    before = simulate(nominal, test_activity, test_noise)
    after = simulate(adjusted.ravel()[pixels], test_activity, test_noise)
    reference = brightness_statistics(test_reference, black, white)
    pre = brightness_statistics(before, black, white)
    post = brightness_statistics(after, black, white)
    errors = [float(np.sqrt(np.mean((np.array(entry['quantiles_adu']) -
                                      reference['quantiles_adu'])**2))) for entry in (pre, post)]
    report = dict(fitted_frames=train.tolist(), validation_frames=test.tolist(),
                  quantile_probabilities=QUANTILES.tolist(), reference=reference,
                  prior=pre, calibrated=post, quantile_rmse_before_after_adu=errors,
                  validation_improved=errors[1] < errors[0], moment_pivot_adu=pivot,
                  mapping_input_adu=input_anchors.tolist(), mapping_output_adu=anchors.tolist(),
                  fitted_moment_offset_adu=float(final_offset), fitted_moment_gain=float(final_gain))
    return adjusted, report


def validate_comparison_gif(path, source_indices, width, height, black, white, *, adaptive=False):
    rng = np.random.default_rng(1167)
    y, x = rng.integers(height, size=20000), rng.integers(width - 1, size=20000)
    held_out = (source_indices // 50) % 2 == 1
    if not held_out.any():
        raise ValueError('No GIF frame comes from a held-out interval. Check --fps and the recording length.')
    samples = [[], []]
    duration = 0
    with Image.open(path) as gif:
        assert gif.n_frames == len(source_indices) and gif.size == (width, 2 * height)
        for i, use in enumerate(held_out):
            gif.seek(i)
            gif.load()
            duration += gif.info['duration']
            if use:
                image = np.asarray(gif.convert('L'))
                for panel in range(2):
                    rows = y + panel * height
                    samples[panel].append(np.stack((image[rows, x], image[rows, x + 1])))
    scale = (white - black) / 255
    metrics = {}
    for panel, data in zip(('real', 'synthetic'), samples):
        codes = np.array(data, dtype=np.uint8)
        values = codes.astype(np.float32) * scale + black
        left, right = values[:, 0], values[:, 1]
        stats = brightness_statistics(left, black, white)
        stats['clipped_black_fraction'] = float(np.mean(codes[:, 0] == 0))
        stats['clipped_white_fraction'] = float(np.mean(codes[:, 0] == 255))
        texture = left - right
        amplitude = 1.4826 * np.median(abs(texture - np.median(texture, axis=0)), axis=0)
        left_median, right_median = np.median(left, axis=0), np.median(right, axis=0)
        mean_level = (left_median + right_median) / 2
        tolerance = .05 * (white - black) if adaptive else 12
        valid = (abs(left_median - right_median) < tolerance) & (
            np.mean((values <= black + scale) | (values >= white - scale), axis=(0, 1)) < .02)
        stats['grain_by_adu'] = []
        edges = np.linspace(black, white, 7).tolist() if adaptive else [115, 150, 185, 220, 260, 310]
        for lo, hi in zip(edges, edges[1:]):
            selection = valid & (mean_level >= lo) & (mean_level < hi)
            stats['grain_by_adu'].append(dict(range_adu=[lo, hi], pairs=int(selection.sum()),
                                             robust_sd_adu=float(np.median(amplitude[selection]))
                                             if selection.any() else None))
        metrics[panel] = stats
    real, synthetic = (metrics[key] for key in ('real', 'synthetic'))
    differences = {key: synthetic[key] - real[key] for key in
                   ('mean_adu', 'std_adu', 'clipped_black_fraction', 'clipped_white_fraction')}
    differences['quantiles_adu'] = np.subtract(synthetic['quantiles_adu'], real['quantiles_adu']).tolist()
    differences['grain_by_adu'] = [
        dict(range_adu=a['range_adu'], robust_sd_adu=b['robust_sd_adu'] - a['robust_sd_adu']
             if a['robust_sd_adu'] is not None and b['robust_sd_adu'] is not None else None)
        for a, b in zip(real['grain_by_adu'], synthetic['grain_by_adu'])
    ]
    return duration, dict(validation_frames=int(held_out.sum()), **metrics, synthetic_minus_real=differences)
