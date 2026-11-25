"""Adjust static somatic brightness using a smooth reference from real Plane20."""
import numpy as np

from spatial_amplitude import predict, weights

RADIUS = 6
MAX_ITERATIONS = 6


def roi_pixels(centers, shape):
    dy, dx = np.mgrid[-RADIUS:RADIUS+1, -RADIUS:RADIUS+1]
    inside = dy**2 + dx**2 <= RADIUS**2
    y, x = np.rint(centers).astype(int).T
    yy = np.clip(y[:, None] + dy[inside], 0, shape[0] - 1)
    xx = np.clip(x[:, None] + dx[inside], 0, shape[1] - 1)
    return yy * shape[1] + xx


def calibrate_baseline(base, real_image, centers, reference_centers, maps):
    if real_image.shape != base.shape:
        raise ValueError('Reference and synthetic Plane20 must share native pixel geometry.')
    reference_pixels = roi_pixels(reference_centers, base.shape)
    synthetic_pixels = roi_pixels(centers, base.shape)
    real_samples = real_image.ravel()[reference_pixels].mean(axis=1)
    low, high = np.quantile(real_samples, [.01, .99])
    reference_map = predict(weights(centers, reference_centers), np.clip(real_samples, low, high))
    synthetic_weights = weights(centers, centers)

    def regional_map(image):
        brightness = image.ravel()[synthetic_pixels].mean(axis=1)
        return predict(synthetic_weights, np.clip(brightness, low, high))

    support = np.asarray(maps.sum(axis=0)).ravel()
    taper = np.minimum(support / np.quantile(support[support > 0], .60), 1)
    scale = np.divide(taper, support, out=np.zeros_like(support), where=support > 0)
    base_flat = base.ravel()
    before = regional_map(base)
    adjusted = base_flat.copy()
    after = before
    # Each increment uses the same spatial soma footprints as the video.
    # Limit total brightness changes to the observed regional mismatch scale.
    cap = max(float(np.quantile(abs(reference_map - before), .99)) * 3, 1.)
    rmse_before = float(np.sqrt(np.mean((before - reference_map)**2)))
    rmse_after = rmse_before
    for _ in range(MAX_ITERATIONS):
        delta = np.asarray(maps.T @ (reference_map - after)).ravel() * scale
        proposed = np.maximum(base_flat + np.clip(adjusted + delta - base_flat, -cap, cap), 0)
        updated_map = regional_map(proposed)
        error = float(np.sqrt(np.mean((updated_map - reference_map)**2)))
        if error >= rmse_after:
            break
        adjusted, after, rmse_after = proposed, updated_map, error
    adjusted = adjusted.reshape(base.shape).astype(np.float32)
    if not np.isfinite(adjusted).all():
        raise RuntimeError('Nonfinite calibrated baseline.')
    np.testing.assert_array_equal(adjusted.ravel()[support == 0], base_flat[support == 0])
    stats = dict(reference_rois=len(reference_centers), disk_radius_px=RADIUS,
                 robust_reference_range_adu=[float(low), float(high)],
                 map_rmse_before_adu=rmse_before, map_rmse_after_adu=rmse_after,
                 median_baseline_adu=[float(np.median(x)) for x in (reference_map, before, after)],
                 correction_cap_adu=cap,
                 correction_p5_p95_adu=np.quantile(adjusted.ravel()[support > 0] -
                                                    base_flat[support > 0], [.05, .95]).tolist(),
                 method='Additive time-independent correction on soma footprints.')
    return adjusted, (reference_map, before, after), stats


def plot_baseline(path, scene, panels):
    import matplotlib.pyplot as plt

    centers = scene['centers']
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), layout='constrained')
    low, high = np.quantile(np.concatenate(panels), [.02, .98])
    for ax, values, title in zip(axes, panels,
                                 ('Real Plane20', 'Synthetic before', 'Synthetic after')):
        ax.imshow(scene['image'].T, cmap='gray', alpha=.65, vmin=low, vmax=high)
        overlay = ax.scatter(centers[:, 0], centers[:, 1], c=values, s=7,
                             cmap='viridis', vmin=low, vmax=high, linewidths=0)
        ax.set_title(title, loc='left')
        ax.axis('off')
    fig.colorbar(overlay, ax=axes, label='Local baseline brightness (ADU)')
    fig.savefig(path, dpi=155)
    plt.close(fig)
