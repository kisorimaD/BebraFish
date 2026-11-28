"""Image and point-pattern summaries on shared real and synthetic windows."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree

from soma_shell_model import normalize_profiles, radial_profiles

PIXEL_LAGS = np.array([1, 2, 4, 8])


def tissue_windows(support, size=32):
    return [np.s_[y:y+size, x:x+size]
            for y in range(0, support.shape[0]-size+1, size)
            for x in range(0, support.shape[1]-size+1, size)
            if support[y:y+size, x:x+size].mean() >= .75]


def image_statistics(image, support, windows, regions):
    contrast = [np.ptp(np.percentile(image[window][support[window]], [5, 95]))
                for window in windows]
    image_float = image.astype(np.float32)
    residual = image_float-gaussian_filter(image_float, 12)
    return contrast, pixel_correlation(residual, [support]), pixel_correlation(residual, regions)


def pixel_correlation(residual, regions):
    correlations = []
    for lag in PIXEL_LAGS:
        left, right = [], []
        for mask in regions:
            for a, b in ((np.s_[:, :-lag], np.s_[:, lag:]), (np.s_[:-lag, :], np.s_[lag:, :])):
                valid = mask[a] & mask[b]
                left.append(residual[a][valid])
                right.append(residual[b][valid])
        x = np.concatenate(left) if left else np.empty(0)
        y = np.concatenate(right) if right else np.empty(0)
        if len(x) < 2:
            correlations.append(np.nan)
            continue
        # Each displacement contributes in both directions.
        mean = (x.mean(dtype=float)+y.mean(dtype=float))/2
        x, y = x-mean, y-mean
        variance = (np.sum(x*x, dtype=float)+np.sum(y*y, dtype=float))/2
        correlations.append(np.sum(x*y, dtype=float)/variance if variance > 0 else np.nan)
    return np.asarray(correlations)


def nearest_curve(points_by_crop, crops, radii):
    """Reduced-sample estimate at each radius, with a matching denominator."""
    hits = np.zeros(len(radii), dtype=int)
    totals = np.zeros_like(hits)
    for points, crop in zip(points_by_crop, crops):
        y, x = crop['y']-.5, crop['x']-.5
        bottom, right = y+crop['height'], x+crop['width']
        inside = ((points[:, 0] >= y) & (points[:, 0] < bottom) &
                  (points[:, 1] >= x) & (points[:, 1] < right))
        points = np.unique(points[inside], axis=0)
        if not len(points):
            continue
        margin = np.minimum(points-[y, x], [bottom, right]-points).min(axis=1)
        distance = cKDTree(points).query(points, k=2)[0][:, 1]
        eligible = margin[:, None] >= radii
        totals += eligible.sum(axis=0)
        hits += (eligible & (distance[:, None] <= radii)).sum(axis=0)
    return np.divide(hits, totals, out=np.full(len(radii), np.nan), where=totals > 0)


def shell_statistics(image, geometry, offset, peak, distance_bins):
    centers, major, minor, angle = geometry
    n_bins = len(distance_bins)-1
    if not len(centers):
        return None, [], [], np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int)
    radii, samples = radial_profiles(image, centers, major, minor, angle,
                                     coordinate_offset=offset, return_samples=True)
    curves = np.median(samples, axis=2)
    _, core, contrast = normalize_profiles(radii, curves)
    visible = contrast > 1
    arcs = samples[visible][:, np.abs(radii-peak) <= .10, :].mean(axis=1)-core[visible, None]
    log_arcs = np.log(np.maximum(arcs, contrast[visible, None]*.1))
    angular = gaussian_filter1d(log_arcs, 2, axis=1, mode='wrap').std(axis=1)

    positions, gains = centers[visible], np.log(contrast[visible])
    pairs = cKDTree(positions).query_pairs(distance_bins[-1], output_type='ndarray')
    a, b = pairs.T
    distances = np.linalg.norm(positions[a]-positions[b], axis=1)
    counts = np.histogram(distances, bins=distance_bins)[0]
    sums = np.histogram(distances, bins=distance_bins, weights=.5*(gains[a]-gains[b])**2)[0]
    semivariance = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
    return (radii, curves), contrast, angular, semivariance, counts
