"""Estimate a median-render profile from a target image and local annotations."""
import json

import numpy as np
import tifffile
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d, label
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, cdist

from .generate import ROOT
from .photometry import spatial_noise, temporal_noise
from analyze_somas import measure
from annotate_regions import compute_mask
from soma_shell_model import normalize_profiles, radial_profiles, shell_patch


def prepare_median(source, target, *, diagnostics=None):
    """Use a bounded, evenly spaced sample with translation registration."""
    with tifffile.TiffFile(source) as stack:
        first = stack.pages[0]
        if len(first.shape) != 2:
            raise ValueError('Нужен 2D TIFF или TIFF-стек одной плоскости.')
        limit = max(1, min(64, (128 * 1024**2) // (4 * np.prod(first.shape))))
        total = len(stack.pages)
        if total > 1 and limit < 2:
            raise ValueError('Кадры слишком велики для регистрации стека.')
        indices = np.linspace(0, total-1, min(total, limit)).round().astype(int)
        frames = np.stack([stack.pages[int(index)].asarray().astype(np.float32)
                           for index in indices])
        noise = temporal_noise(stack.pages, first.shape, len(indices))
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 65535:
        raise ValueError('TIFF должен содержать конечные интенсивности в диапазоне 0–65535.')
    shifts = []
    if len(frames) > 1:
        from scipy.ndimage import shift
        from inspect_movie import registration_fft, pair_shift, peak_to_sidelobe
        reference = registration_fft(frames[len(frames)//2])
        for i, frame in enumerate(frames):
            moving = registration_fft(frame)
            delta = pair_shift(reference, moving) if peak_to_sidelobe(reference, moving) >= 6 else (0., 0.)
            frames[i] = shift(frame, delta, order=1, mode='nearest')
            shifts.append([float(value) for value in delta])
    if diagnostics is not None and len(frames) >= 4:
        half = len(frames)//2
        rng = np.random.default_rng(42)
        groups = [np.arange(half), np.arange(len(frames)-half, len(frames))]
        groups.extend(rng.choice(len(frames), size=half, replace=False) for _ in range(4))
        np.savez_compressed(diagnostics, medians=np.stack([
            np.median(frames[group], axis=0) for group in groups
        ]), pages=np.stack([indices[group] for group in groups]))
    median = np.rint(np.median(frames, axis=0, overwrite_input=True)).astype(np.uint16)
    tifffile.imwrite(target, median, photometric='minisblack')
    return dict(source=str(source), sampled_pages=indices.tolist(), frame_count=total,
                shifts_yx=shifts, intensity_noise=noise or spatial_noise(median))


def morphology(rows):
    diameter = np.array([row['equivalent_diameter_px'] for row in rows])
    excess = np.maximum([row['aspect_ratio']-1 for row in rows], .001)
    angle = np.radians([2*row['orientation_deg'] for row in rows])
    resultant = np.exp(1j*angle).mean()
    mean, variance = excess.mean(), max(excess.var(), .0001)
    return dict(diameter_fit_comparison=[dict(name='normal', parameters=[
                    float(diameter.mean()), float(max(diameter.std(), .1))])],
                aspect_ratio_excess_fit_comparison=[dict(name='gamma', parameters=[
                    float(mean**2/variance), 0, float(variance/mean)])],
                orientation=dict(axial_mean_deg=float(np.degrees(np.angle(resultant))/2),
                                 vonmises_doubled_kappa=float(min(8, 2*abs(resultant) /
                                                                  max(1-abs(resultant)**2, .01)))))


def fit_shell(image, rows):
    centers = np.array([[row['centroid_y'], row['centroid_x']] for row in rows])
    major, minor, theta = [np.array([row[key] for row in rows]) for key in
                           ('major_axis_px', 'minor_axis_px', 'orientation_deg')]
    radii, samples = radial_profiles(image, centers, major, minor, theta,
                                     coordinate_offset=.5, return_samples=True)
    norm, core, contrast = normalize_profiles(radii, np.median(samples, axis=2))
    visible = contrast > 1
    if visible.sum() < 3:
        raise ValueError('Нужно хотя бы три контура с различимой светлой оболочкой.')
    fit_band = radii <= 1.2
    goal = np.median(norm[visible], axis=0)
    axes = np.median(major[visible]), np.median(minor[visible])

    def model(params):
        origin, patch, _ = shell_patch((0., 0.), *axes, 0., True, 0., .6,
                                       peak_radius=params[0], width=params[1])
        _, curves = radial_profiles(patch, [-origin], [axes[0]], [axes[1]], [0.], radii)
        normalized, _, contrast = normalize_profiles(radii, curves * 1000)
        return normalized[0], contrast[0]/1000

    params = least_squares(lambda p: (model(p)[0]-goal)[fit_band], (.8, .2),
                           bounds=((.45, .06), (1.25, .55)), diff_step=.02).x
    peak, width = params
    # The existing renderer adds a small spatially varying blur after the patch.
    unit_contrast = model(params)[1]
    shell = np.mean(samples[:, np.abs(radii-peak) <= .10, :], axis=1) - core[:, None]
    log_arcs = np.log(np.maximum(shell[visible], contrast[visible, None]*.1))
    angular = gaussian_filter1d(log_arcs, 2, axis=1, mode='wrap').std(axis=1)
    gains = np.log(contrast[visible])
    distances = pdist(centers[visible])
    differences = .5*pdist(gains[:, None])**2
    scale = max(1., float(np.median(major)))
    variance = float(np.clip(gains.var(), 0, 1.5))
    length, shared = 2*scale, 0.
    if len(gains) >= 8 and np.sum(distances < 3*scale) >= 8 and np.sum(distances > 3*scale) >= 8:
        use = distances <= 8*scale
        d, observed = distances[use], differences[use]
        result = least_squares(lambda p: variance-p[0]*np.exp(-.5*(d/p[1])**2)-observed,
                               (variance/2, length), bounds=((0., scale/2), (variance+1e-6, 8*scale)),
                               loss='soft_l1', f_scale=max(.02, variance))
        shared, length = result.x
    return dict(width=float(width), peak=float(peak),
                shell_flux=float(width*peak*np.median(contrast[visible])/max(unit_contrast, .01)),
                angular_sd=float(np.clip(np.median(angular), 0., .9)),
                gain_length=float(length), gain_field_sd=float(np.sqrt(shared)),
                gain_jitter_sd=float(np.sqrt(max(0., variance-shared)))), contrast[visible]


def fit_profile(image, soma_doc, region_doc, crops, path, *, intensity_noise=None, grow_somas=False):
    rows = [measure(item['points_xy']) for item in soma_doc['annotations']
            if item['confidence'] == 'confident' and not item['truncated']]
    rows = [row for row in rows if np.all(np.isfinite(list(row.values()))) and row['minor_axis_px'] >= 2]
    if len(rows) < 5:
        raise ValueError('Сохраните хотя бы пять уверенных полных контуров. Рекомендуется около 20.')
    exclusion = compute_mask(image.astype(np.float32), region_doc)
    if not exclusion.any():
        raise ValueError('Отметьте и сохраните хотя бы одну область без сом.')
    envelope = gaussian_filter(image.astype(np.float32), 12)
    floor = float(np.percentile(image[exclusion], 10))
    separation = max(1., .15 * (np.percentile(envelope, 90)-floor))
    tissue = binary_fill_holes(envelope > floor+separation)
    components, _ = label(tissue)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    tissue &= sizes[components] >= 25
    support = tissue & ~exclusion
    if not support.any():
        raise ValueError('После исключения фона не осталось области для размещения сом.')

    sampled = np.zeros(image.shape, bool)
    positions = []
    nearest = []
    for crop in crops:
        x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
        sampled[y:y+h, x:x+w] = True
        points = np.unique(np.asarray(crop['points'], int).reshape(-1, 2), axis=0)
        positions.extend(map(tuple, points))
        if len(points) >= 2:
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            nearest.extend(distances.min(axis=1))
    positions = np.asarray(sorted(set(positions)), int).reshape(-1, 2)
    if len(positions) < 3:
        raise ValueError('Отметьте хотя бы три центра в просмотренных кропах.')
    px, py = positions.T
    if not support[py, px].all():
        raise ValueError('Некоторые центры лежат за пределами оценённой ткани или в размеченном фоне.')
    area = int(np.count_nonzero(sampled & support))
    count = max(1, round(len(positions)*support.sum()/area))
    diameter = np.median([row['equivalent_diameter_px'] for row in rows])
    spacing = min(diameter*.8, float(np.percentile(nearest, 20)) if nearest else diameter*.55)
    if count >= 65535:
        raise ValueError('Оценка плотности слишком велика. Используйте меньшую область изображения.')
    light, contrast = fit_shell(image, rows)
    bands = [gaussian_filter(image.astype(np.float32), a)-gaussian_filter(image.astype(np.float32), b)
             for a, b in [(1, 3), (3, 7)]]
    rms = [float(band[exclusion].std()) for band in bands]
    texture_sigma = float(np.clip(2*rms[1]/max(rms[0], .1), .7, 6.))
    config = dict(size_scale=1., aspect_scale=1., count=int(count), spacing=spacing,
                  candidate_count=min(65534, max(count+8, round(count*1.7))),
                  packing_iterations=12, roi_size=None, density_floor=floor,
                  texture_sigma=texture_sigma, texture_band_rms=rms[1],
                  texture_shell_gain=float(np.clip(np.percentile(contrast, 25)/np.median(contrast), .35, 1)),
                  intensity_noise=intensity_noise or spatial_noise(image),
                  amplitude=1., grow_somas=grow_somas, **light)
    settings = dict(selected=dict(sigma_px=float(max(1., diameter/4)), amplitude_cv=0.))
    np.savez_compressed(path, tissue=tissue, neuron_support=support,
                        density=np.maximum(envelope-floor, 1), normalizer=np.float32(1),
                        reference_background=np.where(support, 0, image),
                        median_envelope=envelope, median_config=np.array(json.dumps(config)),
                        settings_json=np.array(json.dumps(settings)),
                        morphology_json=np.array(json.dumps(morphology(rows))))
    return dict(config, contours=len(rows), centers=len(positions), sampled_area_px=area,
                support_area_px=int(support.sum()))
