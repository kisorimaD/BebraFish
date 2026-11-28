"""Estimate and render brightness-dependent grain for a static image."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .generate import correlated_noise


BACKGROUND_BLEND = 'multiscale_v1'
BACKGROUND_BLEND_PADDING = 12


def reference_background_weights(support):
    """Round the seam and blend brightness more gradually than texture."""
    background = ~np.asarray(support, dtype=bool)
    if background.all():
        full = np.ones(background.shape, dtype=np.float32)
        return full, full
    distance = distance_transform_edt(background).astype(np.float32)
    rounded = gaussian_filter(distance, 2, radius=6)

    def transition(distance, width):
        fraction = np.clip(distance / width, 0, 1)
        return fraction**2 * (3 - 2*fraction)

    edge = transition(distance, 4)
    weights = []
    for width in (12, 32):
        weight = transition(rounded, width) * edge
        weight[distance >= width+6] = 1
        weights.append(weight)
    return tuple(weights)


def blend_reference_background(render, reference, weights):
    fine, broad = weights
    background = fine > 0
    difference = np.where(background, reference - render, 0)
    # Smooth only the difference. Normalization excludes the synthetic side.
    smooth = gaussian_filter(difference, 4, radius=BACKGROUND_BLEND_PADDING)
    support = gaussian_filter(background.astype(np.float32), 4, radius=BACKGROUND_BLEND_PADDING)
    smooth /= np.maximum(support, 1e-6)
    return render + fine*difference + (broad-fine)*smooth


def binned_noise(intensity, detail, *, temporal=False):
    """Bin paired brightness and high-frequency deviations in native ADU."""
    levels, estimates = [], []
    groups = np.array_split(np.argsort(intensity), min(16, max(1, len(intensity)//64)))
    for group in groups:
        values = detail[..., group]
        deviation = (np.median(np.abs(values - np.median(values, axis=0)), axis=0)
                     if temporal else np.abs(values - np.median(values)))
        level, sigma = float(np.median(intensity[group])), float(1.4826*np.median(deviation))
        if not levels or level > levels[-1]:
            levels.append(level)
            estimates.append(sigma)
    return levels, np.maximum.accumulate(estimates).tolist()


def spatial_noise(image):
    """A spatial texture proxy for a lone image or existing temporal average."""
    h, w = (n - n % 2 for n in image.shape)
    blocks = image[:h, :w].astype(np.float32).reshape(h//2, 2, w//2, 2)
    mean = blocks.mean(axis=(1, 3)).ravel()
    detail = ((blocks[:, 0, :, 0] + blocks[:, 1, :, 1] -
               blocks[:, 0, :, 1] - blocks[:, 1, :, 0]) * .5).ravel()
    if not mean.size:
        return dict(method='spatial_2x2', levels_adu=[float(image.mean())], sigma_adu=[0.])
    levels, sigma = binned_noise(mean, detail)
    return dict(method='spatial_2x2', levels_adu=levels, sigma_adu=sigma)


def temporal_noise(pages, shape, median_frames):
    """Estimate raw-frame sigma from consecutive triplets and scale to a median."""
    if len(pages) < 5:
        return None
    width = min(9, len(pages))
    starts = np.linspace(0, len(pages)-width, min(6, len(pages)//width)).round().astype(int)
    rng = np.random.default_rng(51)
    pixels = np.sort(rng.choice(np.prod(shape), min(16000, np.prod(shape)), replace=False))
    windows = [np.stack([pages[int(i)].asarray().ravel()[pixels] for i in range(start, start+width)])
               .astype(np.float32) for start in starts]
    sampled = np.concatenate(windows)
    if not np.isfinite(sampled).all():
        raise ValueError('Кадры для оценки шума должны содержать конечные интенсивности.')
    intensity = np.median(sampled, axis=0)
    detail = np.concatenate([f[1:-1] - .5*(f[:-2]+f[2:]) for f in windows]) / np.sqrt(1.5)
    levels, sigma = binned_noise(intensity, detail, temporal=True)
    estimate_scale = min(1., np.sqrt(np.pi/(2*median_frames)))
    return dict(method='temporal_triplets_scaled_to_median', levels_adu=levels,
                sigma_adu=(np.asarray(sigma)*estimate_scale).tolist(),
                raw_frame_sigma_adu=sigma, median_scale=float(estimate_scale),
                sample_windows=starts.tolist(), pages_per_window=width)


def render_intensity_noise(background, cells, envelope, soma_mask, curve, seed):
    """Use the observed envelope outside somas and rendered light within."""
    clean = background + cells
    levels, sigma = curve['levels_adu'], curve['sigma_adu']
    base = np.interp(np.where(soma_mask, background, envelope), levels, sigma)
    total = np.interp(clean[soma_mask], levels, sigma)
    extra = np.sqrt(np.maximum(total**2 - base[soma_mask]**2, 0))
    base_seed, extra_seed = np.random.SeedSequence(int(seed)).spawn(2)
    base_unit, _ = correlated_noise(clean.shape, base_seed)
    extra_unit, _ = correlated_noise(clean.shape, extra_seed)
    noise = base * base_unit
    noise[soma_mask] += extra * extra_unit[soma_mask]
    return clean + noise
