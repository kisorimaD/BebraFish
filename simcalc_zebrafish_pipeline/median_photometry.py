"""Photometry shared by the Plane20 median and volume renderers."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def smooth_gain_field(shape, length_px, seed):
    """Unit-variance Gaussian field on a regularly spaced grid."""
    rng = np.random.default_rng(seed)
    lengths = np.broadcast_to(length_px, len(shape))
    pads = np.ceil(4 * lengths).astype(int)
    padded = rng.normal(size=tuple(np.asarray(shape) + 2 * pads))
    region = tuple(slice(int(pad), int(pad + size)) for pad, size in zip(pads, shape))
    field = gaussian_filter(padded, lengths / np.sqrt(2), mode='constant')[region]
    return (field - field.mean()) / max(field.std(), 1e-12)


def cell_amplitudes(field, centers, field_sigma, jitter_sigma, seed):
    if not len(centers):
        return np.empty(0)
    rng = np.random.default_rng(seed)
    log_gain = (field_sigma * map_coordinates(field, centers.T, order=1, mode='nearest')
                + jitter_sigma * rng.normal(size=len(centers)))
    amplitudes = np.exp(log_gain - .5 * (field_sigma**2 + jitter_sigma**2))
    return amplitudes / amplitudes.mean()


def median_background(envelope, cell_signal, density_floor):
    """Reserve the low-frequency part of the observed envelope for soma light."""
    return np.maximum(envelope - gaussian_filter(cell_signal, 12.), density_floor)


def median_texture(noise, sigma, highpass_sigma=9.):
    """Correlated, slightly skewed diffuse fluctuations used in the structured median."""
    field = gaussian_filter(noise, sigma)
    field /= max(float(field.std()), 1e-6)
    field += .24 * np.maximum(field - .55, 0)**2
    return field - gaussian_filter(field, highpass_sigma)


def effective_blur(signal, horizontal, vertical, mixture):
    """Mix the two Plane20 blur kernels in native YX order."""
    sharp = gaussian_filter(signal, (horizontal * .5, vertical * .5))
    broad = gaussian_filter(signal, (horizontal * 1.5, vertical * 1.5))
    return sharp * (1 - mixture) + broad * mixture


def median_cell_light(signal, mixture, distance, normalizer, amplitude):
    return (effective_blur(signal, .2, .8, mixture) * normalizer
            * np.clip(distance / 2, 0, 1) * amplitude).astype(np.float32)


def structured_amount(placement, tissue):
    if not tissue.any():
        return np.zeros_like(placement)
    threshold = np.percentile(placement[tissue], 65)
    amount = np.clip((placement - threshold) / 1.2, 0, 1)
    return amount * np.clip(gaussian_filter(tissue.astype(np.float32), 9) * 1.5, 0, 1)


def structured_layer(background, cells, amount, texture, subdued, subdued_gain, shell_gain, floor):
    """The Plane20 structured render redistributes shell light into diffuse light."""
    fraction_removed = 1 - (1 - (1 - subdued_gain) * subdued) * (1 - (1 - shell_gain) * amount)
    removed = (cells * fraction_removed).astype(np.float32)
    diffuse = np.maximum(background + gaussian_filter(removed, 8) + texture, floor)
    return diffuse + cells - removed
