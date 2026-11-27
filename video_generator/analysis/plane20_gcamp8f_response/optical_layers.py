"""Separate soma-local fluorescence from diffuse illumination and detector grain."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening

from training.plane20.photometry import correlated_noise

HALO_SIGMA = 5.
HALO_FRACTION = .45
GLOW_CLEARANCE = 8.
GLOW_FADE = 10.


def halo_light(signal, *, spatial_scale=1.):
    width = HALO_SIGMA * spatial_scale
    sigma = (0, width, width) if signal.ndim == 3 else width
    return HALO_FRACTION * gaussian_filter(signal, sigma)


def optical_layers(scene, profile, reference, maps, glow_strength, *, spatial_scale=1.):
    shape = scene['image'].shape
    soma_mask = np.bincount(scene['instance_pixel_index'], minlength=np.prod(shape)).reshape(shape) > 0
    distance = distance_transform_edt(~soma_mask)
    envelope = profile['median_envelope'].astype(np.float32)
    opening = 2 * round(20 * spatial_scale) + 1
    tissue = gaussian_filter(grey_opening(envelope, size=(opening, opening)), 6 * spatial_scale)
    clearance = GLOW_CLEARANCE * spatial_scale
    fade = GLOW_FADE * spatial_scale
    glow_weight = np.clip((distance - clearance) / fade, 0, 1)
    glow = glow_strength * np.maximum(envelope - tissue, 0) * glow_weight
    assert not np.any(glow[distance <= clearance])

    static = np.asarray(maps.sum(axis=0)).reshape(shape)
    ceiling = max(1., float(np.quantile(reference, .999) - np.median(reference)))
    scale = np.minimum(1, ceiling / np.maximum(static, 1))
    maps = maps.multiply(scale.ravel()).tocsr()
    soma = np.asarray(maps.sum(axis=0)).reshape(shape)
    halo = halo_light(soma, spatial_scale=spatial_scale)
    options = dict(halo_sigma_px=HALO_SIGMA * spatial_scale, halo_integrated_fraction=HALO_FRACTION,
                   background_glow_clearance_px=clearance, background_glow_fade_px=fade,
                   background_glow_strength=glow_strength, soma_pixel_cap_adu=ceiling,
                   background_glow_max_near_soma_adu=float(glow[distance <= clearance].max()),
                   background_glow_mean_outside_adu=float(glow[distance > clearance].mean())
                   if np.any(distance > clearance) else 0.,
                   static_halo_p95_adu=float(np.quantile(halo, .95)))
    return maps, (tissue, glow, soma, halo), options


def add_optical_noise(frame, halo, levels, estimates, base_seed, halo_seed):
    # The recorded intensity-to-noise curve gives the total variance. The
    # incremental variance from soma-scattered light forms the local grain.
    total = np.interp(frame, levels, estimates)
    background = np.interp(frame - np.maximum(halo, 0), levels, estimates)
    extra = np.sqrt(np.maximum(total**2 - background**2, 0))
    base_unit, _ = correlated_noise(frame.shape, base_seed)
    halo_unit, _ = correlated_noise(frame.shape, halo_seed)
    return base_unit * background, halo_unit * extra
