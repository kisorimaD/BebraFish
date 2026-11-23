"""Experimental empirical soma ellipses with independently scaled SimCalc emission.

Does not change frozen Plane16 renderer. Geometry is sampled jointly and
uniformly from whole-soma annotations. Angles use native TIFF X/Y axes.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor/SimCalc'))
from simcalc.simcalclib import makeGaussianFilter

# Radius at the outer half maximum of exp(-r²/2)-exp(-r²).
RING_HALF_MAX_RADIUS = np.sqrt(-2*np.log((1-np.sqrt(.5))/2))
SOLID_HALF_MAX_RADIUS = np.sqrt(2*np.log(2))


def ellipse_patch(center_yx, major, minor, angle_deg, ring, fill, blur):
    """Full soma mask is independent of fluorescence type, fill and blur."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    half = RING_HALF_MAX_RADIUS if ring else SOLID_HALF_MAX_RADIUS
    sigma_major, sigma_minor = major/(2*half), minor/(2*half)
    basis_yx = np.array([[s, c], [c, -s]])
    covariance = basis_yx@np.diag([sigma_major**2, sigma_minor**2])@basis_yx.T
    radius = int(np.ceil(max(major/2, 4*sigma_major)+3*blur))+2
    origin = np.floor(center_yx).astype(int)-radius
    side = 2*radius+1
    local = np.asarray(center_yx)-origin
    kernel = makeGaussianFilter(*local, covariance, side, ring=ring)
    kernel += .2*(kernel > .5)  # SimCalc's bright ROI contribution.
    if ring and fill > 0:
        solid = makeGaussianFilter(*local, covariance, side, ring=False)
        solid += .2*(solid > .5)
        kernel = (1-fill)*kernel+fill*solid
    kernel = gaussian_filter(kernel, blur)
    y, x = np.mgrid[:side, :side].astype(float)
    y -= local[0]; x -= local[1]
    along = x*c+y*s
    across = -x*s+y*c
    mask = (along/(major/2))**2+(across/(minor/2))**2 <= 1
    return origin, kernel.astype(np.float32), mask


def build_empirical_scene(profile, settings, centers, bank, seed, *, geometry=None, patch_renderer=None, amplitudes=None, angular_coefficients=None):
    """Render ellipse geometry; optional amplitudes multiply whole cells."""
    tissue = profile['tissue'].astype(bool)
    n = len(centers)
    if not 0 < n < 65535:
        raise ValueError('Cell count must fit uint16 labels')
    rng = np.random.default_rng(seed)
    if geometry is None:
        chosen = rng.integers(len(bank['major_axis_px']), size=n)
        major = bank['major_axis_px'][chosen]
        minor = bank['minor_axis_px'][chosen]
        angles = bank['orientation_deg'][chosen]
    else:
        chosen = None
        major, minor, angles = (np.asarray(geometry[k]) for k in
                                ['major_axis_px','minor_axis_px','orientation_deg'])
        if any(len(v) != n for v in [major,minor,angles]):
            raise ValueError('Geometry count does not match centers')
    renderer = ellipse_patch if patch_renderer is None else patch_renderer
    if not (np.all(np.isfinite(major)) and np.all(minor > 0) and np.all(major >= minor)):
        raise ValueError('Invalid empirical axes')
    cv = settings['amplitude_cv']
    sampled_amplitudes = rng.lognormal(-cv**2/2, cv, n)
    if amplitudes is None:
        amplitudes = sampled_amplitudes
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (n,) or not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
            raise ValueError('Amplitudes must be finite positive values, one per center')
    if angular_coefficients is not None:
        angular_coefficients=np.asarray(angular_coefficients,float)
        if angular_coefficients.ndim!=3 or angular_coefficients.shape[0]!=n or angular_coefficients.shape[2]!=2 or not np.isfinite(angular_coefficients).all():
            raise ValueError('Angular coefficients must have shape (cells, modes, 2) and be finite')
    rings = rng.random(n) < settings['ring_fraction']
    signal = np.zeros(tissue.shape, dtype=np.float32)
    strength = np.zeros_like(signal)
    labels = np.zeros(tissue.shape, dtype=np.uint16)
    offsets = [0]; footprints = []
    for i, center in enumerate(centers):
        origin, kernel, mask = renderer(center, major[i], minor[i], angles[i],
            bool(rings[i]), settings['ring_center_fill'], settings['blur_px'],
            **({} if angular_coefficients is None else dict(angular_coefficients=angular_coefficients[i])))
        start = np.maximum(origin, 0); stop = np.minimum(origin+kernel.shape, tissue.shape)
        target = tuple(slice(a,b) for a,b in zip(start,stop))
        local = tuple(slice(a-o,b-o) for a,b,o in zip(start,stop,origin))
        footprint = mask[local] & tissue[target]
        intensity = kernel[local]*amplitudes[i]*profile['tissue_field'][target]*profile['cell_gain'][target]
        signal[target] += intensity
        y,x = np.nonzero(footprint)
        pixels = ((y+start[0])*tissue.shape[1]+x+start[1]).astype(np.uint32)
        footprints.append(pixels); offsets.append(offsets[-1]+len(pixels))
        replace = footprint & ((labels[target]==0) | (intensity > strength[target]))
        strength[target][replace] = intensity[replace]
        labels[target][replace] = i+1
    result = dict(signal=signal,labels=labels,
        instance_offsets=np.asarray(offsets,dtype=np.uint32),
        instance_pixel_index=np.concatenate(footprints),
        major_axis_px=major,minor_axis_px=minor,orientation_deg=angles,rings=rings)
    if chosen is not None:
        result.update(empirical_indices=chosen,annotation_numbers=bank['numbers'][chosen])
    return result
