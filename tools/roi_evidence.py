"""Measure local fluorescence evidence for existing Plane20 ROI footprints."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

ANGLES = np.linspace(0, 2*np.pi, 40, endpoint=False)
RADII = np.arange(.15, 1.57, .07)
DIRECTIONS = np.stack([np.cos(ANGLES), np.sin(ANGLES)])


def roi_evidence(image, footprints):
    smooth = gaussian_filter(image, .8)
    measurements = []
    for flat in footprints:
        points = np.column_stack(np.divmod(flat, image.shape[1])).astype(float)
        center = points.mean(axis=0)
        values, basis = (np.linalg.eigh(np.cov(points.T, bias=True))
                         if len(points) > 2 else (np.ones(2), np.eye(2)))
        directions = basis @ (2*np.sqrt(np.maximum(values, 1))[:, None]*DIRECTIONS)
        coordinates = directions[:, None, :]*RADII[None, :, None] + center[:, None, None]
        samples = map_coordinates(smooth, coordinates, order=1, mode='nearest')
        core = np.median(samples[RADII < .4], axis=0)
        rim = np.max(samples[(RADII >= .60) & (RADII <= 1.03)], axis=0)
        outer = np.median(samples[RADII >= 1.20], axis=0)
        relief = rim - np.maximum(core, outer)
        measurements.append((np.percentile(samples, 95), np.median(relief), np.mean(relief > 3)))
    return np.asarray(measurements)
