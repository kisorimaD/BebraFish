"""Bounded competition between 2D soma fronts on the image grid."""

from heapq import heappop, heappush

import numpy as np
from scipy.ndimage import gaussian_filter1d


GROWTH_LIMIT = 1.10
RAY_COUNT = 64


def elliptical_radius(y, x, center, major, minor, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    dy, dx = y - center[0], x - center[1]
    return np.hypot((dx*cosine + dy*sine)/(major/2), (-dx*sine + dy*cosine)/(minor/2))


def grow_somas(allowed, centers, geometry):
    """Grow connected masks by elliptical radius, with first arrival owning a pixel."""
    height, width = allowed.shape
    seeds = np.rint(centers).astype(int)
    owners = np.full(allowed.shape, -1, dtype=np.int32)
    sy, sx = seeds.T
    if not allowed[sy, sx].all() or len(np.unique(sy*width+sx)) != len(seeds):
        raise ValueError('Для роста нужны разные пиксели центров внутри области размещения.')
    owners[sy, sx] = np.arange(len(seeds))

    windows = []
    queue = []
    for index, (point, (y, x), major, minor, angle) in enumerate(zip(
            centers, seeds, geometry['major_axis_px'], geometry['minor_axis_px'],
            np.radians(geometry['orientation_deg']))):
        radius = int(np.ceil(major*GROWTH_LIMIT/2)) + 1
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        local = np.ascontiguousarray(elliptical_radius(yy, xx, point, major, minor, angle))
        local[(local > GROWTH_LIMIT) | ~allowed[y0:y1, x0:x1]] = np.inf
        local[y-y0, x-x0] = np.inf
        windows.append((y0, y1, x0, x1, local))
        heappush(queue, (0., index, int(y), int(x)))

    while queue:
        arrival, index, y, x = heappop(queue)
        owner = owners[y, x]
        if owner >= 0 and owner != index:
            continue
        owners[y, x] = index
        y0, y1, x0, x1, local = windows[index]
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if y0 <= yy < y1 and x0 <= xx < x1 and owners[yy, xx] < 0:
                radius = local[yy-y0, xx-x0]
                if radius <= GROWTH_LIMIT:
                    local[yy-y0, xx-x0] = np.inf
                    heappush(queue, (max(arrival, radius), index, yy, xx))
    return owners


def radial_reaches(owners, centers, geometry):
    """First exit from each mask along 64 rays in local ellipse coordinates."""
    reaches = []
    for start in range(0, len(centers), 128):
        batch = slice(start, start+128)
        reaches.append(_radial_reaches(owners, centers[batch],
                                       {key: values[batch] for key, values in geometry.items()}, start))
    return np.concatenate(reaches)


def _radial_reaches(owners, centers, geometry, start):
    phase = np.arange(RAY_COUNT)*2*np.pi/RAY_COUNT
    distances = np.linspace(0, GROWTH_LIMIT+.25, 109)
    angles = np.radians(geometry['orientation_deg'])[:, None]
    u = geometry['major_axis_px'][:, None]/2*np.cos(phase)
    v = geometry['minor_axis_px'][:, None]/2*np.sin(phase)
    dy = u*np.sin(angles)+v*np.cos(angles)
    dx = u*np.cos(angles)-v*np.sin(angles)
    y = np.rint(centers[:, 0, None, None] + dy[..., None]*distances).astype(int)
    x = np.rint(centers[:, 1, None, None] + dx[..., None]*distances).astype(int)
    height, width = owners.shape
    inside = ((y >= 0) & (y < height) & (x >= 0) & (x < width)
              & (owners[np.clip(y, 0, height-1), np.clip(x, 0, width-1)]
                 == (start+np.arange(len(centers)))[:, None, None]))
    first_exit = np.argmax(~inside, axis=-1)
    reaches = np.where(inside.all(axis=-1), distances[-1],
                       distances[first_exit]-(distances[1]-distances[0])/2)
    return np.minimum(reaches, GROWTH_LIMIT).astype(np.float32)


def light_reaches(reaches):
    """Smooth pixel-grid steps for rendering and keep heavily compressed arcs stable."""
    return np.maximum(.25, gaussian_filter1d(reaches, 1.5, axis=-1, mode='wrap'))
