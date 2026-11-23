"""Center detection and independent full-mask targets on the same Plane20 crops."""
import numpy as np
from .data import CropSource

ROI_SIZE = 24


def roi_anchor(points):
    """Use the same float32 rounding as the GPU decoder, including near-half seeds."""
    return np.floor(np.asarray(points, np.float32)+np.float32(.5)).astype(np.int32)


def ellipse_bounds(scene, indices):
    a, b = scene['major_axis_px'][indices]/2, scene['minor_axis_px'][indices]/2
    theta = np.radians(scene['orientation_deg'][indices]);c, s = np.cos(theta), np.sin(theta)
    return np.column_stack((np.hypot(a*s, b*c), np.hypot(a*c, b*s)))


def check_roi_fit(centers, points, bounds, roi_size):
    displacement = centers-roi_anchor(points)
    if np.any(np.abs(displacement)+bounds >= roi_size/2):
        raise ValueError('A full GT ellipse does not fit its mask ROI; increase ROI_SIZE')


def roi_masks(scene, origin, indices, points, roi_size=ROI_SIZE):
    """Rasterize complete GT ellipses on integer-aligned ROIs around proposed seeds."""
    anchor = roi_anchor(points)
    grid = np.arange(roi_size)-roi_size//2
    yy = anchor[:, 0, None, None]+grid[None, :, None]
    xx = anchor[:, 1, None, None]+grid[None, None, :]
    centers = scene['centers'][indices]-origin
    dy, dx = yy-centers[:, 0, None, None], xx-centers[:, 1, None, None]
    theta = np.radians(scene['orientation_deg'][indices])[:, None, None]
    c, s = np.cos(theta), np.sin(theta)
    a = scene['major_axis_px'][indices, None, None]/2
    b = scene['minor_axis_px'][indices, None, None]/2
    check_roi_fit(centers, points, ellipse_bounds(scene, indices), roi_size)
    return ((((dx*c+dy*s)/a)**2+((-dx*s+dy*c)/b)**2) <= 1).astype(np.float32)[..., None]


def make_heatmap(centers, shape=(128, 128), sigma=.7):
    heat = np.zeros((*shape, 1), np.float32)
    rounded = roi_anchor(centers)
    for y, x in rounded[np.all((rounded >= 0) & (rounded < shape), axis=1)]:
        y0, y1 = max(y-3, 0), min(y+4, shape[0])
        x0, x1 = max(x-3, 0), min(x+4, shape[1])
        dy, dx = np.ogrid[y0-y:y1-y, x0-x:x1-x]
        heat[y0:y1, x0:x1, 0] = np.maximum(heat[y0:y1, x0:x1, 0],
            np.exp(-(dy*dy+dx*dx)/(2*sigma*sigma)))
    return heat


def center_targets(centers, size=128, halo=16, sigma=.7, heatmap=None):
    """Gaussian center heatmap; regression only at each actual center's nearest pixel."""
    heat = make_heatmap(centers, (size, size), sigma) if heatmap is None else heatmap
    rounded = roi_anchor(centers)
    valid = np.all((centers >= halo) & (centers < size-halo), axis=1)
    selected = np.flatnonzero(valid)
    pixels = rounded[selected]
    if len(pixels) != len(np.unique(pixels, axis=0)):
        raise ValueError('Two center targets occupy one pixel; this detector requires distinct center pixels')
    region = np.zeros_like(heat)
    region[halo:size-halo, halo:size-halo] = 1
    outside = rounded[(~valid) & np.all((rounded >= 0) & (rounded < size), axis=1)]
    region[outside[:, 0], outside[:, 1], 0] = 0
    return heat, region, selected, pixels.astype(np.int32), (centers[selected]-pixels).astype(np.float32)


class InstanceSource:
    def __init__(self, directory, split, seed, max_masks=64, jitter=.35, raw_images=False):
        self.source = CropSource(directory, split, seed, use_precomputed=False, keep_scenes=True)
        self.seed, self.max_masks, self.jitter = seed, max_masks, jitter
        self.raw_images = raw_images

    def sample(self, index):
        if self.raw_images:
            image, scene, origin, _ = self.source.raw_crop(index)
        else:
            image, scene, origin = self.source.crop(index)
        centers = scene['centers']-origin
        if '_center_heatmap' not in scene:
            ids = np.arange(len(scene['centers']))
            scene['_center_heatmap'] = make_heatmap(scene['centers'], scene['image'].shape)
            scene['_mask_patches'] = roi_masks(scene, (0, 0), ids, scene['centers'], roi_size=ROI_SIZE+2)[..., 0].astype(bool)
            scene['_mask_bounds'] = ellipse_bounds(scene, ids)
        y, x = origin
        heatmap = scene['_center_heatmap'][y:y+128, x:x+128]
        heat, valid, selected, pixels, offsets = center_targets(centers, heatmap=heatmap)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 197]))
        mask_indices = selected
        if len(mask_indices) > self.max_masks:
            mask_indices = rng.choice(mask_indices, self.max_masks, replace=False)
        points = centers[mask_indices].copy()
        if self.jitter:
            points += np.clip(rng.normal(0, self.jitter, points.shape), -.75, .75)
        points = points.astype(np.float32)
        check_roi_fit(centers[mask_indices], points, scene['_mask_bounds'][mask_indices], ROI_SIZE)
        # Jitter changes the integer ROI anchor by at most one pixel. Slice a
        # cached complete 26px patch instead of re-rasterizing 24px masks per step.
        shift = roi_anchor(points)-roi_anchor(centers[mask_indices])
        if np.any(np.abs(shift) > 1):
            raise ValueError('Seed jitter exceeded the cached mask margin')
        grid = np.arange(ROI_SIZE)
        yy = grid[None, :, None]+(1+shift[:, 0, None, None])
        xx = grid[None, None, :]+(1+shift[:, 1, None, None])
        masks = scene['_mask_patches'][mask_indices[:, None, None], yy, xx].astype(np.float32)[..., None]
        return dict(images=image[..., None], heatmap=heat, valid_region=valid,
            offset_pixels=pixels, offset_targets=offsets,
            points=points.astype(np.float32), masks=masks)


def batch_samples(rows, mask_slots=None):
    data = {k: np.stack([r[k] for r in rows]) for k in ('images', 'heatmap', 'valid_region')}
    data['offset_indices'] = np.concatenate([np.column_stack((np.full(len(r['offset_pixels']), i), r['offset_pixels']))
                                            for i, r in enumerate(rows)]).astype(np.int32)
    data['offset_targets'] = np.concatenate([r['offset_targets'] for r in rows])
    data['points'] = np.concatenate([r['points'] for r in rows])
    data['batch_indices'] = np.concatenate([np.full(len(r['points']), i, np.int32) for i, r in enumerate(rows)])
    data['masks'] = np.concatenate([r['masks'] for r in rows])
    data['mask_valid'] = np.ones(len(data['points']), np.float32)
    count = len(data['points'])
    slots = max(count, 1) if mask_slots is None else mask_slots
    if slots < count:
        raise ValueError('mask_slots cannot discard supervised instances')
    if slots > count:
        # Fixed ROI batch shape avoids cuDNN autotuning for a new instance count
        # on every step. Zero weights preserve the exact per-instance loss.
        data['points'] = np.concatenate((data['points'], np.full((slots-count, 2), 64., np.float32)))
        data['batch_indices'] = np.pad(data['batch_indices'], (0, slots-count))
        data['masks'] = np.pad(data['masks'], ((0, slots-count), (0, 0), (0, 0), (0, 0)))
        data['mask_valid'] = np.pad(data['mask_valid'], (0, slots-count))
    return data
