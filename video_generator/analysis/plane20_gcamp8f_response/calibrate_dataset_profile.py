"""Build a spatial profile from the new reference plane's median and measured ROI."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
import tifffile


def empirical_axes(pixels, offsets, indices, width):
    shapes = []
    for i in indices:
        y, x = np.divmod(pixels[offsets[i]:offsets[i + 1]], width)
        if len(x) < 12:
            continue
        covariance = np.cov(np.column_stack((x, y)), rowvar=False, bias=True)
        eigenvalues, directions = np.linalg.eigh(covariance)
        minor, major = 4 * np.sqrt(np.maximum(eigenvalues, 0))
        if minor < 2 or major > 20:
            continue
        angle = np.degrees(np.arctan2(directions[1, -1], directions[0, -1]))
        shapes.append([major, minor, (angle + 90) % 180 - 90])
    if len(shapes) < 30:
        raise ValueError(f'Only {len(shapes)} plausible soma masks for morphology calibration.')
    return np.asarray(shapes)


def calibrate_profile(reference_dir, bank_path, reference_plane, output, requested_cells=None):
    median = tifffile.imread(reference_dir / 'median.tiff').astype(np.float32)
    shape = median.shape
    with np.load(bank_path) as bank:
        reference = bank['plane'] == reference_plane
        ids = bank['mask_id'][reference]
        donors = int((~reference).sum())
    with np.load(reference_dir / 'traces.npz') as traces:
        centers = traces['centers_yx'][ids]
        raw_baseline = np.median(traces['full_mask_F_adu'][:, ids], axis=0)
    with np.load(reference_dir / 'masks/instances.npz') as masks:
        pixels = masks['instance_pixel_index']
        offsets = masks['instance_offsets']
    axes = empirical_axes(pixels, offsets, ids, shape[1])
    envelope = gaussian_filter(median, 12)
    background_level = float(np.quantile(envelope, .10))
    yy, xx = np.rint(centers).astype(int).T
    impulses = np.zeros(shape, dtype=np.float32)
    np.add.at(impulses, (yy, xx), 1)
    density = gaussian_filter(impulses, 25)
    # Bridge small holes in the optical outline; ROI density keeps dim cell
    # territories eligible even when a single brightness cutoff misses them.
    tissue = binary_fill_holes(binary_closing(
        (envelope > background_level + .07 * (np.quantile(envelope, .95) - background_level)) |
        (density > .035 * density.max()), iterations=4))
    support = tissue & (density > .035 * density.max())
    if support.sum() < 100 or not np.any(~tissue):
        raise ValueError('Could not identify both tissue and background in the calibrated reference plane.')
    relative_density = density / np.mean(density[support])

    # Estimate the approximate soma brightness above its local envelope.
    # This is a prior for the optical renderer; the final frame statistics
    # are fitted separately on the new recording.
    contrast = np.maximum(raw_baseline - envelope[yy, xx], 1)
    typical = float(np.median(contrast))
    lower, upper = np.quantile(contrast, [.15, .85])
    controls = np.zeros(shape, dtype=np.float32)
    np.add.at(controls, (yy, xx), np.clip(contrast, lower, upper))
    counts = gaussian_filter(impulses, 35)
    local = gaussian_filter(controls, 35)
    shrinkage = max(float(np.median(counts[support])) * .20, 1e-8)
    normalizer = (local + shrinkage * typical) / (counts + shrinkage)
    normalizer *= .19 * .78 / (.65 * .36)
    cells = min(len(ids), donors, requested_cells if requested_cells is not None else len(ids))
    if cells < 100:
        raise ValueError('Need at least 100 accepted ROI both on the reference plane and on the donor planes.')
    # The packing stage draws a denser set of candidates and retains those
    # whose sampled soma footprints fit in the reference support.
    packing_count = max(cells, min(cells * 4, int(support.sum() / 13)))
    config = dict(size_scale=1., aspect_scale=1., count=cells, packing_count=packing_count,
                  density_floor=background_level, width=.19, peak=.78, amplitude=.65)
    output = Path(output)
    np.savez_compressed(output, tissue=tissue, neuron_support=support, density=relative_density,
                        median_envelope=envelope, normalizer=normalizer,
                        settings_json=json.dumps({'selected': dict(sigma_px=.6, amplitude_cv=.35)}),
                        median_config=json.dumps(config),
                        morphology_json=json.dumps(dict(empirical_axes_px=axes.tolist())))
    return dict(reference_plane=reference_plane, reference_roi=len(ids), donor_roi=donors,
                morphology_masks=len(axes), requested_scene_cells=cells, candidate_cells=packing_count)
