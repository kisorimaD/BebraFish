"""Segment Convert_Crop medians and measure fluorescence in the original TIFFs."""
import argparse
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import tifffile

HERE = Path(__file__).resolve().parent
PER_PLANE = HERE / 'per_plane'
ROOT = HERE.parents[1]
DATASET = ROOT / '20259815_Soma_GCaMP8f_Convert_Crop' / 'original'
MODEL_CODE = Path('/raid/kisor/projects/zb/plane20_stardist/median_mix_20261120')
CHECKPOINT = ROOT / 'analysis/fish7_f_s_plane37_39/models/pilot'
PLANES = tuple(range(15, 25))
SHAPE = (1000, 1376, 320)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def segment(movie, destination, model_code=MODEL_CODE, checkpoint=CHECKPOINT):
    sys.path.insert(0, str(model_code))
    from training.plane20.center_inference import predict

    with tempfile.TemporaryDirectory(dir=destination) as directory:
        temporary = Path(directory)
        median = np.empty(movie.shape[1:], dtype=np.float32)
        for row in range(0, len(median), 64):
            tile = np.ascontiguousarray(movie[:, row:row + 64].transpose(1, 2, 0))
            median[row:row + 64] = np.median(tile, axis=-1, overwrite_input=True)
        image = temporary / 'median.tiff'
        tifffile.imwrite(image, median)
        predict(Namespace(run=checkpoint, image=image, output=temporary / 'masks',
                          threshold=.3, mask_threshold=.5, frame=0))
        image.replace(destination / image.name)
        (temporary / 'masks').replace(destination / 'masks')
        report_path = destination / 'masks/report.json'
        report = json.loads(report_path.read_text())
        report.update(image=str(destination / image.name), output=str(destination / 'masks'))
        write_json(report_path, report)


def extract(movie, destination):
    with np.load(destination / 'masks/instances.npz') as instances:
        centers = instances['centers']
        scores = instances['scores']
        offsets = instances['instance_offsets'].astype(np.int64)
        pixels = instances['instance_pixel_index']
        shape = tuple(instances['shape_yx'])
    area = np.diff(offsets)
    if shape != movie.shape[1:] or len(centers) == 0 or np.any(area <= 0):
        raise ValueError('Empty masks or incorrect mask coordinates')
    height, width = shape
    pixel_count = height * width
    coverage = np.bincount(pixels, minlength=pixel_count)
    weights = np.repeat(1 / area, area).astype(np.float32)
    mask_matrix = csr_matrix((weights, pixels, offsets), shape=(len(centers), pixel_count))
    np.testing.assert_allclose(np.asarray(mask_matrix.sum(axis=1)), 1, atol=1e-6)
    overlap = np.asarray(mask_matrix @ (coverage > 1).astype(np.float32))

    owner = np.repeat(np.arange(len(centers)), area)
    exclusive = coverage[pixels] == 1
    exclusive_area = np.bincount(owner[exclusive], minlength=len(centers))
    exclusive_matrix = csr_matrix(
        (1.0 / exclusive_area[owner[exclusive]], (owner[exclusive], pixels[exclusive])),
        shape=(len(centers), pixel_count), dtype=np.float32)
    use_exclusive = exclusive_area >= 12

    # Local background annulus. Exclude pixels assigned to any predicted soma.
    dy, dx = np.mgrid[-10:11, -10:11]
    ring = (dy**2 + dx**2 >= 7**2) & (dy**2 + dx**2 <= 10**2)
    integer_centers = np.rint(centers).astype(int)
    y = integer_centers[:, 0, None] + dy[ring]
    x = integer_centers[:, 1, None] + dx[ring]
    inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    indices = np.where(inside, y * width + x, 0)
    valid = inside & (coverage[indices] == 0)
    background_pixels = valid.sum(axis=1)
    use_background = background_pixels >= 12
    rows = np.flatnonzero(use_background)
    # Group rings by size to take exact medians without padding or NaNs.
    groups = []
    for count in np.unique(background_pixels[rows]):
        group = rows[background_pixels[rows] == count]
        group_indices = indices[group][valid[group]].reshape(len(group), count)
        groups.append((group, group_indices))

    raw = np.empty((len(movie), len(centers)), dtype=np.float32)
    exclusive_raw = np.empty_like(raw)
    background = np.full_like(raw, np.nan)
    for frame, image in enumerate(movie):
        flat = image.ravel()
        float_flat = flat.astype(np.float32)
        raw[frame] = mask_matrix @ float_flat
        exclusive_raw[frame] = exclusive_matrix @ float_flat
        for group, indices_in_ring in groups:
            background[frame, group] = np.median(flat[indices_in_ring], axis=1)
    check_frame = len(movie) // 2
    for index in [0, len(centers) // 2, len(centers) - 1]:
        direct = movie[check_frame].ravel()[pixels[offsets[index]:offsets[index + 1]]].mean()
        np.testing.assert_allclose(raw[check_frame, index], direct, rtol=2e-6, atol=1e-4)
    if use_exclusive.any():
        index = np.flatnonzero(use_exclusive)[0]
        direct = movie[check_frame].ravel()[pixels[(owner == index) & exclusive]].mean()
        np.testing.assert_allclose(exclusive_raw[check_frame, index], direct, rtol=2e-6, atol=1e-4)
    if groups:
        group, ring_indices = groups[-1]
        np.testing.assert_equal(background[check_frame, group[0]],
                                np.median(movie[check_frame].ravel()[ring_indices[0]]))

    exclusive_raw[:, exclusive_area == 0] = np.nan
    corrected = exclusive_raw - background
    baseline = np.full(len(centers), np.nan, dtype=np.float32)
    measure = use_background & use_exclusive
    baseline[measure] = np.percentile(corrected[:, measure], 20, axis=0)
    # Same 15 ADU baseline floor as the earlier Plane20 ROI extraction.
    suitable = measure & (baseline >= 15)
    dff = np.full_like(raw, np.nan)
    dff[:, suitable] = corrected[:, suitable] / baseline[suitable] - 1
    assert np.isfinite(raw).all() and np.isfinite(dff[:, suitable]).all()
    temp = destination / 'traces.tmp.npz'
    np.savez_compressed(temp, centers_yx=centers, model_score=scores, mask_area_px=area,
                        mask_overlap_fraction=overlap, background_pixels=background_pixels,
                        exclusive_area_px=exclusive_area, full_mask_F_adu=raw,
                        F_adu=exclusive_raw, background_adu=background, corrected_F_adu=corrected,
                        F0_adu=baseline, dff=dff, valid_dff=suitable)
    temp.replace(destination / 'traces.npz')
    return {'masks': len(centers), 'usable_dff': int(suitable.sum()),
            'insufficient_background': int((~use_background).sum()),
            'insufficient_exclusive_pixels': int((use_background & ~use_exclusive).sum()),
            'below_baseline_floor': int((measure & ~suitable).sum()),
            'mask_area_median_px': float(np.median(area)),
            'usable_dff_p99': float(np.percentile(dff[:, suitable], 99)) if suitable.any() else None}


def combine(summaries):
    parts = {key: [] for key in ['dff', 'F_adu', 'background_adu', 'F0_adu', 'centers_yx',
                                 'model_score', 'mask_area_px', 'mask_overlap_fraction',
                                 'background_pixels', 'exclusive_area_px', 'plane', 'mask_id']}
    for row in summaries:
        plane = row['plane']
        with np.load(PER_PLANE / f'plane_{plane}' / 'traces.npz') as data:
            keep = data['valid_dff']
            for key in ('dff', 'F_adu', 'background_adu'):
                parts[key].append(data[key][:, keep])
            for key in ('F0_adu', 'centers_yx', 'model_score', 'mask_area_px',
                        'mask_overlap_fraction', 'background_pixels', 'exclusive_area_px'):
                parts[key].append(data[key][keep])
            parts['plane'].append(np.full(keep.sum(), plane))
            parts['mask_id'].append(np.flatnonzero(keep))
    combined = {key: np.concatenate(items, axis=1 if key in
                ('dff', 'F_adu', 'background_adu') else 0) for key, items in parts.items()}
    n = sum(row['usable_dff'] for row in summaries)
    assert combined['dff'].shape == (1000, n)
    for key in ('dff', 'F_adu', 'background_adu', 'F0_adu'):
        assert np.isfinite(combined[key]).all(), key
    np.testing.assert_allclose(
        (combined['F_adu'] - combined['background_adu']) / combined['F0_adu'] - 1,
        combined['dff'])
    temp = HERE / 'trace_bank.tmp.npz'
    np.savez_compressed(temp, **combined)
    temp.replace(HERE / 'trace_bank.npz')
    write_json(HERE / 'summary.json', {'frames': 1000, 'fps': None, 'total_usable_dff': n,
                                      'trace_layout': 'frame × ROI',
                                      'F_adu': 'Mean of exclusive mask pixels in original TIFF',
                                      'background_adu': 'Median of unmasked pixels 7–10 px from center',
                                      'F0_adu': '20th percentile of F_adu minus background_adu',
                                      'dff': '(F_adu - background_adu) / F0_adu - 1',
                                      'minimum_exclusive_pixels': 12, 'minimum_background_pixels': 12,
                                      'minimum_F0_adu': 15, 'planes': summaries})
    print(f'Trace bank verified: 1000 frames × {n} ROI', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planes', type=int, nargs='+', choices=PLANES, default=[20])
    parser.add_argument('--combine', action='store_true', help='Combine all ten completed planes')
    parser.add_argument('--dataset-manifest', type=Path)
    args = parser.parse_args()
    if args.dataset_manifest is not None:
        if args.combine:
            parser.error('--combine is reserved for the original Convert_Crop extraction.')
        manifest = json.loads(args.dataset_manifest.read_text())
        sources = {int(k): Path(v) for k, v in manifest['sources'].items()}
        output = args.dataset_manifest.parent / 'per_plane'
        model_code, checkpoint = Path(manifest['model_code']), Path(manifest['checkpoint'])
        shape = tuple(manifest['shape_tyx'])
    else:
        sources = {plane: DATASET / f'Plane_Aligned_{plane}.tiff' for plane in args.planes}
        output = PER_PLANE
        model_code, checkpoint, shape = MODEL_CODE, CHECKPOINT, SHAPE
    for plane, source in sources.items():
        destination = output / f'plane_{plane}'
        destination.mkdir(parents=True, exist_ok=True)
        if args.dataset_manifest and (destination / 'summary.json').is_file() and (destination / 'traces.npz').is_file():
            print(f'Plane {plane}: reusing extracted traces', flush=True)
            continue
        with tifffile.TiffFile(source) as tif:
            movie = tif.asarray(key=slice(None))
        if movie.shape != shape or movie.dtype != np.uint16:
            raise ValueError(f'Unexpected TIFF geometry: {movie.shape}, {movie.dtype}')
        print(f'Plane {plane}: verified {movie.shape} {movie.dtype}', flush=True)
        if not (destination / 'masks/instances.npz').exists():
            segment(movie, destination, model_code, checkpoint)
        result = {'plane': plane, 'source': str(source), **extract(movie, destination)}
        write_json(destination / 'summary.json', result)
        print(result, flush=True)
    if args.combine:
        summaries = [json.loads((PER_PLANE / f'plane_{plane}' / 'summary.json').read_text())
                     for plane in PLANES]
        combine(summaries)


if __name__ == '__main__':
    main()
