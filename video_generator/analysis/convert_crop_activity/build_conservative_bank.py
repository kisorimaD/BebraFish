"""Keep ROI unless both its brightness and temporal activity resemble empty background."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / 'foreground_conservative'
SOURCE = ROOT / '20259815_Soma_GCaMP8f_Convert_Crop' / 'original'
DY, DX = np.mgrid[-6:7, -6:7]
DISK = DY**2 + DX**2 <= 36
WIDE_Y, WIDE_X = np.mgrid[-25:26, -25:26]
WIDE_RING = (WIDE_Y**2 + WIDE_X**2 >= 10**2) & (WIDE_Y**2 + WIDE_X**2 <= 25**2)


def temporal_evidence(raw):
    band = gaussian_filter1d(raw, 1, axis=0) - gaussian_filter1d(raw, 12, axis=0)
    return np.quantile(np.abs(band), .95, axis=0) / np.sqrt(np.maximum(np.median(raw, axis=0), 1))


def analyze_plane(plane, rng, directory, source, output):
    image = tifffile.imread(directory / 'median.tiff')
    height, width = image.shape
    with np.load(directory / 'masks/instances.npz') as masks:
        pixels = masks['instance_pixel_index']
        offsets = masks['instance_offsets'].astype(int)
    with np.load(directory / 'traces.npz') as data:
        raw_full = data['full_mask_F_adu']
        raw_exclusive = data['F_adu']
        background = data['background_adu']
        centers = data['centers_yx']
        exclusive_area = data['exclusive_area_px']
        old = data['valid_dff']
    coverage = np.bincount(pixels, minlength=image.size).reshape(image.shape)
    # The reference consists of dark, spatially even patches outside masks.
    local_mean = uniform_filter(image, 13)
    roughness = np.sqrt(np.maximum(uniform_filter(image**2, 13) - local_mean**2, 0))
    free = uniform_filter((coverage > 0).astype(np.float32), 13) == 0
    free[:8] = free[-8:] = False
    free[:, :8] = free[:, -8:] = False
    if not np.any(free):
        raise ValueError(f'Plane {plane} has no unmasked 13×13 patches for background calibration.')
    candidates = np.argwhere(free & (local_mean <= np.quantile(local_mean[free], .45)) &
                            (roughness <= np.quantile(roughness[free], .60)))
    if len(candidates) < 30:
        raise ValueError(f'Plane {plane} has only {len(candidates)} usable background controls.')
    control_centers = candidates[rng.choice(len(candidates), min(800, len(candidates)), replace=False)]
    control_pixels = ((control_centers[:, 0, None] + DY[DISK]) * width +
                      control_centers[:, 1, None] + DX[DISK])
    control_static = np.quantile(image.ravel()[control_pixels], .85, axis=1)
    control_raw = np.empty((len(raw_full), len(control_centers)), dtype=np.float32)
    fallback_rows, fallback_pixels = [], []
    for roi in np.flatnonzero(~np.isfinite(background).all(axis=0)):
        y, x = np.rint(centers[roi]).astype(int)
        yy, xx = y + WIDE_Y[WIDE_RING], x + WIDE_X[WIDE_RING]
        in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy, xx = yy[in_bounds], xx[in_bounds]
        available = coverage[yy, xx] == 0
        yy, xx = yy[available], xx[available]
        if len(yy) >= 12:
            near = np.argsort((yy - y)**2 + (xx - x)**2)[:12]
            fallback_rows.append(roi)
            fallback_pixels.append(yy[near] * width + xx[near])
    fallback_pixels = np.asarray(fallback_pixels, dtype=int).reshape(-1, 12)
    with tifffile.TiffFile(source) as movie:
        for frame, page in enumerate(movie.pages):
            pixels_flat = page.asarray().ravel()
            control_raw[frame] = pixels_flat[control_pixels].mean(axis=1)
            background[frame, fallback_rows] = np.median(pixels_flat[fallback_pixels], axis=1)
    control_dynamic = temporal_evidence(control_raw)
    detected_static = np.array([np.quantile(image.ravel()[pixels[a:b]], .85)
                                for a, b in zip(offsets, offsets[1:])])
    detected_dynamic = temporal_evidence(raw_full)
    calibration = rng.permutation(len(control_static))
    split = int(.625 * len(calibration))
    train, test = calibration[:split], calibration[split:]
    limits = [float(np.quantile(feature[train], .98)) for feature in (control_static, control_dynamic)]
    # Keep any ROI with either evidence of a structure or variation above
    # the quiet reference. Ambiguous ROI remain eligible.
    keep = (detected_static > limits[0]) | (detected_dynamic > limits[1])
    control_kept = (control_static[test] > limits[0]) | (control_dynamic[test] > limits[1])

    # Prefer nonoverlapping pixels and local background if they were measured.
    # When either is missing, retain a measured trace and record the method.
    raw = np.where((exclusive_area >= 12)[None, :], raw_exclusive, raw_full)
    measured_background = np.isfinite(background).all(axis=0)
    wide_background = np.isin(np.arange(len(centers)), fallback_rows)
    corrected = raw - np.where(measured_background[None, :], background, 0)
    scale = np.median(raw, axis=0)
    keep &= scale > 0
    relative = (corrected - np.median(corrected, axis=0)) / np.maximum(scale, 1)
    assert np.isfinite(relative[:, keep]).all()
    np.savez_compressed(output / f'plane_{plane}_foreground.npz', accepted=keep,
                        static=detected_static, temporal=detected_dynamic, thresholds=limits)
    row = dict(plane=plane, detected=len(keep), accepted=int(keep.sum()),
               old=int(old.sum()), control_count=len(test), controls_passing=int(control_kept.sum()),
               full_mask=int(np.count_nonzero(keep & (exclusive_area < 12))),
               wider_background=int(np.count_nonzero(keep & wide_background)),
               without_local_background=int(np.count_nonzero(keep & ~measured_background)),
               accepted_below_old_floor=int(np.count_nonzero(keep & ~old)), limits=limits)
    print(row, flush=True)
    return row, dict(dff=relative[:, keep], plane=np.full(keep.sum(), plane),
                     mask_id=np.flatnonzero(keep), centers_yx=centers[keep],
                     full_mask=(exclusive_area < 12)[keep],
                     local_background=measured_background[keep],
                     wide_background=wide_background[keep])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path)
    args = parser.parse_args()
    if args.dataset_manifest:
        manifest = json.loads(args.dataset_manifest.read_text())
        sources = {int(k): Path(v) for k, v in manifest['sources'].items()}
        output = args.dataset_manifest.parent / 'foreground'
        per_plane = args.dataset_manifest.parent / 'per_plane'
    else:
        sources = {p: SOURCE / f'Plane_Aligned_{p}.tiff' for p in range(15, 25)}
        output = OUTPUT
        per_plane = HERE / 'per_plane'
    output.mkdir(exist_ok=True)
    rows, parts = zip(*(analyze_plane(plane, np.random.default_rng(6020 + plane),
                                    per_plane / f'plane_{plane}', source, output)
                        for plane, source in sources.items()))
    combined = {key: np.concatenate([part[key] for part in parts], axis=1 if key == 'dff' else 0)
                for key in parts[0]}
    assert np.isfinite(combined['dff']).all()
    np.savez_compressed(output / 'trace_bank.npz', **combined)
    report = dict(roi=len(combined['plane']), frames=combined['dff'].shape[0],
                  filter='Rejected only when both static mask brightness and temporal variation fall '
                         'within the range of separately sampled dark, spatially uniform patches outside masks.',
                  relative_signal='(F - background - median(F - background)) / median(F); '
                                  'background is zero when it could not be estimated locally.',
                  peak_prominence=max(.001, round(float(np.median(np.quantile(combined['dff'], .95, axis=0))), 3)),
                  planes=rows)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
