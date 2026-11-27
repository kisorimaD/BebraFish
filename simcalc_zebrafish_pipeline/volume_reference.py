"""Build local Plane18–22 priors and compare synthetic and measured stacks."""
import argparse
import itertools
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .generate_volume import write_ome
from .volume import Acquisition, Optics, Profile, TissueMaps

ACQUISITION = Acquisition(5)
PLANE_NUMBERS = range(18, 23)
SHAPE_YX = (1376, 320)
CALIBRATION_STOP_Y = SHAPE_YX[0] // 2


def _temporal_median(path):
    with tifffile.TiffFile(path) as tif:
        pages = list(tif.pages)
        if not pages or any(p.shape != SHAPE_YX or p.dtype != np.uint16 for p in pages):
            raise ValueError(f'{path}: expected frames of shape {SHAPE_YX} and dtype uint16')
        offsets = np.array([p.dataoffsets[0] for p in pages])
        spans = np.diff(offsets)
        frame_stride = int(spans[0]) if len(spans) else 2 * np.prod(SHAPE_YX)
        dtype = np.dtype(tif.byteorder + 'u2')
        # This export has a small IFD between contiguous frames. Map its
        # pixels as a strided view, with a decoder fallback for other layouts.
        if all(p.is_memmappable for p in pages) and np.all(spans == frame_stride) and frame_stride > 0:
            source = np.memmap(path, mode='r', dtype=np.uint8)
            frames = np.ndarray((len(pages), *SHAPE_YX), dtype=dtype, buffer=source,
                                offset=int(offsets[0]), strides=(frame_stride, 2 * SHAPE_YX[1], 2))
        else:
            frames = tif.asarray(key=slice(None), out='memmap').reshape(-1, *SHAPE_YX)
        median = np.empty(SHAPE_YX, dtype=np.float32)
        for start in range(0, SHAPE_YX[0], 32):
            band = np.ascontiguousarray(frames[:, start:start + 32].transpose(1, 2, 0))
            median[start:start + 32] = np.median(band, axis=-1, overwrite_input=True)
        return median, {'path': str(path), 'frames': len(pages), 'shape_yx': list(SHAPE_YX),
                        'dtype': 'uint16', 'depth_source': 'relative acquisition grid'}


def _tissue_priors(stack):
    envelopes = np.asarray([gaussian_filter(plane, 12) for plane in stack])
    low, high = np.percentile(envelopes, [10, 60])
    span = max(high - low, 1)
    tissue = np.clip((envelopes - low) / (.45 * span), 0, 1)
    # Exclude dark areas and use local fine-scale energy as a density proxy.
    texture = np.asarray([plane - gaussian_filter(plane, 4) for plane in stack])
    energy = np.sqrt(np.asarray([gaussian_filter(plane**2, 13) for plane in texture]))
    in_tissue = tissue >= .8
    scale = np.percentile(energy[in_tissue], 80) if in_tissue.any() else 1.
    density = np.clip(energy / max(scale, 1), .15, 1) * tissue
    return tuple(a.astype(np.float32) for a in (tissue, density, envelopes))


def _registration(stack):
    band = [gaussian_filter(p, 10) - gaussian_filter(p, 50) for p in stack]
    shifts = []
    radius = 16
    for before, after in itertools.pairwise(band):
        template = before[128:-128, 64:-64].astype(np.float64)
        template -= template.mean()
        search = after[128-radius:-128+radius, 64-radius:-64+radius].astype(np.float64)
        ones = np.ones_like(template)
        sums = fftconvolve(search, ones, mode='valid')
        squares = fftconvolve(search**2, ones, mode='valid')
        cross = fftconvolve(search, template[::-1, ::-1], mode='valid')
        denom = np.sqrt(np.maximum(squares - sums**2 / template.size, 1e-12) * np.sum(template**2))
        correlation = cross / np.maximum(denom, 1e-12)
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
        shifts.append({'dy_px': int(y - radius), 'dx_px': int(x - radius), 'r': float(correlation[y, x])})
    return shifts


def _diameters(annotation):
    diameters = []
    for soma in annotation['annotations']:
        if soma['truncated'] or soma['confidence'] != 'confident':
            continue
        x, y = np.asarray(soma['points_xy']).T
        area = .5 * abs(x @ np.roll(y, 1) - y @ np.roll(x, 1))
        diameters.append(2 * np.sqrt(area / np.pi))
    return np.asarray(diameters)


def build_reference(data, annotation_path, output, pixel_yx_um, scale_source):
    if not np.isfinite(pixel_yx_um).all() or min(pixel_yx_um) <= 0:
        raise ValueError('TIFF XY pitches must be finite and positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    annotation_text = Path(annotation_path).read_text()
    annotation = json.loads(annotation_text)
    if annotation['dataset']['shape_yx'] != list(SHAPE_YX):
        raise ValueError('Annotation grid differs from native TIFF grid')
    files = {}
    for path in Path(data).glob('Plane_Aligned_*'):
        if path.suffix.lower() not in ('.tif', '.tiff') or not path.stem.rsplit('_', 1)[-1].isdigit():
            continue
        number = int(path.stem.rsplit('_', 1)[-1])
        if number in PLANE_NUMBERS:
            if number in files:
                raise ValueError(f'Duplicate plane {number}: {files[number]} and {path}')
            files[number] = path
    if missing := set(PLANE_NUMBERS) - files.keys():
        raise ValueError(f'Missing TIFF planes: {sorted(missing)}')
    planes, inventory = [], []
    depths = ACQUISITION.depths()
    for number, depth in zip(PLANE_NUMBERS, depths):
        median, row = _temporal_median(files[number])
        row.update(plane=number, z_um=float(depth))
        inventory.append(row)
        planes.append(median)
        print(f'Plane {number}: exact median of {row["frames"]} frames', flush=True)
    stack = np.asarray(planes)
    support, density, background = _tissue_priors(stack)
    digest = sha256(np.rint(stack[2]).astype('<u2').tobytes()).hexdigest()
    metadata = {
        'format_version': 1, 'files': inventory, 'coordinate_system': annotation['coordinate_system'],
        'pixel_yx_um': list(pixel_yx_um), 'xy_scale': scale_source,
        'z_reference': 'plane 18 = 0 µm, increasing file number follows positive relative Z',
        'registration_yx_next_relative_to_previous': _registration(stack),
        'maps': 'image-derived priors: envelope, dark-region exclusion and local fine-scale energy',
        'annotation_source': str(annotation_path), 'annotation_matches_rounded_plane20': (
            annotation['dataset']['frame_sha256'] == digest),
        'computed_rounded_plane20_sha256': digest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'reference'
        pending.mkdir()
        np.savez_compressed(
            pending / 'reference.npz', median_adu=stack, support=support, density=density,
            background_adu=background, z_positions_um=depths, annotation_json=np.array(annotation_text),
            metadata_json=np.array(json.dumps(metadata)),
        )
        write_ome(pending / 'median.ome.tif', stack, ACQUISITION, pixel_yx_um)
        (pending / 'reference.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        pending.rename(output)


def read_reference_profile(path):
    with np.load(path, allow_pickle=False) as reference:
        meta = json.loads(str(reference['metadata_json']))
        annotation = json.loads(str(reference['annotation_json']))
        maps = TissueMaps(reference['z_positions_um'], reference['support'], reference['density'],
                          reference['background_adu'])
        fit = reference['median_adu'][:, :CALIBRATION_STOP_Y]
    tissue = maps.support[:, :CALIBRATION_STOP_Y] >= .8
    band = gaussian_filter(fit, (0, 3, 3)) - gaussian_filter(fit, (0, 7, 7))
    texture_rms = .45 * float(band[tissue].std()) if tissue.any() else 0.
    diameters = _diameters(annotation)
    dy, dx = meta['pixel_yx_um']
    # For uniformly sampled offsets across an ellipsoid, the median section
    # diameter is sqrt(3)/2 of its equatorial diameter before censoring.
    diameter_xy = np.median(diameters) * np.sqrt(dy * dx) / np.sqrt(.75)
    profile = replace(Profile(), name='real18-22', shape_yx=SHAPE_YX, pixel_yx_um=(dy, dx),
                      depth_range_um=(0., 40.), acquisition=ACQUISITION,
                      depth_margin_um=27.,
                      density_per_um3=.0032, min_center_distance_um=2.45,
                      diameter_xy_um=float(.86 * diameter_xy), diameter_z_um=6.4, amplitude_adu=38.,
                      median_style='structured', emission_model='projected',
                      texture_band_rms_adu=texture_rms,
                      background_floor_adu=float(np.percentile(fit, 2)),
                      optics=Optics(sigma_zyx_um=(1.65, .6 * dy, .6 * dx)))
    meta.update(anatomy='local TIFF priors; held at end planes for optical margin',
                morphology='XY prior from Plane20 polygons scaled by 0.86 for smaller somas; axial size is modeled',
                photometry='shared Plane20 baseline and structured compositors; '
                           'texture amplitude estimated on the upper half of the TIFF grid; '
                           'texture darkening capped at 10% of the local envelope above the background floor',
                density='denser placement for Plane20 appearance; absolute density remains a model value',
                psf='Plane20 0.6 px base blur and spatially varying lateral blur; assumed axial PSF')
    return profile, maps, meta


def _quantiles(values):
    return np.percentile(values, [10, 50, 90]).tolist() if len(values) else None


def _intensity_stats(stack, support, rows):
    image = stack[:, rows]
    tissue = support[:, rows] >= .8
    if not tissue.any():
        return None
    return {
        'p5_p50_p95_adu': np.percentile(image[tissue], [5, 50, 95]).tolist(),
        'band_rms_adu': {
            f'{a}-{b}_px': float((gaussian_filter(image, (0, a, a))
                                - gaussian_filter(image, (0, b, b)))[tissue].std())
            for a, b in ((0, 1), (1, 3), (3, 7), (7, 12))
        },
    }


def _correlations_for_depths(stack, support, z_positions_um):
    band = np.asarray([p - gaussian_filter(p, 4) for p in stack])
    result = {}
    for i, z in enumerate(z_positions_um):
        for j in range(i + 1, len(stack)):
            mask = (support[i] >= .8) & (support[j] >= .8)
            a, b = band[i][mask], band[j][mask]
            if len(a) and a.std() > 0 and b.std() > 0:
                result.setdefault(float(z_positions_um[j] - z), []).append(float(np.corrcoef(a, b)[0, 1]))
    return {distance: {'mean': float(np.mean(values)), 'pair_r': values} for distance, values in result.items()}


def _compare_metrics(reference, synthetic, gt, visibility):
    shape = tuple(gt['shape_zyx'])
    if shape != reference['median_adu'].shape or synthetic.shape != shape:
        raise ValueError('Compare requires the same native TIFF grid in both stacks')
    depths = gt['z_positions_um']
    if not np.array_equal(depths, reference['z_positions_um']):
        raise ValueError('Compare requires identical physical depths in both stacks')
    metadata = json.loads(str(reference['metadata_json']))
    dy, dx = metadata['pixel_yx_um']
    if [dy, dx] != json.loads(str(gt['metadata_json']))['profile']['pixel_yx_um']:
        raise ValueError('Compare requires identical XY pixel sizes')
    annotated = _diameters(json.loads(str(reference['annotation_json'])))
    support = reference['support']
    area_mm2 = (support >= .8).sum(axis=(1, 2)) * dy * dx / 1e6
    visible = gt['cell_peak_adu'] >= visibility
    mask_counts = np.zeros(visible.shape, dtype=np.int64)
    voxels = gt['instance_voxel_index']
    for i, (start, stop) in enumerate(zip(gt['instance_offsets'][:-1], gt['instance_offsets'][1:])):
        mask_counts[i] = np.bincount(voxels[start:stop] // (shape[1] * shape[2]), minlength=shape[0])
    middle = list(depths).index(20.)
    diameters = 2 * np.sqrt(mask_counts[:, middle][mask_counts[:, middle] > 0] / np.pi)
    count = np.count_nonzero(visible, axis=1)
    observed = count[count > 0]
    return {
        'depth_um': depths.tolist(), 'diameter_p10_p50_p90_px': {
            'real_plane20': _quantiles(annotated), 'synthetic_plane20': _quantiles(diameters)},
        'real_untruncated_plane20_annotations': len(annotated),
        'real_visible_density_per_mm2': None, 'real_planes_per_soma': None,
        'real_tracking_status': 'Plane20 polygons are not an exhaustive count or matched multi-plane tracks',
        'synthetic_visibility_threshold_adu': visibility,
        'synthetic_visibility_basis': 'isolated soma contribution before median layer compositing',
        'synthetic_visible_per_plane': visible.sum(axis=0).tolist(),
        'synthetic_visible_per_mm2': [float(n / a) if a else None for n, a in zip(visible.sum(axis=0), area_mm2)],
        'synthetic_visible_planes_histogram': np.bincount(observed, minlength=len(depths) + 1)[1:].tolist(),
        'synthetic_single_plane_fraction': float(np.mean(observed == 1)) if len(observed) else None,
        'synthetic_visible_without_section': int(np.count_nonzero(visible & (mask_counts == 0))),
        'tissue_highpass_correlation': {
            'real': _correlations_for_depths(reference['median_adu'], support, depths),
            'synthetic': _correlations_for_depths(synthetic, support, depths)},
        'annotation_matches_rounded_plane20': metadata['annotation_matches_rounded_plane20'],
        'intensity_in_tissue': {
            region: {
                'real': _intensity_stats(reference['median_adu'], support, rows),
                'synthetic': _intensity_stats(synthetic, support, rows),
            }
            for region, rows in (
                ('upper_half', slice(None, CALIBRATION_STOP_Y)),
                ('lower_half', slice(CALIBRATION_STOP_Y, None)),
            )
        },
    }


def _plot(path, stacks, pixel_yx_um, depths, origin, size=128, xz_row=None, limits=None):
    import matplotlib.pyplot as plt

    y, x = origin
    dy, dx = pixel_yx_um
    row = size // 2 if xz_row is None else xz_row
    labels = list(stacks)
    crops = [stack[:, y:y+size, x:x+size] for stack in stacks.values()]
    if any(a.shape != (len(depths), size, size) for a in crops) or not 0 <= row < size:
        raise ValueError('Crop or XZ row is outside the volume')
    lo, hi = np.percentile(crops, [2, 99]) if limits is None else limits
    n = len(depths)
    fig, axes = plt.subplots(len(labels), n + 1, squeeze=False, figsize=(2.1 * (n + 1), 2.45 * len(labels)),
                             layout='constrained')
    half_step = (depths[1] - depths[0]) / 2 if n > 1 else .5
    for label, crop, panels in zip(labels, crops, axes):
        for k, ax in enumerate(panels[:-1]):
            image = ax.imshow(crop[k], cmap='gray', vmin=lo, vmax=hi,
                              extent=(x*dx, (x+size)*dx, (y+size)*dy, y*dy))
            ax.axhline((y+row+.5)*dy, color='orange', lw=.6, alpha=.8)
            ax.set_title(f'{label} · Z {depths[k]:g} µm', fontsize=9)
            ax.set(xlabel='X (µm)', ylabel='Y (µm)' if k == 0 else '')
        ax = panels[-1]
        ax.imshow(crop[:, row], cmap='gray', vmin=lo, vmax=hi, interpolation='nearest',
                  extent=(x*dx, (x+size)*dx, depths[-1]+half_step, depths[0]-half_step))
        ax.set(title=f'XZ · TIFF row {y+row}', xlabel='X (µm)', ylabel='Z (µm)')
    fig.colorbar(image, ax=axes, label='ADU', shrink=.75)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def compare(reference_path, synthetic_path, output, visibility):
    if not np.isfinite(visibility) or visibility <= 0:
        raise ValueError('Visibility threshold must be finite and positive')
    with np.load(reference_path, allow_pickle=False) as source:
        reference = dict(source)
    with np.load(synthetic_path / 'instances.npz', allow_pickle=False) as source:
        gt = dict(source)
    synthetic = np.load(synthetic_path / 'clean_adu.npy', allow_pickle=False)
    metrics = _compare_metrics(reference, synthetic, gt, visibility)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(metrics, indent=2) + '\n')
    annotation = json.loads(str(reference['annotation_json']))
    pixel_yx_um = json.loads(str(reference['metadata_json']))['pixel_yx_um']
    limits = np.percentile(reference['median_adu'], [2, 99])
    centers = np.asarray([np.mean(s['points_xy'], axis=0) for s in annotation['annotations']])
    for index, group in enumerate(np.array_split(centers[centers[:, 1].argsort()], 2), 1):
        midpoint = np.median(group, axis=0)
        x, y = group[np.argmin(np.sum((group - midpoint)**2, axis=1))]
        y0 = int(np.clip(np.floor(y) - 64, 0, SHAPE_YX[0] - 128))
        x0 = int(np.clip(np.floor(x) - 64, 0, SHAPE_YX[1] - 128))
        _plot(output / f'comparison_{index}.png', {'Real': reference['median_adu'], 'Synthetic': synthetic},
              pixel_yx_um, gt['z_positions_um'], (y0, x0), xz_row=int(y) - y0, limits=limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('build')
    build.add_argument('--data', type=Path, default=Path('data'))
    build.add_argument('--annotations', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--dx-um', type=float)
    build.add_argument('--dy-um', type=float)
    build.add_argument('--scale-source', help='Provenance for an explicitly measured TIFF scale')
    cmp = actions.add_parser('compare')
    cmp.add_argument('--reference', type=Path, required=True)
    cmp.add_argument('--synthetic', type=Path, required=True)
    cmp.add_argument('--output', type=Path, required=True)
    cmp.add_argument('--visibility-adu', type=float, default=10.)
    args = parser.parse_args()
    try:
        if args.action == 'build':
            known = (args.dx_um is not None, args.dy_um is not None, bool(args.scale_source))
            if any(known) and not all(known):
                parser.error('Specify both pitches and --scale-source together')
            scale = ((args.dy_um, args.dx_um) if all(known) else (.576, .576))
            source = args.scale_source or ('Assumed retained acquisition pitch of 0.576 µm/px on both TIFF axes. '
                                           'Export transform is undocumented and TIFF has no physical scale tags.')
            build_reference(args.data, args.annotations, args.output, scale, source)
        else:
            compare(args.reference, args.synthetic, args.output, args.visibility_adu)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
