"""Compare visible-soma candidates and manual outlines on the Plane20 TIFF grid."""
import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate, distance_transform_edt, gaussian_filter, maximum_filter

from analyze_somas import ellipse_iou, measure


SIZE = 64
PAD = 12
RADII = (4.5, 5.5, 6.5, 7.5)
THRESHOLDS = (6., 8., 12.)
GAP_BINS = (0, 2, 4, 6, 8, 12, 16, 24, 32, np.inf)
ANGLES = np.linspace(0., 2 * np.pi, 720, endpoint=False)


def choose_crops(real, tissue):
    """Two spatially separated examples per contrast stratum, based on real data only."""
    band = gaussian_filter(real, 1.5) - gaussian_filter(real, 5.)
    bright = real > np.percentile(real, 99)
    best = []
    occupied = []
    for part, (start, stop) in enumerate(((0, real.shape[0] // 2), (real.shape[0] // 2, real.shape[0])), 1):
        grid = []
        for y in range(start + PAD, stop - SIZE - PAD + 1, 16):
            for x in range(PAD, real.shape[1] - SIZE - PAD + 1, 16):
                patch = np.s_[y:y+SIZE, x:x+SIZE]
                if tissue[patch].mean() >= .96 and bright[patch].mean() <= .03:
                    grid.append((float(band[patch].std()), y, x))
        grid.sort()
        for stratum, quantile in (('high', .9), ('typical', .5)):
            target = round(quantile * (len(grid) - 1))
            ordered = sorted(enumerate(grid), key=lambda row: (abs(row[0] - target), row[0]))
            index, (score, y, x) = next((i, row) for i, row in ordered if all(
                abs(row[1] - yy) >= SIZE or abs(row[2] - xx) >= SIZE for yy, xx in occupied))
            occupied.append((y, x))
            best.append((f'{stratum}-{part}', y, x, score, index / (len(grid) - 1)))
    return best


def ring_proposals(image, tissue):
    """Local nuclei with a brighter, substantially complete shell."""
    yy, xx = np.mgrid[-8:9, -8:9]
    distance = np.hypot(xx, yy)
    sector = (np.floor((np.arctan2(yy, xx) + np.pi) * 12 / (2 * np.pi)).astype(int)) % 12
    candidates = []
    for radius in RADII:
        core = distance <= .35 * radius
        shell = (distance >= .58 * radius) & (distance <= .93 * radius)
        core_mean = correlate(image, core / core.sum(), mode='nearest')
        shell_mean = correlate(image, shell / shell.sum(), mode='nearest')
        score = shell_mean - core_mean
        y, x = np.nonzero((score >= min(THRESHOLDS)) & (score == maximum_filter(score, 5)) & tissue)
        if not len(y):
            continue
        visible = np.zeros(len(y), dtype=int)
        for k in range(12):
            mask = shell & (sector == k)
            values = correlate(image, mask / mask.sum(), mode='nearest')[y, x]
            visible += values >= core_mean[y, x] + 2.
        for yi, xi, count in zip(y, x, visible):
            if count >= 8:
                candidates.append((float(score[yi, xi]), int(yi), int(xi), radius))
    kept = []
    for score, y, x, radius in sorted(candidates, reverse=True):
        if all(np.hypot(y - v, x - u) >= .65 * (radius + r) for _, v, u, r in kept):
            kept.append((score, y, x, radius))
    return kept


def crop_metrics(proposals, tissue, threshold):
    yy, xx = np.ogrid[:tissue.shape[0], :tissue.shape[1]]
    covered = np.zeros_like(tissue)
    within = 0
    for strength, y, x, radius in proposals:
        if strength >= threshold:
            covered |= (yy - y)**2 + (xx - x)**2 <= radius**2
            within += PAD <= y < PAD + SIZE and PAD <= x < PAD + SIZE
    region = np.s_[PAD:PAD+SIZE, PAD:PAD+SIZE]
    roi = tissue[region]
    empty = roi & ~covered[region]
    # Diameter of a free disc centered on each empty tissue pixel (area-weighted).
    free_diameter = 2 * distance_transform_edt(np.pad(tissue & ~covered, 1))[1:-1, 1:-1][region]
    gap_values = free_diameter[empty]
    p50, p90 = np.percentile(gap_values, [50, 90]).tolist() if empty.any() else [0., 0.]
    hist, _ = np.histogram(gap_values, GAP_BINS)
    row = dict(candidates=within, provisional_coverage_pct=float(100 * covered[region][roi].mean()),
               gap_p50_px=p50, gap_p90_px=p90)
    row.update({f'gap_pixels_{lo:g}_to_{hi:g}_px': int(n)
                for lo, hi, n in zip(GAP_BINS[:-1], GAP_BINS[1:], hist)})
    return row


def polygon_radii(points, center, theta):
    """First intersection of rays from the centroid with the polygon boundary."""
    start = points - center
    side = np.roll(start, -1, axis=0) - start
    direction = np.array([np.cos(theta), np.sin(theta)]).T
    denom = direction[:, 0, None] * side[:, 1] - direction[:, 1, None] * side[:, 0]
    numerator = start[:, 0] * side[:, 1] - start[:, 1] * side[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        radius = numerator / denom
        segment = (start[:, 0] * direction[:, 1, None] - start[:, 1] * direction[:, 0, None]) / denom
    hits = np.where((radius >= 0.) & (segment >= 0.) & (segment <= 1.), radius, np.inf)
    result = hits.min(axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Could not measure polygon along every ray')
    return result


def ellipse_radii(shape, theta):
    phi = theta - shape['angle']
    a, b = shape['axes']
    return 1 / np.hypot(np.cos(phi) / a, np.sin(phi) / b)


def shape_analysis(annotation):
    shapes = {}
    for cell in annotation['annotations']:
        if cell['truncated'] or cell['confidence'] != 'confident':
            continue
        points = np.array(cell['points_xy'])
        m = measure(points)
        shapes[cell['number']] = dict(points=points, center=np.array([m['centroid_x'], m['centroid_y']]),
                                      axes=np.array([m['major_axis_px'], m['minor_axis_px']]) / 2,
                                      angle=np.deg2rad(m['orientation_deg']), iou=ellipse_iou(points, m),
                                      diameter=m['equivalent_diameter_px'])
    cutoff = float(np.median([obj['diameter'] for obj in shapes.values()]) / 2)
    rows = []
    for key, shape in shapes.items():
        def edge_gap(other):
            vector = other['center'] - shape['center']
            theta = np.arctan2(vector[1], vector[0])
            return np.linalg.norm(vector) - ellipse_radii(shape, theta) - ellipse_radii(other, theta + np.pi)

        near = min((k for k in shapes if k != key), key=lambda k: edge_gap(shapes[k]))
        other = shapes[near]
        delta = other['center'] - shape['center']
        theta = np.arctan2(delta[1], delta[0])
        gap = edge_gap(other)
        measured = polygon_radii(shape['points'], shape['center'], ANGLES)
        residual = ellipse_radii(shape, ANGLES) - measured
        face = np.cos(ANGLES - theta)
        front = float(residual[face >= np.cos(np.deg2rad(35))].mean())
        back = float(residual[face <= -np.cos(np.deg2rad(35))].mean())
        polygon_gap = (np.linalg.norm(delta)
                       - polygon_radii(shape['points'], shape['center'], np.array([theta]))[0]
                       - polygon_radii(other['points'], other['center'], np.array([theta + np.pi]))[0])
        rows.append(dict(cell=key, neighbor=near, ellipse_gap_px=float(gap), polygon_gap_px=float(polygon_gap),
                         close=bool(gap <= cutoff), inward_facing_px=front, inward_opposite_px=back,
                         inward_difference_px=front-back, ellipse_iou=shape['iou']))
    return shapes, rows, cutoff


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_pair(ax, pair, shapes, rows, real, limits):
    pts = np.concatenate([shapes[key]['points'] for key in pair])
    cx, cy = (pts.min(axis=0) + pts.max(axis=0)) / 2
    size = int(np.ceil(max(np.ptp(pts, axis=0)) + 8))
    x, y = np.maximum(0, np.floor([cx - size/2, cy - size/2]).astype(int))
    ax.imshow(real[y:y+size, x:x+size], cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest',
              extent=(x, x+size, y+size, y))
    left, right = (shapes[k] for k in pair)
    vector = right['center'] - left['center']
    theta = np.arctan2(vector[1], vector[0])
    direction = vector / np.linalg.norm(vector)
    ends = []
    for key, obj, ang, sign in zip(pair, (left, right), (theta, theta + np.pi), (1, -1)):
        poly = np.vstack([obj['points'], obj['points'][0]])
        ax.plot(*poly.T, color='#42d9cc', lw=1.45)
        curve = obj['center'] + ellipse_radii(obj, ANGLES)[:, None] * np.stack(
            (np.cos(ANGLES), np.sin(ANGLES)), axis=1)
        ax.plot(*curve.T, '--', color='#ffd05d', lw=1.35)
        ax.annotate(f'#{key}', obj['center'], xytext=(0, 0), textcoords='offset points',
                    color='white', ha='center', va='center', fontsize=8)
        end = polygon_radii(obj['points'], obj['center'], np.array([ang]))[0]
        ends.append(obj['center'] + sign * end * direction)
    ax.plot(*np.asarray(ends).T, 'o-', ms=3, color='#fa6fa8', lw=1.8)
    actual = next(row for row in rows if (row['cell'], row['neighbor']) in (pair, pair[::-1]))
    gap = actual['polygon_gap_px']
    relation = 'зазор' if gap >= 0 else 'перекрытие'
    ax.set_title(f'#{pair[0]} и #{pair[1]} · {relation} {abs(gap):.1f} px', fontsize=10)
    ax.set_axis_off()


def run(reference, uniform, fitted, output):
    with np.load(reference, allow_pickle=False) as ref:
        real = ref['median_adu'][2]
        tissue = ref['support'][2] >= .8
        annotation = json.loads(str(ref['annotation_json']))
        meta = json.loads(str(ref['metadata_json']))
    synthetic = np.load(uniform / 'clean_adu.npy', mmap_mode='r')[2]
    images = {'real': real, 'uniform': synthetic}
    gain = None
    if fitted:
        images['uniform_fitted'] = np.load(fitted / 'clean_adu.npy', mmap_mode='r')[2]
        with np.load(uniform / 'instances.npz') as original, np.load(fitted / 'instances.npz') as control:
            ordinary_meta = json.loads(str(original['metadata_json']))
            control_meta = json.loads(str(control['metadata_json']))
            old_amp = ordinary_meta['profile'].pop('amplitude_adu')
            new_amp = control_meta['profile'].pop('amplitude_adu')
            if ordinary_meta['profile'] != control_meta['profile']:
                raise ValueError('The control profile must differ only in soma amplitude')
            gain = new_amp / old_amp
            for name in ('shape_zyx', 'z_positions_um', 'instance_offsets', 'instance_voxel_index'):
                np.testing.assert_array_equal(original[name], control[name])
            for name in original['cells'].dtype.names:
                if name != 'amplitude_adu':
                    np.testing.assert_array_equal(original['cells'][name], control['cells'][name])
            np.testing.assert_allclose(control['cells']['amplitude_adu'], original['cells']['amplitude_adu'] * gain)
    if any(image.shape != real.shape for image in images.values()):
        raise ValueError('Plane20 TIFF-grid shapes must agree')
    limits = np.percentile(real, [2, 99])
    crops = choose_crops(real, tissue)
    report, proposals = [], []
    main = {}
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output / 'manual_review.zip', 'w') as review:
        for name, y, x, score, rank in crops:
            padded = np.s_[y-PAD:y+SIZE+PAD, x-PAD:x+SIZE+PAD]
            patch_tissue = tissue[padded]
            for source, image in images.items():
                patch = image[padded]
                cells = ring_proposals(patch, patch_tissue)
                for strength, v, u, radius in cells:
                    proposals.append(dict(crop=name, source=source, x_tiff=u+x-PAD+.5, y_tiff=v+y-PAD+.5,
                                          outer_radius_px=radius, contrast_adu=strength,
                                          center_in_core=PAD <= v < PAD+SIZE and PAD <= u < PAD+SIZE))
                for cut in THRESHOLDS:
                    report.append(dict(crop=name, y=y, x=x, size_px=SIZE, contrast_rms_adu=score,
                                       contrast_rank=rank, source=source, threshold_adu=cut,
                                       **crop_metrics(cells, patch_tissue, cut)))
                core = image[y:y+SIZE, x:x+SIZE]
                main[name, source] = core
                if source != 'uniform_fitted':
                    with BytesIO() as png:
                        plt.imsave(png, patch, format='png', cmap='gray', vmin=limits[0], vmax=limits[1])
                        review.writestr(f'{name}-{source}.png', png.getvalue())
        write_csv(output / 'candidates.csv', proposals)
        review.write(output / 'candidates.csv', 'candidates.csv')
        review.writestr('README.txt', 'Review both sources with the same rule: a dark core and a brighter shell in '
                        'at least 2/3 of directions. Mark every visible outer contour in the central 64 x 64 px. '
                        'The image is 88 x 88 px with a 12 px context border. '
                        'Do not infer invisible cells from synthetic GT. Mark ambiguous contours separately. '
                        'Use the origin in crops.csv and candidate centers (native TIFF pixel-edge coordinates) '
                        'as suggestions. Correct missed and false candidates, including at the ROI boundary.\n')
        write_csv(output / 'crops.csv', report)
        review.write(output / 'crops.csv', 'crops.csv')
    shapes, shape_rows, cutoff = shape_analysis(annotation)
    write_csv(output / 'shape.csv', shape_rows)
    pairs = []
    for row in sorted(shape_rows, key=lambda r: abs(r['ellipse_gap_px'])):
        pair = tuple(sorted((row['cell'], row['neighbor'])))
        if row['close'] and all(not set(pair).intersection(chosen) for chosen in pairs):
            pairs.append(pair)
        if len(pairs) == 3:
            break
    if len(pairs) < 3:
        for row in sorted(shape_rows, key=lambda r: abs(r['ellipse_gap_px'])):
            pair = tuple(sorted((row['cell'], row['neighbor'])))
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) == 3:
                break

    sources = [('real', 'Реальная медиана'), ('uniform', 'Uniform-shell')]
    if gain is not None:
        sources.append(('uniform_fitted', f'Те же сомы · ×{gain:.2f} к амплитуде'))
    fig = plt.figure(figsize=(12.2, 10.9 + 2.8 * (len(sources) - 2)), layout='constrained')
    grid = fig.add_gridspec(len(sources) + 2, 1, height_ratios=(.17, *([1.] * len(sources)), 1.02))
    header = fig.add_subplot(grid[0])
    header.set_axis_off()
    header.text(0., .78, 'Plane20 · упаковка и форма сом', fontsize=20, weight='bold')
    header.text(0., .08, f'64 × 64 px · одна шкала {limits[0]:.0f}–{limits[1]:.0f} ADU · '
                'контраст используется только для выбора участков', fontsize=10)
    for band, (source, label) in enumerate(sources, 1):
        panels = grid[band].subgridspec(1, 4, wspace=.04)
        for col, (name, y, x, score, rank) in enumerate(crops):
            ax = fig.add_subplot(panels[col])
            ax.imshow(main[name, source], cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest')
            category = 'Контрастный' if name.startswith('high') else 'Обычный'
            ax.set_title(f'{label}\n{category} {name[-1]} · Y={y}, X={x}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    bottom = grid[-1].subgridspec(2, 3, height_ratios=(1, .12))
    for idx, pair in enumerate(pairs):
        draw_pair(fig.add_subplot(bottom[0, idx]), pair, shapes, shape_rows, real, limits)
    legend = fig.add_subplot(bottom[1, :])
    legend.set_axis_off()
    legend.text(.01, .68, 'Бирюзовый — ручной контур', color='#087d76', fontsize=10)
    legend.text(.30, .68, 'Жёлтый пунктир — эллипс', color='#9a6a00', fontsize=10)
    legend.text(.58, .68, 'Розовый — зазор вдоль линии центров', color='#ad2861', fontsize=10)
    legend.text(.01, .05, 'Фон под контурами — пересчитанная медиана TIFF. Архивный кадр разметки имеет другой хеш.',
                fontsize=9)
    fig.savefig(output / 'plane20_packing.png', dpi=165)
    plt.close(fig)
    close = [r for r in shape_rows if r['close']]
    delta = np.array([r['inward_difference_px'] for r in close])
    close_pairs = {frozenset((r['cell'], r['neighbor'])) for r in close}
    summary = dict(shape_count=len(shapes), median_iou=float(np.median([s['iou'] for s in shapes.values()])),
                   close_cutoff_ellipse_gap_px=cutoff, close_count=len(close),
                   close_median_inward_difference_px=float(np.median(delta)),
                   close_inward_difference_quartiles_px=np.percentile(delta, [25, 75]).tolist(),
                   close_positive_count=int(np.count_nonzero(delta > 0)),
                   close_pairs=len(close_pairs), gain_only_control=gain,
                   display_limits_adu=limits.tolist(), scale=meta['xy_scale'], pairs=pairs,
                   packing_status='automatic provisional ring candidates, no exhaustive manual labels')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--uniform', type=Path, required=True)
    parser.add_argument('--fitted', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.uniform, args.fitted, args.output)
