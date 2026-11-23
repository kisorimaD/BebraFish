"""Compare inherited placement to dense random and relaxed SimCalc layouts.

Experimental only. Reuses SimCalc kernels/detector; never alters frozen sources.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_erosion
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'simcalc_zebrafish_pipeline'))
from calibrated_simcalc import detector
from simcalc_core import _build_scene, sample_head_centres
from hybrid_high_model import CELL_DEFAULTS


def relaxed_centers(support, density, count, seed, iterations=35):
    """Weighted Lloyd relaxation with random jitter: local evenness, no lattice.

    A fixed cloud samples the coarse spatial prior. Each center moves toward
    its Voronoi region's centroid. Projection keeps centers inside support.
    """
    rng = np.random.default_rng(seed)
    pixels = np.argwhere(support)
    weights = density[support].astype(float)
    weights /= weights.sum()
    centers = sample_head_centres(support, count, 2.5, rng, density)
    cloud = pixels[rng.choice(len(pixels), size=180000, p=weights)].astype(float)
    cloud += rng.uniform(-.45, .45, cloud.shape)
    _, projection = distance_transform_edt(~support, return_indices=True)
    for _ in range(iterations):
        owner = cKDTree(centers).query(cloud)[1]
        mass = np.bincount(owner, minlength=count)
        updated = np.stack([np.bincount(owner, weights=cloud[:, d], minlength=count)
                            / np.maximum(mass, 1) for d in (0, 1)], axis=1)
        active = mass > 0
        centers[active] = .25*centers[active] + .75*updated[active]
        iy, ix = np.rint(centers).astype(int).T
        outside = ~support[iy, ix]
        centers[outside] = projection[:, iy[outside], ix[outside]].T
    centers += rng.normal(0, .25, centers.shape)
    np.clip(centers, 0, np.asarray(support.shape)-1, out=centers)
    iy, ix = np.rint(centers).astype(int).T
    outside = ~support[iy, ix]
    centers[outside] = projection[:, iy[outside], ix[outside]].T
    return centers.astype(np.float32)


def render_centers(profile, settings, centers, seed, *, scene=None):
    """Experimental photometry preserving coarse mean at higher cell density.

    Kept outside calibrated renderer so its code hash and exports stay valid.
    """
    mask = profile['tissue'].astype(bool)
    p = CELL_DEFAULTS | dict(shape_yx=list(mask.shape), sigma_px=settings['sigma_px'],
        size_cv=settings['size_cv'], amplitude_cv=settings['amplitude_cv'],
        eccentricity=settings['eccentricity'], ring_fraction=settings['ring_fraction'],
        blur_px=settings['blur_px'], background=0., background_texture=0., neuropil_filters=0,
        fill_soma_masks=True, keep_instance_footprints=True,
        ring_center_fill=settings['ring_center_fill'])
    if scene is None:
        scene = _build_scene(p, seed, centres=centers, tissue=profile['tissue_field'],
                             emission_gain=profile['cell_gain'])
    signal = scene['signal']
    # Positive cell texture with the observed coarse fluorescence envelope.
    # Additive compensation clips at zero and cannot preserve brightness once
    # expected cellular emission exceeds the observed envelope at high density.
    local_mean = gaussian_filter(signal, float(profile['compensation_sigma_px']))
    diffuse = profile['diffuse_target']
    texture = diffuse * (signal/np.maximum(local_mean, 1e-6)-1)
    base = profile['background'] + diffuse
    h0 = (base-gaussian_filter(base, 4))[mask]
    h1 = (texture-gaussian_filter(texture, 4))[mask]
    a = float(np.var(h1))
    b = float(2*np.mean((h0-h0.mean())*(h1-h1.mean())))
    c = float(np.var(h0)-settings['cell_contrast_std_adu']**2)
    fraction = float(np.clip((-b+np.sqrt(max(b*b-4*a*c, 0)))/max(2*a, 1e-8), 0, 1))
    clean = base + fraction*texture
    noisy = detector(clean, profile['background'], settings['shot_gain_adu'],
                     settings['read_variance_adu2'], np.random.default_rng(seed+1),
                     settings['excess_variance_fraction'])
    return dict(image=np.rint(np.clip(noisy, 0, 65535)).astype(np.uint16),
                clean=clean, centers=centers, labels=scene['labels'],
                instance_offsets=scene['instance_offsets'], instance_pixel_index=scene['instance_pixel_index'],
                cell_texture_fraction=np.array(fraction))


def contact_metrics(result, interior, shape):
    """Approximate soma radii from full footprint areas, not label ownership.

    A near neighbor has edge gap <= 1 px (can overlap in projection).
    'Surrounded' means no angular gap between such neighbors exceeds 180deg.
    This is a geometric proxy, not a measurement of real cell contacts.
    """
    centers = result['centers']
    areas = np.diff(result['instance_offsets'].astype(np.int64))
    radii = np.sqrt(areas/np.pi)
    tree = cKDTree(centers)
    neighbors = tree.query_ball_point(centers, float(2*radii.max()+1))
    neighbor_counts = []
    surrounded = []
    for i, near in enumerate(neighbors):
        y, x = np.rint(centers[i]).astype(int)
        if not interior[y, x]:
            continue
        ids = np.array([j for j in near if j != i], dtype=int)
        delta = centers[ids]-centers[i]
        dist = np.linalg.norm(delta, axis=1)
        near_contact = dist <= radii[i]+radii[ids]+1
        angles = np.sort(np.arctan2(delta[near_contact, 0], delta[near_contact, 1]))
        neighbor_counts.append(len(angles))
        surrounded.append(bool(len(angles) >= 3 and
                           np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) < np.pi))
    distances = tree.query(centers, k=2)[0][:, 1]
    union = np.zeros(np.prod(shape), dtype=bool)
    union[result['instance_pixel_index']] = True
    return dict(cell_count=len(centers), median_nearest_center_px=float(np.median(distances)),
                median_equivalent_soma_diameter_px=float(2*np.median(radii)),
                mean_near_neighbors=float(np.mean(neighbor_counts)),
                surrounded_fraction=float(np.mean(surrounded)),
                evaluated_interior_cells=len(surrounded),
                tissue_soma_coverage=float(union.reshape(shape)[interior].mean()),
                footprint_overlap_fraction=float(1-union.sum()/max(areas.sum(), 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, default=ROOT/'outputs/plane20_trial')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/plane20_packing')
    p.add_argument('--seed', type=int, default=20261113)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.fit/'report.json').read_text())
    settings = report['selected']
    with np.load(args.fit/'profile.npz') as z:
        profile = {k:z[k] for k in z.files}
    with np.load(args.fit/'frame_fit.npz') as z:
        real = z['real_frame'].copy()
        baseline = {k:z[src].copy() for k, src in dict(image='synthetic_image', clean='synthetic_clean', centers='centers_yx',
            labels='synthetic_labels', instance_offsets='instance_offsets', instance_pixel_index='instance_pixel_index').items()}
    support = distance_transform_edt(profile['tissue']) > 1.5*settings['sigma_px']
    interior = binary_erosion(profile['tissue'], iterations=15)
    variants = [('Original random 1x', baseline)]
    # Retain the same coarse density field for fair placement comparisons.
    for multiplier in [2., 2.5]:
        count = round(settings['cell_count']*multiplier)
        random_centers = sample_head_centres(support, count, max(2., settings['sigma_px']),
            np.random.default_rng(args.seed), profile['density'])
        random = render_centers(profile, settings, random_centers, args.seed)
        variants.append((f'Random {multiplier:g}x', random))
        centers = relaxed_centers(support, profile['density'], count, args.seed)
        packed = render_centers(profile, settings, centers, args.seed)
        variants.append((f'Relaxed {multiplier:g}x', packed))
        print('Rendered', multiplier, flush=True)
    metrics = {}
    for i, (name, result) in enumerate(variants):
        metrics[name] = contact_metrics(result, interior, real.shape)
        metrics[name]['mean_tissue_adu'] = float(np.mean(result['image'][interior]))
        clean = result['clean']
        hp = clean-gaussian_filter(clean, 4)
        metrics[name]['intrinsic_hp_std_adu'] = float(np.std(hp[profile['tissue']]))
        if 'cell_texture_fraction' in result:
            metrics[name]['cell_texture_fraction'] = float(result['cell_texture_fraction'])
        np.savez_compressed(args.output/f'variant_{i}.npz', **result)
    y0, y1, x0, x1 = report['crop_yx']
    roi = (slice(y0, y1), slice(x0, x1))
    vmin, vmax = np.percentile(real, [1, 99.5])
    fig, axs = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, (name, result) in zip(axs.flat, [('Real frame 500', dict(image=real))]+variants):
        ax.imshow(result['image'][roi].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(name); ax.axis('off')
    fig.savefig(args.output/'packing_crops.png', dpi=150); plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
    for ax, (name, result) in zip(axs, [('Real frame 500', dict(image=real)), variants[0], variants[-1]]):
        ax.imshow(result['image'].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(name); ax.axis('off')
    fig.savefig(args.output/'packing_full.png', dpi=150); plt.close(fig)
    output = dict(seed=args.seed, settings=settings, metrics=metrics,
        real_mean_tissue_adu=float(real[interior].mean()),
        method='Weighted Lloyd relaxation of centers, 35 iterations, same coarse density prior and SimCalc kernels.',
        contact_definition='Equivalent-area footprint radii, edge gap <=1px; surrounded: >=3 neighbors and maximum angular gap <180deg; excludes 15px tissue boundary.',
        limitations=['No real-cell segmentation: no measured real density or real contact fraction.',
            'Density multipliers are hypotheses based on visual feedback.',
            'Projected footprints may overlap; packing does not deform somas.',
            'Texture fraction is capped at 1: dense relaxed layouts may not reach the target contrast.',
            'Dense variants use mean-normalized cell texture to preserve the observed coarse brightness; texture fraction matches the prior contrast where feasible. Original 1x is retained unchanged.'])
    (args.output/'report.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
