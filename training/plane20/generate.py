"""Export the calibrated profile or generate independently seeded full scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'simcalc_zebrafish_pipeline')]
from compare_packing import relaxed_centers
from empirical_morphology import build_empirical_scene
from soma_shell_model import sample_morphology, sample_angular_coefficients
from try_thinner_medium import thinner_shell_patch
from try_correlated_cell_gain import smooth_gain_field, cell_amplitudes
from try_heterogeneous_scene import effective_blur, correlated_noise
from fit_contrast_sharpness import contrast_signal


def profile_arrays():
    arrays = {}
    for source, keys in [
        ('plane20_contrast_sharpness/synthetic.npz', ['background', 'noise_sigma']),
        ('plane20_masked_background/synthetic.npz', ['neuron_support', 'exclusion_mask']),
        ('plane20_trial/profile.npz', ['density', 'tissue']),
        ('plane20_angular_shells/synthetic.npz', ['normalizer'])]:
        with np.load(ROOT/'outputs'/source) as z:
            arrays.update({k: z[k] for k in keys})
    arrays['settings_json'] = np.array((ROOT/'outputs/plane20_trial/report.json').read_text())
    arrays['morphology_json'] = np.array((ROOT/'outputs/soma_analysis/report.json').read_text())
    return arrays


def export_profile(path):
    arrays = profile_arrays()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def generate_one(job, profile_path, ablation=False):
    split, index, seed, output = job
    with np.load(profile_path) as z:
        profile = {k: z[k] for k in z.files}
    children = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(8)]
    allowed = profile['neuron_support'].astype(bool)
    settings = json.loads(str(profile['settings_json']))['selected'] | dict(
        ring_fraction=1., ring_center_fill=0., blur_px=.6)
    # Reuse the calibrated placement algorithm and candidate count, not old centers.
    support = distance_transform_edt(profile['tissue'].astype(bool)) > 1.5*settings['sigma_px']
    centers = relaxed_centers(support, profile['density'], 6830, children[0])
    geometry = sample_morphology(json.loads(str(profile['morphology_json'])), len(centers), children[1])
    # Exact full-mask support test, as in the selected renderer's scene filtering.
    from soma_shell_model import shell_patch
    keep = []
    for k, center in enumerate(centers):
        origin, _, mask = shell_patch(center, geometry['major_axis_px'][k],
            geometry['minor_axis_px'][k], geometry['orientation_deg'][k], True, 0, .6, width=.306)
        y, x = np.nonzero(mask); y, x = y+origin[0], x+origin[1]
        inside = (y >= 0) & (x >= 0) & (y < allowed.shape[0]) & (x < allowed.shape[1])
        keep.append(bool(inside.all() and allowed[y, x].all()))
    keep = np.asarray(keep)
    centers = centers[keep]
    geometry = {k: v[keep] for k, v in geometry.items() if k in
                ('major_axis_px', 'minor_axis_px', 'orientation_deg')}
    field = smooth_gain_field(allowed.shape, 27.932934624762108, children[2])
    amplitudes = cell_amplitudes(field, centers, .55, .35, children[3])
    angular = .5*sample_angular_coefficients(len(centers), children[4])
    ones = np.ones(allowed.shape, np.float32)
    focus = smooth_gain_field(allowed.shape, 40., children[6])
    mixture = 1/(1+np.exp(-1.5*focus))
    unit, _ = correlated_noise(allowed.shape, children[7])
    sampling_centers = centers.copy()
    rows = []
    fractions = (1., .5) if ablation else (1.,)
    for fraction in fractions:
        selected = np.arange(len(sampling_centers))
        if fraction < 1:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 8675309]))
            selected = np.sort(rng.choice(selected, int(len(selected)*fraction), replace=False))
        selected_geometry = {k: v[selected] for k, v in geometry.items()}
        scene = build_empirical_scene(dict(tissue=allowed, tissue_field=ones, cell_gain=ones),
            settings, sampling_centers[selected], None, children[5], geometry=selected_geometry,
            amplitudes=amplitudes[selected], angular_coefficients=angular[selected],
            patch_renderer=thinner_shell_patch)
        cells = effective_blur(scene['signal'], .2, .8, mixture)*profile['normalizer']
        cells *= np.clip(distance_transform_edt(allowed)/2, 0, 1)
        cells = contrast_signal(cells.astype(np.float32), 1.8)
        assert not cells[profile['exclusion_mask'].astype(bool)].any()
        for noise_scale in ((1., 0.) if ablation else (1.,)):
            variant = ('normal' if fraction == 1 else 'half')+('_noisy' if noise_scale else '_clean')
            directory = Path(output)/variant if ablation else Path(output)
            image = np.rint(np.clip(profile['background']+cells+noise_scale*unit*profile['noise_sigma'], 0, 65535)).astype(np.uint16)
            path = directory/split/f'scene_{index:05d}.npz'
            extras = dict(sampling_centers=sampling_centers, parent_indices=selected,
                          noise_scale=np.array(noise_scale), density_fraction=np.array(fraction)) if ablation else {}
            np.savez_compressed(path, image=image, centers=sampling_centers[selected], **selected_geometry,
                instance_offsets=scene['instance_offsets'], instance_pixel_index=scene['instance_pixel_index'],
                amplitudes=amplitudes[selected], angular_coefficients=angular[selected],
                seed=np.array(seed, np.uint64), **extras)
            rows.append(dict(split=split, index=index, seed=seed, child_seeds=children,
                cells=len(selected), path=str(path.relative_to(directory)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **(dict(variant=variant) if ablation else {})))
    return rows if ablation else rows[0]



def add_generation_arguments(p):
    p.add_argument('--export-profile', type=Path)
    p.add_argument('--profile', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--val', type=int, default=8)
    p.add_argument('--test', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261113)
    p.add_argument('--workers', type=int, default=8)


def generate_dataset(args, worker, profile_policy):
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for split, count in [('train', args.train), ('val', args.val), ('test', args.test)]:
        (args.output/split).mkdir()
        for i in range(count):
            # Global index keeps all splits seed-disjoint.
            seed = int(np.random.SeedSequence([args.seed, len(jobs)]).generate_state(1, dtype=np.uint64)[0])
            jobs.append((split, i, seed, str(args.output)))
    config = vars(args) | dict(profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(), profile_policy=profile_policy)
    (args.output/'config.json').write_text(json.dumps(config, default=str, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'manifest.jsonl').open('w') as manifest:
        for row in pool.map(partial(worker, profile_path=args.profile), jobs):
            manifest.write(json.dumps(row)+'\n'); manifest.flush()
            print(json.dumps(row), flush=True)
    (args.output/'COMPLETE').write_text(f'{len(jobs)} scenes\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(p)
    args = p.parse_args()
    if args.export_profile:
        export_profile(args.export_profile)
        return
    if args.profile is None or args.output is None:
        p.error('--profile and --output are required')
    generate_dataset(args, generate_one,
        'Fixed calibrated anatomy/background/normalizer/noise_sigma; new packing, morphology, gains, shell patterns, focus and noise. No additional parameter randomization.')


if __name__ == '__main__':
    main()
