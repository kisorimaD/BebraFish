"""Run motion-free synthetic fluorescence calibration and validation from aligned TIFFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import tifffile

from calibrate_dataset_profile import calibrate_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEGMENT_HOME = Path('/raid/kisor/projects/zb/plane20_stardist')
BUNDLED_MODEL = ROOT / 'segmentation'
CONVERT = HERE.parent / 'convert_crop_activity'
PACKAGE = ROOT / 'plane20_synthetic' / 'plane20_synthetic'
ORIGINAL_PLANE20 = ROOT/'20259815_Soma_GCaMP8f_Convert_Crop'/'original'/'Plane_Aligned_20.tiff'
ORIGINAL_SCENE = PACKAGE/'smoke_median'/'train'/'scene_00000.npz'
ORIGINAL_PROFILE = PACKAGE/'profiles'/'plane20.npz'
ORIGINAL_OPTICAL_MODEL = HERE/'movie_foreground_conservative'
ORIGINAL_REPORT = HERE/'movie_decoupled_dynamics'
FILENAME = re.compile(r'Plane(?:_Aligned)?_?(\d+)\.tiff?', re.I)


def discover(directory, requested):
    sources = {}
    for source in sorted(directory.iterdir()):
        match = FILENAME.fullmatch(source.name)
        if not match:
            continue
        plane = int(match[1])
        if requested and plane not in requested:
            continue
        if plane in sources:
            raise ValueError(f'Two TIFFs have the plane number {plane}.')
        sources[plane] = source.resolve()
    if requested and set(sources) != set(requested):
        raise ValueError(f'Missing TIFF for planes {sorted(set(requested) - set(sources))}.')
    if len(sources) < 2:
        raise ValueError('Supply at least two TIFFs named PlaneN.tif or Plane_Aligned_N.tiff.')
    return dict(sorted(sources.items()))


def inspect_sources(sources):
    shapes = set()
    for plane, path in sources.items():
        with tifffile.TiffFile(path) as file:
            page = file.pages[0]
            if len(page.shape) != 2 or page.dtype.name != 'uint16':
                raise ValueError(f'Plane {plane} requires a grayscale uint16 TIFF with one frame per page.')
            current = (len(file.pages), *page.shape)
            if current[0] < 100 or min(current[1:]) < 40:
                raise ValueError(f'Plane {plane} needs at least 100 frames and 40 pixels per image axis.')
            shapes.add(current)
    if len(shapes) != 1:
        raise ValueError(f'Input planes must share the same frame count and native image shape, got {shapes}.')
    return list(shapes.pop())


def run_stage(name, executable, script, arguments, environment):
    print(f'\n{name}', flush=True)
    subprocess.run([str(executable), str(script), *map(str, arguments)], check=True, env=environment)


def read_geometry(scene, profile, shape):
    scene, profile = scene.resolve(), profile.resolve()
    if not scene.is_file() or not profile.is_file():
        raise ValueError(f'Expected saved geometry at {scene} and {profile}.')
    with (np.load(scene, allow_pickle=False) as scene_data,
          np.load(profile, allow_pickle=False) as profile_data):
        if tuple(scene_data['image'].shape) != tuple(shape[1:]) or profile_data['tissue'].shape != scene_data['image'].shape:
            raise ValueError(f'The scene and profile at {scene} must match the TIFF frame geometry.')
        count = len(scene_data['centers'])
        if count < 100 or len(scene_data['instance_offsets']) != count+1:
            raise ValueError(f'The scene at {scene} needs at least 100 valid soma masks.')
    return dict(scene=str(scene), profile=str(profile), cells=count,
                scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())


def run_plane(args, reference_plane, reference_tiff, bank, per_plane, output, geometry, saved_report, environment):
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seed + reference_plane if args.all_planes else args.seed
    print(f'\nRender plane {reference_plane}', flush=True)
    reference_dir = per_plane / f'plane_{reference_plane}'
    profile = output / 'profile.npz'
    scene = output / 'scene' / 'scene_00000.npz'
    scene_report = output / 'profile_summary.json'
    if geometry is not None:
        scene, profile = Path(geometry['scene']), Path(geometry['profile'])
        with np.load(bank, allow_pickle=False) as donor_bank:
            planes = donor_bank['plane']
            donor_count = len(planes) if args.all_planes else np.count_nonzero(planes != reference_plane)
            if donor_count < geometry['cells']:
                raise ValueError('The donor planes contain fewer usable traces than the supplied scene has masks.')
        if not scene_report.is_file():
            scene_report.write_text(json.dumps({'geometry': 'reused', **geometry}, indent=2)+'\n')
        print(f'\n3/7 Reuse saved scene with {geometry["cells"]} soma masks', flush=True)
    elif not all(path.is_file() for path in (profile, scene, scene_report)):
        print('\n3/7 Calibrate tissue profile and sample new soma geometry', flush=True)
        if str(PACKAGE) not in sys.path:
            sys.path.insert(0, str(PACKAGE))
        from training.plane20.generate_median import generate_pair
        calibration = calibrate_profile(reference_dir, bank, reference_plane, profile, args.cells,
                                        include_reference_donors=args.all_planes)
        generated = generate_pair(('scene', 0, seed, output), profile, structured=False)
        if generated['cells'] < 100:
            raise ValueError(f'Only {generated["cells"]} soma masks fit in the calibrated support.')
        scene_report.write_text(json.dumps({**calibration, **generated}, indent=2) + '\n')
        print(f'Packed {generated["cells"]} somas', flush=True)

    resamples = output / 'activity_resamples'
    if not (resamples / 'comparison.json').is_file():
        sample_args = ['--bank', bank, '--scene', scene, '--reference-plane', reference_plane,
                       '--output', resamples, '--no-figure', '--seeds', *args.activity_seeds]
        if args.all_planes:
            sample_args.append('--all-plane-donors')
        run_stage('4/7 Sample activity variants', sys.executable, HERE/'sample_plane20_activity.py',
                  sample_args, environment)
    coordinated = output / 'activity_resamples_coordinated'
    if not (coordinated / 'comparison.json').is_file():
        run_stage('5/7 Calibrate spatial coactivity and rare population events', sys.executable,
                  HERE/'coordinate_sampled_activity.py', ['--input', resamples, '--output', coordinated], environment)

    sample = coordinated / f'activity_seed_{args.activity_seeds[0]}.npz'
    optical = ORIGINAL_OPTICAL_MODEL if saved_report else output/'optical_model'
    if saved_report is None and not (optical / 'generation.json').is_file():
        render_args = ['--output', optical, '--bank', bank, '--scene', scene, '--profile', profile,
                       '--reference-plane', reference_plane, '--reference-tiff', reference_tiff,
                       '--reference-median', reference_dir / 'median.tiff', '--seed', seed,
                       '--activity-sample', sample, '--calibration-only']
        if args.fps is not None:
            render_args.extend(('--source-fps', args.fps))
        run_stage('6/7 Fit brightness, soma halos and detector noise', sys.executable,
                  HERE / 'render_recorded_bank.py', render_args, environment)
    report, figures = output/'report', output/'paper_figures'
    if not (report/'validation.json').is_file():
        label = '6–7/7 Restore and verify the saved Plane20 render' if saved_report else (
            '7/7 Match activity in image units and verify the rendered recording')
        restore_args = ['--saved-report', saved_report] if saved_report else []
        run_stage(label, sys.executable,
                  HERE/'render_calibrated_dataset.py', ['--source', optical, '--output', report,
                  '--figures', figures, '--seed', seed, '--diagnostic-seconds', args.diagnostic_seconds,
                  *restore_args], environment)
    for suffix in ('.pdf', '.png'):
        name = 'activity_comparison'+suffix
        shutil.copyfile(coordinated/name, figures/name)
    print(f'\nFinished: {report}\nPublication figures: {figures}', flush=True)


def main():
    bundled = (BUNDLED_MODEL / 'training/plane20/center_inference.py').is_file()
    default_checkpoint = ROOT / 'analysis/fish7_f_s_plane37_39/models/pilot'
    if bundled:
        model_dir = BUNDLED_MODEL / 'model'
        default_checkpoint = model_dir if (model_dir / 'weights_best.h5').is_file() else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Folder containing the source TIFF planes.')
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference-plane', type=int, help='Render one plane using donors from the other planes.')
    mode.add_argument('--all-planes', action='store_true', help='Render every selected plane from the common trace bank.')
    parser.add_argument('--planes', type=int, nargs='+', help='Use only these source plane IDs.')
    parser.add_argument('--fps', type=float, help='Rate of TIFF frames, if confirmed. Display defaults to 12.5 otherwise.')
    parser.add_argument('--cells', type=int, help='Maximum scene cell count. Defaults to the reference ROI count, limited by donor capacity.')
    parser.add_argument('--scene', type=Path, help='Reuse this saved synthetic scene and its masks.')
    parser.add_argument('--profile', type=Path, help='Optical profile for the supplied --scene.')
    parser.add_argument('--geometry-dir', type=Path,
                        help='With --all-planes, use plane_N/scene/scene_00000.npz and plane_N/profile.npz.')
    parser.add_argument('--refit-render', action='store_true',
                        help='Fit new render parameters on the archived Plane20 scene.')
    parser.add_argument('--seed', type=int, default=20261125)
    parser.add_argument('--activity-seeds', type=int, nargs='+', default=[27, 63, 105],
                        help='Seeds for activity samples. The first sample is rendered.')
    parser.add_argument('--diagnostic-seconds', type=int, choices=range(1, 61), default=15)
    parser.add_argument('--segment-python', type=Path,
                        default=Path(sys.executable) if bundled else SEGMENT_HOME / 'env/bin/python')
    parser.add_argument('--model-code', type=Path, default=BUNDLED_MODEL if bundled else
                        SEGMENT_HOME / 'median_mix_20261120')
    parser.add_argument('--checkpoint', type=Path, default=default_checkpoint,
                        help='Directory with weights_best.h5 and model_config.json.')
    parser.add_argument('--resume', action='store_true', help='Resume the same run from validated completed stages.')
    args = parser.parse_args()
    if args.fps is not None and not 3 <= args.fps < float('inf'):
        parser.error('--fps must be finite and at least 3.')
    if args.cells is not None and args.cells < 100:
        parser.error('--cells must be at least 100.')
    if (args.scene is None) != (args.profile is None):
        parser.error('--scene and --profile must be used together.')
    if args.all_planes and args.scene is not None:
        parser.error('--scene and --profile describe a single plane. Use --reference-plane to reuse them.')
    if args.geometry_dir is not None and not args.all_planes:
        parser.error('--geometry-dir requires --all-planes.')
    if args.cells is not None and (args.scene is not None or args.geometry_dir is not None):
        parser.error('--cells applies only when generating a new scene.')
    if min(args.activity_seeds) < 0 or len(set(args.activity_seeds)) != len(args.activity_seeds):
        parser.error('--activity-seeds must be distinct nonnegative integers.')
    if args.checkpoint is None:
        parser.error('To segment new TIFFs, set --checkpoint to a directory with weights_best.h5 and model_config.json.')
    sources = discover(args.input, args.planes)
    if not args.all_planes and args.reference_plane not in sources:
        parser.error('The reference plane must be among the selected source planes.')
    shape = inspect_sources(sources)
    fixed_scene, fixed_profile = args.scene, args.profile
    original_input = (args.reference_plane == 20 and set(sources) == set(range(15, 25)) and
                      all(path == (ORIGINAL_PLANE20.parent/f'Plane_Aligned_{plane}.tiff').resolve()
                          for plane, path in sources.items()))
    original_geometry = original_input and fixed_scene is None and args.cells is None
    original_scene = (ORIGINAL_SCENE if ORIGINAL_SCENE.is_file() else
                      ROOT / 'results' / 'Plane20' / ORIGINAL_SCENE.name)
    if original_geometry:
        fixed_scene, fixed_profile = original_scene, ORIGINAL_PROFILE
    saved_report = None
    legacy_artifacts = (ORIGINAL_OPTICAL_MODEL / 'generation.json', ORIGINAL_REPORT / 'validation.json',
                        ORIGINAL_REPORT / 'activity.npz', ORIGINAL_REPORT / 'real_vs_synthetic.gif')
    if (original_geometry and args.fps is None and not args.refit_render and
            all(path.is_file() for path in legacy_artifacts)):
        saved_report = ORIGINAL_REPORT
    geometries = {}
    try:
        if fixed_scene is not None:
            geometries[args.reference_plane] = read_geometry(fixed_scene, fixed_profile, shape)
        elif args.geometry_dir is not None:
            for plane in sources:
                layer = args.geometry_dir / f'plane_{plane}'
                geometries[plane] = read_geometry(layer / 'scene/scene_00000.npz', layer / 'profile.npz', shape)
        elif args.all_planes and args.cells is None and sources.get(20) == ORIGINAL_PLANE20.resolve():
            geometries[20] = read_geometry(original_scene, ORIGINAL_PROFILE, shape)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.segment_python.is_file() or not (args.model_code / 'training/plane20/center_inference.py').is_file():
        parser.error('Segmentation runtime missing. Set --segment-python and --model-code.')
    checkpoint = args.checkpoint.resolve()
    for name in ('weights_best.h5', 'model_config.json'):
        if not (checkpoint / name).is_file():
            parser.error(f'Missing {name} in --checkpoint.')
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error('Use a new output folder or --resume to continue the same run.')

    manifest = dict(sources={str(p): str(path) for p, path in sources.items()},
                    source_fingerprints={str(p): [path.stat().st_size, path.stat().st_mtime_ns]
                                         for p, path in sources.items()},
                    shape_tyx=shape, reference_plane=args.reference_plane, all_planes=args.all_planes, fps=args.fps,
                    cells=args.cells, seed=args.seed, diagnostic_seconds=args.diagnostic_seconds,
                    activity_seeds=args.activity_seeds, fixed_geometry=geometries.get(args.reference_plane),
                    saved_render=str(saved_report) if saved_report is not None else None,
                    segment_python=str(args.segment_python.absolute()), model_code=str(args.model_code.resolve()),
                    checkpoint=str(checkpoint),
                    checkpoint_sha256={name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                                       for name in ('weights_best.h5', 'model_config.json')})
    if args.all_planes and geometries:
        manifest['plane_geometries'] = {str(plane): geometry for plane, geometry in geometries.items()}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'dataset_manifest.json'
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        stored.setdefault('all_planes', False)
        stored.setdefault('activity_seeds', [27, 63, 105])
        stored.setdefault('fixed_geometry', None)
        stored.setdefault('saved_render', None)
        if stored != manifest:
            parser.error('The input files or run parameters changed. Choose a new output folder.')
    elif any(output.iterdir()):
        parser.error('Cannot resume a nonempty folder without a dataset_manifest.json.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    environment.setdefault('TF_NUM_INTEROP_THREADS', '2')
    if not all((output / 'per_plane' / f'plane_{plane}' / 'summary.json').is_file() and
               (output / 'per_plane' / f'plane_{plane}' / 'traces.npz').is_file() for plane in sources):
        run_stage('1/7 Segment somata and extract raw fluorescence', args.segment_python,
                  CONVERT / 'extract.py', ['--dataset-manifest', manifest_path], environment)

    foreground = output / 'foreground'
    bank = foreground / 'trace_bank.npz'
    if not bank.is_file() or not (foreground / 'summary.json').is_file():
        run_stage('2/7 Filter background and build trace bank', sys.executable,
                  CONVERT / 'build_conservative_bank.py', ['--dataset-manifest', manifest_path], environment)

    targets = sources if args.all_planes else (args.reference_plane,)
    for plane in targets:
        plane_output = output / 'planes' / f'plane_{plane}' if args.all_planes else output
        run_plane(args, plane, sources[plane], bank, output / 'per_plane', plane_output,
                  geometries.get(plane), saved_report, environment)


if __name__ == '__main__':
    main()
