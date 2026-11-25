"""Run segmentation, trace and photometry calibration, and GIF rendering for a new set of TIFFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import tifffile

from calibrate_dataset_profile import calibrate_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEGMENT_HOME = Path('/raid/kisor/projects/zb/plane20_stardist')
BUNDLED_MODEL = ROOT / 'segmentation'
CONVERT = HERE.parent / 'convert_crop_activity'
PACKAGE = ROOT / 'plane20_synthetic' / 'plane20_synthetic'
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


def main():
    bundled = (BUNDLED_MODEL / 'training/plane20/center_inference.py').is_file()
    default_checkpoint = ROOT / 'analysis/fish7_f_s_plane37_39/models/pilot'
    if bundled:
        model_dir = BUNDLED_MODEL / 'model'
        default_checkpoint = model_dir if (model_dir / 'weights_best.h5').is_file() else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Folder containing the source TIFF planes.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-plane', type=int, required=True)
    parser.add_argument('--planes', type=int, nargs='+', help='Use only these source plane IDs.')
    parser.add_argument('--fps', type=float, help='Rate of TIFF frames, if confirmed. Display defaults to 12.5 otherwise.')
    parser.add_argument('--cells', type=int, help='Maximum scene cell count. Defaults to the reference ROI count, limited by donor capacity.')
    parser.add_argument('--seed', type=int, default=20261125)
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
    if args.checkpoint is None:
        parser.error('To segment new TIFFs, set --checkpoint to a directory with weights_best.h5 and model_config.json.')
    sources = discover(args.input, args.planes)
    if args.reference_plane not in sources:
        parser.error('The reference plane must be among the selected source planes.')
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
                    shape_tyx=inspect_sources(sources), reference_plane=args.reference_plane, fps=args.fps,
                    cells=args.cells, seed=args.seed, diagnostic_seconds=args.diagnostic_seconds,
                    segment_python=str(args.segment_python.absolute()), model_code=str(args.model_code.resolve()),
                    checkpoint=str(checkpoint),
                    checkpoint_sha256={name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                                       for name in ('weights_best.h5', 'model_config.json')})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'dataset_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
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
        run_stage('1/6 Segment and extract raw fluorescence', args.segment_python,
                  CONVERT / 'extract.py', ['--dataset-manifest', manifest_path], environment)

    foreground = output / 'foreground'
    bank = foreground / 'trace_bank.npz'
    if not bank.is_file() or not (foreground / 'summary.json').is_file():
        run_stage('2/6 Filter background and build trace bank', sys.executable,
                  CONVERT / 'build_conservative_bank.py', ['--dataset-manifest', manifest_path], environment)
    if not all((foreground / name).is_file() for name in ('coordination.json', 'coordination.png')):
        run_stage('3/6 Measure spatial and temporal coactivity', sys.executable,
                  CONVERT / 'measure_coordination.py', ['--bank', bank], environment)

    reference_dir = output / 'per_plane' / f'plane_{args.reference_plane}'
    profile = output / 'profile.npz'
    scene = output / 'scene' / 'scene_00000.npz'
    scene_report = output / 'profile_summary.json'
    if not all(path.is_file() for path in (profile, scene, scene_report)):
        print('\n4/6 Calibrate profile and sample new soma geometry', flush=True)
        sys.path.insert(0, str(PACKAGE))
        from training.plane20.generate_median import generate_pair
        calibration = calibrate_profile(reference_dir, bank, args.reference_plane, profile, args.cells)
        generated = generate_pair(('scene', 0, args.seed, output), profile, structured=False)
        if generated['cells'] < 100:
            raise ValueError(f'Only {generated["cells"]} soma masks fit in the calibrated support.')
        scene_report.write_text(json.dumps({**calibration, **generated}, indent=2) + '\n')
        print(f'Packed {generated["cells"]} somas', flush=True)

    gif_dir = output / 'render'
    reference_tiff = sources[args.reference_plane]
    if not (gif_dir / 'generation.json').is_file():
        render_args = ['--output', gif_dir, '--bank', bank, '--scene', scene, '--profile', profile,
                       '--reference-plane', args.reference_plane, '--reference-tiff', reference_tiff,
                       '--reference-median', reference_dir / 'median.tiff', '--seed', args.seed]
        if args.fps is not None:
            render_args.extend(('--source-fps', args.fps))
        run_stage('5/6 Match activity, fit noise and brightness, render comparison GIF', sys.executable,
                  HERE / 'render_recorded_bank.py', render_args, environment)
    if not all((gif_dir / name).is_file() for name in
               ('plane20_signal_residual.gif', 'plane20_signal_residual_median.png')):
        run_stage('6/6 Render four-panel diagnostic and temporal medians', sys.executable,
                  HERE / 'plot_render_components.py',
                  ['--directory', gif_dir, '--gif-seconds', args.diagnostic_seconds], environment)
    print(f'\nFinished: {gif_dir}', flush=True)


if __name__ == '__main__':
    main()
