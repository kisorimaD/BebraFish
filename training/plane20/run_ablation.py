"""Run four independent training processes on one GPU, with prefixed terminal logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

VARIANTS = ('normal_noisy', 'normal_clean', 'half_noisy', 'half_clean')


def stream(process, name, path):
    with path.open('a') as log:
        for line in process.stdout:
            log.write(line); log.flush()
            # Large startup metadata stays in each complete log file.
            if not line.startswith('{"data":'):
                print(f'[{name}] {line.rstrip()}', flush=True)


def stop(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('../dataset_ablation'))
    p.add_argument('--output', type=Path, default=Path('../runs/ablation_center32'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--val-steps', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--gpu-memory-mb', type=int, default=16384)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-evaluation', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.steps, args.val_steps, args.batch_size, args.gpu_memory_mb) < 1:
        p.error('counts and GPU cap must be positive')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    args.data, args.output = args.data.resolve(), args.output.resolve()
    for name in VARIANTS:
        if not (args.data/name/'COMPLETE').is_file():
            p.error(f'Incomplete dataset: {args.data/name}')
        if (args.output/name).exists() and not args.resume:
            p.error(f'Run exists: {args.output/name}; use --resume or another --output')
        if (args.output/name).exists() and not (args.output/name/'state/checkpoint').is_file():
            p.error(f'No resumable checkpoint in {args.output/name}')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output/'logs'; logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', TF_NUM_INTRAOP_THREADS='2',
               TF_NUM_INTEROP_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='2')
    commands = {}
    for name in VARIANTS:
        command = [sys.executable, '-m', 'training.plane20.train_center_mask',
                   '--data', str(args.data/name), '--output', str(args.output/name),
                   '--epochs', str(args.epochs), '--steps', str(args.steps),
                   '--val-steps', str(args.val_steps), '--batch-size', str(args.batch_size),
                   '--seed', str(args.seed), '--gpu-memory-mb', str(args.gpu_memory_mb)]
        if args.resume and (args.output/name).exists():
            command.append('--resume')
        commands[name] = command
    (args.output/'launcher.json').write_text(json.dumps(dict(arguments=vars(args), commands=commands,
        cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'], environment={k: env[k] for k in
        ('OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        started=time.time()), default=str, indent=2)+'\n')
    processes, threads = [], []
    try:
        for name, command in commands.items():
            print(f'[{name}] starting; GPU cap {args.gpu_memory_mb} MiB; log {logs/name}.log', flush=True)
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True)
            processes.append(process)
            thread = threading.Thread(target=stream, args=(process, name, logs/f'{name}.log'))
            thread.start(); threads.append(thread)
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('A training process failed; stopping the other runs. See prefixed logs.')
            time.sleep(.5)
        for thread in threads:
            thread.join()
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('Training failed; see logs')
        # GPU memory is released before evaluations. Same threshold/crops for all.
        if not args.skip_evaluation:
            for name in VARIANTS:
                command = [sys.executable, '-m', 'training.plane20.center_inference', 'evaluate',
                    '--run', str(args.output/name), '--data', str(args.data/name), '--split', 'val',
                    '--crops', '24', '--start-index', '32', '--seed', '991',
                    '--threshold', '0.3', '--mask-threshold', '0.5',
                    '--output', str(args.output/name/'validation_instances.json')]
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
                processes.append(process)
                stream(process, name+'/eval', logs/f'{name}_eval.log')
                if process.wait() != 0:
                    raise RuntimeError(f'Evaluation failed: {name}')
            report = {name: json.loads((args.output/name/'validation_instances.json').read_text()) for name in VARIANTS}
            (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'All four runs finished. Results: {args.output}', flush=True)
    finally:
        stop(processes)
        for thread in threads:
            thread.join()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted: stopped all child processes; use --resume to continue.', file=sys.stderr)
        sys.exit(130)
