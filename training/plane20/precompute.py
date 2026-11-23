"""Precompute exact full-scene StarDist targets into uncompressed mmap arrays."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import targets, target_cache_config


def build_scene(job):
    directory, row, rays = job
    directory = Path(directory)
    source = directory/row['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError(f'Scene checksum mismatch: {source}')
    base = directory/'target_cache'/Path(row['path']).with_suffix('')
    base.mkdir(parents=True, exist_ok=True)
    expected = dict(source_sha256=row['sha256'], rays=rays)
    if (base/'complete.json').is_file():
        if json.loads((base/'complete.json').read_text()) == expected and all(
                (base/f'{name}.npy').is_file() for name in ('prob', 'dist')):
            return dict(path=row['path'], reused=True)
        raise ValueError(f'Existing target cache has different metadata: {base}')
    with np.load(source) as z:
        scene = {k: z[k] for k in ('image', 'centers', 'major_axis_px', 'minor_axis_px', 'orientation_deg')}
    h, w = scene['image'].shape
    prob = np.lib.format.open_memmap(base/'prob.tmp.npy', mode='w+', dtype=np.float32, shape=(h, w))
    dist = np.lib.format.open_memmap(base/'dist.tmp.npy', mode='w+', dtype=np.float32, shape=(h, w, rays))
    # Reuse the reference target function. Tile boundaries never truncate ellipses:
    # targets() uses full scene geometry; only the evaluated pixel grid is cropped.
    for y in range(0, h, 128):
        for x in range(0, w, 128):
            p, d = targets(scene, (y, x), size=128, rays=rays, halo=0)
            dy, dx = min(128, h-y), min(128, w-x)
            prob[y:y+dy, x:x+dx] = p[:dy, :dx, 0]
            dist[y:y+dy, x:x+dx] = d[:dy, :dx, :-1]
    prob.flush(); dist.flush()
    del prob, dist
    (base/'prob.tmp.npy').replace(base/'prob.npy')
    (base/'dist.tmp.npy').replace(base/'dist.npy')
    (base/'complete.json').write_text(json.dumps(expected, indent=2)+'\n')
    return dict(path=row['path'], reused=False)


def precompute(directory, rays=32, workers=4):
    directory = Path(directory).resolve()
    if not (directory/'COMPLETE').is_file():
        raise ValueError('Scene dataset is incomplete')
    cache = directory/'target_cache'
    cache.mkdir(exist_ok=True)
    config = target_cache_config(directory, rays)
    if (cache/'config.json').exists():
        if json.loads((cache/'config.json').read_text()) != config:
            raise ValueError('Existing target cache has incompatible metadata; use a new cache directory')
    else:
        (cache/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    rows = [json.loads(line) for line in (directory/'manifest.jsonl').read_text().splitlines()]
    jobs = [(str(directory), row, rays) for row in rows]
    if workers == 1:
        for job in jobs:
            print(json.dumps(build_scene(job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(build_scene, jobs):
                print(json.dumps(row), flush=True)
    (cache/'COMPLETE').write_text(f'{len(rows)} scene target maps\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--rays', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.rays < 3:
        p.error('workers must be positive and rays >= 3')
    precompute(args.data, args.rays, args.workers)


if __name__ == '__main__':
    main()
