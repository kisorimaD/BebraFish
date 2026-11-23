"""Generate paired density x additive-noise ablation without changing the old dataset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import hashlib
import json
from pathlib import Path
import numpy as np
from .generate import generate_one

VARIANTS = ('normal_noisy', 'normal_clean', 'half_noisy', 'half_clean')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--val', type=int, default=8)
    p.add_argument('--test', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261113)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if min(args.train, args.val, args.test, args.workers) < 1:
        p.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | dict(schema='plane20-paired-ablation-v1',
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        variants=VARIANTS, low_density_fraction=.5,
        policy='Paired scenes; exact random half-subset after morphology and photometry sampling. '
               'Same per-cell amplitudes, optics, focus, background, normalizer and noise field. '
               'Clean removes only explicit additive correlated noise, not background or blur. '
               'Same crop locations use full-density sampling_centers. P5-P95 normalization unchanged.')
    jobs = []
    for variant in VARIANTS:
        for split in ('train', 'val', 'test'):
            (args.output/variant/split).mkdir(parents=True)
        (args.output/variant/'config.json').write_text(json.dumps(config | dict(variant=variant), default=str, indent=2)+'\n')
    (args.output/'config.json').write_text(json.dumps(config, default=str, indent=2)+'\n')
    for split, count in [('train', args.train), ('val', args.val), ('test', args.test)]:
        for i in range(count):
            seed = int(np.random.SeedSequence([args.seed, len(jobs)]).generate_state(1, dtype=np.uint64)[0])
            jobs.append((split, i, seed, str(args.output)))
    with ExitStack() as stack:
        manifests = {v: stack.enter_context((args.output/v/'manifest.jsonl').open('w')) for v in VARIANTS}
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers))
        for i, rows in enumerate(pool.map(partial(generate_one, profile_path=args.profile, ablation=True), jobs)):
            for row in rows:
                manifests[row['variant']].write(json.dumps(row)+'\n')
                manifests[row['variant']].flush()
            print(f'{i+1}/{len(jobs)} paired scenes: '+json.dumps({r['variant']: r['cells'] for r in rows}), flush=True)
    for variant in VARIANTS:
        (args.output/variant/'COMPLETE').write_text(f'{len(jobs)} scenes\n')
    (args.output/'COMPLETE').write_text(f'{len(jobs)} paired scenes, four variants\n')


if __name__ == '__main__':
    main()
