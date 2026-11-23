"""Generate images and whole-soma GT with the frozen fish4 Plane16 profile."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'simcalc_zebrafish_pipeline'))
from calibrated_simcalc import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/generated')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    print(generate(ROOT / 'profiles/fish4_plane16', args.output, args.count, args.seed))


if __name__ == '__main__':
    main()
