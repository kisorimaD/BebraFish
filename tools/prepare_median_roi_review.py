"""Prepare an independent review card for each filtered Plane20 median mask."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from review_somas import ROOT, ReviewSession, atomic_write, read_frame

MIX = ROOT/'outputs/plane20_mixed_training'
PROBE = MIX/'wide64/roi_review'
MEDIAN = ROOT/'outputs/plane20_median_resolution/local.tiff'
SESSION = ROOT/'annotations/plane20_median_filtered_roi_review'
KEYS = ('centers', 'scores', 'instance_offsets', 'instance_pixel_index', 'shape_yx')


def load_npz(path):
    with np.load(path) as z:
        return {key:z[key] for key in KEYS}


def bind_subset(original, screened, destination, name, filter_settings):
    full = original/'instances.npz'
    original_report = json.loads((original/'report.json').read_text())
    source = load_npz(full)
    subset = load_npz(screened)
    lookup = {tuple(center):i for i,center in enumerate(source['centers'])}
    indices = np.array([lookup[tuple(center)] for center in subset['centers']], np.int32)
    valid = (len(np.unique(indices)) == len(indices)
             and np.array_equal(subset['scores'], source['scores'][indices])
             and np.array_equal(subset['shape_yx'], source['shape_yx']))
    if not valid:
        raise ValueError(f'Filtered ROI do not match source predictions: {screened}')
    original_offsets, filtered_offsets = source['instance_offsets'],subset['instance_offsets']
    for i,j in enumerate(indices):
        before = source['instance_pixel_index'][original_offsets[j]:original_offsets[j+1]]
        after = subset['instance_pixel_index'][filtered_offsets[i]:filtered_offsets[i+1]]
        if not np.array_equal(before,after):
            raise ValueError(f'Mask differs from source: {screened}, index {i}')
    destination.mkdir(parents=True,exist_ok=True)
    target = destination/'instances.npz'
    np.savez_compressed(target,**subset,source_indices=indices)
    atomic_write(destination/'report.json',original_report | {
        'name':name, 'instances':len(indices), 'source_instances':len(source['centers']),
        'source_predictions_sha256':hashlib.sha256(full.read_bytes()).hexdigest(),
        'filter':filter_settings,
    })
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=SESSION)
    parser.add_argument('--inputs',type=Path,default=PROBE/'review_inputs')
    args = parser.parse_args()
    if args.session.exists():
        parser.error(f'Сессия уже существует: {args.session}')
    summary = json.loads((PROBE/'roi_filter_report.json').read_text())
    black = dict(maximum_neighborhood_p95_adu=summary['black_limit_adu'])
    rim = summary['wide64_screened_example']
    paths = [
        bind_subset(MIX/'pilot/median_prediction',PROBE/'pilot_black_filtered.npz',
                    args.inputs/'pilot','Пилот · фильтр фона',black),
        bind_subset(MIX/'wide64/analysis/median_tuned',PROBE/'wide64_screened.npz',
                    args.inputs/'wide64','Wide64 · фильтр оболочки',
                    black | dict(weak_shell=dict(score_below=rim['score_below'],relief_below_adu=rim['relief_below_adu']))),
    ]
    data, frame = read_frame(MEDIAN,0)
    prior = ROOT/'annotations/Plane20_median_registered_uint16_frame0_776e568f5575.json'
    support = ROOT/'outputs/plane20_masked_background/synthetic.npz'
    session = ReviewSession(args.session,data,frame,paths,prior,support,review_rois=True)
    print(f'{len(session.groups)} ROI · {session.path}')


if __name__ == '__main__':
    main()
