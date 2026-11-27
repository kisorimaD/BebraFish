"""Align sampled soma activity while preserving the values available in every trace."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import cdist

from sample_plane20_activity import (BANK, HERE, draw_figure, loss, serializable, spatial_pairs,
                                     stabilized_bank, statistics)


def standardized(values):
    return (values-values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)


def regional_fields(traces, centers, rng):
    # Carrier traces come from randomly chosen somata in this bank sample.
    # Separate time offsets keep the regional field from adding a second global pulse.
    _, anchor = np.unique((centers // 16).astype(int), axis=0, return_index=True)
    carriers = standardized(traces[:, rng.choice(traces.shape[1], len(anchor), replace=False)])
    offsets = rng.integers(len(traces), size=len(anchor))
    time = (np.arange(len(traces))[:, None] + offsets) % len(traces)
    carriers = np.take_along_axis(carriers, time, axis=0)
    squared_distance = cdist(centers[anchor], centers, 'sqeuclidean')
    fields = {}
    for scale in (12, 24, 45):
        weights = np.exp(-.5*squared_distance/scale**2).astype(traces.dtype)
        field = carriers @ weights
        field = standardized(field)
        fields[f'gaussian_{scale}px'] = standardized(field-np.median(field, axis=1, keepdims=True))
    mixed = .8*fields['gaussian_12px']+.6*fields['gaussian_45px']
    fields['mixed_12px_45px'] = standardized(mixed)
    return fields


def population_pulse(traces, percentile):
    # Gate the already occurring donor events, with no imposed recording timeline.
    population = 100*np.mean(traces > .1, axis=1)
    gate = np.quantile(population, percentile)
    scale = max(np.quantile(population, .99)-gate, 1e-3)
    return (np.clip((population-gate)/scale, 0, 3)[:, None]).astype(traces.dtype)


def align_rare_rises(traces, target_percent):
    """Move nearby strong observations onto the few existing population peaks."""
    frames, cells = traces.shape
    strength = np.count_nonzero(traces > .2, axis=1)
    goal = int(np.ceil(cells*target_percent/100))
    # The 99th percentile depends on the 11 highest frames in a 1000-frame trace.
    count = frames-int(np.floor(.99*(frames-1)))
    events = np.argsort(strength, kind='stable')[-count:][::-1]
    order = np.broadcast_to(np.arange(frames)[:, None], traces.shape).copy()
    adjusted = traces.copy()
    time = np.arange(frames)
    reserved = np.isin(time, events)
    moves = 0
    for event in events:
        need = goal-strength[event]
        if need <= 0:
            continue
        # Both observations exceed 10% ΔF/F. The swap leaves lower-threshold
        # population counts exactly as they were and preserves each soma's values.
        eligible = np.flatnonzero((adjusted[event] > .10) & (adjusted[event] <= .2))
        nearby = np.flatnonzero((np.abs(time-event) <= 50) & ~reserved)
        ready = adjusted[np.ix_(nearby, eligible)]
        if not len(eligible) or not len(nearby):
            continue
        distance = np.abs(nearby-event)[:, None]/50
        cost = np.where(ready > .2, distance + (ready-adjusted[event, eligible])/.15, np.inf)
        best_frame = np.argmin(cost, axis=0)
        best_cost = cost[best_frame, np.arange(len(eligible))]
        selected = np.argsort(best_cost, kind='stable')[:need]
        selected = selected[np.isfinite(best_cost[selected])]
        origins = nearby[best_frame[selected]]
        somas = eligible[selected]
        adjusted[event, somas], adjusted[origins, somas] = adjusted[origins, somas], adjusted[event, somas]
        order[event, somas], order[origins, somas] = order[origins, somas], order[event, somas]
        moves += len(somas)
    return adjusted, order, moves


def calibration_loss(result, target):
    p99 = result['coactivity'][-1, -1]
    tail = ((p99-target['coactivity'][-1, -1])/2.5)**2
    return loss(result, target) + .23*tail


def coordinated(traces, centers, pairs, target, seed):
    rng = np.random.default_rng(seed)
    ranks = np.argsort(traces, axis=0, kind='stable')
    values = np.take_along_axis(traces, ranks, axis=0)
    base = standardized(traces)
    fields = regional_fields(traces, centers, rng)
    pulses = {percentile: population_pulse(traces, percentile) for percentile in (.92, .96, .98)}
    before = statistics(traces, pairs)
    best_by_gain = {}

    def evaluate(name, regional_gain, gate, population_gain):
        key = population_gain > 0
        score = base + regional_gain*fields[name] + population_gain*pulses[gate]
        order = np.argsort(score, axis=0, kind='stable')
        output = np.empty_like(traces)
        np.put_along_axis(output, order, values, axis=0)
        summary = statistics(output, pairs)
        error = calibration_loss(summary[0], target)
        if key not in best_by_gain or error < best_by_gain[key][0]:
            best_by_gain[key] = (error, dict(field=name, regional_gain=regional_gain, pulse_percentile=gate,
                                             population_gain=population_gain), summary, order, output)

    for name in fields:
        for gain in (.25, .40, .55, .73):
            evaluate(name, gain, .96, 0.)
    choice = best_by_gain[False][1].copy()
    for gate in pulses:
        for gain in (.10, .24, .42, .65, .90, 1.25, 1.70, 2.30):
            evaluate(choice['field'], choice['regional_gain'], gate, gain)
    choice = best_by_gain[True][1].copy()
    for name in fields:
        for delta in (-.08, 0., .08):
            evaluate(name, choice['regional_gain']+delta,
                     choice['pulse_percentile'], choice['population_gain'])

    candidates = []
    for error, parameters, summary, order, output in best_by_gain.values():
        permutation = np.empty_like(ranks)
        np.put_along_axis(permutation, order, ranks, axis=0)
        candidates.append((error, {**parameters, 'nearby_value_swaps': 0}, summary, permutation, output))
        target_tail = target['coactivity'][-1, -1]
        if summary[0]['coactivity'][-1, -1] < target_tail - .3:
            adjusted, local_order, moves = align_rare_rises(output, target_tail)
            if moves:
                aligned = statistics(adjusted, pairs)
                candidate_indices = np.take_along_axis(permutation, local_order, axis=0)
                candidates.append((calibration_loss(aligned[0], target), {**parameters, 'nearby_value_swaps': moves},
                                   aligned, candidate_indices, adjusted))
    error, parameters, summary, permutation, output = min(candidates, key=lambda item: item[0])
    np.testing.assert_array_equal(output, np.take_along_axis(traces, permutation, axis=0))
    np.testing.assert_array_equal(np.sort(output, axis=0), values)
    return output, permutation, summary, before, parameters, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'activity_resamples')
    parser.add_argument('--output', type=Path, default=HERE/'activity_resamples_coordinated')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a new output directory.')

    source = json.loads((args.input/'comparison.json').read_text())
    bank = stabilized_bank(Path(source.get('bank', BANK)))
    plane = source.get('reference_plane', 20)
    ref = bank['plane'] == plane
    pairs = spatial_pairs(bank['centers_yx'][ref], np.random.default_rng(193))
    real = statistics(bank['dff'][:, ref], pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.activity-coordinate-') as tmp:
        stage = Path(tmp)/output.name
        stage.mkdir()
        rows, results, previous = [], [], []
        for sample in source['samples']:
            seed = int(sample['seed'])
            name = f'activity_seed_{seed}.npz'
            with np.load(args.input/name, allow_pickle=False) as arrays:
                activity = dict(arrays)
            centers = activity['centers_yx']
            pairs = spatial_pairs(centers, np.random.default_rng(194))
            traces, permutation, summary, before, parameters, score = coordinated(
                activity['dff'], centers, pairs, real[0], seed)
            for key in ('donor_frames', 'previous_donor_frames'):
                activity[key] = np.take_along_axis(activity[key], permutation, axis=0)
            activity['blend_weight'] = activity['blend_weight'][permutation]
            activity['input_common_source_blocks'] = activity.pop('common_source_blocks')
            activity['dff'] = traces
            activity['sample_frame_index'] = permutation.astype(np.int32)
            np.savez_compressed(stage/name, **activity)
            rows.append(dict(seed=seed, coordination=parameters, score=score,
                             before=serializable(before[0]), statistics=serializable(summary[0])))
            previous.append(before)
            results.append(summary)
            print(f'Seed {seed}: nearest r {before[0]["correlation"][0]:.3f} → {summary[0]["correlation"][0]:.3f}, '
                  f'P99 rise share {before[0]["coactivity"][-1,-1]:.2f}% → {summary[0]["coactivity"][-1,-1]:.2f}%.',
                  flush=True)
        report = {**source, 'samples': rows, 'input_resamples': str(args.input),
                  'method': 'Within-soma rank remapping with regional donor carriers and a population pulse.'}
        (stage/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        draw_figure(stage/'activity_comparison', real, results, before=previous, reference_label=f'Plane {plane}')
        stage.rename(output)
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
