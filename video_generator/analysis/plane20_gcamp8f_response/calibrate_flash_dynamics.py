"""Calibrate coordinated fluorescence dynamics without changing the source render."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import tifffile

from plot_render_components import display_values, prepare_render, render_layers
from optical_layers import add_optical_noise, halo_light
from render_recorded_bank import HERE, PROFILE

BLOCK = 50
TILE = 32


def raw_signal(root, plane, ids, full):
    with np.load(root / f'plane_{plane}/traces.npz') as traces:
        fluorescence = np.where(full, traces['full_mask_F_adu'][:, ids], traces['F_adu'][:, ids])
    return fluorescence


def median_center(values):
    return values - np.median(values, axis=0)


def stabilize_background(raw, corrected, train):
    baseline = np.median(raw[train], axis=0)
    relative = raw / baseline
    # The bank stores (raw - background - median) / the full-series raw baseline.
    corrected = corrected * (np.median(raw, axis=0) / baseline)
    inferred = median_center(relative - corrected)
    adjacent = train[1:] == train[:-1] + 1
    delta_raw = np.diff(relative[train], axis=0)[adjacent]
    delta_background = np.diff(inferred[train], axis=0)[adjacent]
    limit = np.maximum(np.quantile(np.abs(delta_background), .85, axis=0), .001)
    delta_background = np.clip(delta_background, -limit, limit)
    target_limit = np.maximum(np.quantile(np.abs(delta_raw), .85, axis=0), .001)
    delta_raw = np.clip(delta_raw, -target_limit, target_limit)
    variance = np.sum(delta_background**2, axis=0)
    slope = np.divide(np.sum(delta_background * delta_raw, axis=0), variance,
                      out=np.zeros_like(variance), where=variance > 0)
    slope = np.clip(slope, 0., 1.)
    relative = relative - np.median(relative[train], axis=0)
    values = relative - slope * (inferred - np.median(inferred[train], axis=0))
    values -= np.median(values[train], axis=0)
    # Preserve observed darkening in the raw signal and limit extrapolation
    # from subtracting an unusually bright annulus.
    raw_tail = np.quantile(relative[train], .005, axis=0)
    return np.maximum(values, np.minimum(relative, 1.25 * raw_tail)), slope


def sample_blocks(indices, train_starts, group, rng):
    blocks = (len(indices) + BLOCK - 1) // BLOCK
    planned = np.concatenate([rng.permutation(train_starts) for _ in
                              range((blocks + len(train_starts) - 1) // len(train_starts))])[:blocks]
    offsets = rng.integers(-2, 3, size=group.max() + 1)
    offsets = offsets[group]
    positions = np.arange(len(indices)) % BLOCK
    windows = np.arange(len(indices)) // BLOCK
    timeline = planned[windows, None] + np.clip(positions[:, None] + offsets, 0, BLOCK - 1)
    return planned[windows] + positions, timeline


def read_reference(meta, source_dir, bank, reference, train, shape):
    ids = bank['mask_id'][reference]
    reference_plane = int(meta.get('reference_plane', 20))
    with np.load(source_dir / f'plane_{reference_plane}/traces.npz') as traces:
        annuli = traces['background_adu'][:, ids]
    usable = np.isfinite(annuli).all(axis=0)
    centers = bank['centers_yx'][reference][usable]
    annuli = annuli[:, usable] - np.median(annuli[train][:, usable], axis=0)
    radius = 1.1
    grid_shape = tuple(v // TILE for v in shape)
    grid_yx = np.minimum((centers / TILE).astype(int), np.array(grid_shape) - 1)
    flat = np.ravel_multi_index(grid_yx.T, grid_shape)
    count = np.bincount(flat, minlength=np.prod(grid_shape)).reshape(grid_shape)
    pooled = np.array([np.bincount(flat, weights=frame, minlength=count.size) for frame in annuli])
    pooled = pooled.reshape(-1, *grid_shape).astype(np.float32)
    count = gaussian_filter(count.astype('float32'), radius)
    pooled = gaussian_filter(pooled, (0, radius, radius))
    common = np.median(annuli, axis=1)
    # Shrink poorly sampled tiles towards the common background change.
    field = (pooled + 2 * common[:, None, None]) / (count + 2)
    return field


def tile_mean(values):
    height, width = (v // TILE for v in values.shape)
    return values[:height * TILE, :width * TILE].reshape(height, TILE, width, TILE).mean(axis=(1, 3))


def mask_matrix(scene):
    pixels = scene['instance_pixel_index']
    offsets = scene['instance_offsets'].astype(int)
    weights = np.repeat(1 / np.diff(offsets), np.diff(offsets)).astype(np.float32)
    return csr_matrix((weights, pixels, offsets), shape=(len(offsets) - 1, scene['image'].size))


def temporal_amplitude(values):
    return float(np.median(np.quantile(values, .95, axis=0) - np.median(values, axis=0)))


def render_modulation(values, gain):
    # The negative side is calibrated against the uncorrected fluorescence
    # tail. The gain restores weak positive events diluted by image background.
    return np.maximum(values + (gain-1) * np.maximum(values, 0), -1.)


def dark_weights(state):
    low, high = (state['meta'][key] for key in ('display_black_adu', 'display_white_adu'))
    return np.clip(1 - (state['base']-low)/max(1., .18*(high-low)), 0, 1)


def calibrated_baselines(state, tone, offset, dark_offset=0.):
    base = state['base']
    background = state['background_alone']
    calibrated = 150 + tone*(base-150) + offset
    # The diffuse baseline is the calibrated image with somata switched off.
    somas = tone*(base-background)
    diffuse = 150 + tone*(background-150) + offset
    if dark_offset:
        adjustment = dark_offset*dark_weights(state)
        calibrated += adjustment
        diffuse += adjustment
    return calibrated, somas, diffuse


def render_flash_frame(state, index, signal, dynamic_halo, broad_change, baselines,
                       grain_correction=None):
    base, soma_base, background_base = baselines
    clean = base + signal + broad_change
    halo = state['halo_static'] + dynamic_halo
    config = state['meta']['detector_noise']
    noise, halo_noise = add_optical_noise(clean, halo, config['intensity_adu'],
                                         config['sigma_adu'], state['seeds'][index], state['halo_seeds'][index])
    if grain_correction is not None:
        correction = np.interp(clean, grain_correction['intensity_adu'], grain_correction['scale'])
        noise *= correction
        halo_noise *= correction
    synthetic = clean + noise + halo_noise
    return synthetic, soma_base + signal, background_base + broad_change + noise + halo_noise


def prepare_flash_replay(source, activity_dir):
    state = prepare_render(source)
    metadata = state['meta']
    config = json.loads((activity_dir / 'validation.json').read_text())
    with np.load(activity_dir / 'activity.npz', allow_pickle=False) as arrays:
        modulation = arrays['render_modulation']
        diffuse_tiles = arrays['diffuse_tile_changes']
    with np.load(source / metadata.get('profile', PROFILE), allow_pickle=False) as profile:
        tissue = profile['tissue'].astype(np.float32)
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coordinates = ((y + .5)/TILE-.5, (x + .5)/TILE-.5)
    support = gaussian_filter(tissue, 12)
    baselines = calibrated_baselines(state, config['static_gain'], config['static_offset_adu'],
                                    config.get('dark_offset_adu', 0.))

    def render(index):
        if not 0 <= index < len(modulation):
            raise ValueError(f'Frame {index} is outside the saved activity.')
        diffuse = map_coordinates(diffuse_tiles[index], coordinates, order=1, mode='nearest')
        diffuse *= support
        diffuse *= config['diffuse_gain']
        activity = (modulation[index] @ state['maps']).reshape(tissue.shape)
        halo = halo_light(activity)
        return render_flash_frame(state, index, activity+halo, halo, diffuse, baselines,
                                  config.get('grain_correction'))

    return config, metadata, render


def replay_frame(source, activity_dir, index):
    _, _, render = prepare_flash_replay(source, activity_dir)
    return render(index)


def flash_metrics(tiles, rois, pixels, low, high):
    delta = median_center(tiles)
    relative = median_center(rois) / np.median(rois, axis=0)
    fraction = np.mean(relative > .2, axis=1)
    shown = np.clip(pixels, low, high)
    return dict(roi_median_p95_minus_p50_adu=temporal_amplitude(rois),
                roi_simultaneous_up_20pct_p95=float(100 * np.quantile(fraction, .95)),
                roi_simultaneous_up_20pct_max=float(100 * fraction.max()),
                tile_delta_p99_adu=float(np.quantile(delta, .99)),
                pct_frames_with_wide_flash=float(100 * np.mean(np.mean(delta >= 10, axis=1) >= .1)),
                displayed_mean_adu=float(shown.mean()), displayed_std_adu=float(shown.std()),
                displayed_quantiles_adu=np.quantile(shown, [.02, .05, .1, .25, .5, .75, .9, .95, .98, .995]).tolist(),
                pct_clipped_black=float(100 * np.mean(pixels <= low)),
                pct_clipped_white=float(100 * np.mean(pixels >= high)))


def grain_metrics(left, right, low, high, edges=None):
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    median_left = np.median(left, axis=0)
    median_right = np.median(right, axis=0)
    level = (median_left+median_right)/2
    difference = left-right
    deviation = 1.4826*np.median(np.abs(difference-np.median(difference,axis=0)),axis=0)
    valid = np.abs(median_left-median_right)<12
    if edges is None:
        edges = [max(115,low),150,185,220,260,min(310,high)]
    return [dict(range_adu=[a,b],pairs=int(pairs.sum()),
                 robust_sd_adu=float(np.median(deviation[pairs])) if pairs.any() else None)
            for a,b in zip(edges,edges[1:]) if b > a
            for pairs in [valid & (level>=a) & (level<b)]]


def make_gif(path, dimensions):
    width, height = dimensions
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                             '-f', 'rawvideo', '-pixel_format', 'gray', '-video_size', f'{width}x{height}',
                             '-framerate', '3', '-i', 'pipe:0', '-filter_complex_threads', '1',
                             '-filter_complex',
                             'split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none',
                             '-loop', '0', str(path)], stdin=subprocess.PIPE)


def finish_gif(writer, path, frames):
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError(f'GIF encoding failed: {path}')
    with Image.open(path) as gif:
        assert gif.n_frames == frames
        duration = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            duration += gif.info['duration']
        assert abs(duration - frames / 3 * 1000) <= 10 and duration <= 60000


def save_temporal_median(path):
    with Image.open(path) as gif:
        frames = np.empty((gif.n_frames, gif.height, gif.width), dtype=np.uint8)
        for index in range(gif.n_frames):
            gif.seek(index)
            frames[index] = np.asarray(gif.convert('L'))
    median = np.rint(np.median(frames, axis=0, overwrite_input=True)).astype(np.uint8)
    Image.fromarray(median).save(path.with_name(path.stem + '_median.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE/'movie_foreground_conservative')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261126)
    parser.add_argument('--replay-frame', type=int, help='Write this saved synthetic frame to a TIFF.')
    parser.add_argument('--activity-dir', type=Path, help='Directory containing saved flash activity and validation.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path.')
    if args.replay_frame is not None:
        image, _, _ = replay_frame(args.source, args.activity_dir or args.source, args.replay_frame)
        tifffile.imwrite(args.output, image.astype(np.float32))
        return
    state = prepare_render(args.source)
    meta = state['meta']
    if state['halo_static'] is None:
        raise ValueError('Expected the current optical render with soma halos.')
    profile = args.source / meta.get('profile', PROFILE)
    with np.load(profile) as p:
        tissue = p['tissue'].astype(bool)
    with np.load(args.source / meta['scene']) as f:
        scene = dict(f)
    with np.load(args.source / 'activity.npz') as f:
        before_traces, columns = f['dff'], f['bank_columns']
    with np.load(args.source / meta['bank']) as f:
        bank = dict(f)
    source_dir = Path(meta['bank']).parent.parent / 'per_plane'
    reference_plane = int(meta.get('reference_plane', 20))
    ref = bank['plane'] == reference_plane
    corrected = bank['dff'][:, columns]
    frames, cells = corrected.shape
    starts = np.arange(0, frames - BLOCK + 1, BLOCK)
    # Reverse the assignment in successive pairs, so a repeating event
    # cadence cannot consistently fall into only one side of the split.
    train_starts = starts[np.isin(np.arange(len(starts)) % 4, [0,3])]
    if len(train_starts) < 3:
        raise ValueError('Need at least three independent training windows.')
    block_start = np.arange(starts[-1] + BLOCK)//BLOCK*BLOCK
    train = np.flatnonzero(np.isin(block_start,train_starts))
    test = np.flatnonzero(~np.isin(block_start,train_starts))
    donors = bank['plane'][columns]
    raw = np.empty_like(corrected)
    for plane in np.unique(donors):
        which = donors == plane
        indices = columns[which]
        raw[:, which] = raw_signal(source_dir, plane, bank['mask_id'][indices], bank['full_mask'][indices])
    stable, slopes = stabilize_background(raw, corrected, train)
    # A common window preserves broad events. Somata in the same spatial
    # group use the same small delay relative to that window.
    _, group = np.unique((scene['centers'] / 48).astype(int), axis=0, return_inverse=True)
    rng = np.random.default_rng(args.seed)
    shared_frames = np.empty(frames, dtype=int)
    grouped_frames = np.empty((frames, cells), dtype=np.int32)
    previous_frames = np.empty_like(grouped_frames)
    weights = np.ones(frames, np.float32)
    sampled = np.empty_like(stable)
    for split in (train, test):
        timeline, local = sample_blocks(split, train_starts, group, rng)
        shared_frames[split] = timeline
        grouped_frames[split] = local
    unused = np.setdiff1d(np.arange(frames), np.r_[train, test])
    if len(unused):
        grouped_frames[unused] = grouped_frames[train[-1]]
        shared_frames[unused] = shared_frames[train[-1]]
    previous_frames[:] = grouped_frames
    for start in range(BLOCK, frames, BLOCK):
        count = min(3,frames-start)
        previous_frames[start:start+count] = grouped_frames[start-1]
        weights[start:start+count] = np.arange(1,count+1)/4
    sampled[:] = (weights[:,None] * stable[grouped_frames, np.arange(cells)] +
                  (1-weights[:,None]) * stable[previous_frames, np.arange(cells)])
    assert np.isin(grouped_frames, train).all() and np.isin(shared_frames, train).all()
    sampled -= np.median(sampled[train], axis=0)
    broad = read_reference(meta, source_dir, bank, ref, train, tissue.shape)
    background = broad[shared_frames].copy()
    # Remap the coarse field around each sampled source window. Preserve its
    # broad time dependence without replaying native background pixels.
    for i in range(0, frames, BLOCK):
        block = range(i, min(i+BLOCK, frames))
        shift = rng.uniform(-.75, .75, size=2)
        yy, xx = np.indices(background.shape[1:], dtype=np.float32)
        background[block] = np.array([map_coordinates(frame, (yy+shift[0], xx+shift[1]),
                                      order=1, mode='nearest') for frame in background[block]])
        if i:
            s = slice(i,i+3)
            blend = weights[s,None,None]
            background[s] = blend*background[s] + (1-blend)*background[i-1]
    # Preserve the existing static luminance as a starting point.
    background -= np.mean(background[train], axis=0)
    y, x = np.indices(tissue.shape, dtype=np.float32)
    coordinates = ((y + .5)/TILE-.5, (x + .5)/TILE-.5)
    field_support = gaussian_filter(tissue.astype(np.float32), 12)

    def diffuse(i):
        return map_coordinates(background[i], coordinates, order=1, mode='nearest') * field_support

    def dynamic(values):
        activity = (values @ state['maps']).reshape(tissue.shape)
        halo = halo_light(activity)
        return activity + halo, halo

    masks = mask_matrix(scene)
    base = state['base']
    baseline_roi = masks @ base.ravel()
    valid_tiles = tile_mean(tissue) > .7
    baseline_tile = tile_mean(base)[valid_tiles]
    pixel_pool = np.flatnonzero((np.arange(base.size) % base.shape[1]) < base.shape[1]-1)
    pixels = rng.choice(pixel_pool, min(14000, len(pixel_pool)), replace=False)
    low, high = meta['display_black_adu'], meta['display_white_adu']
    fit = train[np.linspace(0, len(train)-1, min(140,len(train)), dtype=int)]
    held = test[np.linspace(0, len(test)-1, min(220,len(test)), dtype=int)]
    reference_roi = raw_signal(source_dir, reference_plane, bank['mask_id'][ref],
                               np.ones(int(ref.sum()), dtype=bool))
    with tifffile.TiffFile(meta['real_source']) as tif:
        def observation(i):
            im = tif.pages[int(i)].asarray()
            return im, tile_mean(im)[valid_tiles], im.ravel()[pixels]

        def projections(im):
            return masks @ im.ravel(), tile_mean(im)[valid_tiles], im.ravel()[pixels]

        def arrays(rows):
            return tuple(np.asarray(v) for v in zip(*rows))

        fit_real_tile, fit_real_pixels = arrays(observation(i)[1:] for i in fit)
        somatic_roi, somatic_tile, somatic_pixels = arrays(
            projections(dynamic(sampled[i])[0]) for i in fit)
        positive_roi, positive_tile, positive_pixels = arrays(
            projections(dynamic(np.maximum(sampled[i],0))[0]) for i in fit)
        broad_roi, broad_tile, broad_pixels = arrays(projections(diffuse(i)) for i in fit)
        print('Fitting somatic and broad changes on training frames.',flush=True)
        real_metrics = flash_metrics(fit_real_tile, reference_roi[fit], fit_real_pixels, low, high)
        config = meta['detector_noise']
        base_pixels = base.ravel()[pixels]
        noise_rng = np.random.default_rng(args.seed + 1)
        pixel_noise = noise_rng.normal(size=somatic_pixels.shape) * np.interp(
            base_pixels, config['intensity_adu'], config['sigma_adu'])
        roi_sd = (masks @ np.interp(base.ravel(), config['intensity_adu'], config['sigma_adu']))
        areas = np.maximum(1, np.diff(masks.indptr))
        roi_noise = noise_rng.normal(size=somatic_roi.shape) * roi_sd * np.sqrt(2/areas)
        tile_noise = noise_rng.normal(0, .6, size=somatic_tile.shape)
        fit_metrics = ('roi_median_p95_minus_p50_adu','tile_delta_p99_adu',
                       'displayed_mean_adu','displayed_std_adu','pct_clipped_black','pct_clipped_white',
                       'pct_frames_with_wide_flash','roi_simultaneous_up_20pct_p95')
        target = np.array([real_metrics[key] for key in fit_metrics])

        def loss(parameters):
            gs, gb, gain, offset = parameters
            roi = baseline_roi + somatic_roi + (gs-1)*positive_roi + gb*broad_roi + roi_noise
            tile = baseline_tile + somatic_tile + (gs-1)*positive_tile + gb*broad_tile + tile_noise
            intensity = (150 + gain*(base_pixels-150) + offset + somatic_pixels +
                         (gs-1)*positive_pixels + gb*broad_pixels + pixel_noise)
            shown = intensity.clip(low, high)
            tile_delta = median_center(tile)
            roi_relative = median_center(roi)/np.median(roi,axis=0)
            metrics = np.array([temporal_amplitude(roi),
                                np.quantile(tile_delta,.99),
                                shown.mean(),shown.std(),
                                100*np.mean(intensity<=low),100*np.mean(intensity>=high),
                                100*np.mean(np.mean(tile_delta>=10,axis=1)>=.1),
                                100*np.quantile(np.mean(roi_relative>.2,axis=1),.95)])
            return np.sum(((metrics-target)/[2.,3.,1.,1.5,2.,.7,5.,2.])**2)

        fit_gains = minimize(loss, [3.,1.,1.,0.], method='Powell',
                             bounds=[(.1,9.),(.1,3.),(.8,1.2),(-12.,12.)],
                             options={'maxiter':12,'xtol':.025,'ftol':.008})
        gs, gb, tone, offset = fit_gains.x
        print('Fitted gains:',fit_gains.x,flush=True)
        baselines = calibrated_baselines(state,tone,offset)
        somatic_base = base - state['background_alone']
        modulation = render_modulation(sampled, gs)

        def render(i):
            signal, dynamic_halo = dynamic(modulation[i])
            return render_flash_frame(state,i,signal,dynamic_halo,gb*diffuse(i),baselines)

        width, height = meta['width'],meta['panel_height']
        transpose = meta.get('transpose_for_display',True)
        gif_indices = meta['displayed_source_frames'][:180]
        diagnostic_frames = min(45,len(gif_indices))
        print('Rendering GIFs and measuring both versions on held-out frames.',flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
            staged = Path(temporary)/args.output.name
            staged.mkdir()
            comp = staged/'real_vs_synthetic.gif'
            diag = staged/'signal_and_background.gif'
            writers = [make_gif(comp,(width,2*height)),make_gif(diag,(width,4*height))]
            measured_old, measured_new, measured_ref = [], [], []
            clean_old, clean_new = [], []
            neighbors = {name: [] for name in ('real','old','new')}
            view = {frame:i for i,frame in enumerate(gif_indices)}
            hold = set(held)
            for i in sorted(hold|view.keys()):
                syn, neuron, other = render(i)
                if i in hold:
                    old, old_activity, _, old_halo, _ = render_layers(state,i)
                    old_roi, old_tile, old_pixels = projections(old)
                    new_roi, new_tile, new_pixels = projections(syn)
                    measured_old.append((old_tile,old_roi,old_pixels))
                    measured_new.append((new_tile,new_roi,new_pixels))
                    real_im, real_tile, real_pixels = observation(i)
                    measured_ref.append((real_tile,reference_roi[i],real_pixels))
                    clean_old.append(masks @ (somatic_base+old_activity+old_halo-state['halo_static']).ravel())
                    clean_new.append(masks @ neuron.ravel())
                    for key, im in (('real',real_im),('old',old),('new',syn)):
                        neighbors[key].append(im.ravel()[pixels+1])
                if i in view:
                    real = tif.pages[i].asarray()
                    pairs = [display_values(a,low,high,transpose) for a in (real,syn)]
                    writers[0].stdin.write(np.concatenate(pairs).tobytes())
                    if view[i] < diagnostic_frames:
                        np.testing.assert_allclose(neuron+other,syn,atol=.001)
                        pairs += [display_values(a,0,high,transpose) for a in (neuron,other)]
                        writers[1].stdin.write(np.concatenate(pairs).tobytes())
            finish_gif(writers[0],comp,len(gif_indices))
            finish_gif(writers[1],diag,diagnostic_frames)
            save_temporal_median(diag)
            averaged = {name:flash_metrics(*arrays(values),low,high) for name,values in
                         (('real',measured_ref),('old',measured_old),('new',measured_new))}
            for name, values in (('real',measured_ref),('old',measured_old),('new',measured_new)):
                averaged[name]['grain_by_brightness'] = grain_metrics(
                    np.array([row[2] for row in values]), np.array(neighbors[name]),low,high)
            ref_raw = reference_roi / np.median(reference_roi[train],axis=0) - 1
            def negative(values):
                return float(100*np.mean(values<-.2))
            def synchrony(values):
                return float(100*np.quantile(np.mean(values>.2,axis=1),.95))
            report = dict(seed=args.seed, training_source_blocks=train_starts.tolist(),
                          heldout_reference_frames=held.tolist(), reference_gif_frames=gif_indices,
                          somatic_gain=float(gs), diffuse_gain=float(gb), static_gain=float(tone),
                          static_offset_adu=float(offset), correction_beta_quantiles=np.quantile(slopes,[.1,.5,.9]).tolist(),
                          component_split='Static contrast gain applies to both the soma and diffuse baselines.',
                          pct_samples_below_minus_20={'real_roi_raw':negative(ref_raw[held]),
                              'real_roi_corrected_old':negative(bank['dff'][held][:,ref]),
                              'old_assigned_traces':negative(before_traces[held]),
                              'new_assigned_traces':negative(sampled[held]),
                              'new_render_modulation':negative(modulation[held])},
                          pct_synchronous_rises_above_20_p95={'real_roi_corrected_old':synchrony(bank['dff'][held][:,ref]),
                              'old_assigned_traces':synchrony(before_traces[held]),'new_assigned_traces':synchrony(sampled[held]),
                              'new_render_modulation':synchrony(modulation[held])},
                          clean_soma_roi_median_p95_minus_p50_adu={
                              'old':temporal_amplitude(clean_old),'new':temporal_amplitude(clean_new)},
                          training_targets=dict(zip(fit_metrics,target.tolist())),
                          training_objective=float(fit_gains.fun), heldout=averaged)
            np.savez_compressed(staged/'activity.npz', dff=sampled, render_modulation=modulation,
                                bank_columns=columns, group_id=group, donor_frames=grouped_frames,
                                previous_donor_frames=previous_frames, blend_weight=weights,
                                diffuse_sample_frames=shared_frames, diffuse_tile_changes=background)
            (staged/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            staged.replace(args.output)
    print(f'Finished: {args.output}',flush=True)


if __name__ == '__main__':
    main()
