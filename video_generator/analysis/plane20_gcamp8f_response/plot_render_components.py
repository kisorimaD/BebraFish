"""Visualize additive layers of an existing Plane20 GIF frame."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

from glare import redistribute_glare
from render_recorded_bank import HERE, PROFILE, REFERENCE_MEDIAN
from soma_maps import soma_maps
from optical_layers import add_optical_noise, halo_light, optical_layers
from spatial_amplitude import plane20_reference
from spatial_baseline import calibrate_baseline
from training.plane20.photometry import correlated_noise


def prepare_render(directory):
    meta = json.loads((directory / 'generation.json').read_text())
    with np.load(directory / meta['scene'], allow_pickle=False) as arrays:
        scene = dict(arrays)
    with np.load(directory / meta.get('profile', PROFILE), allow_pickle=False) as arrays:
        profile = dict(arrays)
    maps = soma_maps(scene, profile)
    original = scene['image'].astype(np.float32)
    reference_path = directory / meta.get('reference_median', REFERENCE_MEDIAN)
    reference_median = tifffile.imread(reference_path)
    optics = meta.get('optical_layers')

    base = original.copy()
    if meta['spatial_baseline'] is not None:
        with np.load(directory / meta['bank']) as bank:
            reference_centers = bank['centers_yx'][plane20_reference(bank, meta.get('reference_plane', 20))]
        base, _, _ = calibrate_baseline(base, reference_median, scene['centers'], reference_centers, maps)
    light = np.asarray(maps.sum(axis=0)).reshape(base.shape)
    static_soma = np.clip(light, 0, original)
    diffuse = original - static_soma
    glare = np.zeros_like(base)
    halo_static = None
    if optics is not None:
        maps, parts, _ = optical_layers(scene, profile, reference_median, maps,
                                        optics['background_glow_strength'])
        diffuse, glare, static_soma, halo_static = parts
        base = sum(parts)
    elif meta['glare'] is not None:
        with np.load(reference_path.parent / 'masks/instances.npz') as masks:
            base, config, removed, added = redistribute_glare(
                base, reference_median, masks['instance_pixel_index'], profile['tissue'].astype(bool),
                maps, meta['glare']['reduction_fraction'], meta['display_white_adu'], return_layers=True)
        original_glare = np.maximum(diffuse - config['diffuse_threshold_adu'], 0)
        diffuse -= original_glare
        glare = np.maximum(original_glare - removed, 0)
        static_soma = static_soma + added

    tone = meta['photometry']
    x, y = np.array(tone['mapping_input_adu']), np.array(tone['mapping_output_adu'])
    pivot = tone.get('moment_pivot_adu', 150.)
    def calibrated(values):
        mapped = values + np.interp(values, x, y - x)
        return (pivot + (mapped - pivot) * tone['fitted_moment_gain'] + tone['fitted_moment_offset_adu']).astype(np.float32)

    background_alone = calibrated(diffuse + glare)
    base = calibrated(base)
    corrections = base.astype(np.float64) - diffuse - glare - static_soma
    if halo_static is not None:
        corrections -= halo_static
    with np.load(directory / 'activity.npz') as arrays:
        light_traces = np.maximum(1 + arrays['dff'], 0)
    modulation = (light_traces / np.median(light_traces, axis=0) - 1).astype(np.float32)
    rng = np.random.default_rng(meta['detector_noise']['seed'])
    seeds = rng.integers(2**32, size=len(light_traces))
    halo_seeds = rng.integers(2**32, size=len(light_traces))
    mask = np.bincount(scene['instance_pixel_index'], minlength=base.size).reshape(base.shape) > 0
    return dict(meta=meta, base=base, maps=maps, diffuse=diffuse, glare=glare, soma=static_soma,
                corrections=corrections, halo_static=halo_static, modulation=modulation,
                seeds=seeds, halo_seeds=halo_seeds, mask=mask, background_alone=background_alone)


def render_layers(state, source_index):
    base = state['base']
    activity = (state['modulation'][source_index] @ state['maps']).reshape(base.shape)
    clean = activity + base
    config = state['meta']['detector_noise']
    seed = state['seeds'][source_index]
    halo, halo_noise = None, None
    if state['halo_static'] is not None:
        halo_delta = halo_light(activity)
        halo = state['halo_static'] + halo_delta
        clean += halo_delta
        halo_seed = state['halo_seeds'][source_index]
        noise, halo_noise = add_optical_noise(clean, halo, config['intensity_adu'],
                                             config['sigma_adu'], seed, halo_seed)
    else:
        unit, _ = correlated_noise(base.shape, seed)
        noise = unit * np.interp(clean, config['intensity_adu'], config['sigma_adu'])
    synthetic = clean + noise
    total = state['diffuse'].astype(np.float64) + state['glare'] + state['soma'] + activity + noise + state['corrections']
    if halo is not None:
        total += halo + halo_noise
        synthetic += halo_noise
    np.testing.assert_allclose(synthetic, total, atol=.01, rtol=1e-6)
    return synthetic, activity, noise, halo, halo_noise


def display_values(frame, black, white, transpose=True):
    oriented = frame.T if transpose else frame
    scaled = np.round((oriented.astype(np.float32) - black) * 255 / (white - black))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def reconstruct(directory, index):
    state = prepare_render(directory)
    meta = state['meta']
    source_index = meta['displayed_source_frames'][index]
    synthetic, activity, noise, halo, halo_noise = render_layers(state, source_index)
    lo, hi = meta['display_black_adu'], meta['display_white_adu']
    displayed = display_values(synthetic, lo, hi, meta.get('transpose_for_display', True)).astype(float)
    with Image.open(directory / 'plane20_recorded_activity.gif') as gif:
        gif.seek(index)
        decoded = np.asarray(gif.convert('L'))[meta['panel_height']:]
    error = float(np.mean(abs(decoded - displayed)))
    if error > 1.5:
        raise RuntimeError(f'Reconstructed frame differs from GIF by {error:.2f} grayscale levels.')
    with tifffile.TiffFile(directory / meta['real_source']) as recording:
        real = recording.pages[source_index].asarray()
    return (source_index, (lo, hi), error, real, synthetic, state['diffuse'], state['glare'], state['soma'],
            activity, state['corrections'], noise, halo, halo_noise)


def render_component_gif(directory, start, seconds, split):
    state = prepare_render(directory)
    meta = state['meta']
    fps = meta['playback_fps']
    indices = meta['displayed_source_frames']
    count = min(seconds * fps, len(indices) - start)
    if not 0 <= start < len(indices) or not 0 < count <= 60 * fps:
        raise ValueError('Choose an existing frame and a GIF length between 1 and 60 seconds.')
    width, height = meta['width'], meta['panel_height']
    black, white = meta['display_black_adu'], meta['display_white_adu']
    transpose = meta.get('transpose_for_display', True)
    filename = {'segmentation': 'plane20_segmentation_signal.gif',
                'optical': 'plane20_components.gif', 'signal': 'plane20_signal_residual.gif'}[split]
    output = directory / filename
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pixel_format', 'gray', '-video_size', f'{width}x{height * 4}',
               '-framerate', str(fps), '-i', 'pipe:0', '-filter_complex_threads', '1',
               '-filter_complex', 'split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none',
               '-loop', '0']
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        result = Path(staging) / output.name
        with Image.open(directory / 'plane20_recorded_activity.gif') as reference:
            with subprocess.Popen([*command, str(result)], stdin=subprocess.PIPE) as writer:
                with writer.stdin:
                    for index in range(start, start + count):
                        synthetic, activity, noise, halo, halo_noise = render_layers(state, indices[index])
                        neural = state['soma'] + activity
                        if split == 'segmentation':
                            neural = np.where(state['mask'], neural, 0)
                            remainder = synthetic - neural
                            assert not np.any(neural[~state['mask']])
                        elif split == 'signal':
                            # The counterfactual frame with all somas off uses
                            # the same global calibration. Its complement
                            # contains the full optical footprint and never
                            # slices the signal at an annotation boundary.
                            neural = (state['base'] - state['background_alone']).astype(np.float64) + activity
                            if halo is not None:
                                neural += halo - state['halo_static']
                                noise = noise + halo_noise
                            remainder = state['background_alone'] + noise
                        else:
                            if halo is not None:
                                neural = neural + halo
                                noise = noise + halo_noise
                            remainder = state['diffuse'] + state['glare'] + state['corrections'] + noise
                        np.testing.assert_allclose(neural + remainder, synthetic, atol=.01, rtol=1e-6)
                        reference.seek(index)
                        displayed = np.asarray(reference.convert('L'))
                        difference = abs(displayed[height:].astype(float) -
                                         display_values(synthetic, black, white, transpose))
                        if difference.mean() > 1.5:
                            raise RuntimeError(f'Reconstruction mismatch at GIF frame {index + 1}.')
                        panels = np.concatenate((displayed, display_values(neural, 0, white, transpose),
                                                 display_values(remainder, 0, white, transpose)), axis=0)
                        writer.stdin.write(panels.tobytes())
                if writer.wait():
                    raise RuntimeError('Component GIF encoding failed.')
        with Image.open(result) as gif:
            assert gif.n_frames == count and gif.size == (width, height * 4)
            duration = 0
            decoded_frames = np.empty((count, height * 4, width), dtype=np.uint8)
            for frame in range(count):
                gif.seek(frame)
                gif.load()
                duration += gif.info['duration']
                decoded_frames[frame] = np.asarray(gif.convert('L'))
            assert abs(duration - 1000 * count / fps) <= 20 and duration <= 60000
        median = np.rint(np.median(decoded_frames, axis=0)).astype(np.uint8)
        median_path = result.with_name(result.stem + '_median.png')
        Image.fromarray(median).save(median_path)
        median_path.replace(output.with_name(median_path.name))
        result.replace(output)
    print(f'{output}: {count} frames and {duration / 1000:.1f} seconds, verified additive decomposition.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE / 'movie_foreground_conservative')
    parser.add_argument('--frame', type=int, default=0, help='Zero-based GIF frame index.')
    parser.add_argument('--gif-seconds', type=int, choices=range(1, 61))
    parser.add_argument('--split', choices=('optical', 'segmentation', 'signal'), default='signal',
                        help='Choose an optical, annotation-boundary, or smooth signal/residual split.')
    args = parser.parse_args()
    if args.gif_seconds is not None:
        render_component_gif(args.directory, args.frame, args.gif_seconds, args.split)
        return
    (index, display, error, real, synthetic, diffuse, glare, somas,
     activity, correction, noise, halo, halo_noise) = reconstruct(args.directory, args.frame)
    components = [
        ('Реальный кадр', real, 'gray', *display),
        ('Синтетический кадр', synthetic, 'gray', *display),
        ('Фон ткани', diffuse, 'gray', float(np.quantile(diffuse, .01)), float(np.quantile(diffuse, .99))),
        ('Фоновое свечение вне сом', glare, 'inferno', 0,
         float(np.quantile(glare[glare > 0], .99)) if np.any(glare > 0) else 1),
        ('Постоянное свечение сом', somas, 'inferno', 0, float(np.quantile(somas, .995))),
    ]
    signed = [('Изменение яркости сом от активности', activity),
              ('Поправки яркости', correction), ('Основной шум кадра', noise)]
    if halo is not None:
        components.append(('Свечение вокруг сом', halo, 'inferno', 0, float(np.quantile(halo, .995))))
        signed.append(('Шум от свечения сом', halo_noise))
    for title, component in signed:
        limit = max(float(np.quantile(abs(component), .995)), .01)
        components.append((title, component, 'coolwarm', -limit, limit))
    rows = len(components) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 2.25 * rows), layout='constrained')
    for ax, (title, data, palette, low, high) in zip(axes.flat, components):
        image = ax.imshow(data.T, cmap=palette, vmin=low, vmax=high, interpolation='nearest')
        ax.set_title(title, loc='left')
        ax.axis('off')
        fig.colorbar(image, ax=ax, label='ADU', shrink=.88)
    fig.suptitle(f'Plane20 • кадр GIF {args.frame + 1}, отсчёт TIFF {index}')
    output = args.directory / f'render_components_frame_{args.frame + 1:03d}.png'
    fig.savefig(output, dpi=175)
    plt.close(fig)
    print(f'{output}: verified reconstruction, GIF grayscale MAE {error:.3f}/255')


if __name__ == '__main__':
    main()
