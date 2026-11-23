"""SimCalc cell kernels with fitted ADU illumination and effective camera noise.

No histogram remapping after the detector; no neuropil or real cell texture.
The fitted field is coarse fluorescence/illumination, not anatomical GT.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from hybrid_high_model import CELL_DEFAULTS
from simcalc_core import _build_scene, sample_head_centres
from dataset_io import file_hash


def detector(clean, background, gain, variance_floor, rng, excess_variance=0.):
    signal = np.maximum(clean - background, 0)
    if gain > 1e-8:
        shot = rng.poisson(signal / gain) * gain - signal
    else:
        shot = np.zeros_like(signal)
    return clean + shot + rng.normal(0, np.sqrt(variance_floor+excess_variance*signal**2), clean.shape)


def render(profile, settings, seed, *, count=None, sigma=None, ring=None, temporal_gain=None):
    rng = np.random.default_rng(seed)
    mask = profile['tissue'].astype(bool)
    sigma = float(settings['sigma_px'] if sigma is None else sigma)
    ring = float(settings['ring_fraction'] if ring is None else ring)
    count = int(settings['cell_count'] if count is None else count)
    support = distance_transform_edt(mask) > max(2., 1.5*sigma)
    centers = sample_head_centres(support, count, max(2., sigma), rng, profile['density'])
    p = CELL_DEFAULTS | dict(shape_yx=list(mask.shape), sigma_px=sigma,
        size_cv=settings['size_cv'], amplitude_cv=settings['amplitude_cv'],
        eccentricity=settings['eccentricity'], ring_fraction=ring,
        blur_px=settings['blur_px'], background=0., background_texture=0., neuropil_filters=0)
    p['fill_soma_masks']=settings.get('gt_mode','fluorescence')=='soma'
    p['keep_instance_footprints']=p['fill_soma_masks']
    p['ring_center_fill']=settings.get('ring_center_fill',0.)
    scene = _build_scene(p, seed, centres=centers, tissue=profile['tissue_field'],
                         emission_gain=profile['cell_gain'])
    signal = scene['signal']
    # Match high-pass contrast in ADU, with camera contribution subtracted.
    hp = signal - gaussian_filter(signal, 4.)
    intrinsic = float(np.std(hp[mask]))
    amplitude = float(settings['cell_contrast_std_adu']) / max(intrinsic, 1e-5)
    amplitude *= float(settings.get('contrast_scale', 1.))
    # Compensate only the expected coarse cell contribution, not each rendered cell.
    area = max(int(support.sum()), 1)
    unit_integral = float(signal.sum()) / count
    expected = amplitude * count / area * unit_integral * profile['density'] * profile['cell_gain']
    expected = gaussian_filter(expected*profile['tissue_field'],float(profile.get('compensation_sigma_px',32.)))
    diffuse = np.maximum(profile['diffuse_target'] - expected, 0)
    temporal_gain = float(rng.choice(profile['temporal_gains'])) if temporal_gain is None else float(temporal_gain)
    clean = profile['background'] + temporal_gain * (amplitude*signal + diffuse)
    noisy = detector(clean, profile['background'], settings['shot_gain_adu'],
                     settings['read_variance_adu2'], np.random.default_rng(seed+1),
                     settings.get('excess_variance_fraction',0.))
    image = np.rint(np.clip(noisy,0,65535)).astype(np.uint16)
    result = dict(image=image, clean=clean.astype(np.float32), labels=scene['labels'],
                tissue=mask, centers=centers.astype(np.float32),
                seed=seed, count=count, sigma_px=sigma, ring_fraction=ring,
                amplitude_adu=amplitude, temporal_gain=temporal_gain)
    for key in ['instance_offsets','instance_pixel_index']:
        if key in scene:result[key]=scene[key]
    return result


def generate(fit_dir, output, count=8, seed=20261031):
    fit_dir, output = Path(fit_dir), Path(output)
    config = json.loads((fit_dir/'calibration.json').read_text())
    if config.get('format') != 'plane16-simcalc-adu-fit-v2':
        raise ValueError('Calibration format changed; refit into a new directory')
    if file_hash(fit_dir/'profile.npz') != config['profile_sha256']:
        raise ValueError('Fitted profile changed')
    for name in ['calibrated_simcalc.py','simcalc_core.py']:
        if file_hash(Path(__file__).parent/name) != config['source_code_sha256'][name]:
            raise ValueError(f'Renderer changed ({name}); refit before generation')
    output.mkdir(parents=True, exist_ok=False)
    with np.load(fit_dir/'profile.npz', allow_pickle=False) as z:
        profile = {k:z[k] for k in z.files}
    rows = [render(profile, config['selected'], seed+i) for i in range(count)]
    footprints=[]
    if config['selected'].get('gt_mode')=='soma':
        (output/'footprints').mkdir()
        for row in rows:
            name=f'footprints/seed{row["seed"]}.npz'
            np.savez_compressed(output/name,offsets=row['instance_offsets'],pixel_index=row['instance_pixel_index'],
                                shape_yx=np.asarray(row['image'].shape,dtype=np.uint32))
            footprints.append(name)
    metadata = dict(model='SimCalc kernels + fitted ADU detector v1', calibration=config,
        label_definition=('Filled SimCalc soma footprint including dark center; full overlaps in separate footprint files'
                          if config['selected'].get('gt_mode')=='soma'
                          else 'SimCalc half-maximum spatial kernel before optical blur; single-owner overlaps'),
        full_instance_footprints=footprints,
        limitations=['Provisional calibration pending human review; not measured real-cell GT',
                     'Static images; no calcium time dynamics or motion',
                     'Coarse out-of-focus fluorescence is not modeled neuropil',
                     'Effective noise parameters, including excess variance, are not measured electron gain or read noise'])
    np.savez_compressed(output/'simcalc_plane16.npz',
        synthetic_images=np.stack([r['image'] for r in rows]),
        synthetic_labels=np.stack([r['labels'] for r in rows]),
        synthetic_tissue_masks=np.stack([r['tissue'] for r in rows]),
        synthetic_seeds=np.asarray([r['seed'] for r in rows],dtype=np.uint32),
        metadata_json=np.asarray(json.dumps(metadata)))
    return output/'simcalc_plane16.npz'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261031)
    a = p.parse_args()
    if a.count < 1:
        p.error('count must be positive')
    print(generate(a.fit_dir,a.output,a.count,a.seed))
