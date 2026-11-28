"""Read-only soma light previews using a saved scene's layout and random draws."""
from base64 import b64encode
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import tifffile

from training.plane20.generate_median import AXES, light_fields, light_strength, render_somas
from training.plane20.soma_growth import light_reaches
from training.plane20.photometry import (BACKGROUND_BLEND, BACKGROUND_BLEND_PADDING,
                                         blend_reference_background, reference_background_weights)
from analyze_somas import measure
from soma_shell_model import normalize_profiles, radial_profiles, shell_patch_radius
from try_heterogeneous_scene import effective_blur
from .preview import display
from .spatial_metrics import pixel_correlation

# Bounds are shared by the sliders and the server.
CONTROLS = (
    ('peak', 'Радиус пика', .3, 1.5, .01, .8, '× радиус маски'),
    ('width', 'Ширина оболочки', .025, .65, .005, .2, '× радиус маски'),
    ('core_fill', 'Доля свечения в центре', 0., .9, .01, 0., ''),
    ('blur_px', 'Размытие сомы', 0., 2.5, .05, .6, 'px'),
    ('amplitude', 'Множитель яркости', .1, 4., .025, 1., '×'),
    ('angular_sd', 'Неравномерность по ободку', 0., 1.8, .025, .5, ''),
    ('gain_jitter_sd', 'Разброс яркости между сомами', 0., 1.5, .025, .35, 'σ лог-яркости'),
    ('gain_field_sd', 'Общий разброс яркости соседей', 0., 1.5, .025, .55, 'σ лог-яркости'),
)


def png(image, limits, *, overview=False):
    result = display(image, limits)
    if overview:
        result.thumbnail((480, 320))
    stream = BytesIO()
    result.save(stream, format='PNG')
    return 'data:image/png;base64,' + b64encode(stream.getvalue()).decode('ascii')


def annotation_geometry(soma):
    rows = [measure(item['points_xy']) for item in soma['annotations']
            if item['confidence'] == 'confident' and not item['truncated']]
    rows = [row for row in rows if all(np.isfinite(list(row.values()))) and row['minor_axis_px'] >= 2]
    return (np.array([(r['centroid_y']-.5, r['centroid_x']-.5) for r in rows]).reshape(-1, 2),
            *(np.array([r[key] for r in rows]) for key in AXES))


def mask_geometry(path):
    with np.load(path) as data:
        centers = data['centers']
        offsets, pixels, shape = (data[key] for key in (
            'instance_offsets', 'instance_pixel_index', 'shape_yx'))
    major, minor, theta = [], [], []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        points = np.column_stack(np.unravel_index(pixels[start:stop], shape))[:, ::-1]
        centered = points - points.mean(axis=0)
        eig, axes = np.linalg.eigh(centered.T @ centered / len(points) + np.eye(2)/12)
        major.append(4*np.sqrt(eig[1]))
        minor.append(4*np.sqrt(eig[0]))
        theta.append(np.degrees(np.arctan2(axes[1, 1], axes[0, 1])))
    return centers, np.array(major), np.array(minor), np.array(theta)


class LightPreview:
    def __init__(self, reference, profile, scene, real_geometry, crops, placement):
        self.reference = reference
        self.clean = scene['image']
        self.centers = scene['centers']
        self.geometry = {key: scene[key] for key in AXES}
        self.real_geometry = real_geometry
        self.normalizer = profile['normalizer']
        self.settings = json.loads(str(profile['settings_json']))['selected']
        self.config = json.loads(str(profile['median_config']))
        self.defaults = {key: self.config.get(key, default)
                         for key, _, _, _, _, default, _ in CONTROLS}
        self.seed = int(scene['seed'])
        self.crops = crops
        self.growth_reach = scene.get('growth_reach')
        self.growth_owners = None
        if self.growth_reach is not None:
            owners = np.full(self.clean.shape, -1, dtype=np.int32)
            owners.ravel()[scene['instance_pixel_index']] = np.repeat(
                np.arange(len(self.centers)), np.diff(scene['instance_offsets']))
            self.growth_owners = owners
        self.allowed = (np.ones(reference.shape, bool) if placement == 'fixed'
                        else profile['neuron_support'].astype(bool))
        self.edge_gain = (np.ones(reference.shape, np.float32) if placement == 'fixed'
                          else np.clip(distance_transform_edt(self.allowed)/2, 0, 1))
        support = profile.get('reference_background_support', profile.get('neuron_support', self.allowed))
        if placement == 'fixed' and 'instance_pixel_index' in scene:
            support = support.astype(bool, copy=True)
            support.ravel()[scene['instance_pixel_index']] = True
        self.background_weights = reference_background_weights(support)
        self.reference_background_saved = str(scene.get('background_blend', '')) == BACKGROUND_BLEND
        self.noise = None
        if self.config.get('intensity_noise') and 'image_structured' in scene:
            self.noise = scene['image_structured'].astype(np.float32) - self.clean
        if self.reference.shape != self.clean.shape or self.reference.shape != self.allowed.shape:
            raise ValueError('Размеры референса, профиля и сцены должны совпадать.')

    @lru_cache(maxsize=3)
    def fields(self):
        return light_fields(self.reference.shape, len(self.centers), self.config, self.seed)

    def metadata(self):
        self.fields()
        size = np.minimum(self.reference.shape, 96)
        default_crop = 0
        for index, crop in enumerate(self.crops):
            position = np.array([crop['y'], crop['x']])
            extent = np.array([crop['height'], crop['width']])
            start = np.clip(np.floor(position + (extent-96)/2 + .5), 0, self.reference.shape-size)
            if complete_profiles((*start, *size), self.real_geometry).any():
                default_crop = index
                break
        return dict(shape=self.reference.shape, seed=str(self.seed), noise=self.noise is not None,
                    defaults=self.defaults, crops=self.crops, default_crop=default_crop,
                    controls=[dict(name=key, label=label, min=lo, max=hi, step=step, unit=unit)
                              for key, label, lo, hi, step, _, unit in CONTROLS],
                    overview=png(self.reference, np.percentile(self.reference, [2, 99]), overview=True))

    def light(self, crop, config):
        y, x, h, w = crop
        # The optical blur needs only a short pixel margin. Select every soma
        # whose emission patch can reach that padded rectangle.
        start = np.maximum([0, 0], [y-8, x-8])
        stop = np.minimum(self.reference.shape, [y+h+8, x+w+8])
        zone = np.s_[start[0]:stop[0], start[1]:stop[1]]
        scale = None if self.growth_reach is None else light_reaches(self.growth_reach).max(axis=1)
        radius = shell_patch_radius(self.geometry['major_axis_px'], config['peak'], config['width'],
                                    config.get('blur_px', .6), scale_max=scale)[:, None]
        centers = self.centers
        pixels = np.floor(centers)
        keep = np.all((pixels+radius >= start) & (pixels-radius < stop), axis=1)
        if not keep.any():
            return np.zeros((h,w), np.float32)
        field, brightness, angles, mixture = self.fields()
        amplitudes = light_strength(field, centers, config, brightness)
        growth = None
        if self.growth_owners is not None:
            # Stored owners are global indices. Remap them to the cells selected
            # for this crop, keeping unselected cells and background unowned.
            remap = np.full(len(centers)+1, -1, dtype=np.int32)
            remap[np.flatnonzero(keep)] = np.arange(keep.sum())
            growth = remap[self.growth_owners[zone]], self.growth_reach[keep]
        scene = render_somas(self.allowed[zone], centers[keep]-start,
                             {key: value[keep] for key, value in self.geometry.items()},
                             config | {'grow_somas': False}, self.settings, self.seed,
                             amplitudes[keep], angles[keep], growth=growth)
        light = (effective_blur(scene['signal'], .2, .8, mixture[zone])*self.normalizer
                 *self.edge_gain[zone]*config['amplitude'])
        dy, dx = [y,x]-start
        return light[dy:dy+h, dx:dx+w]

    @lru_cache(maxsize=16)
    def nominal(self, crop):
        return self.light(crop, self.config)

    def parse_settings(self, supplied):
        if not isinstance(supplied, dict):
            raise ValueError('Ожидались параметры свечения.')
        settings = {}
        for key, _, lo, hi, _, _, _ in CONTROLS:
            value = float(supplied.get(key, self.defaults[key]))
            if not min(lo, self.defaults[key]) <= value <= max(hi, self.defaults[key]):
                raise ValueError('Значение вне диапазона: ' + key)
            settings[key] = value
        return settings

    def render(self, data):
        crop = tuple(data['crop'])
        if len(crop) != 4 or any(type(n) is not int for n in crop):
            raise ValueError('Кроп должен задаваться целыми y, x, высотой и шириной.')
        y, x, h, w = crop
        if not (0 <= y <= self.reference.shape[0]-h and 0 <= x <= self.reference.shape[1]-w
                and 1 <= min(h, w) <= max(h, w) <= 256):
            raise ValueError('Кроп должен быть внутри изображения и не больше 256 px.')
        settings = self.parse_settings(data.get('settings', {}))
        pad = BACKGROUND_BLEND_PADDING
        y0, x0 = max(0, y-pad), max(0, x-pad)
        y1, x1 = min(self.reference.shape[0], y+h+pad), min(self.reference.shape[1], x+w+pad)
        padded_crop = (y0, x0, y1-y0, x1-x0)
        zone = np.s_[y0:y1, x0:x1]
        inner = np.s_[y-y0:y-y0+h, x-x0:x-x0+w]
        reference = self.reference[zone]
        baseline = self.clean[zone].astype(np.float32)
        changed = baseline.copy()
        weights = tuple(weight[zone] for weight in self.background_weights)
        if settings != self.defaults:
            light_delta = self.light(padded_crop, self.config | settings) - self.nominal(padded_crop)
            if self.reference_background_saved:
                light_delta = blend_reference_background(light_delta, 0, weights)
            changed += light_delta
        if data.get('noise') and self.noise is not None:
            baseline += self.noise[zone]
            changed += self.noise[zone]
        if not self.reference_background_saved:
            baseline = blend_reference_background(baseline, reference, weights)
            changed = blend_reference_background(changed, reference, weights)
        reference = reference[inner]
        baseline = np.clip(baseline[inner], 0, 65535)
        changed = np.clip(changed[inner], 0, 65535)
        limits = np.percentile(reference, [2, 98])
        synth_geometry = (self.centers, *self.geometry.values())
        values = (reference, baseline, changed)
        stats = [crop_statistics(image, crop, geometry) for image, geometry in
                 zip(values, (self.real_geometry, synth_geometry, synth_geometry))]
        target = stats[0]['profile']
        for result in stats:
            result['profile_error'] = None
            if target and result['profile'] and result is not stats[0]:
                use = np.asarray(target['radius']) <= 1.5
                residual = np.asarray(result['profile']['adu']) - target['adu']
                result['profile_error'] = float(np.sqrt(np.mean(residual[use]**2)))
        return dict(images=[png(image, limits) for image in values], stats=stats, settings=settings)


def complete_profiles(crop, geometry):
    y, x, h, w = crop
    centers, major, _, _ = geometry
    local = centers-[y,x]
    margin = .9*major[:, None]
    return np.all((local-margin >= 0) & (local+margin <= [h-1,w-1]), axis=1)


def crop_statistics(image, crop, geometry):
    centers, major, minor, angles = geometry
    local = centers-crop[:2]
    keep = complete_profiles(crop, geometry)
    curves = None
    contrast = None
    if keep.any():
        radii, profiles = radial_profiles(image, local[keep], major[keep], minor[keep], angles[keep])
        _, core, shell = normalize_profiles(radii, profiles)
        curves = dict(radius=radii.tolist(), adu=np.median(profiles-core[:, None], axis=0).tolist())
        contrast = float(np.median(shell))
    residual = image.astype(np.float32)-gaussian_filter(image.astype(np.float32), 12)
    acf = pixel_correlation(residual, [np.ones(image.shape, bool)])[2]
    return dict(n=int(keep.sum()), profile=curves, shell_contrast=contrast,
                contrast=float(np.ptp(np.percentile(image, [5, 95]))),
                acf4=float(acf) if np.isfinite(acf) else None)


def arrays(source):
    with np.load(source) as data:
        return {key: data[key] for key in data.files}


@lru_cache(maxsize=3)
def open_run(folder, run):
    report = json.loads((run/'report.json').read_text())
    first = report['scenes'][0]
    with zipfile.ZipFile(run/'dataset.zip') as archive:
        profile = arrays(BytesIO(archive.read('profile.npz')))
        scene = arrays(BytesIO(archive.read(first['path'])))
        real_geometry = annotation_geometry(json.loads(archive.read('soma.json')))
        names = set(archive.namelist())
        crops = json.loads(archive.read('centers.json')) if 'centers.json' in names else []
        held = json.loads(archive.read('heldout.json')) if 'heldout.json' in names else []
    return LightPreview(tifffile.imread(folder/'reference.tiff'), profile, scene, real_geometry,
                        [dict(c, label='Оценка') for c in held]+crops, first.get('placement'))


@lru_cache(maxsize=1)
def open_manifest(path):
    data = json.loads(path.read_text())
    base = path.parent
    return LightPreview(tifffile.imread(base/data['reference']), arrays(base/data['profile']),
                        arrays(base/data['scene']), mask_geometry(base/data['masks']),
                        data.get('crops', []), data.get('placement'))
