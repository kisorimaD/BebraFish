"""Compare a uniform 3D shell with Plane20 medians and attribute its out-of-section light."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import minimize_scalar

from simcalc_zebrafish_pipeline.generate_volume import write_ome
from simcalc_zebrafish_pipeline.volume import Acquisition, render_stack, sample_scene
from simcalc_zebrafish_pipeline.volume_reference import CALIBRATION_STOP_Y, read_reference_profile


def split_soma_light(scene):
    """Additive detected light with zero envelope and floor, before the median background."""
    profile = scene.profile
    depths = profile.depths()
    maps = replace(scene.tissue_maps, background_adu=np.zeros_like(scene.tissue_maps.background_adu))
    dark_scene = replace(scene, tissue_maps=maps, profile=replace(profile, background_floor_adu=0.))
    catalog = scene.cells
    rz = catalog['semiaxes_zyx_um'][:, 0]
    reach = rz * (catalog['shell_peak_radius'] + 4 * catalog['shell_width'])
    reach += profile.optics.quadrature()[0][-1]
    groups = np.zeros((3, len(depths), *profile.shape_yx), np.float32)
    for k, depth in enumerate(depths):
        dz = np.abs(catalog['center_zyx_um'][:, 0] - depth)
        intersects = dz <= rz
        far = dz >= profile.acquisition.z_step_um
        for j, mask in enumerate((intersects, ~intersects & ~far, ~intersects & far)):
            cells = catalog[mask & (dz <= reach)]
            if len(cells):
                component = render_stack(replace(dark_scene, cells=cells), Acquisition(1, depth))
                groups[j, k] = component.clean_adu[0]
    return groups


def bandpass(stack, a, b):
    return gaussian_filter(stack, (0, a, a)) - gaussian_filter(stack, (0, b, b))


def contour_ellipse_iou(annotation):
    """Compare a hand-drawn polygon with its area-moment ellipse on a 1/8 px grid."""
    measure = annotation['measurements']
    polygon = np.asarray(annotation['points_xy'])
    center = np.array([measure['centroid_x'], measure['centroid_y']])
    a, b = measure['major_axis_px'] / 2, measure['minor_axis_px'] / 2
    lower = np.floor(np.minimum(polygon.min(0), center-a))
    upper = np.ceil(np.maximum(polygon.max(0), center+a))
    x, y = np.meshgrid(*(np.arange(start, stop, .125) + .0625 for start, stop in zip(lower, upper)))
    inside = np.zeros(x.shape, bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if y1 != y2:
            inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (x2-x1)*(y-y1)/(y2-y1))
    x, y = x-center[0], y-center[1]
    theta = np.deg2rad(measure['orientation_deg'])
    c, s = np.cos(theta), np.sin(theta)
    ellipse = ((c*x+s*y)/a)**2 + ((-s*x+c*y)/b)**2 <= 1.
    return np.count_nonzero(inside & ellipse) / np.count_nonzero(inside | ellipse)


def adjacent_correlations(real_band, mask, shifts):
    """Compare common tissue at coarse anatomical shifts and two displaced XY controls."""
    results = []
    for k, shift in enumerate(shifts):
        dy, dx = shift['dy_px'], shift['dx_px']
        y, x = np.nonzero(mask[k])
        h, w = mask.shape[1:]
        inside = (y + dy >= 0) & (y + dy < h) & (x + dx - 32 >= 0) & (x + dx + 32 < w)
        y, x = y[inside], x[inside]
        valid = np.logical_and.reduce([mask[k+1, y+dy, x+dx+d] for d in (-32, 0, 32)])
        y, x = y[valid], x[valid]
        a = real_band[k, y, x]
        values = [float(np.corrcoef(a, real_band[k+1, y+dy, x+dx+d])[0, 1]) for d in (-32, 0, 32)]
        results.append(dict(planes=[18+k, 19+k], n=len(a), shifted_r=values[1], controls_r=values[::2]))
    return results


def preview(path, images, real):
    from PIL import Image, ImageDraw, ImageFont

    lo, hi = np.percentile(real[2], [2, 99])
    canvas = Image.new('RGB', (len(images) * 254 + 14, 555), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for i, (name, stack) in enumerate(images.items()):
        x = 10 + i * 254
        draw.text((x, 6), name, fill='black', font=font)
        for j, (y, left) in enumerate(((480, 80), (928, 80))):
            patch = stack[2, y:y+128, left:left+128]
            gray = np.rint(255 * np.clip((patch - lo) / (hi - lo), 0, 1)).astype(np.uint8)
            canvas.paste(Image.fromarray(gray).resize((242, 242), Image.Resampling.NEAREST),
                         (x, 29 + 247*j))
    draw.text((10, 528), f'Plane 20, 128 x 128 px crops at (Y, X) = (480, 80) and (928, 80). '
              f'Shared range: {lo:.0f} to {hi:.0f} ADU.', fill='black', font=font)
    canvas.save(path)
    return [float(lo), float(hi)]


def single_soma_preview(path, scene):
    """Show a representative model soma under the same PSF on a shared intensity scale."""
    from PIL import Image, ImageDraw, ImageFont

    radii = scene.cells['semiaxes_zyx_um']
    median = np.median(radii, axis=0)
    index = np.argmin(np.linalg.norm(np.log(radii / median), axis=1))
    cells = scene.cells[index:index+1].copy()
    profile = replace(scene.profile, shape_yx=(32, 32))
    dy, dx = profile.pixel_yx_um
    cells['center_zyx_um'] = (20., 16*dy, 16*dx)
    isolated = replace(scene, cells=cells, profile=profile, tissue_maps=None, texture=None)
    scan = Acquisition(11, 20., 1.)
    base = render_stack(replace(isolated, cells=cells[:0]), scan).clean_adu
    light = render_stack(isolated, scan).clean_adu - base
    canvas = Image.new('RGB', (854, 220), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for i, dz in enumerate((0, 2, 3, 5, 10)):
        x = 8 + i * 170
        gray = np.rint(255 * np.clip(light[dz] / light.max(), 0, 1)).astype(np.uint8)
        canvas.paste(Image.fromarray(gray).resize((158, 158), Image.Resampling.NEAREST), (x, 30))
        draw.text((x, 6), f'dZ = {dz} um', fill='black', font=font)
    draw.text((8, 196), 'One synthetic soma, distance from its centre. Same intensity scale in all panels.',
              fill='black', font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--structured', type=Path, required=True)
    parser.add_argument('--uniform', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    profile, maps, provenance = read_reference_profile(args.reference)
    with np.load(args.structured/'instances.npz') as old, np.load(args.uniform/'instances.npz') as new:
        settings = json.loads(str(new['metadata_json']))
        profile = replace(profile, emission_model='uniform-shell', median_style='baseline')
        scene = sample_scene(profile, settings['seed'], tissue_maps=maps)
        np.testing.assert_array_equal(scene.cells, new['cells'])
        for field in ('center_zyx_um', 'semiaxes_zyx_um', 'angle_rad', 'cell_id'):
            np.testing.assert_array_equal(old['cells'][field], new['cells'][field])
        np.testing.assert_allclose(old['cells']['amplitude_adu'], new['cells']['amplitude_adu'],
                                   atol=1e-10, rtol=1e-12)
        for field in ('instance_voxel_index', 'instance_offsets', 'z_positions_um'):
            np.testing.assert_array_equal(old[field], new[field])
        np.testing.assert_array_equal(new['z_positions_um'], profile.depths())
        old_profile = json.loads(str(old['metadata_json']))['profile']
        if (old_profile['median_style'], old_profile['emission_model']) != ('structured', 'projected'):
            raise ValueError('Expected the projected + structured comparison render')
    with np.load(args.reference) as reference:
        real = reference['median_adu']
        annotation = json.loads(str(reference['annotation_json']))
    contours = [a for a in annotation['annotations']
                if not a['truncated'] and a['confidence'] == 'confident']
    structured = np.load(args.structured/'clean_adu.npy')
    uniform = np.load(args.uniform/'clean_adu.npy')
    section, near, far = split_soma_light(scene)
    total = section + near + far
    coarse = gaussian_filter(total, (0, 12, 12))
    background = np.maximum(maps.background_adu - coarse, profile.background_floor_adu)
    np.testing.assert_allclose(background + total, uniform, atol=.001, rtol=0)
    interior = np.stack([distance_transform_edt(s >= .8) >= 12 for s in maps.support])
    train = np.s_[:, :CALIBRATION_STOP_Y]
    fit_mask = interior[train].copy()
    fit_mask[:, -48:] = False
    target = bandpass(real[train], 3, 7)[fit_mask].std()

    def contrast_error(gain):
        base = np.maximum(maps.background_adu[train] - gain*coarse[train], profile.background_floor_adu)
        candidate = base + gain*total[train]
        return (bandpass(candidate, 3, 7)[fit_mask].std() - target)**2

    fit = minimize_scalar(contrast_error, bounds=(0., 8.), method='bounded')
    if not fit.success:
        raise RuntimeError(fit.message)
    gain = float(fit.x)
    calibrated_background = np.maximum(maps.background_adu - gain*coarse, profile.background_floor_adu)
    calibrated = calibrated_background + gain*total
    images = {'Real median': real, 'Current structured': structured, 'Uniform, original gain': uniform,
              'Fitted, intersecting only': calibrated_background + gain*section,
              'Fitted, all somas': calibrated}
    mask = interior.copy()
    mask[:, :CALIBRATION_STOP_Y + 48] = False
    signal_sum = total[mask].sum(dtype=np.float64)
    report = dict(seed=int(scene.seed), cells=len(scene.cells), contours=len(contours),
                  gain_calibration=dict(parameter='soma amplitude', fit='3–7 px contrast on upper half, 48 px gap',
                                        original_adu=profile.amplitude_adu, gain=gain,
                                        fitted_adu=gain * profile.amplitude_adu, allowed_gain=[0., 8.]),
                  ellipse_iou_p10_median_p90=np.percentile([contour_ellipse_iou(a) for a in contours],
                                                          [10, 50, 90]).tolist(),
                  morphology_p10_median_p90={k: np.percentile([a['measurements'][k] for a in contours], [10,50,90]).tolist()
                                             for k in ('equivalent_diameter_px','aspect_ratio','circularity')},
                  evaluation='all 5 planes, tissue interior >=12 px and y >= calibration split +48 px',
                  band_rms_adu={name: {f'{a}-{b}': float(bandpass(stack,a,b)[mask].std())
                                      for a,b in ((1,3),(3,7),(7,12))} for name, stack in images.items()},
                  out_of_section_fraction=float((near+far)[mask].sum(dtype=np.float64)/signal_sum),
                  out_of_section_at_least_10um_fraction=float(far[mask].sum(dtype=np.float64)/signal_sum),
                  fraction_denominator='total detected soma light before additive background, on evaluation pixels',
                  adjacent_real_band_3_7=adjacent_correlations(bandpass(real, 3, 7), mask,
                                            provenance['registration_yx_next_relative_to_previous']))
    args.output.mkdir(parents=True)
    write_ome(args.output/'calibrated.ome.tif', np.rint(np.clip(calibrated, 0, 65535)).astype(np.uint16),
              profile.acquisition, profile.pixel_yx_um)
    report['preview_range_adu'] = preview(args.output/'comparison.png', images, real)
    single_soma_preview(args.output/'single_soma.png', scene)
    (args.output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
