"""Run small soma-contact experiments or reslice a saved growth scene."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import distance_transform_edt

from .close_contact_gaps import close_dense_gaps, load_gap_fill
from .generate_volume import write_ome
from .growth_photometry import GrowthOptics, render_growth
from .soma_growth import (
    GrowthConfig, Somata, grow, load_growth, rasterize, save_growth, save_masks,
)
from .soma_id_matching import remap_ids
from .volume import Acquisition


ANNOTATIONS = Path('annotations/Plane20_median_registered_uint16_frame0_776e568f5575.json')
GRID_UM = .4
PLOT_UM = .2
COLORS = ('#377ba8', '#e09243', '#4a997c', '#987dba', '#be6268', '#a3a34b',
          '#529aa5', '#957766', '#dd80a3')


def _shape_prior(path):
    # Reuse the Plane20 polygon moments and the same section-to-equator model.
    from tools.analyze_somas import measure

    data = json.loads(path.read_text())
    rows = [measure(a['points_xy']) for a in data['annotations']
            if a['confidence'] == 'confident' and not a['truncated']]
    if not rows:
        raise ValueError('Need at least one confident, complete Plane20 contour')
    equiv = np.median([r['equivalent_diameter_px'] for r in rows]) * .576 / np.sqrt(.75)
    aspect = np.median([r['aspect_ratio'] for r in rows])
    return len(rows), float(equiv), float(aspect)


def _catalog(centers, diameter, aspect, seed):
    n = len(centers)
    rng = np.random.default_rng(seed)
    diameter = diameter * np.clip(rng.lognormal(0., .06, n), .88, 1.12)
    aspect = np.clip(aspect * rng.lognormal(0., .08, n), 1.05, 1.55)
    axial = np.full(n, .85)
    orientation = np.column_stack((rng.normal(0., .12, (n, 2)), rng.uniform(0., np.pi, n)))
    volume = 4 * np.pi / 3 * (diameter / 2)**3 * axial
    return Somata(np.asarray(centers), volume, aspect, axial, orientation, np.full(n, .65),
                  np.arange(1, n + 1, dtype=np.uint32))


def _scenes(diameter, aspect, seed):
    group_center = np.array([8., 9.2, 9.2])
    corners = np.array(np.meshgrid([-3.1, 3.1], [-3.15, 3.15], [-3.15, 3.15], indexing='ij'))
    cluster = np.vstack((corners.reshape(3, -1).T + group_center, group_center))
    return {
        'single': (_catalog([[6., 6., 6.]], diameter, aspect, seed), (12., 12., 12.), 6., 5.9),
        'pair': (_catalog([[6., 6., 6.], [6., 6., 11.]], diameter, aspect, seed), (12., 12., 17.2), 6., 5.9),
        'group': (_catalog(cluster, diameter, aspect, seed+1), (16., 18.4, 18.4), 4.9, 6.1),
    }


def _survey_depths(tissue_depth, dz):
    return Acquisition(int(np.floor(tissue_depth / dz)), dz / 2, dz).depths()


def _measure_masks(masks, dz):
    counts = masks.occupancy()
    empty = counts == 0
    depths = distance_transform_edt(empty, sampling=(dz, *masks.pixel_yx_um))[empty]
    voxel_volume = dz * np.prod(masks.pixel_yx_um)
    stats = {
        'occupied_fraction': float(np.mean(~empty)),
        'overlap_voxels': int(np.count_nonzero(counts > 1)),
        'overlap_um3': float(np.count_nonzero(counts > 1) * voxel_volume),
        'excess_instance_volume_um3': float(np.maximum(counts - 1, 0).sum() * voxel_volume),
        'distance_to_soma_p10_p50_p90_um': np.percentile(depths, [10, 50, 90]).tolist(),
    }
    return stats, depths


def _support_gaps(growth):
    """Lower bound on distance between meshes if the projected gap is positive."""
    i, j = np.triu_indices(len(growth.somata.cell_id), 1)
    if not len(i):
        return None
    rel = growth.vertices_zyx_um - growth.centers_zyx_um[:, :, None, :]
    delta = growth.centers_zyx_um[:, j] - growth.centers_zyx_um[:, i]
    dist = np.linalg.norm(delta, axis=2)
    normal = delta / dist[:, :, None]
    left = np.einsum('tpc,tpvc->tpv', normal, rel[:, i]).max(axis=2)
    right = np.einsum('tpc,tpvc->tpv', -normal, rel[:, j]).max(axis=2)
    return (dist - left - right).min(axis=1).tolist()


def _planes_for_plot(scene, z_slice, y_slice, stage):
    z_grid = _survey_depths(scene.tissue_zyx_um[0], PLOT_UM)
    z = np.unique(np.r_[z_grid, z_slice])
    masks = rasterize(scene, z, (PLOT_UM, PLOT_UM), stage=stage)
    zi = int(np.searchsorted(z, z_slice))
    yi = int(np.rint(y_slice / PLOT_UM - .5))
    xy, xz = [], []
    for a, b in zip(masks.instance_offsets[:-1], masks.instance_offsets[1:]):
        points = masks.instance_voxel_index[a:b]
        index_z, index_y, index_x = np.unravel_index(points, masks.shape_zyx)
        xy_cell = np.zeros(masks.shape_zyx[1:], bool)
        xz_cell = np.zeros((len(z), masks.shape_zyx[2]), bool)
        xy_cell[index_y[index_z == zi], index_x[index_z == zi]] = True
        xz_cell[index_z[index_y == yi], index_x[index_y == yi]] = True
        xy.append(xy_cell)
        xz.append(xz_cell[np.searchsorted(z, z_grid)])
    return xy, xz


def _draw_section(ax, masks, *, section, title):
    from matplotlib.colors import to_rgb

    height, width = masks[0].shape
    rgb = np.ones((height, width, 3)) * .965
    coverage = np.zeros((height, width), dtype=int)
    for i, mask in enumerate(masks):
        coverage += mask
        rgb[mask] = np.array(to_rgb(COLORS[i % len(COLORS)])) * .76 + .24
    rgb[coverage > 1] = [.29, .18, .23]
    ax.imshow(rgb, origin='lower', interpolation='nearest',
              extent=(0., width * PLOT_UM, 0., height * PLOT_UM))
    for i, mask in enumerate(masks):
        if mask.any() and not mask.all():
            ax.contour((np.arange(width) + .5)*PLOT_UM, (np.arange(height) + .5)*PLOT_UM,
                       mask.astype(float), [.5], colors=[COLORS[i % len(COLORS)]], linewidths=.8)
    ax.set(xlim=(0, width*PLOT_UM), ylim=(0, height*PLOT_UM),
           xlabel='X, мкм', ylabel=f'{section[1]}, мкм')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _plots(output, results, coordinates, stats, distances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = [('single', 'Одна сома'), ('pair', 'Пара'), ('group', 'Группа')]
    count = results['pair', 'deform'].scale.size
    steps = np.rint(np.linspace(0, count - 1, min(count, 5))).astype(int)
    fig, axes = plt.subplots(5, len(steps), figsize=(2.8 * len(steps), 12.2), layout='constrained')
    fig.suptitle('Рост сом · доля конечного целевого объёма', fontsize=17)
    row = 0
    for name, title in order:
        scene = results[name, 'deform']
        for col, stage in enumerate(steps):
            xy, xz = _planes_for_plot(scene, *coordinates[name], stage)
            label = f'{scene.scale[stage]**3:.0%}'
            _draw_section(axes[row, col], xy, section='XY', title=f'{title}, XY · {label}')
            if name != 'single':
                _draw_section(axes[row+1, col], xz, section='XZ', title=f'{title}, XZ · {label}')
        row += 1 if name == 'single' else 2
    fig.savefig(output / 'growth.png', dpi=155)
    plt.close(fig)

    fig = plt.figure(figsize=(13.2, 9.1), layout='constrained')
    grid = fig.add_gridspec(3, 4, height_ratios=(1., 1., .8))
    fig.suptitle('Упаковка с одинаковыми начальными центрами и объёмами\n'
                 'Тёмные участки на сечениях показывают пересечения', fontsize=14)
    for row, (name, title) in enumerate(order[1:]):
        for col, (mode, text) in enumerate((('shift_only', 'Сдвиг'), ('deform', 'Сдвиг + деформация'))):
            xy, xz = _planes_for_plot(results[name, mode], *coordinates[name], -1)
            _draw_section(fig.add_subplot(grid[row, col]), xy, section='XY', title=f'{title}, XY · {text}')
            _draw_section(fig.add_subplot(grid[row, col+2]), xz, section='XZ', title=f'{title}, XZ · {text}')
    ax = fig.add_subplot(grid[2, :2])
    modes = ('shift_only', 'deform')
    names = [name for name, _ in order]
    for i, (mode, label, color) in enumerate(zip(modes, ('Сдвиг', 'Сдвиг + деформация'), COLORS)):
        coverage = [100 * stats[name][mode]['occupied_fraction'] for name in names]
        bars = ax.bar(np.arange(3) + (i - .5)*.32, coverage, width=.29, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=8)
    ax.set_xticks(np.arange(3), [s for _, s in order])
    full = max(stats[name][mode]['occupied_fraction'] for name in names for mode in modes)
    ax.set(ylabel='Доля занятой ткани, %', ylim=(0, 120 * full))
    ax.legend(frameon=False, fontsize=8)
    gl = fig.add_subplot(grid[2, 2:])
    for name, label in order[1:]:
        for j, mode in enumerate(modes):
            gap = np.sort(distances[name, mode])
            gl.plot(gap, np.arange(1, len(gap)+1) / len(gap), color=COLORS[j], lw=1.7,
                    linestyle='-' if name == 'group' else '--',
                    label=f'{label}, {("сдвиг" if j == 0 else "деформация")}')
    gl.set(xlabel='Расстояние пустой точки до ближайшей сомы, мкм', ylabel='Доля пустых точек')
    gl.legend(frameon=False, fontsize=8)
    fig.savefig(output / 'packing_comparison.png', dpi=155)
    plt.close(fig)


def run_demo(output, annotations, seed, config):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    count, diameter, aspect = _shape_prior(annotations)
    output.mkdir(parents=True)
    results, coordinates, distances, stats = {}, {}, {}, {}
    for name, (somata, tissue, z, y) in _scenes(diameter, aspect, seed).items():
        stats[name] = {}
        coordinates[name] = (z, y)
        for mode in ('shift_only', 'deform'):
            growth = grow(somata, tissue, deform=mode == 'deform', config=config)
            results[name, mode] = growth
            folder = output / name / mode
            folder.mkdir(parents=True)
            save_growth(folder / 'scene.npz', growth)
            masks = rasterize(growth, _survey_depths(tissue[0], GRID_UM), (GRID_UM, GRID_UM))
            save_masks(folder / 'instances.npz', masks)
            summary, distances[name, mode] = _measure_masks(masks, GRID_UM)
            targets = growth.scale[:, None]**3 * somata.target_volume_um3
            errors = 100 * (growth.volumes_um3 / targets - 1.)
            summary.update(
                final_volume_error_pct=errors[-1].tolist(),
                final_median_abs_volume_error_pct=float(np.median(np.abs(errors[-1]))),
                max_abs_volume_error_pct_by_stage=np.max(np.abs(errors), axis=1).tolist(),
                max_center_shift_um=float(np.linalg.norm(growth.centers_zyx_um[-1] -
                                                       somata.centers_zyx_um, axis=1).max()),
                centerline_support_gap_um_by_stage=_support_gaps(growth),
            )
            stats[name][mode] = summary
            print(name, mode, f'{100*summary["occupied_fraction"]:.2f}% occupied',
                  f'{summary["overlap_um3"]:.2f} µm³ overlap')
    report = {'seed': seed, 'config': asdict(config), 'grid_um': GRID_UM,
              'shape_prior': {'annotations': str(annotations), 'n': count, 'xy_diameter_um': diameter,
                              'xy_aspect': aspect, 'axial_ratio_assumption': .85,
                              'tiff_pitch_um_assumption': .576},
              'plot_planes': {name: {'z_for_xy_um': z, 'y_for_xz_um': y}
                              for name, (z, y) in coordinates.items()},
              'scenes': stats}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    _plots(output, results, coordinates, stats, distances)


def _write_fluorescence(output, scene_path, scene, acquisition, pixel_yx_um, stage):
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    optics = GrowthOptics()
    clean, masks = render_growth(scene, acquisition.depths(), pixel_yx_um, stage=stage, optics=optics)
    meta = {
        'scene': str(scene_path), 'stage': stage % len(scene.scale), 'optics': asdict(optics),
        'acquisition': asdict(acquisition), 'pixel_yx_um': pixel_yx_um, 'axes': 'ZYX',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'render'
        pending.mkdir()
        image = np.rint(np.clip(clean, 0, 65535)).astype(np.uint16)
        write_ome(pending / 'image.ome.tif', image, acquisition, pixel_yx_um)
        np.save(pending / 'clean_adu.npy', clean, allow_pickle=False)
        save_masks(pending / 'instances.npz', masks)
        (pending / 'render.json').write_text(json.dumps(meta, indent=2) + '\n')
        pending.rename(output)
    print(f'{masks.shape_zyx} ZYX, {len(masks.cell_id)} IDs → {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo')
    demo.add_argument('--output', required=True, type=Path)
    demo.add_argument('--annotations', default=ANNOTATIONS, type=Path)
    demo.add_argument('--seed', default=42, type=int)
    demo.add_argument('--mobility', default=GrowthConfig.mobility, type=float)
    demo.add_argument('--max-shift-um', default=GrowthConfig.max_shift_um, type=float)
    demo.add_argument('--steps', default=GrowthConfig.steps, type=int)
    for command in ('reslice', 'render'):
        scan = commands.add_parser(command)
        scan.add_argument('--scene', required=True, type=Path)
        scan.add_argument('--output', required=True, type=Path)
        scan.add_argument('--n-planes', required=True, type=int)
        scan.add_argument('--z-start-um', required=True, type=float)
        scan.add_argument('--z-step-um', required=True, type=float)
        scan.add_argument('--pixel-um', default=GRID_UM, type=float)
        scan.add_argument('--stage', default=-1, type=int)
        if command == 'reslice':
            scan.add_argument('--dense-gap-fill', type=Path, help='Include the saved dense gap partition')
            scan.add_argument('--soma-id-map', type=Path, help='Apply the saved count-matched grouping')
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            run_demo(args.output, args.annotations, args.seed, GrowthConfig(
                mobility=args.mobility, max_shift_um=args.max_shift_um, steps=args.steps))
        else:
            scene = load_growth(args.scene)
            acquisition = Acquisition(args.n_planes, args.z_start_um, args.z_step_um)
            pixel_yx_um = (args.pixel_um, args.pixel_um)
            if args.command == 'render':
                _write_fluorescence(args.output, args.scene, scene, acquisition, pixel_yx_um, args.stage)
            else:
                masks = rasterize(scene, acquisition.depths(), pixel_yx_um, stage=args.stage)
                if args.dense_gap_fill is not None:
                    if args.stage not in (-1, len(scene.scale) - 1):
                        raise ValueError('Dense gap closure is defined at the final growth stage')
                    masks = close_dense_gaps(scene, masks, load_gap_fill(args.dense_gap_fill))
                if args.soma_id_map is not None:
                    with np.load(args.soma_id_map, allow_pickle=False) as mapping:
                        masks = remap_ids(masks, mapping['source_cell_id'], mapping['owner_cell_id'])
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('xb') as stream:
                    save_masks(stream, masks)
                print(f'{masks.shape_zyx} ZYX, {len(masks.cell_id)} IDs')
    except (ValueError, OSError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
