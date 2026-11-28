"""Descriptive real/synthetic comparisons for a completed median run."""
import io
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, label
import tifffile

from analyze_somas import measure
from annotate_regions import compute_mask
from soma_shell_model import normalize_profiles

from .spatial_metrics import (PIXEL_LAGS, image_statistics, nearest_curve,
                              shell_statistics, tissue_windows)

REAL = '#226789'
SYNTH = '#b64d2e'
BASE = '#686191'
RENDERS = (('image', BASE, 'Базовый'), ('image_structured', SYNTH, 'Структурированный'))
GROUPS = ('Рабочая медиана', 'Первая и вторая половины', 'Смешанные подвыборки')
COLORS = (REAL, '#76529c', '#208068')


def background_regions(mask):
    components, count = label(mask)
    sizes = np.bincount(components.ravel())
    ids = np.argsort(-sizes[1:])[:8]+1
    return [components == index for index in ids if index <= count and sizes[index] >= 64]


def background_values(image, regions, interiors):
    image = np.asarray(image, dtype=np.float32)
    smooth = [gaussian_filter(image, sigma) for sigma in (1, 3, 7)]
    bands = [smooth[0]-smooth[1], smooth[1]-smooth[2]]
    return np.array([[np.median(image[region]),
                      *(np.std(band[inside]) if inside.sum() >= 16 else np.nan for band in bands)]
                     for region, inside in zip(regions, interiors)])


def area_density(centers, crops, support):
    values = []
    for crop in crops:
        x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
        area = int(support[y:y+h, x:x+w].sum())
        cy, cx = centers.T
        in_crop = (cy >= y-.5) & (cy < y+h-.5) & (cx >= x-.5) & (cx < x+w-.5)
        values.append(1000*in_crop.sum()/area if area else np.nan)
    return values


def missing(axis, message):
    axis.set_axis_off()
    axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes)


def scatter_groups(axis, groups, labels, colors):
    for index, (values, name, color) in enumerate(zip(groups, labels, colors)):
        values = np.asarray(values)
        if values.size:
            axis.plot(np.arange(values.shape[1])+1+(index-(len(labels)-1)/2)*.13,
                      values.T, 'o', color=color, alpha=.65, markersize=4)
            axis.plot([], [], 'o', color=color, label=name, markersize=4)


def ecdf(axis, values, color, name):
    values = np.sort(values)
    if len(values):
        axis.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                  color=color, label=f'{name} (n={len(values)})')


def shell_curves(axis, radii, curves, color, name):
    if len(curves):
        low, median, high = np.percentile(curves, [25, 50, 75], axis=0)
        axis.fill_between(radii, low, high, color=color, alpha=.16)
        axis.plot(radii, median, color=color, label=f'{name} (n={len(curves)})')


def comparison_curve(axis, x, real, synthetic, color, name):
    if real is not None and np.isfinite(real).any():
        axis.plot(x, real, 'o-', color=color, label=name)
    # Each scene contributes a curve. Sparse bins can be absent in some scenes.
    bands = np.full((3, len(x)), np.nan)
    for i, column in enumerate(np.asarray(synthetic).T):
        finite = column[np.isfinite(column)]
        if len(finite):
            bands[:, i] = np.percentile(finite, [25, 50, 75])
    low, median, high = bands
    if np.isfinite(median).any():
        axis.fill_between(x, low, high, color=color, alpha=.14)
        axis.plot(x, median, 'o-', color=color, label=name)


def write_metrics(folder, attempt):
    reference = tifffile.imread(folder/'reference.tiff')
    with zipfile.ZipFile(attempt/'dataset.zip') as archive:
        soma, region, crops, report = [json.loads(archive.read(name+'.json')) for name in
                                      ('soma', 'region', 'centers', 'report')]
        with np.load(io.BytesIO(archive.read('profile.npz'))) as profile:
            support = profile['neuron_support']
        regions = background_regions(compute_mask(reference.astype(np.float32), region))
        interiors = [binary_erosion(mask, iterations=7) for mask in regions]
        # Use the same background interiors for the two texture bands and pixel pairs.
        texture_regions = [mask for mask in interiors if mask.sum() >= 16]
        windows = tissue_windows(support)
        real = [measure(item['points_xy']) for item in soma['annotations']
                if item['confidence'] == 'confident' and not item['truncated']]
        real = [row for row in real if np.all(np.isfinite(list(row.values()))) and row['minor_axis_px'] >= 2]
        axes = ('major_axis_px', 'minor_axis_px', 'orientation_deg')
        annotated = (np.array([(row['centroid_y'], row['centroid_x']) for row in real]).reshape(-1, 2),
                     *(np.array([row[key] for row in real]) for key in axes))
        diameter = np.median([row['equivalent_diameter_px'] for row in real]) if real else 10.
        distance_bins = diameter*np.array([0, 2, 4, 6, 8])
        nn_radii = np.arange(1, 2*diameter+1)
        peak = report['calibration'].get('peak', .8)
        real_shell, real_contrast, real_arcs, real_pairs, pair_counts = shell_statistics(
            reference, annotated, .5, peak, distance_bins)
        image_real = image_statistics(reference, support, windows, texture_regions)
        crop_points = [np.asarray(crop['points'], float).reshape(-1, 2)[:, ::-1] for crop in crops]
        real_nn = nearest_curve(crop_points, crops, nn_radii)

        rng = np.random.default_rng(42)
        scenes = report['scenes']
        selected = np.linspace(0, len(scenes)-1, min(16, len(scenes))).round().astype(int)
        synthetic_density, synthetic_nn = [], []
        diameters, aspects = [], []
        backgrounds = {key: [] for key, *_ in RENDERS}
        images = {key: [] for key, *_ in RENDERS}
        shells = {key: [] for key, *_ in RENDERS}
        for index in selected:
            with np.load(io.BytesIO(archive.read(scenes[index]['path']))) as scene:
                centers = scene['centers']
                major, minor, angle = (scene[key] for key in axes)
                diameters.extend(np.sqrt(major*minor))
                aspects.extend(major/minor)
                take = rng.choice(len(centers), min(128, len(centers)), replace=False)
                geometry = (centers[take], major[take], minor[take], angle[take])
                for key, *_ in RENDERS:
                    image = scene[key]
                    if regions:
                        backgrounds[key].append(background_values(image, regions, interiors))
                    images[key].append(image_statistics(image, support, windows, texture_regions))
                    shells[key].append(shell_statistics(image, geometry, 0., peak, distance_bins))
                synthetic_density.append(area_density(centers, crops, support))
                synthetic_nn.append(nearest_curve([centers]*len(crops), crops, nn_radii))

    fig, grid = plt.subplots(7, 2, figsize=(13, 23), layout='constrained')
    try:
        fig.suptitle(f'Сравнение с рабочей медианой · {len(selected)} из {len(scenes)} синтетических сцен',
                     fontsize=15)
        temporal = folder/'median-diagnostics.npz'
        background_axes = (*grid[0], grid[1, 0])
        if regions:
            baseline = background_values(reference, regions, interiors)
            repeats = []
            if temporal.is_file():
                with np.load(temporal) as diagnostics:
                    repeats = [background_values(m, regions, interiors) for m in diagnostics['medians']]
            groups = ([baseline], repeats[:2], repeats[2:], *(backgrounds[key] for key, *_ in RENDERS))
            labels = (*GROUPS, *(name for _, _, name in RENDERS))
            colors = (*COLORS, *(color for _, color, _ in RENDERS))
            titles = ('Уровень фона', 'Текстура фона · полоса 1–3 px',
                      'Текстура фона · полоса 3–7 px')
            for i, (ax, title) in enumerate(zip(background_axes, titles)):
                values = [np.asarray(group)[:, :, i] if len(group) else [] for group in groups]
                if any(np.isfinite(group).any() for group in values):
                    scatter_groups(ax, values, labels, colors)
                    ax.set(xlabel='Размеченная область фона', ylabel='ADU',
                           xticks=np.arange(len(regions))+1)
                else:
                    missing(ax, 'Мало пикселей внутри размеченных областей')
                ax.set_title(title)
            if not repeats:
                background_axes[0].text(.02, .98, 'Нет медиан из подвыборок',
                                        transform=background_axes[0].transAxes, va='top', fontsize=8)
        else:
            for ax in background_axes:
                missing(ax, 'Недостаточно данных о фоне')

        for ax, signal, key, title in zip((grid[1, 1], grid[2, 0]), (diameters, aspects),
                                          ('equivalent_diameter_px', 'aspect_ratio'),
                                          ('Эквивалентный диаметр, px', 'Отношение осей')):
            ecdf(ax, [row[key] for row in real], REAL, 'Ручные контуры')
            ecdf(ax, signal, SYNTH, 'Нарисованные сомы')
            ax.set(xlabel=title, ylabel='Доля сом')

        density, nearest = grid[2, 1], grid[3, 0]
        visible = [area_density(points, [crop], support)[0] for points, crop in zip(crop_points, crops)]
        if np.isfinite(visible).any():
            scatter_groups(density, ([visible], synthetic_density),
                           ('Видимые по разметке', 'Все нарисованные'), (REAL, SYNTH))
            density.set(title='Плотность в просмотренных кропах', xlabel='Кроп',
                        ylabel='Сом на 1000 px² области размещения', xticks=np.arange(len(crops))+1)
        else:
            missing(density, 'Нужны полностью просмотренные кропы')
        if np.isfinite(real_nn).any():
            comparison_curve(nearest, nn_radii, real_nn, [], REAL, 'Размеченные центры')
            comparison_curve(nearest, nn_radii, None, synthetic_nn, SYNTH, 'Нарисованные центры')
            nearest.set(title='Расстояние до ближайшей сомы с поправкой на край',
                        xlabel='Расстояние, px', ylabel='Доля центров с соседом в радиусе', ylim=(-.02, 1.02))
        else:
            missing(nearest, 'Недостаточно данных о ближайших соседях')

        for i, ax in enumerate((grid[3, 1], grid[4, 0])):
            sources = [(real_shell, REAL, 'Ручные контуры')]
            for key, color, name in RENDERS:
                curves = [row[0] for row in shells[key] if row[0] is not None]
                if curves:
                    sources.append(((curves[0][0], np.concatenate([item[1] for item in curves])),
                                    color, name))
            for source, color, name in sources:
                if source is None:
                    continue
                r, curves = source
                normalized, core, contrast = normalize_profiles(r, curves)
                values = curves-core[:, None] if i == 0 else normalized[contrast > 1]
                shell_curves(ax, r, values, color, name)
            ax.set_title(('Свет оболочки · абсолютный контраст', 'Свет оболочки · форма профиля')[i])
            ax.axvline(1, color='0.6', linestyle=':', linewidth=1)
            ax.set(xlabel='Радиус в долях радиуса маски',
                   ylabel=('Сигнал над центром, ADU', 'Относительный сигнал')[i])

        for ax, values, results, index, title, unit, ylabel in (
                (grid[4, 1], real_contrast, shells, 1, 'Контраст отдельных сом', 'Пик минус центр, ADU', 'Доля сом'),
                (grid[5, 0], real_arcs, shells, 2, 'Неравномерность свечения по дугам', 'SD лог-яркости', 'Доля сом'),
                (grid[5, 1], image_real[0], images, 0, 'Локальный контраст ткани', 'P95 − P5 в окнах 32×32, ADU', 'Доля окон')):
            ecdf(ax, values, REAL, 'Реальная медиана')
            for key, color, name in RENDERS:
                ecdf(ax, np.concatenate([row[index] for row in results[key]]), color, name)
            ax.set(title=title, xlabel=unit, ylabel=ylabel)
            if not ax.get_legend_handles_labels()[0]:
                missing(ax, 'Недостаточно данных')

        ax = grid[6, 0]
        for index, linestyle, domain in ((1, '-', 'ткань'), (2, '--', 'фон')):
            for values, color, name in [(image_real[index], REAL, 'Медиана'),
                                         *((np.nanmedian(np.asarray([row[index] for row in images[key]]), axis=0),
                                            color, name) for key, color, name in RENDERS
                                           if np.isfinite([row[index] for row in images[key]]).any())]:
                if np.isfinite(values).any():
                    ax.plot(PIXEL_LAGS, values, linestyle, marker='o', color=color, label=f'{name} · {domain}')
        ax.axhline(0, color='0.8', linewidth=1)
        ax.set(title='Автокорреляция остатка после вычитания G₁₂', xlabel='Сдвиг, px',
               ylabel='Корреляция', xticks=PIXEL_LAGS)
        if not np.isfinite(image_real[2]).any():
            ax.text(.98, .04, 'Мало внутренних пикселей фона', ha='right', transform=ax.transAxes, fontsize=8)

        ax = grid[6, 1]
        x = (distance_bins[:-1]+distance_bins[1:])/2
        comparison_curve(ax, x, real_pairs, [], REAL, 'Реальная медиана')
        for key, color, name in RENDERS:
            comparison_curve(ax, x, None, [row[3] for row in shells[key]], color, name)
        labels = [f'{lo:.0f}–{hi:.0f}\n({n} пар)' for lo, hi, n in
                  zip(distance_bins[:-1], distance_bins[1:], pair_counts)]
        ax.set(title='Различия контраста соседних сом', ylabel='½ среднее Δ(log C)²',
               xlabel='Интервал расстояния, px (число реальных пар)', xticks=x, xticklabels=labels)
        for ax in grid.flat:
            if ax.axison and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)

        temp = attempt/'metrics.tmp'
        fig.savefig(temp, format='png', dpi=130)
        temp.replace(attempt/'metrics.png')
    finally:
        plt.close(fig)
    return dict(image_real=image_real, images=images, real_nn=real_nn, nn_radii=nn_radii,
                synthetic_nn=synthetic_nn, real_contrast=real_contrast, real_arcs=real_arcs,
                real_pairs=real_pairs, pair_counts=pair_counts, shells=shells)


def write_heldout_metrics(folder, attempt):
    """Compare matched seeds on evaluation crops fixed by the source run."""
    heldout = json.loads((attempt/'heldout.json').read_text())
    if not heldout:
        raise ValueError('В исходном прогоне нет контрольных кропов.')
    report = json.loads((attempt/'report.json').read_text())
    baseline = folder/report['refinement']['source_run']
    before = json.loads((baseline/'report.json').read_text())
    reference = tifffile.imread(folder/'reference.tiff')
    with np.load(baseline/'profile.npz') as profile:
        support = profile['neuron_support'].copy()
    heldout_mask = np.zeros(reference.shape, bool)
    for crop in heldout:
        x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
        heldout_mask[y:y+h, x:x+w] = True
    support &= heldout_mask
    windows = tissue_windows(support)
    radii = np.arange(1, 21)
    points = [np.asarray(crop['points'], float).reshape(-1, 2)[:, ::-1] for crop in heldout]
    contrast, corr, _ = image_statistics(reference, support, windows, [])
    real_nn = nearest_curve(points, heldout, radii)
    seeds = {row['seed']: row['path'] for row in before['scenes']}
    matched = [row for row in report['scenes'] if row['seed'] in seeds]
    selected = np.linspace(0, len(matched)-1, min(16, len(matched))).round().astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    try:
        fig.suptitle(f'Контрольные кропы · {len(heldout)} кропов · '
                     f'{sum(map(len, points))} отмеченных центров · {len(selected)} пар сцен')
        ecdf(axes[0], contrast, REAL, 'Рабочая медиана')
        axes[1].plot(PIXEL_LAGS, corr, 'o-', color=REAL, label='Рабочая медиана')
        axes[2].plot(radii, real_nn, color=REAL, label='Рабочая медиана')
        for run, color, name, paths in (
                (baseline, BASE, 'До доразметки', [seeds[matched[i]['seed']] for i in selected]),
                (attempt, SYNTH, 'После доразметки', [matched[i]['path'] for i in selected])):
            contrasts, correlations, neighbors = [], [], []
            with zipfile.ZipFile(run/'dataset.zip') as archive:
                for path in paths:
                    with np.load(io.BytesIO(archive.read(path))) as scene:
                        values, correlation, _ = image_statistics(scene['image_structured'], support, windows, [])
                        contrasts.extend(values)
                        correlations.append(correlation)
                        neighbors.append(nearest_curve([scene['centers']]*len(heldout), heldout, radii))
            ecdf(axes[0], contrasts, color, name)
            for ax, x, values in ((axes[1], PIXEL_LAGS, correlations), (axes[2], radii, neighbors)):
                lines = np.asarray(values)
                if lines.size:
                    valid = np.isfinite(lines).any(axis=0)
                    if valid.any():
                        q = np.nanpercentile(lines[:, valid], [25, 50, 75], axis=0)
                        ax.plot(x[valid], q[1], color=color, label=name)
                        ax.fill_between(x[valid], q[0], q[2], color=color, alpha=.16)
        axes[0].set(title=f'Локальный контраст · {len(windows)} окон', xlabel='P95–P5, ADU',
                    ylabel='Доля окон ≤ x')
        axes[1].set(title='Автокорреляция ткани', xlabel='Сдвиг, px', ylabel='Корреляция')
        axes[2].set(title='Ближайший сосед', xlabel='Радиус, px', ylabel='Доля центров')
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        temp = attempt/'heldout-metrics.tmp'
        fig.savefig(temp, format='png', dpi=130)
        temp.replace(attempt/'heldout-metrics.png')
    finally:
        plt.close(fig)
