"""Prepare model proposals for the existing Soma Lab editor."""
import json
import math
import os
import shlex
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import tifffile
from skimage.measure import approximate_polygon, find_contours, points_in_poly

from review_somas import read_frame, read_predictions
from soma_shell_model import normalize_profiles, radial_profiles


def training_python():
    return os.environ.get('BEBRAFISH_TRAINING_PYTHON', sys.executable)


def train_and_predict(folder, run, session, update):
    python = training_python()
    seed = str(json.loads((run/'report.json').read_text())['seed'])
    log_path = session/'training.log'

    def command(*args):
        argv = [python, *map(str, args)]
        with log_path.open('ab') as log:
            log.write((shlex.join(argv)+'\n').encode())
            log.flush()
            subprocess.run(argv, cwd=Path(__file__).resolve().parents[1],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    update('Проверяем GPU для TensorFlow…')
    try:
        command('-c', "import tensorflow as tf; raise SystemExit(0 if tf.config.list_physical_devices('GPU') else 1)")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('TensorFlow не видит доступную GPU или среда обучения не запускается.') from error
    try:
        update('Генерируем обучающие сцены…')
        command('-m', 'training.plane20.generate_median', '--profile', run/'profile.npz',
                '--output', session/'dataset', '--seed', seed, '--workers', 2,
                '--train', 16, '--val', 4, '--test', 1, '--fit-mask-roi')
        update('Обучаем модель центров и масок…')
        command('-m', 'training.plane20.train_center_mask', '--data', session/'dataset',
                '--output', session/'model', '--seed', seed, '--epochs', 20,
                '--steps', 100, '--val-steps', 8, '--batch-size', 8)
        update('Ищем сомы на рабочей медиане…')
        command('-m', 'training.plane20.center_inference', 'predict', '--run', session/'model',
                '--image', folder/'reference.tiff', '--frame', 0, '--output', session/'prediction')
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('Подготовка датасета, обучение или инференс завершились ошибкой.') from error
    return session/'prediction'/'instances.npz'


def contains(crop, x, y):
    return crop['x'] <= x < crop['x']+crop['width'] and crop['y'] <= y < crop['y']+crop['height']


def intersects(a, b):
    return (a['x'] < b['x']+b['width'] and b['x'] < a['x']+a['width']
            and a['y'] < b['y']+b['height'] and b['y'] < a['y']+a['height'])


def contour(runs):
    """Convert native raster pixels to a verified pixel-edge polygon."""
    y, x, end = np.asarray(runs).T
    top, left = y.min(), x.min()
    mask = np.zeros((y.max()-top+3, end.max()-left+2), bool)
    for row, start, stop in runs:
        mask[row-top+1, start-left+1:stop-left+1] = True
    loops = find_contours(mask, .5)
    outer = max(loops, key=len)
    polygon = approximate_polygon(outer, 0)[:-1]
    if len(polygon) > 20000:
        return None, False
    complete = len(loops) == 1
    if complete:
        pixels = np.indices(mask.shape).reshape(2, -1).T
        complete = np.array_equal(points_in_poly(pixels, polygon).reshape(mask.shape), mask)
    return (polygon[:, ::-1]+[left-.5, top-.5]).tolist(), complete


def suggest_crops(image, tissue, objects, reserved):
    h, w = image.shape
    size = min(96, h, w)
    occupancy = np.zeros_like(tissue)
    for obj in objects:
        y, x = obj['center_yx']
        occupancy[min(h-1, math.floor(y+.5)), min(w-1, math.floor(x+.5))] = True
    options = []
    for y in range(0, h-size+1, size):
        for x in range(0, w-size+1, size):
            crop = dict(x=x, y=y, width=size, height=size)
            if any(intersects(crop, item) for item in reserved):
                continue
            roi = np.s_[y:y+size, x:x+size]
            if tissue[roi].mean() < .5:
                continue
            contrast = float(np.ptp(np.percentile(image[roi][tissue[roi]], [5, 95])))
            options.append((contrast, int(occupancy[roi].sum()), crop))
    options.sort(key=lambda option: option[0])
    crops = []
    for band in np.array_split(np.arange(len(options)), 3):
        remaining = sorted((options[i] for i in band), key=lambda option: option[1])
        for index in dict.fromkeys([0, len(remaining)//2, len(remaining)-1]):
            if remaining:
                crops.append(remaining[index][2])
    return crops


def prepare_review(folder, run, predictions):
    dataset, _ = read_frame(folder/'reference.tiff', 0)
    model, objects, _ = read_predictions(predictions, dataset, 'model')
    with zipfile.ZipFile(run/'dataset.zip') as archive:
        soma = json.loads(archive.read('soma.json'))
    baseline = json.loads((run/'centers.json').read_text())
    report = json.loads((run/'report.json').read_text())
    heldout = json.loads((run/'heldout.json').read_text()) if (run/'heldout.json').exists() else []
    with np.load(run/'profile.npz') as profile:
        suggestions = suggest_crops(tifffile.imread(folder/'reference.tiff'), profile['tissue'],
                                    objects, baseline+heldout)
    if not suggestions:
        suggestions = [crop for crop in soma['crops'] if not any(intersects(crop, item) for item in heldout)][:6]
    session_id = predictions.parent.parent.name
    review_crops = [dict(crop, id=f'{session_id}-crop-{i}', name=f'Проверка {i+1}')
                    for i, crop in enumerate(suggestions)]
    if not review_crops:
        raise ValueError('Нет кропов для доразметки вне контрольных областей.')
    manual = []
    for cell in soma['annotations']:
        original = cell.copy()
        if 'decision' in original:
            original['prior_review'] = {key: original.pop(key) for key in
                                        ('decision', 'model_center_xy', 'score', 'exact_mask', 'edited')
                                        if key in original}
        manual.append(original)
    manual_polygons = [np.asarray(cell['points_xy']) for cell in manual]
    annotations = []
    h, w = dataset['shape_yx']
    for obj in objects:
        y, x = obj['center_yx']
        x, y = min(x+.5, w-.5), min(y+.5, h-.5)
        if any(contains(crop, x, y) for crop in heldout):
            continue
        crop = next((crop for crop in review_crops if contains(crop, x, y)), None)
        if crop is None:
            continue
        if any(points_in_poly([[x, y]], polygon)[0] for polygon in manual_polygons):
            continue
        polygon, exact = contour(obj['mask_runs_yx'])
        if polygon is None:
            continue
        truncated = any(px <= 0 or px >= w or py <= 0 or py >= h for px, py in polygon)
        annotations.append(dict(id=f'{session_id}:{obj["instance_index"]}', number=soma['next_number']+len(annotations),
                                points_xy=polygon, crop_id=crop['id'], confidence='uncertain',
                                truncated=truncated, decision='pending', exact_mask=exact,
                                model_center_xy=[x, y], score=obj['score']))
    return dict(soma, annotations=manual+annotations, crops=review_crops+soma['crops'],
                next_number=soma['next_number']+len(annotations), source_run=run.name,
                model=model, original_ids=[cell['id'] for cell in manual],
                reviewed_crops=[], baseline=baseline, heldout=heldout,
                count=len(report['scenes']), seed=report['seed'])


def refinement_input(doc, initial, image, support):
    """Use checked crops for density and explicit contour decisions for morphology."""
    from analyze_somas import measure

    original_ids = set(initial['original_ids'])
    original = [cell for cell in doc['annotations'] if cell['id'] in original_ids]
    centered = []
    additions = []
    added_by_hand = []
    h, w = image.shape
    reviewed = [crop for crop in doc['crops'] if crop['id'] in doc['reviewed_crops']]
    if any(intersects(crop, hold) for crop in reviewed for hold in initial['heldout']):
        raise ValueError('Просмотренный кроп пересекает контрольную область.')
    if any(intersects(crop, other) for i, crop in enumerate(reviewed) for other in reviewed[:i]):
        raise ValueError('Выберите непересекающиеся кропы для оценки плотности.')
    for cell in doc['annotations']:
        decision = cell.get('decision')
        if decision in ('pending', 'bad_mask', 'uncertain'):
            x, y = cell['model_center_xy']
            if decision == 'pending' and any(contains(crop, x, y) for crop in reviewed):
                raise ValueError('В просмотренном кропе остались предложения без решения.')
            if decision == 'bad_mask':
                # The user has confirmed a soma exists here, even if its contour is unusable.
                centered.append((math.floor(x), math.floor(y)))
            continue
        if decision == 'no_soma':
            continue
        row = measure(cell['points_xy'])
        cx, cy = row['centroid_x'], row['centroid_y']
        if not (math.isfinite(cx) and math.isfinite(cy)):
            continue
        if cell['truncated'] and cell['id'] not in original_ids:
            # A contour cut by the crop edge cannot reliably locate a cell center.
            continue
        centered.append((min(w-1, math.floor(cx)), min(h-1, math.floor(cy))))
        if (cell['id'] not in original_ids and cell['confidence'] == 'confident'
                and not cell['truncated'] and row['minor_axis_px'] >= 2
                and not any(contains(crop, cx, cy) for crop in initial['heldout'])):
            additions.append(cell)
            if decision is None:
                radii, profile = radial_profiles(image, [[cy, cx]], [row['major_axis_px']],
                                                [row['minor_axis_px']], [row['orientation_deg']],
                                                coordinate_offset=.5)
                _, _, contrast = normalize_profiles(radii, profile)
                added_by_hand.append(dict(id=cell['id'], equivalent_diameter_px=row['equivalent_diameter_px'],
                                          center_xy=[cx, cy], shell_contrast_adu=float(contrast[0])))

    crops = [crop for crop in initial['baseline'] if not any(intersects(crop, other) for other in reviewed)]
    outside_support = 0
    for crop in reviewed:
        points = {(x, y) for x, y in centered if contains(crop, x, y)}
        supported = {(x, y) for x, y in points if support[y, x]}
        outside_support += len(points)-len(supported)
        crops.append(dict(crop, points=sorted(supported)))
    if not additions and not reviewed:
        raise ValueError('Подтвердите хотя бы один полный контур или отметьте просмотренный кроп.')
    summary = dict(accepted=len(additions)-len(added_by_hand), drawn=len(added_by_hand),
                   drawn_details=added_by_hand, reviewed_crops=len(reviewed),
                   centers_outside_support=outside_support)
    return dict(doc, annotations=original+additions), crops, summary
