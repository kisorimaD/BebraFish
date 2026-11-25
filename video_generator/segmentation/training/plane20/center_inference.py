"""End-to-end prediction/evaluation of independent center-conditioned soma masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .center_model import build_model
from .center_data import roi_masks, ROI_SIZE, roi_anchor
from .data import CropSource, normalize
from .evaluate import instance_metrics


def load_model(run):
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    config = json.loads((Path(run)/'model_config.json').read_text())
    if config.get('schema') != 'plane20-center-mask-v1' or config.get('roi_size') != ROI_SIZE:
        raise ValueError('Checkpoint uses a different center-mask format or ROI size')
    model = build_model(config['base_filters'])
    model.load_weights(str(Path(run)/'weights_best.h5'))
    return model


def masks_at(model, features, images, points):
    values = []
    for start in range(0, len(points), 128):
        p = tf.convert_to_tensor(points[start:start+128], tf.float32)
        count = len(p)
        p = tf.pad(p, ((0, 128-count), (0, 0)), constant_values=64.)
        logits = model.decode_masks(features, images, p, tf.zeros(128, tf.int32), training=False)
        values.append(tf.sigmoid(logits)[:count, ..., 0].numpy())
    probabilities = np.concatenate(values) if values else np.empty((0, ROI_SIZE, ROI_SIZE), np.float32)
    return probabilities, roi_anchor(points)-ROI_SIZE//2


def patch_prediction(model, image, threshold=.3, oracle_points=None):
    images = tf.convert_to_tensor(image[None, ..., None], tf.float32)
    features, logits, offsets = model.encoder(images, training=False)
    probability = tf.sigmoid(logits)
    peaks = (probability == tf.nn.max_pool2d(probability, ksize=3, strides=1, padding='SAME')).numpy()[0, ..., 0]
    heat, offsets = probability.numpy()[0, ..., 0], offsets.numpy()[0]
    y, x = np.nonzero(peaks & (heat >= threshold))
    scores = heat[y, x]
    points = np.column_stack((y, x))+np.clip(offsets[y, x], -.5, .5)
    take = np.all((points >= 16) & (points < 112), axis=1)
    points, scores = points[take], scores[take]
    order = np.argsort(-scores)[:1024]
    points, scores = points[order].astype(np.float32), scores[order]
    masks, origins = masks_at(model, features, images, points)
    result = dict(centers=points, scores=scores, probabilities=masks, origins=origins, heatmap=heat)
    if oracle_points is not None:
        om, oo = masks_at(model, features, images, np.asarray(oracle_points, np.float32))
        result['oracle'] = dict(centers=np.asarray(oracle_points), probabilities=om, origins=oo)
    return result


def rasterize(mask, origin, shape):
    result = np.zeros(shape, bool)
    start = np.maximum(origin, 0); stop = np.minimum(np.asarray(origin)+mask.shape, shape)
    if np.all(stop > start):
        result[start[0]:stop[0], start[1]:stop[1]] = mask[
            start[0]-origin[0]:stop[0]-origin[0], start[1]-origin[1]:stop[1]-origin[1]]
    return result


def summarize(total):
    tp, fp, fn = (total[k] for k in ('tp', 'fp', 'fn'))
    return total | dict(precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
        f1=2*tp/max(2*tp+fp+fn, 1), matched_mean_iou=total['matched_iou_sum']/max(tp, 1))


def evaluate(args, restore=None):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_model(args.run); source = CropSource(args.data, args.split, args.seed, use_precomputed=False, keep_scenes=True)
    totals = {name: dict(tp=0, fp=0, fn=0, matched_iou_sum=0.) for name in ('predicted_centers', 'oracle_centers')}
    center_counts = dict(tp=0, fp=0, fn=0)
    for i in range(args.start_index, args.start_index+args.crops):
        image, scene, origin, _ = source.raw_crop(i)
        image = normalize(image if restore is None else restore(image))
        centers = scene['centers']-origin
        ids = np.flatnonzero(np.all((centers >= 16) & (centers < 112), axis=1)); truth = centers[ids]
        gm = roi_masks(scene, origin, ids, truth)[..., 0]
        go = roi_anchor(truth)-ROI_SIZE//2
        gt = [rasterize(m.astype(bool), o, (128, 128)) for m, o in zip(gm, go)]
        prediction = patch_prediction(model, image, args.threshold, oracle_points=truth)
        for name, p in [('predicted_centers', prediction), ('oracle_centers', prediction['oracle'])]:
            masks = [rasterize(m >= args.mask_threshold, o, (128, 128)) for m, o in zip(p['probabilities'], p['origins'])]
            masks = [mask for mask in masks if mask.any()]
            row = instance_metrics(gt, masks)
            for k in row:
                totals[name][k] += row[k]
        pred = prediction['centers'];ng, npred = len(truth), len(pred); tp = 0
        if ng and npred:
            distance = np.linalg.norm(truth[:, None]-pred[None], axis=-1)
            rows, cols = linear_sum_assignment(-((distance <= 2)*(min(ng, npred)+1)+np.maximum(1-distance/2, 0)))
            tp = int(np.sum(distance[rows, cols] <= 2))
        for k, v in dict(tp=tp, fp=npred-tp, fn=ng-tp).items():
            center_counts[k] += v
    tp, fp, fn = (center_counts[k] for k in ('tp', 'fp', 'fn'))
    report = vars(args) | {k: summarize(v) for k, v in totals.items()} | dict(
        center_detection=center_counts | dict(tolerance_px=2., precision=tp/max(tp+fp, 1),
            recall=tp/max(tp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1)),
        scope='Synthetic only. Full overlapping masks, one-to-one IoU>=0.5. Oracle centers are diagnostic, not deployable performance.')
    args.output.write_text(json.dumps(report, default=str, indent=2)+'\n')
    print(json.dumps(report, default=str, indent=2))
    return report


def predict(args):
    denoiser_path = getattr(args, 'denoiser', None)
    denoiser_sha256 = None
    if denoiser_path:
        from .train_denoiser import build_model as build_denoiser, restore_batch

        denoiser_sha256 = hashlib.sha256(denoiser_path.read_bytes()).hexdigest()
        config = json.loads((Path(args.run) / 'model_config.json').read_text())
        expected = config.get('denoiser_weights_sha256')
        if expected and expected != denoiser_sha256:
            raise ValueError('Denoiser weights differ from detector training configuration')
    model = load_model(args.run)
    if denoiser_path:
        denoiser = build_denoiser()
        denoiser.load_weights(str(denoiser_path))
    args.output.mkdir(parents=True, exist_ok=False)
    with tifffile.TiffFile(args.image) as tif:
        input_page = args.frame if len(tif.pages) > 1 else 0
        image = tif.pages[input_page].asarray()
    h, w = image.shape;padded = np.pad(image, ((16, 144), (16, 144)), mode='reflect')
    points, scores, origins, masks = [], [], [], []
    for y in range(0, h, 96):
        for x in range(0, w, 96):
            tile = padded[y:y+128, x:x+128]
            if denoiser_path:
                tile = restore_batch(denoiser, tile[None])[0]
            p = patch_prediction(model, normalize(tile), args.threshold)
            for point, score, origin, probability in zip(p['centers'], p['scores'], p['origins'], p['probabilities']):
                point = point+[y-16, x-16]
                if not np.all((point >= 0) & (point < [h, w])):
                    continue
                origin = origin+[y-16, x-16];mask = probability >= args.mask_threshold
                start, stop = np.maximum(origin, 0), np.minimum(origin+mask.shape, [h, w])
                if not np.all(stop > start) or not mask[start[0]-origin[0]:stop[0]-origin[0], start[1]-origin[1]:stop[1]-origin[1]].any():
                    continue
                points.append(point);scores.append(score);origins.append(origin);masks.append(mask)
    points = np.asarray(points, np.float32).reshape(-1, 2);scores = np.asarray(scores, np.float32)
    suppressed = set()
    if len(points):
        neighbors = cKDTree(points).query_ball_point(points, 1.)
        for i in np.argsort(-scores):
            if i in suppressed:
                continue
            for j in neighbors[i]:
                if i != j and scores[j] <= scores[i]:
                    suppressed.add(j)
    keep = [i for i in range(len(points)) if i not in suppressed]
    pixels, offsets = [], [0];preview = np.zeros((h, w), np.uint32)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 20))
    ax.imshow(image, cmap='gray', vmin=np.percentile(image, 5), vmax=np.percentile(image, 95))
    for k, i in enumerate(keep):
        ry, rx = np.nonzero(masks[i]);ry, rx = ry+origins[i][0], rx+origins[i][1]
        valid = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w);ry, rx = ry[valid], rx[valid]
        flat = (ry*w+rx).astype(np.uint32);pixels.append(flat);offsets.append(offsets[-1]+len(flat));preview[ry, rx] = k+1
        ax.contour(np.arange(ROI_SIZE)+origins[i][1], np.arange(ROI_SIZE)+origins[i][0], masks[i], levels=[.5], colors='lime', linewidths=.4)
    ax.axis('off');fig.savefig(args.output/'overlay.png', dpi=160, bbox_inches='tight');plt.close(fig)
    np.savez_compressed(args.output/'instances.npz', centers=points[keep], scores=scores[keep],
        instance_offsets=np.array(offsets, np.uint64),
        instance_pixel_index=np.concatenate(pixels) if pixels else np.array([], np.uint32), shape_yx=np.array([h, w]))
    tifffile.imwrite(args.output/'labels_preview.tiff', preview)
    report = vars(args) | dict(
        instances=len(keep), input_page=input_page, input_dtype=image.dtype.name,
        input_sha256=hashlib.sha256(image.astype(image.dtype.newbyteorder('<'), copy=False).tobytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((Path(args.run) / 'weights_best.h5').read_bytes()).hexdigest(),
        denoiser_sha256=denoiser_sha256,
        representation='Independent masks; preview has one owner only',
    )
    (args.output/'report.json').write_text(json.dumps(report, default=str, indent=2)+'\n')
    print(json.dumps(report, default=str, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'predict'):
        q = sub.add_parser(name);q.add_argument('--run', type=Path, required=True);q.add_argument('--output', type=Path, required=True)
        q.add_argument('--threshold', type=float, default=.3);q.add_argument('--mask-threshold', type=float, default=.5)
        if name == 'evaluate':
            q.add_argument('--data', type=Path, required=True);q.add_argument('--split', choices=['train', 'val', 'test'], default='val')
            q.add_argument('--crops', type=int, default=24);q.add_argument('--start-index', type=int, default=32);q.add_argument('--seed', type=int, default=991)
        else:
            q.add_argument('--image', type=Path, required=True);q.add_argument('--frame', type=int, default=500)
            q.add_argument('--denoiser', type=Path, help='Optional denoiser weights, applied tile by tile before normalization')
    args = p.parse_args()
    (evaluate if args.command == 'evaluate' else predict)(args)


if __name__ == '__main__':
    main()
