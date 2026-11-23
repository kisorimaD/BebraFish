"""Instance metrics on held-out synthetic crops, including full overlaps."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon
from stardist.models import StarDist2D
from .data import normalize


def instance_metrics(gt, pred, threshold=.5):
    ng, npred = len(gt), len(pred)
    if not ng or not npred:
        return dict(tp=0, fp=npred, fn=ng, matched_iou_sum=0.)
    g, p = np.asarray(gt, np.float32).reshape(ng, -1), np.asarray(pred, np.float32).reshape(npred, -1)
    intersection = g@p.T
    iou = intersection/np.maximum(g.sum(1)[:, None]+p.sum(1)[None, :]-intersection, 1)
    # Maximize valid match count before IoU; no double credit per object.
    reward = (iou >= threshold)*(min(ng, npred)+1)+iou
    rows, cols = linear_sum_assignment(-reward)
    good = iou[rows, cols] >= threshold
    tp = int(good.sum())
    return dict(tp=tp, fp=npred-tp, fn=ng-tp, matched_iou_sum=float(iou[rows[good], cols[good]].sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--crops', type=int, default=32)
    p.add_argument('--threshold', type=float, default=.4)
    p.add_argument('--nms', type=float, default=.7)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = StarDist2D(None, name='model', basedir=str(args.run))
    model.load_weights('weights_best.h5')
    paths = sorted((args.data/args.split).glob('*.npz'))
    if not paths:
        raise ValueError('No held-out scenes')
    rng = np.random.default_rng(1024)
    totals = dict(tp=0, fp=0, fn=0, matched_iou_sum=0.)
    yy, xx = np.mgrid[:128, :128]
    for i in range(args.crops):
        with np.load(paths[i % len(paths)]) as z:
            image = z['image']
            y, x = int(rng.integers(image.shape[0]-127)), int(rng.integers(image.shape[1]-127))
            centers = z['centers']-[y, x]
            take = np.flatnonzero(np.all((centers >= 16) & (centers < 112), axis=1))
            gt = []
            for k in take:
                theta = np.radians(z['orientation_deg'][k]); c, s = np.cos(theta), np.sin(theta)
                dy, dx = yy-centers[k, 0], xx-centers[k, 1]
                gt.append(((dx*c+dy*s)/(z['major_axis_px'][k]/2))**2+
                          ((-dx*s+dy*c)/(z['minor_axis_px'][k]/2))**2 <= 1)
        _, details = model.predict_instances(normalize(image[y:y+128, x:x+128]),
            prob_thresh=args.threshold, nms_thresh=args.nms, verbose=False)
        take = np.all((details['points'] >= 16) & (details['points'] < 112), axis=1)
        pred = []
        for contour in details['coord'][take]:
            mask = np.zeros((128, 128), bool)
            ry, rx = polygon(contour[0], contour[1], shape=mask.shape); mask[ry, rx] = True
            pred.append(mask)
        row = instance_metrics(gt, pred)
        for key in totals:
            totals[key] += row[key]
    tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
    result = vars(args) | totals | dict(iou_threshold=.5,
        precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
        f1=2*tp/max(2*tp+fp+fn, 1), matched_mean_iou=totals['matched_iou_sum']/max(tp, 1),
        scope='Synthetic only; seeded crops may overlap within a held-out scene. Not evidence of real-data accuracy.')
    args.output.write_text(json.dumps(result, default=str, indent=2)+'\n')
    print(json.dumps(result, default=str, indent=2))


if __name__ == '__main__':
    main()
