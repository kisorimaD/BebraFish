"""Tile inference with crop normalization and complete overlapping polygon masks."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from skimage.draw import polygon
from stardist.models import StarDist2D
from .data import normalize


def predict(model, image, threshold=.4, nms=.7):
    """Keep central 96px of each crop; never remove objects merely for mask overlap."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('Select a single YX frame')
    h, w = image.shape
    padded = np.pad(image, ((16, 144), (16, 144)), mode='reflect')
    all_coords, all_points, all_scores = [], [], []
    for y in range(0, h, 96):
        for x in range(0, w, 96):
            _, details = model.predict_instances(normalize(padded[y:y+128, x:x+128]),
                prob_thresh=threshold, nms_thresh=nms, verbose=False)
            points = details['points']
            take = np.all((points >= 16) & (points < 112), axis=1)
            points = points[take]+[y-16, x-16]
            take2 = np.all((points >= 0) & (points < [h, w]), axis=1)
            all_points.extend(points[take2])
            all_coords.extend(details['coord'][take][take2]+np.array([y-16, x-16])[None, :, None])
            all_scores.extend(details['prob'][take][take2])
    points = np.asarray(all_points, np.float32).reshape(-1, 2)
    coords = np.asarray(all_coords, np.float32).reshape(-1, 2, model.config.n_rays)
    scores = np.asarray(all_scores, np.float32)
    # Seam duplicates only: close centers AND almost identical polygons.
    suppressed = set()
    if len(points):
        neighbors = cKDTree(points).query_ball_tree(cKDTree(points), 1.5)
        for i in np.argsort(-scores):
            if i in suppressed:
                continue
            for j in neighbors[i]:
                if j != i and scores[j] <= scores[i] and np.mean(np.linalg.norm(coords[i]-coords[j], axis=0)) < 1.5:
                    suppressed.add(j)
    keep = [i for i in range(len(points)) if i not in suppressed]
    return points[keep], coords[keep], scores[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--frame', type=int, default=500)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.4)
    p.add_argument('--nms', type=float, default=.7)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with tifffile.TiffFile(args.image) as tif:
        image = tif.pages[args.frame if len(tif.pages) > 1 else 0].asarray()
    model = StarDist2D(None, name='model', basedir=str(args.run))
    model.load_weights('weights_best.h5')
    points, coords, scores = predict(model, image, args.threshold, args.nms)
    pixels, offsets = [], [0]
    labels = np.zeros(image.shape, np.uint32)
    for i, contour in enumerate(coords):
        y, x = polygon(contour[0], contour[1], shape=image.shape)
        indices = (y*image.shape[1]+x).astype(np.uint32)
        pixels.append(indices); offsets.append(offsets[-1]+len(indices))
        labels[y, x] = i+1
    np.savez_compressed(args.output/'instances.npz', centers=points, contours_yx=coords,
        scores=scores, instance_offsets=np.array(offsets, np.uint64),
        instance_pixel_index=np.concatenate(pixels) if pixels else np.array([], np.uint32),
        shape_yx=np.array(image.shape))
    tifffile.imwrite(args.output/'labels_preview.tiff', labels)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 20))
    ax.imshow(image, cmap='gray', vmin=np.percentile(image, 5), vmax=np.percentile(image, 95))
    for contour in coords:
        ax.plot(np.r_[contour[1], contour[1, 0]], np.r_[contour[0], contour[0, 0]], lw=.4, color='lime')
    ax.axis('off'); fig.savefig(args.output/'overlay.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    (args.output/'report.json').write_text(json.dumps(vars(args) | dict(instances=len(points)), default=str, indent=2)+'\n')
    print(f'{len(points)} instances -> {args.output}')


if __name__ == '__main__':
    main()
