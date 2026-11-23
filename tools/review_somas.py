"""Review model-proposed soma masks against a real Plane20 TIFF frame."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

import numpy as np
from scipy.spatial import cKDTree
import tifffile

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'soma_review'
DEFAULT_INPUT = ROOT / 'data/Plane_Aligned_20.tiff'
DEFAULT_SESSION = ROOT / 'annotations/plane20_model_review_frame500'


def encode(doc):
    return (json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()


def atomic_write(path, doc):
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(encode(doc))
    tmp.replace(path)


def read_frame(path, page):
    with tifffile.TiffFile(path) as tif:
        if not 0 <= page < len(tif.pages):
            raise ValueError(f'Frame index outside 0..{len(tif.pages) - 1}')
        frame = tif.pages[page].asarray()
        frame_count = len(tif.pages)
    dtype = frame.dtype.name
    if frame.ndim != 2 or dtype not in ('uint16', 'float32') or not np.isfinite(frame).all():
        raise ValueError('Expected a finite two-dimensional uint16 or float32 TIFF page')
    data = frame.astype(frame.dtype.newbyteorder('<'), copy=False).tobytes()
    dataset = dict(source_name=path.name, frame_index=page, frame_count=frame_count,
                   frame_sha256=hashlib.sha256(data).hexdigest(), shape_yx=list(frame.shape),
                   dtype=dtype, byte_order='little',
                   display_range=np.percentile(frame, [2, 99]).tolist())
    if dtype == 'float32':
        rounded = np.clip(np.rint(frame), 0, 65535).astype('<u2')
        dataset['rounded_uint16_sha256'] = hashlib.sha256(rounded.tobytes()).hexdigest()
    return dataset, data


def mask_runs(pixels, width):
    """Encode sorted native flat pixel indices as [y, x_start, x_stop)."""
    y, x = np.divmod(pixels, width)
    cuts = np.r_[0, np.flatnonzero((np.diff(y) != 0) | (np.diff(x) != 1)) + 1, len(y)]
    return np.column_stack((y[cuts[:-1]], x[cuts[:-1]], x[cuts[1:] - 1] + 1)).tolist()


def read_predictions(path, dataset, model_id):
    report = json.loads((path.parent / 'report.json').read_text())
    if (report.get('input_sha256') != dataset['frame_sha256'] or report.get('input_dtype') != dataset['dtype']
            or dataset['frame_index'] not in (report.get('input_page'), report.get('frame'))):
        raise ValueError(f'{path} was generated from a different frame or lacks input provenance')
    contents = path.read_bytes()
    with np.load(path) as array:
        centers = array['centers']
        scores = array['scores']
        offsets = array['instance_offsets']
        pixels = array['instance_pixel_index']
        shape = array['shape_yx']
        source_indices = array['source_indices'] if 'source_indices' in array else None
    h, w = dataset['shape_yx']
    n = len(centers)
    if (not np.array_equal(shape, [h, w]) or centers.shape != (n, 2) or scores.shape != (n,)
            or not np.isfinite(centers).all() or np.any((centers < 0) | (centers >= [h, w]))
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or offsets.shape != (n + 1,) or offsets.dtype.kind not in 'ui' or offsets[0] != 0
            or np.any(offsets[1:] <= offsets[:-1]) or offsets[-1] != len(pixels)
            or pixels.ndim != 1 or pixels.dtype.kind not in 'ui' or np.any((pixels < 0) | (pixels >= h*w))
            or report.get('instances') != n):
        raise ValueError(f'Invalid centers or sparse masks: {path}')
    model = {key: report[key] for key in
             ('run', 'checkpoint_sha256', 'denoiser_sha256', 'threshold', 'mask_threshold')}
    model |= dict(id=model_id, name=Path(report['run']).name, instances=n,
                  predictions_sha256=hashlib.sha256(contents).hexdigest())
    for key in ('name', 'filter', 'source_predictions_sha256', 'source_instances'):
        if key in report:
            model[key] = report[key]
    if source_indices is not None and (source_indices.shape != (n,) or source_indices.dtype.kind not in 'ui'
                                       or np.any(source_indices < 0) or len(np.unique(source_indices)) != n):
        raise ValueError(f'Invalid source indices: {path}')
    objects, footprints = [], []
    for i, ((y, x), score) in enumerate(zip(centers, scores)):
        footprint = np.unique(pixels[offsets[i]:offsets[i+1]])
        objects.append(dict(id=f'{model_id}:{i}', model_id=model_id, instance_index=i,
                            center_yx=[float(y), float(x)], score=float(score),
                            mask_runs_yx=mask_runs(footprint, w), area_px=int(footprint.size)))
        if source_indices is not None:
            objects[-1]['source_instance_index'] = int(source_indices[i])
        footprints.append(footprint)
    return model, objects, footprints


def pair_models(objects, footprints):
    """Pair nearby masks across models, keeping both original renderings."""
    if len(objects) == 1 or not all(objects):
        return [[obj] for collection in objects for obj in collection]
    first, second = objects
    centers = [np.array([obj['center_yx'] for obj in collection]) for collection in objects]
    options = []
    for i, neighbors in enumerate(cKDTree(centers[1]).query_ball_point(centers[0], 2.5)):
        a = footprints[0][i]
        for j in neighbors:
            b = footprints[1][j]
            intersection = np.intersect1d(a, b, assume_unique=True).size
            iou = intersection / (len(a) + len(b) - intersection)
            if iou >= .25:
                distance = np.linalg.norm(centers[0][i] - centers[1][j])
                options.append((-iou, distance, i, j))
    paired, used = {}, set()
    for _, _, i, j in sorted(options):
        if i not in paired and j not in used:
            paired[i] = j
            used.add(j)
    groups = [[obj, second[paired[i]]] if i in paired else [obj] for i, obj in enumerate(first)]
    return groups + [[obj] for j, obj in enumerate(second) if j not in used]


def prior_matches(path, dataset, groups):
    if path is None:
        return {}, None
    raw = path.read_bytes()
    prior = json.loads(raw)
    allowed_hashes = {dataset['frame_sha256']}
    if 'rounded_uint16_sha256' in dataset:
        allowed_hashes.add(dataset['rounded_uint16_sha256'])
    if (prior.get('format') != 'soma-annotations-v1'
            or prior['dataset']['frame_sha256'] not in allowed_hashes
            or prior['dataset']['frame_index'] != dataset['frame_index']):
        raise ValueError('Earlier manual annotations belong to another frame')
    entries = prior['annotations']
    if not entries:
        return {}, dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest())
    points = [[a['measurements']['centroid_y'] - .5, a['measurements']['centroid_x'] - .5]
              for a in entries]
    tree = cKDTree(points)
    matches = {}
    for i, group in enumerate(groups):
        distance, j = tree.query([p['center_yx'] for p in group])
        best = int(np.argmin(distance))
        if distance[best] <= 2.5:
            entry = entries[j[best]]
            matches[i] = dict(id=entry['id'], number=entry['number'])
    return matches, dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest())


def build_queue(paths, dataset, prior, support_path, review_rois=False):
    batches = [read_predictions(path, dataset, f'model-{i+1}') for i, path in enumerate(paths)]
    models, candidates, footprints = zip(*batches)
    groups = pair_models(candidates, footprints)
    location_ids = []
    if review_rois:
        original_groups = groups
        groups = []
        for i, group in enumerate(original_groups, 1):
            groups.extend([candidate] for candidate in group)
            location_ids.extend([f'location-{i:05d}']*len(group))
    if not groups:
        raise ValueError('Prediction files contain no nonempty masks')
    matches, prior_source = prior_matches(prior, dataset, groups)
    support = None
    if support_path:
        with np.load(support_path) as profile:
            support = profile['neuron_support']
        if list(support.shape) != dataset['shape_yx']:
            raise ValueError('Tissue support shape differs from TIFF frame')
    # Interleave coarse spatial tiles so early decisions cover multiple regions.
    tiles = {}
    for i, variants in enumerate(groups):
        y, x = variants[0]['center_yx']
        outside = support is not None and not bool(support[int(y), int(x)])
        tile = (outside, i in matches, int(y // 64), int(x // 64))
        group = dict(id=f'soma-{i+1:05d}', candidates=variants,
                     nearby_manual_annotation=matches.get(i), outside_priority_region=outside)
        if review_rois:
            group['location_id'] = location_ids[i]
        tiles.setdefault(tile, []).append(group)
    rng = np.random.default_rng(500)
    ranked = []
    keys = sorted(tiles)
    for tile_rank, index in enumerate(rng.permutation(len(keys))):
        tile = keys[index]
        records = tiles[tile]
        rng.shuffle(records)
        ranked.extend(((order, tile[:2]) if review_rois else (tile[:2], order), tile_rank, group)
                      for order, group in enumerate(records))
    queue = [item[-1] for item in sorted(ranked, key=lambda item: item[:-1])]
    doc = dict(format='soma-review-proposals-v1', dataset=dataset, models=models,
                prior_annotations=prior_source,
                priority_region=(dict(name=support_path.name, array='neuron_support',
                    sha256=hashlib.sha256(support_path.read_bytes()).hexdigest()) if support_path else None),
                coordinates='Native integer-centered YX; mask runs are [y, x_start, x_stop_exclusive]',
                mask_target='Whole soma including dark center; intended contour roughly mid-shell',
                group_pairing=dict(max_center_distance_px=2.5, minimum_mask_iou=.25),
                proposals=queue)
    if review_rois:
        doc.update(review_rois=True,
                   mask_target='Visible outer contour of the full soma, including the dark center')
    return doc


class ReviewSession:
    def __init__(self, path, dataset, frame, predictions, prior, support=None, review_rois=False):
        self.frame = frame
        self.path = path
        self.lock = threading.Lock()
        proposals_path = path / 'proposals.json'
        if predictions:
            proposals = build_queue(predictions, dataset, prior, support, review_rois)
            path.mkdir(parents=True, exist_ok=False)
            atomic_write(proposals_path, proposals)
        self.proposals_bytes = proposals_path.read_bytes()
        self.proposals = json.loads(self.proposals_bytes)
        if (self.proposals['format'] != 'soma-review-proposals-v1'
                or self.proposals['dataset'] != dataset):
            raise ValueError('Review session belongs to a different TIFF frame')
        self.digest = hashlib.sha256(self.proposals_bytes).hexdigest()
        self.groups = {p['id']: {c['id'] for c in p['candidates']} for p in self.proposals['proposals']}
        self.decisions_path = path / 'decisions.json'
        if not self.decisions_path.exists():
            atomic_write(self.decisions_path, dict(format='soma-review-decisions-v1',
                         proposals_sha256=self.digest, decisions={}))
        self.read_decisions()

    def read_decisions(self):
        state = json.loads(self.decisions_path.read_bytes())
        if state.get('format') != 'soma-review-decisions-v1' or state.get('proposals_sha256') != self.digest:
            raise ValueError('Decisions belong to another proposal set')
        for group, decision in state['decisions'].items():
            if (group not in self.groups or decision['candidate_id'] not in self.groups[group]
                    or not self.valid_verdict(decision['status'], decision.get('reason'))):
                raise ValueError(f'Invalid decision for {group}')
        return state

    def valid_verdict(self, status, reason):
        if status == 'rejected':
            return (reason in ('no_soma', 'bad_mask') if self.proposals.get('review_rois')
                    else reason is None)
        return reason is None and (status == 'accepted' or
                                   (status == 'uncertain' and self.proposals.get('review_rois', False)))

    def update(self, payload):
        group = payload['id']
        candidate = payload['candidate_id']
        status, reason = payload['status'], payload.get('reason')
        if group not in self.groups or candidate not in self.groups[group]:
            raise ValueError('Unknown proposal or mask')
        if not (status is None and reason is None) and not self.valid_verdict(status, reason):
            raise ValueError('Invalid decision')
        with self.lock:
            state = self.read_decisions()
            if status is None:
                state['decisions'].pop(group, None)
            else:
                state['decisions'][group] = dict(status=status, candidate_id=candidate,
                                                reviewed_at=datetime.now(timezone.utc).isoformat())
                if reason is not None:
                    state['decisions'][group]['reason'] = reason
            atomic_write(self.decisions_path, state)
            return state


def handler_for(session):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=WEB, **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def send_data(self, data, content_type='application/json; charset=utf-8', attachment=None):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/frame':
                self.send_data(session.frame, 'application/octet-stream')
            elif path == '/api/proposals':
                self.send_data(session.proposals_bytes)
            elif path == '/api/decisions':
                self.send_data(encode(session.read_decisions()))
            elif path == '/api/export':
                state = session.read_decisions()
                doc = session.proposals | dict(format='soma-review-export-v1',
                    proposals_sha256=session.digest, decisions=state['decisions'])
                self.send_data(encode(doc), attachment='soma-review.json')
            elif path in ('/', '/index.html', '/app.js', '/style.css'):
                super().do_GET()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/api/decision':
                self.send_error(404)
                return
            if self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', '')):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                self.send_data(encode(session.update(payload)))
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send_error(400, str(error))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--frame', type=int, default=500, help='Zero-based TIFF page')
    parser.add_argument('--predictions', type=Path, nargs='+', help='One or two instances.npz with sibling reports; creates a new session')
    parser.add_argument('--prior', type=Path, help='Optional manual annotations for identifying already reviewed locations')
    parser.add_argument('--support-profile', type=Path, help='NPZ with neuron_support for prioritizing candidate locations')
    parser.add_argument('--session', type=Path, default=DEFAULT_SESSION)
    parser.add_argument('--review-rois', action='store_true', help='Review each candidate and record the reason for rejection')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    if args.predictions and len(args.predictions) > 2:
        parser.error('Use one or two model outputs')
    if (args.prior or args.support_profile or args.review_rois) and not args.predictions:
        parser.error('--prior, --support-profile and --review-rois are only read when importing predictions')
    with ThreadingHTTPServer(('127.0.0.1', args.port), SimpleHTTPRequestHandler) as server:
        dataset, frame = read_frame(args.input, args.frame)
        session = ReviewSession(args.session, dataset, frame, args.predictions, args.prior, args.support_profile,
                                args.review_rois)
        server.RequestHandlerClass = handler_for(session)
        url = f'http://127.0.0.1:{args.port}'
        print(f'Soma Review: {url}\nProposals: {len(session.groups)}\nSession: {session.path.resolve()}',
              flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
