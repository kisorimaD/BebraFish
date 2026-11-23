"""Serve the interactive Plane20 ROI filter and save user marks locally."""
import argparse
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import numpy as np
from skimage.measure import find_contours
import tifffile

from review_somas import atomic_write, encode
from roi_evidence import roi_evidence

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT/'roi_filter_lab'
OUTPUTS = ROOT/'outputs'
MIX = OUTPUTS/'plane20_mixed_training'
MODELS = {
    'wide64': ('Wide64', MIX/'wide64/analysis/median_tuned'),
    'pilot': ('Пилот', MIX/'pilot/median_prediction'),
}
CROPS = [
    ('Ровные сомы', 489, 103), ('Слабый сигнал', 267, 112), ('Плотная ткань', 947, 115),
    ('Край ткани', 1169, 150), ('Рассеянное свечение', 783, 255),
    ('Тёмный участок', 994, 192), ('Размытый фон', 658, 145), ('Смешанная граница', 840, 62),
    ('Внешний фон', 60, 180),
]


def outline(flat, width):
    y, x = np.divmod(flat, width)
    y0, x0 = int(y.min()), int(x.min())
    tile = np.zeros((int(y.max())-y0+3, int(x.max())-x0+3), bool)
    tile[y-y0+1, x-x0+1] = True
    # Display coordinates are native Y horizontally and native X vertically.
    return [(path+[y0-1, x0-1]).ravel().tolist() for path in find_contours(tile, .5)]


def prepare_model(path, image, input_hash):
    report = json.loads((path/'report.json').read_text())
    if report['input_sha256'] != input_hash:
        raise ValueError(f'Предсказания относятся к другой картинке: {path}')
    archive = path/'instances.npz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with np.load(archive) as z:
        centers, scores, offsets, pixels = [z[k] for k in ('centers','scores','instance_offsets','instance_pixel_index')]
        if tuple(z['shape_yx']) != image.shape or len(offsets) != len(centers)+1:
            raise ValueError(f'Некорректные размеры ROI: {archive}')
    footprints = np.split(pixels, offsets[1:-1].astype(np.int64))
    evidence = roi_evidence(image, footprints)
    candidates = []
    for center, score, feature, flat in zip(centers, scores, evidence, footprints):
        candidates.append([*map(float, center), float(score), *map(float, feature), outline(flat, image.shape[1])])
    meta = dict(digest=digest, count=len(candidates), center_threshold=report['threshold'],
                mask_threshold=report['mask_threshold'])
    return meta, gzip.compress(encode(candidates), compresslevel=4)


class Session:
    def __init__(self, marks):
        image = tifffile.imread(OUTPUTS/'plane20_median_resolution/local.tiff').astype('<f4', copy=False)
        self.frame = image.tobytes()
        self.frame_digest = hashlib.sha256(self.frame).hexdigest()
        self.meta = dict(shape=image.shape, crops=CROPS,
                         display_range=np.percentile(image, [1,99.5]).tolist(), models={})
        self.models = {}
        self.marks = marks
        self.lock = Lock()
        for key,(name,path) in MODELS.items():
            info, self.models[key] = prepare_model(path, image, self.frame_digest)
            self.meta['models'][key] = info | dict(name=name)
        self.meta_bytes = encode(self.meta)

    def read_marks(self):
        saved = json.loads(self.marks.read_text()) if self.marks.exists() else {}
        if saved.get('frame_sha256') != self.frame_digest:
            return {}
        return saved.get('models', {})

    def marks_for_model(self, model):
        digest = self.meta['models'][model]['digest']
        return sorted(self.read_marks().get(digest, {}), key=int)

    def set_mark(self, payload):
        model, index, marked = payload['model'], payload['id'], payload['marked']
        info = self.meta['models'][model]
        if type(marked) is not bool or type(index) is not int or not 0 <= index < info['count']:
            raise ValueError('Некорректная отметка ROI')
        with self.lock:
            models = self.read_marks()
            entries = models.setdefault(info['digest'], {})
            if marked:
                entries[str(index)] = True
            else:
                entries.pop(str(index), None)
            self.marks.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self.marks, dict(frame_sha256=self.frame_digest, models=models))
            return sorted(entries, key=int)


def handler_for(session):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB), **kwargs)

        def send_bytes(self, data, content_type='application/json', compressed=False):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/meta':
                self.send_bytes(session.meta_bytes)
            elif path == '/api/frame':
                self.send_bytes(session.frame, 'application/octet-stream')
            elif path.startswith('/api/roi/') and path.split('/')[-1] in MODELS:
                self.send_bytes(session.models[path.split('/')[-1]], compressed=True)
            elif path.startswith('/api/marks/') and path.split('/')[-1] in MODELS:
                with session.lock:
                    self.send_bytes(encode(session.marks_for_model(path.split('/')[-1])))
            elif path in ('/', '/index.html', '/app.js', '/style.css'):
                super().do_GET()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/api/mark':
                self.send_error(404)
                return
            if self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', '')):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('Некорректный запрос')
                self.send_bytes(encode(session.set_mark(json.loads(self.rfile.read(length)))))
            except (KeyError, TypeError, ValueError, OSError):
                self.send_error(400, 'Could not save mark')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8772)
    parser.add_argument('--marks', type=Path, default=ROOT/'annotations/plane20_roi_filter_marks.json')
    args = parser.parse_args()
    session = Session(args.marks)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(session)) as server:
        print(f'ROI Lab: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
