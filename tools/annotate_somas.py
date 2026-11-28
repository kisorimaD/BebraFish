"""Local soma annotation sandbox. Run from any directory; no web dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import threading
import webbrowser

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT/'annotation_sandbox'


def prepare(input_path, frame_index, annotation_dir, *, web_dir=None):
    with tifffile.TiffFile(input_path) as tif:
        if not 0 <= frame_index < len(tif.pages):
            raise ValueError(f'Frame index outside 0..{len(tif.pages)-1}')
        frame = tif.pages[frame_index].asarray()
        frame_count = len(tif.pages)
    if frame.ndim != 2 or frame.dtype != np.uint16:
        raise ValueError('Expected a 2D uint16 TIFF page')
    data = frame.astype('<u2', copy=False).tobytes()
    digest = hashlib.sha256(data).hexdigest()
    h, w = frame.shape
    meta = dict(source_name=input_path.name, source_path=str(input_path.resolve()),
                frame_index=frame_index, frame_count=frame_count, shape_yx=[h, w],
                frame_sha256=digest, dtype='uint16', byte_order='little', pixel_size_um=None,
                display_range=np.percentile(frame, [1, 99.5]).round().astype(int).tolist())
    assets = (WEB if web_dir is None else web_dir)/'assets'; assets.mkdir(exist_ok=True)
    (assets/'frame.bin').write_bytes(data)
    (assets/'dataset.json').write_text(json.dumps(meta, indent=2)+'\n')
    crops = []
    size = min(192, h, w)
    # A systematic grid across the frame, not a selection of attractive cells.
    for y in np.linspace(0, h-size, min(6, max(1, h//size))).round().astype(int):
        for x in sorted(set([0, w-size])):
            index = len(crops)+1
            crops.append(dict(id=f'grid-{index}', name=f'Кроп {index:02d}',
                              x=int(x), y=int(y), width=size, height=size, custom=False))
    doc = dict(format='soma-annotations-v1', dataset=meta,
               coordinate_system='TIFF pixel-edge coordinates; points_xy=[x,y]; native shape_yx',
               annotation_target='Whole soma, including dark nucleus/center',
               created_at=datetime.now(timezone.utc).isoformat(), updated_at='',
               next_number=1, crops=crops, annotations=[])
    annotation_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_dir/f'{input_path.stem}_frame{frame_index}_{digest[:12]}.json'
    if path.exists():
        existing = json.loads(path.read_text())
        validate(existing, meta)
        shutil.copy2(path, path.with_suffix('.session-backup.json'))
    else:
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False)+'\n')
    return meta, path


def validate(doc, meta):
    if not isinstance(doc, dict) or doc.get('format') != 'soma-annotations-v1':
        raise ValueError('Unsupported annotation format')
    dataset = doc.get('dataset', {})
    if dataset.get('frame_sha256') != meta['frame_sha256'] or dataset.get('frame_index') != meta['frame_index']:
        raise ValueError('Annotations belong to another frame')
    if dataset.get('shape_yx') != meta['shape_yx']:
        raise ValueError('Shape mismatch')
    h, w = meta['shape_yx']
    crops = doc.get('crops')
    if not isinstance(crops, list) or not crops or len(crops) > 10000:
        raise ValueError('Invalid crop list')
    crop_ids = set()
    for c in crops:
        if not isinstance(c, dict) or not isinstance(c.get('id'), str) or c['id'] in crop_ids:
            raise ValueError('Invalid/duplicate crop ID')
        crop_ids.add(c['id'])
        if not all(type(c.get(k)) is int for k in ['x', 'y', 'width', 'height']):
            raise ValueError('Crop bounds must be integers')
        if not (0 <= c['x'] < c['x']+c['width'] <= w and 0 <= c['y'] < c['y']+c['height'] <= h):
            raise ValueError('Crop outside image')
    cells = doc.get('annotations')
    if not isinstance(cells, list) or len(cells) > 50000:
        raise ValueError('Invalid annotations')
    ids, numbers = set(), set()
    for a in cells:
        if not isinstance(a, dict) or not isinstance(a.get('id'), str) or a['id'] in ids:
            raise ValueError('Invalid/duplicate cell ID')
        ids.add(a['id'])
        number = a.get('number')
        if type(number) is not int or number < 1 or number in numbers:
            raise ValueError('Invalid/duplicate cell number')
        numbers.add(number)
        if a.get('confidence') not in ['confident', 'uncertain'] or type(a.get('truncated')) is not bool:
            raise ValueError('Invalid cell attributes')
        if a.get('crop_id') not in crop_ids:
            raise ValueError('Cell refers to missing crop')
        points = a.get('points_xy')
        if not isinstance(points, list) or not 3 <= len(points) <= 20000:
            raise ValueError('Invalid contour')
        for pt in points:
            if not isinstance(pt, list) or len(pt) != 2 or not all(type(v) in [int, float] and math.isfinite(v) for v in pt):
                raise ValueError('Invalid contour vertex')
            if not (0 <= pt[0] <= w and 0 <= pt[1] <= h):
                raise ValueError('Contour outside frame')
    if type(doc.get('next_number')) is not int or doc['next_number'] <= max(numbers, default=0):
        raise ValueError('Invalid cell counter')


def handler_for(meta, annotation_path, *, validator=validate, web_dir=None):
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB if web_dir is None else web_dir), **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_GET(self):
            if self.path == '/api/annotations':
                with lock:
                    data = annotation_path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != '/api/annotations':
                self.send_error(404); return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                self.send_error(403, 'Cross-origin write rejected'); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 20*1024*1024:
                    raise ValueError('Invalid request size')
                doc = json.loads(self.rfile.read(length))
                validator(doc, meta)
                # Fixed target, atomic replace. Previous session copy retained.
                with lock:
                    tmp = annotation_path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
                    tmp.replace(annotation_path)
                data = json.dumps(dict(saved=True, path=str(annotation_path))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            except (ValueError, TypeError, KeyError, OSError) as e:
                self.send_error(400, str(e))
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'data/Plane_Aligned_20.tiff')
    p.add_argument('--frame', type=int, default=500, help='TIFF page index, zero-based')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--annotations', type=Path, default=ROOT/'annotations')
    p.add_argument('--open', action='store_true', help='Open in the default browser')
    args = p.parse_args()
    # Bind first to avoid replacing an active server\'s dataset assets.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), SimpleHTTPRequestHandler)
    try:
        meta, path = prepare(args.input, args.frame, args.annotations)
        server.RequestHandlerClass = handler_for(meta, path)
        url = f'http://127.0.0.1:{args.port}'
        print(f'Soma Lab: {url}\nAnnotations: {path}\nCtrl+C to stop.', flush=True)
        if args.open:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped. Annotations remain on disk.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
