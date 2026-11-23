"""Proof of frame coordinates and annotation persistence; no browser required."""
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('annotate_somas', ROOT/'tools/annotate_somas.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class AnnotationTests(unittest.TestCase):
    def test_frame_export_and_persistent_roundtrip_reject_foreign_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); web = root/'web'; web.mkdir()
            source = root/'test.tiff'
            frame = np.arange(30*24, dtype=np.uint16).reshape(24, 30)
            tifffile.imwrite(source, frame)
            with patch.object(app, 'WEB', web):
                meta, path = app.prepare(source, 0, root/'annotations')
                raw = (web/'assets/frame.bin').read_bytes()
                self.assertEqual(meta['shape_yx'], [24, 30])
                self.assertEqual(meta['frame_sha256'], hashlib.sha256(raw).hexdigest())
                np.testing.assert_array_equal(np.frombuffer(raw, '<u2').reshape(24, 30), frame)
                doc = json.loads(path.read_text())
                doc['annotations'] = [dict(id='test-cell', number=1, crop_id=doc['crops'][0]['id'],
                    points_xy=[[2,3],[8,3],[8,9],[2,9]], confidence='confident', truncated=False, comment='')]
                doc['next_number'] = 2
                server = ThreadingHTTPServer(('127.0.0.1', 0), app.handler_for(meta, path))
                thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                base = f'http://127.0.0.1:{server.server_port}'
                try:
                    def post(d):
                        return urlopen(Request(base+'/api/annotations', data=json.dumps(d).encode(),
                            headers={'Content-Type':'application/json'}, method='POST'))
                    with post(doc) as r:
                        self.assertEqual(r.status, 200)
                    with urlopen(base+'/api/annotations') as r:
                        self.assertEqual(json.load(r)['annotations'][0]['points_xy'], doc['annotations'][0]['points_xy'])
                    self.assertEqual(json.loads(path.read_text())['annotations'][0]['id'], 'test-cell')
                    doc['dataset']['frame_sha256'] = 'foreign'
                    with self.assertRaises(HTTPError) as caught:
                        post(doc)
                    self.assertEqual(caught.exception.code, 400)
                    self.assertNotEqual(json.loads(path.read_text())['dataset']['frame_sha256'], 'foreign')
                finally:
                    server.shutdown(); server.server_close(); thread.join()
                app.prepare(source, 0, root/'annotations')
                self.assertTrue(path.with_suffix('.session-backup.json').exists())
                self.assertEqual(len(json.loads(path.read_text())['annotations']), 1)

    def test_out_of_frame_contour_is_rejected(self):
        meta=dict(frame_sha256='abc', frame_index=0, shape_yx=[20, 30])
        doc=dict(format='soma-annotations-v1', dataset=meta, next_number=2,
            crops=[dict(id='c',x=0,y=0,width=20,height=20)],
            annotations=[dict(id='a',number=1,crop_id='c',points_xy=[[1,1],[32,1],[2,2]],confidence='confident',truncated=False)])
        with self.assertRaisesRegex(ValueError, 'outside frame'):
            app.validate(doc, meta)


if __name__ == '__main__':
    unittest.main()
