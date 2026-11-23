"""Crop eligibility, blind payloads, shared rendering and answer persistence."""
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import numpy as np
from matplotlib.image import imread
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from crop_quiz import Quiz,handler_for,valid_origins,crop_png


class CropQuizTests(unittest.TestCase):
    def test_exact_crop_filter_matches_brute_force_including_borders(self):
        rng=np.random.default_rng(12);mask=rng.random((19,23))<.05
        for h,w in [(1,1),(3,5),(8,9),(19,23)]:
            eligible,shape=valid_origins(mask,h,w)
            brute=[y*shape[1]+x for y in range(shape[0]) for x in range(shape[1]) if not mask[y:y+h,x:x+w].any()]
            np.testing.assert_array_equal(eligible,brute)
        with self.assertRaises(ValueError):valid_origins(mask,20,5)

    def test_png_uses_same_scale_and_transposes_native_axes(self):
        image=np.array([[100,150],[200,250],[125,175]],np.uint16)
        png=crop_png(image,100,250)
        actual=imread(BytesIO(png),format='png')
        expected=np.rint(np.clip((image.astype(float)-100)/150,0,1)*255).astype(np.uint8).T
        self.assertEqual(actual.shape[:2],(2,3))
        np.testing.assert_allclose(actual[:,:,0]*255,expected,atol=1e-4)
        self.assertEqual(png,crop_png(image.astype(np.float32),100,250))

    def test_normalized_png_removes_positive_affine_intensity_difference(self):
        y,x=np.indices((20,25));image=(100+y*3+x).astype(float)
        self.assertEqual(crop_png(image,0,1000,'normalized'),crop_png(image*2.5+70,0,1000,'normalized'))
        self.assertNotEqual(crop_png(image,0,1000),crop_png(image*2.5+70,0,1000))

    def test_http_does_not_reveal_truth_before_answer_and_saves_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);y,x=np.indices((40,48));real=(100+y*2+x).astype(np.uint16);synthetic=(110+y+x*2).astype(np.uint16)
            mask=np.zeros(real.shape,np.uint8);mask[:,0:7]=255;mask[18:23,25:30]=255
            for name,array in [('real',real),('synthetic',synthetic),('mask',mask)]:tifffile.imwrite(root/(name+'.tiff'),array)
            quiz=Quiz(root/'real.tiff',root/'synthetic.tiff',root/'mask.tiff',root/'logs',frame=0)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(quiz));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def request(path,body=None,headers=None):
                req=Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers or {})
                with urlopen(req) as response:return response.read()
            try:
                first=json.loads(request('/api/trial',dict(width=12,height=10,strict=True)))
                self.assertEqual(set(first),{'id','image_url','width','height','eligible_positions','contrast'})
                history=json.loads(request('/api/history'));self.assertEqual(history['answers'],[])
                state=quiz.trials[first['id']];self.assertFalse(mask[state['y']:state['y']+12,state['x']:state['x']+10].any())
                png=request(first['image_url']);expected_source=real if state['truth']=='real' else synthetic
                self.assertEqual(png,crop_png(expected_source[state['y']:state['y']+12,state['x']:state['x']+10],quiz.black,quiz.white))
                guess='synthetic' if state['truth']=='real' else 'real'
                reply=json.loads(request('/api/answer',dict(id=first['id'],guess=guess)))
                self.assertEqual(reply['result']['truth'],state['truth']);self.assertFalse(reply['result']['correct']);self.assertEqual(reply['stats']['total'],1)
                again=json.loads(request('/api/answer',dict(id=first['id'],guess=guess)));self.assertEqual(again['stats']['total'],1)
                with self.assertRaises(HTTPError):request('/api/answer',dict(id=first['id'],guess=state['truth']))
                with self.assertRaises(HTTPError):request('/api/trial',dict(width=40,height=48,strict=True))
                with self.assertRaises(HTTPError):request('/api/trial',dict(width=12,height=10,strict=True),{'Origin':'http://foreign.example'})
                log=[json.loads(line) for line in quiz.log.read_text().splitlines()]
                self.assertEqual([row['event'] for row in log],['session','answer'])
                history=json.loads(request('/api/history'));self.assertEqual(history['stats']['total'],1)
                self.assertEqual(len(history['answers']),1)
                normalized=json.loads(request('/api/trial',dict(width=12,height=10,strict=True,contrast='normalized')))
                nstate=quiz.trials[normalized['id']];source=real if nstate['truth']=='real' else synthetic
                crop=source[nstate['y']:nstate['y']+12,nstate['x']:nstate['x']+10]
                self.assertEqual(request(normalized['image_url']),crop_png(crop,quiz.black,quiz.white,'normalized'))
                nreply=json.loads(request('/api/answer',dict(id=normalized['id'],guess=nstate['truth'])))
                self.assertEqual(nreply['result']['contrast'],'normalized')
                self.assertEqual(nreply['stats']['by_contrast']['normalized']['total'],1)
                self.assertIn(b'Crop Quiz',request('/'))
            finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
