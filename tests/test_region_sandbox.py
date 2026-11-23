"""Independent seed thresholds, persistence and browser/backend parity."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import annotate_regions as regions


class RegionSandboxTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is required only for browser/backend parity tests')
    def test_backend_matches_browser_flood_with_independent_thresholds(self):
        rng=np.random.default_rng(12)
        image=rng.integers(0,10,(21,30)).astype(np.float32)
        points=[dict(id=str(i),number=i+1,x=x,y=y,threshold=t,enabled=en) for i,(x,y,t,en) in enumerate([(0,0,4,True),(15,10,6,True),(29,20,8,False),(3,7,3,True)])]
        js="""import {flood,unionMasks} from './region_sandbox/flood.mjs';
let input='';for await(const chunk of process.stdin)input+=chunk;
const d=JSON.parse(input),image=Float32Array.from(d.image);
const masks=d.points.map(p=>flood(image,d.width,d.height,p,d.connectivity,d.crops?.find(c=>c.id===p.crop_id)).mask);
process.stdout.write(JSON.stringify([...unionMasks(masks,image.length)]));"""
        for connectivity in [4,8]:
            doc=dict(points=points,connectivity=connectivity)
            expected=regions.compute_mask(image,doc)
            payload=json.dumps(dict(image=image.ravel().tolist(),width=30,height=21,points=points,connectivity=connectivity))
            result=subprocess.run(['node','--input-type=module','-e',js],input=payload,text=True,capture_output=True,cwd=ROOT,check=True)
            actual=np.array(json.loads(result.stdout),dtype=bool).reshape(image.shape)
            np.testing.assert_array_equal(actual,expected)
            crops=regions.make_grid(image.shape,(11,9))
            cropped=[{**p,'crop_id':next(c['id'] for c in crops if c['x']<=p['x']<c['x']+c['width'] and c['y']<=p['y']<c['y']+c['height'])} for p in points]
            crop_doc=dict(format='region-flood-v2',points=cropped,connectivity=connectivity,crops=crops)
            payload=json.dumps(dict(image=image.ravel().tolist(),width=30,height=21,points=cropped,connectivity=connectivity,crops=crops))
            result=subprocess.run(['node','--input-type=module','-e',js],input=payload,text=True,capture_output=True,cwd=ROOT,check=True)
            actual=np.array(json.loads(result.stdout),dtype=bool).reshape(image.shape)
            np.testing.assert_array_equal(actual,regions.compute_mask(image,crop_doc))

    def test_polygon_bounds_parity_and_independence(self):
        # The bright wall could be bypassed via the dark exterior, which is forbidden.
        image=np.ones((7,7),np.float32);image[1:6,3]=9
        polygon=[[1,1],[5,1],[5,5],[1,5]]
        points=[dict(id='a',number=1,x=1,y=3,threshold=2,enabled=True,polygon=polygon)]
        js="""import {flood,unionMasks} from './region_sandbox/flood.mjs';
let input='';for await(const c of process.stdin)input+=c;const d=JSON.parse(input);
process.stdout.write(JSON.stringify([...unionMasks(d.points.map(p=>flood(Float32Array.from(d.image),7,7,p,d.connectivity).mask),49)]));"""
        for connectivity in [4,8]:
            doc=dict(format='region-flood-v3',points=points,connectivity=connectivity)
            expected=np.zeros((7,7),bool);expected[1:6,1:3]=True
            np.testing.assert_array_equal(regions.compute_mask(image,doc),expected)
            # Concave and diagonal boundaries exercise the shared raster convention.
            for poly in [polygon,[[0,0],[6,0],[4,3],[6,6],[0,6],[2,3]],[[0,0],[6,3],[0,6]]]:
                points[0]['polygon']=poly
                payload=json.dumps(dict(image=image.ravel().tolist(),points=points,connectivity=connectivity))
                result=subprocess.run(['node','--input-type=module','-e',js],input=payload,text=True,capture_output=True,cwd=ROOT,check=True)
                np.testing.assert_array_equal(np.array(json.loads(result.stdout),bool).reshape(image.shape),regions.compute_mask(image,doc))
            points[0]['polygon']=polygon
            points.append(dict(id='b',number=2,x=6,y=3,threshold=2,enabled=True))
            self.assertGreater(regions.compute_mask(image,doc).sum(),expected.sum())
            points.pop()

    def test_crop_path_cannot_leave_and_reenter(self):
        image=np.ones((5,5),np.float32);image[1:4,2]=9
        doc=dict(format='region-flood-v2',connectivity=8,crops=[dict(id='c',x=1,y=1,width=3,height=3)],
            points=[dict(x=1,y=2,threshold=2,enabled=True,crop_id='c')])
        mask=regions.compute_mask(image,doc)
        expected=np.zeros((5,5),bool);expected[1:4,1]=True
        np.testing.assert_array_equal(mask,expected)
        doc['points'][0]['threshold']=100
        mask=regions.compute_mask(image,doc)
        self.assertEqual(mask.sum(),9)
        self.assertFalse(mask[0].any());self.assertFalse(mask[:,0].any())

    def test_legacy_migration_backup_removes_grid(self):
        with self.assertRaisesRegex(ValueError,'Too many'):regions.make_grid((1376,320),(1,1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);web=root/'web';web.mkdir();source=root/'q90.tiff'
            tifffile.imwrite(source,np.ones((202,205),np.float32))
            with patch.object(regions,'WEB',web):
                meta,path=regions.prepare(source,root/'annotations')
                legacy=dict(format='region-flood-v1',dataset=meta,connectivity=4,next_number=3,updated_at='original',
                    points=[dict(id='a',number=1,x=0,y=0,threshold=137.2,enabled=True),
                            dict(id='b',number=2,x=204,y=201,threshold=180.,enabled=False)])
                original=json.dumps(legacy,indent=2);path.write_text(original)
                regions.prepare(source,root/'annotations');new=json.loads(path.read_text())
                self.assertEqual(path.with_suffix('.pre-polygons.json').read_text(),original)
                for old,p in zip(legacy['points'],new['points']):
                    for key in old:self.assertEqual(p[key],old[key])
                self.assertEqual(new['format'],'region-flood-v3')
                self.assertNotIn('crops',new)
                self.assertNotIn('crop_id',new['points'][1])
                regions.validate(new,meta)
                saved=path.read_text();regions.prepare(source,root/'annotations')
                self.assertEqual(path.read_text(),saved)
                self.assertEqual(path.with_suffix('.pre-polygons.json').read_text(),original)

    def test_save_reload_and_export_do_not_use_soma_annotation_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);web=root/'web';web.mkdir();image=np.array([[100,105,200],[100,110,200]],np.float32)
            source=root/'q90.tiff';tifffile.imwrite(source,image)
            with patch.object(regions,'WEB',web):
                meta,path=regions.prepare(source,root/'annotations');doc=json.loads(path.read_text())
                self.assertEqual(doc['format'],'region-flood-v3')
                self.assertTrue(path.name.startswith('regions_'))
                doc['points']=[dict(id='seed',number=1,x=0,y=0,threshold=105.,enabled=True,polygon=[[0,0],[1,0],[1,1]])];doc['next_number']=2
                handler=regions.handler_for(meta,path,validator=regions.validate_active,web_dir=web)
                server=ThreadingHTTPServer(('127.0.0.1',0),handler)
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    url=f'http://127.0.0.1:{server.server_port}/api/annotations'
                    with urlopen(Request(url,data=json.dumps(doc).encode(),headers={'Content-Type':'application/json'},method='POST')) as res:self.assertEqual(res.status,200)
                    with urlopen(url) as res:self.assertEqual(json.load(res)['points'][0]['threshold'],105.)
                    legacy=json.loads(json.dumps(doc));legacy['format']='region-flood-v1'
                    with self.assertRaises(HTTPError):urlopen(Request(url,data=json.dumps(legacy).encode(),method='POST'))
                    invalid=json.loads(json.dumps(doc));invalid['dataset']['image_sha256']='foreign'
                    with self.assertRaises(HTTPError):urlopen(Request(url,data=json.dumps(invalid).encode(),method='POST'))
                finally:server.shutdown();server.server_close();thread.join()
                regions.export_annotations(path,root/'export')
                mask=tifffile.imread(root/'export/exclusion_mask.tiff')
                np.testing.assert_array_equal(mask,np.array([[255,255,0],[0,0,0]],np.uint8))
                self.assertEqual(mask.shape,image.shape)
                regions.prepare(source,root/'annotations')
                self.assertEqual(len(json.loads(path.read_text())['points']),1)
                self.assertTrue(path.with_suffix('.session-backup.json').exists())


if __name__=='__main__':unittest.main()
