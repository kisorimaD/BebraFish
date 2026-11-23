"""Local per-seed intensity flood-fill sandbox for q90; independent of soma labels."""
import argparse
import copy
import hashlib
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json
import math
from pathlib import Path
import shutil
import webbrowser

import numpy as np
from scipy.ndimage import label,generate_binary_structure
import tifffile

from annotate_somas import handler_for

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'region_sandbox'


def load_image(path):
    image=tifffile.imread(path)
    if image.ndim!=2 or not np.isfinite(image).all() or np.min(image)<0:
        raise ValueError('Expected one finite nonnegative 2D intensity image')
    return np.asarray(image,dtype='<f4')


def make_grid(shape,size=(192,192)):
    h,w=shape;cw,ch=size
    if not all(type(v) is int and 1<=v<=16384 for v in size):raise ValueError('Invalid crop size')
    if math.ceil(h/ch)*math.ceil(w/cw)>10000:raise ValueError('Too many crops')
    crops=[]
    for y in range(0,h,ch):
        for x in range(0,w,cw):
            crops.append(dict(id=f'tile-{y}-{x}',x=x,y=y,width=min(cw,w-x),height=min(ch,h-y),name=f'Кроп {len(crops)+1:02d}'))
    if len(crops)>10000:raise ValueError('Too many crops')
    return crops


def upgrade_document(doc):
    if doc.get('format')=='region-flood-v3':return doc
    upgraded=copy.deepcopy(doc);upgraded['format']='region-flood-v3'
    upgraded.pop('crops',None);upgraded.pop('crop_size_xy',None)
    for point in upgraded['points']:point.pop('crop_id',None)
    return upgraded


def polygon_mask(shape,polygon):
    """Integer pixel centers, even/odd interior with boundary included; matches JS."""
    if polygon is None:return np.ones(shape,bool)
    y,x=np.indices(shape);inside=np.zeros(shape,bool);boundary=np.zeros(shape,bool)
    for (a,b),(c,d) in zip(polygon[-1:]+polygon[:-1],polygon):
        boundary|=(np.abs((x-a)*(d-b)-(y-b)*(c-a))<1e-8)&(x>=min(a,c))&(x<=max(a,c))&(y>=min(b,d))&(y<=max(b,d))
        if d!=b:inside^=((b>y)!=(d>y))&(x<(c-a)*(y-b)/(d-b)+a)
    return inside|boundary


def prepare(input_path,annotation_dir):
    image=load_image(input_path);payload=image.tobytes()
    meta=dict(source_path=str(input_path.resolve()),source_name=input_path.name,
        shape_yx=list(image.shape),image_sha256=hashlib.sha256(payload).hexdigest(),dtype='float32',byte_order='little',
        intensity_range_adu=[float(image.min()),float(image.max())],
        display_range=np.percentile(image,[1,99.5]).astype(float).tolist())
    assets=WEB/'assets';assets.mkdir(exist_ok=True)
    (assets/'image.bin').write_bytes(payload);(assets/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n')
    annotation_dir.mkdir(parents=True,exist_ok=True)
    target=annotation_dir/f'regions_{input_path.stem}_{meta["image_sha256"][:12]}.json'
    if target.exists():
        existing=json.loads(target.read_text());validate(existing,meta)
        shutil.copy2(target,target.with_suffix('.session-backup.json'))
        if existing['format']!='region-flood-v3':
            backup=target.with_suffix('.pre-polygons.json')
            if not backup.exists():shutil.copy2(target,backup)
            upgraded=upgrade_document(existing);validate(upgraded,meta)
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(upgraded,indent=2,ensure_ascii=False)+'\n');tmp.replace(target)
    else:
        target.write_text(json.dumps(upgrade_document(dict(format='region-flood-v1',dataset=meta,connectivity=4,
            next_number=1,points=[],updated_at='',point_coordinate_system='Native TIFF integer pixel indices: x=column, y=row',mask_definition='1 = user-selected exclusion region; 0 = unselected/unknown, not confirmed neurons')),indent=2,ensure_ascii=False)+'\n')
    return meta,target


def validate(doc,meta):
    if not isinstance(doc,dict) or doc.get('format') not in ['region-flood-v1','region-flood-v2','region-flood-v3']:raise ValueError('Unsupported region format')
    dataset=doc.get('dataset',{})
    if dataset.get('image_sha256')!=meta['image_sha256'] or dataset.get('shape_yx')!=meta['shape_yx']:
        raise ValueError('Regions belong to a different image')
    if doc.get('connectivity') not in [4,8]:raise ValueError('Connectivity must be 4 or 8')
    points=doc.get('points')
    if not isinstance(points,list) or len(points)>2000:raise ValueError('Invalid point list')
    ids=set();numbers=set();height,width=meta['shape_yx']
    crop_map={}
    if doc['format']=='region-flood-v2':
        size=doc.get('crop_size_xy')
        if not isinstance(size,list) or len(size)!=2:raise ValueError('Missing crop size')
        expected=make_grid(meta['shape_yx'],size);crops=doc.get('crops')
        if not isinstance(crops,list) or len(crops)!=len(expected):raise ValueError('Invalid crop partition')
        for c,e in zip(crops,expected):
            if not isinstance(c,dict) or any(c.get(k)!=e[k] for k in ['id','x','y','width','height']):raise ValueError('Crop partition mismatch')
            crop_map[c['id']]=c
    for p in points:
        if not isinstance(p,dict) or not isinstance(p.get('id'),str) or p['id'] in ids:raise ValueError('Invalid/duplicate point ID')
        ids.add(p['id'])
        if type(p.get('number')) is not int or p['number']<1 or p['number'] in numbers:raise ValueError('Invalid point number')
        numbers.add(p['number'])
        if type(p.get('x')) is not int or type(p.get('y')) is not int or not (0<=p['x']<width and 0<=p['y']<height):raise ValueError('Point outside image')
        if doc['format']=='region-flood-v2':
            c=crop_map.get(p.get('crop_id'))
            if c is None or not(c['x']<=p['x']<c['x']+c['width'] and c['y']<=p['y']<c['y']+c['height']):raise ValueError('Point outside its crop')
        if doc['format']=='region-flood-v3' and p.get('polygon') is not None:
            polygon=p['polygon']
            if not isinstance(polygon,list) or not 3<=len(polygon)<=2000 or any(not isinstance(v,list) or len(v)!=2 or any(type(n) is not int for n in v) or not(0<=v[0]<width and 0<=v[1]<height) for v in polygon):raise ValueError('Invalid polygon')
            # Evaluate only the seed rather than rasterizing a whole frame for validation.
            shifted=[[x-p['x'],y-p['y']] for x,y in polygon]
            if not polygon_mask((1,1),shifted)[0,0]:raise ValueError('Point outside its polygon')
        threshold=p.get('threshold')
        if type(threshold) not in [int,float] or not math.isfinite(threshold) or not (0<=threshold<=1e7):raise ValueError('Invalid intensity threshold')
        if type(p.get('enabled')) is not bool:raise ValueError('Invalid enabled flag')
    if type(doc.get('next_number')) is not int or doc['next_number']<=max(numbers,default=0):raise ValueError('Invalid point counter')


def validate_active(doc,meta):
    if not isinstance(doc,dict) or doc.get('format')!='region-flood-v3':
        raise ValueError('Polygon format required: restart server and reload Region Lab')
    validate(doc,meta)


def compute_mask(image,doc):
    """SciPy connected-component reference implementation of browser flood fill."""
    result=np.zeros(image.shape,bool);structure=generate_binary_structure(2,1 if doc['connectivity']==4 else 2)
    cache={};crop_map={c['id']:c for c in doc.get('crops',[])}
    for p in doc['points']:
        if not p['enabled']:continue
        t=p['threshold']
        if image[p['y'],p['x']]>t+1e-4:continue
        c=crop_map[p['crop_id']] if doc.get('format')=='region-flood-v2' else dict(x=0,y=0,width=image.shape[1],height=image.shape[0])
        y0,x0=c['y'],c['x'];y1,x1=y0+c['height'],x0+c['width'];key=(t,y0,x0,y1,x1,json.dumps(p.get('polygon')))
        if key not in cache:
            if len(cache)>=4:cache.clear()
            # Label the crop itself. Clipping a global flood afterwards would
            # allow forbidden paths that leave the crop and re-enter elsewhere.
            allowed=image[y0:y1,x0:x1]<=t+1e-4
            if doc.get('format')=='region-flood-v3':allowed&=polygon_mask(image.shape,p.get('polygon'))[y0:y1,x0:x1]
            cache[key]=label(allowed,structure=structure)[0]
        components=cache[key];component=components[p['y']-y0,p['x']-x0]
        if component:result[y0:y1,x0:x1]|=components==component
    return result


def export_annotations(annotation_path,output):
    doc=json.loads(annotation_path.read_text());image=load_image(Path(doc['dataset']['source_path']))
    meta=dict(image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),shape_yx=list(image.shape))
    validate(doc,meta);mask=compute_mask(image,doc)
    output.mkdir(parents=True,exist_ok=False)
    tifffile.imwrite(output/'exclusion_mask.tiff',mask.astype(np.uint8)*255)
    np.savez_compressed(output/'exclusion_mask.npz',exclusion_mask=mask,shape_yx=np.array(image.shape),annotation_json=np.asarray(json.dumps(doc)))
    (output/'points.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
    print(f'Exported {mask.sum()} selected pixels to {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'outputs/plane20_movie_diagnostics/q90.tiff')
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--annotations',type=Path,default=ROOT/'annotations')
    p.add_argument('--open',action='store_true')
    p.add_argument('--export',type=Path,help='Export a saved region JSON to TIFF/NPZ instead of starting server')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/region_export')
    args=p.parse_args()
    if args.export:
        export_annotations(args.export,args.output);return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),SimpleHTTPRequestHandler)
    try:
        meta,path=prepare(args.input,args.annotations)
        server.RequestHandlerClass=handler_for(meta,path,validator=validate_active,web_dir=WEB)
        url=f'http://127.0.0.1:{args.port}'
        print(f'Region Lab: {url}\nSaved points: {path}\nCtrl+C to stop.',flush=True)
        if args.open:webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:print('\nStopped. Points remain on disk.')
    finally:server.server_close()


if __name__=='__main__':main()
