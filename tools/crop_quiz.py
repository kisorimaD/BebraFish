"""Local blind real/synthetic crop quiz. No frontend packages required."""
import argparse
from collections import OrderedDict
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse,parse_qs
import webbrowser

import matplotlib
matplotlib.use('Agg')
from matplotlib.image import imsave
import numpy as np
import tifffile

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'crop_quiz'


def image_hash(image):return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def crop_png(image,black,white,contrast='fixed'):
    """Identical grayscale transfer and orientation for both sources."""
    if contrast not in ['fixed','normalized']:raise ValueError('Unknown contrast mode')
    if contrast=='normalized':
        black,white=map(float,np.percentile(image,[5,95]))
        if white<=black:white=black+1
    gray=np.rint(np.clip((image.astype(float)-black)/(white-black),0,1)*255).astype(np.uint8)
    output=BytesIO();imsave(output,gray.T,cmap='gray',vmin=0,vmax=255,format='png',origin='upper')
    return output.getvalue()


def valid_origins(mask,height_y,width_x):
    """Exact zero-exclusion crop test using summed areas; no rejection loop."""
    h,w=mask.shape
    if type(height_y) is not int or type(width_x) is not int or not(1<=height_y<=h and 1<=width_x<=w):raise ValueError('Размер кропа выходит за границы кадра.')
    integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    counts=integral[height_y:,width_x:]-integral[:-height_y,width_x:]-integral[height_y:,:-width_x]+integral[:-height_y,:-width_x]
    return np.flatnonzero(counts.ravel()==0),counts.shape


class Quiz:
    def __init__(self,real_path,synthetic_path,mask_path,output,frame=500,random_frames=False):
        self.real_path=Path(real_path);self.random_frames=random_frames;self.frame=frame
        with tifffile.TiffFile(real_path) as tif:
            self.frame_count=len(tif.pages)
            if not 0<=frame<self.frame_count:raise ValueError('Реальный кадр вне фильма.')
            self.real=tif.pages[frame].asarray()
        self.synthetic=tifffile.imread(synthetic_path);self.mask=tifffile.imread(mask_path)!=0
        if self.real.ndim!=2 or self.synthetic.shape!=self.real.shape or self.mask.shape!=self.real.shape:raise ValueError('Кадр, синтетика и маска должны быть одинаковыми 2D изображениями.')
        if not(np.isfinite(self.real).all() and np.isfinite(self.synthetic).all()):raise ValueError('Изображения содержат нечисловые значения.')
        self.black,self.white=map(float,np.percentile(self.real,[1,99.5]))
        if self.white<=self.black:self.white=self.black+1
        self.lock=threading.RLock();self.trials={};self.pngs=OrderedDict();self.origins=OrderedDict();self.answers=[]
        self.session=secrets.token_hex(8);output=Path(output);output.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.log=output/f'quiz_{stamp}_{self.session}.jsonl'
        self.append(dict(event='session',session=self.session,real_path=str(self.real_path.resolve()),real_frame=frame,random_frames=random_frames,
            synthetic_path=str(Path(synthetic_path).resolve()),mask_path=str(Path(mask_path).resolve()),shape_yx=list(self.real.shape),
            display_range=[self.black,self.white],real_reference_hash=image_hash(self.real),synthetic_hash=image_hash(self.synthetic),mask_hash=image_hash(self.mask)))

    def append(self,entry):
        with self.log.open('a',encoding='utf-8') as file:file.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')

    def metadata(self):
        h,w=self.real.shape
        return dict(api_version=2,session=self.session,max_width=h,max_height=w,display_range=[self.black,self.white],
            real_sampling='Случайный кадр фильма' if self.random_frames else f'Реальный кадр {self.frame}',
            synthetic_sampling='Одна последняя синтетическая сцена',results_file=self.log.name,
            orientation='По горизонтали исходная Y, по вертикали X. Размеры указаны как на экране.')

    def stats(self):
        n=len(self.answers);correct=sum(a['correct'] for a in self.answers)
        by_class={label:dict(total=sum(a['truth']==label for a in self.answers),correct=sum(a['truth']==label and a['correct'] for a in self.answers)) for label in ['real','synthetic']}
        by_contrast={mode:dict(total=sum(a.get('contrast','fixed')==mode for a in self.answers),correct=sum(a.get('contrast','fixed')==mode and a['correct'] for a in self.answers)) for mode in ['fixed','normalized']}
        return dict(total=n,correct=correct,accuracy=correct/n if n else None,by_class=by_class,by_contrast=by_contrast)

    def new_trial(self,settings):
        width=settings.get('width');height=settings.get('height');strict=settings.get('strict');contrast=settings.get('contrast','fixed')
        if contrast not in ['fixed','normalized']:raise ValueError('Некорректный режим контраста.')
        if type(width) is not int or type(height) is not int or type(strict) is not bool:raise ValueError('Некорректные параметры кропа.')
        h,w=self.real.shape
        if not(8<=width<=h and 8<=height<=w):raise ValueError(f'Допустимые размеры: ширина 8–{h}, высота 8–{w} px.')
        # Width on screen is native Y; height on screen is native X.
        key=(width,height)
        if strict:
            if key not in self.origins:
                if len(self.origins)>=8:self.origins.popitem(last=False)
                self.origins[key]=valid_origins(self.mask,width,height)
            eligible,shape=self.origins[key]
            if not len(eligible):raise ValueError('Нет кропов такого размера без маски исключения. Уменьшите размер или выключите фильтр.')
            y,x=map(int,np.unravel_index(int(eligible[secrets.randbelow(len(eligible))]),shape));available=int(len(eligible))
        else:y=secrets.randbelow(h-width+1);x=secrets.randbelow(w-height+1);available=(h-width+1)*(w-height+1)
        truth='real' if secrets.randbelow(2)==0 else 'synthetic'
        frame=None
        if truth=='real':
            frame=secrets.randbelow(self.frame_count) if self.random_frames else self.frame
            if frame==self.frame:image=self.real
            else:
                with tifffile.TiffFile(self.real_path) as tif:image=tif.pages[frame].asarray()
                if image.shape!=self.real.shape:raise ValueError('Кадры фильма различаются по размеру.')
        else:image=self.synthetic
        crop=image[y:y+width,x:x+height]
        png=crop_png(crop,self.black,self.white,contrast);identity=secrets.token_hex(16)
        trial=dict(id=identity,truth=truth,frame=frame,y=y,x=x,width=width,height=height,strict=strict,
            contrast=contrast,created=time.monotonic(),crop_sha256=image_hash(crop),answered=None)
        # Bound RAM while keeping answers (small) and their original ids.
        if len(self.pngs)>=32:self.pngs.popitem(last=False)
        pending=[k for k,t in self.trials.items() if t['answered'] is None]
        for k in pending[:-31]:self.trials.pop(k,None)
        self.trials[identity]=trial;self.pngs[identity]=png
        return dict(id=identity,image_url='/api/crop?id='+identity,width=width,height=height,eligible_positions=available,contrast=contrast)

    def answer(self,payload):
        identity=payload.get('id');guess=payload.get('guess')
        if not isinstance(identity,str) or guess not in ['real','synthetic']:raise ValueError('Некорректный ответ.')
        trial=self.trials.get(identity)
        if trial is None:raise ValueError('Этот кроп устарел. Запросите следующий.')
        if trial['answered'] is not None:
            if trial['answered']['guess']!=guess:raise ValueError('Ответ уже записан.')
            return dict(result=trial['answered'],stats=self.stats())
        result=dict(id=identity,guess=guess,truth=trial['truth'],correct=guess==trial['truth'],
            frame=trial['frame'],crop_yx=[trial['y'],trial['y']+trial['width'],trial['x'],trial['x']+trial['height']],
            width=trial['width'],height=trial['height'],strict=trial['strict'],contrast=trial['contrast'],crop_sha256=trial['crop_sha256'],
            response_seconds=round(time.monotonic()-trial['created'],3),answered_at=datetime.now(timezone.utc).isoformat())
        self.append(dict(event='answer',**result));trial['answered']=result;self.answers.append(result)
        return dict(result=result,stats=self.stats())


def handler_for(quiz):
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,status=200,content_type='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)

        def do_GET(self):
            url=urlparse(self.path)
            with quiz.lock:
                if url.path=='/api/meta':self.send(quiz.metadata())
                elif url.path=='/api/history':self.send(dict(stats=quiz.stats(),answers=quiz.answers,session=quiz.session))
                elif url.path=='/api/crop':
                    identity=parse_qs(url.query).get('id',[''])[0];png=quiz.pngs.get(identity)
                    if png is None:self.send(dict(error='Кроп устарел.'),404)
                    else:self.send(png,content_type='image/png')
                elif url.path in ['/','/index.html','/style.css','/app.js']:
                    file=WEB/('index.html' if url.path=='/' else url.path.lstrip('/'))
                    self.send(file.read_bytes(),content_type={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8'}[file.suffix])
                else:self.send(dict(error='Не найдено.'),404)

        def do_POST(self):
            if self.headers.get('Origin') and self.headers['Origin']!='http://'+self.headers.get('Host',''):
                self.send(dict(error='Cross-origin request rejected.'),403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Некорректный размер запроса.')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('Ожидался объект JSON.')
                with quiz.lock:
                    if self.path=='/api/trial':self.send(quiz.new_trial(data))
                    elif self.path=='/api/answer':self.send(quiz.answer(data))
                    else:self.send(dict(error='Не найдено.'),404)
            except (ValueError,TypeError,KeyError,OSError) as error:self.send(dict(error=str(error)),400)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real',type=Path,default=ROOT/'data/Plane_Aligned_20.tiff')
    p.add_argument('--synthetic',type=Path,default=ROOT/'outputs/plane20_contrast_sharpness/synthetic.tiff')
    p.add_argument('--mask',type=Path,default=ROOT/'outputs/plane20_masked_background/exclusion_mask.tiff')
    p.add_argument('--frame',type=int,default=500)
    p.add_argument('--random-frames',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/crop_quiz')
    p.add_argument('--port',type=int,default=8767);p.add_argument('--open',action='store_true');args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),BaseHTTPRequestHandler)
    try:
        quiz=Quiz(args.real,args.synthetic,args.mask,args.output,args.frame,args.random_frames)
        server.RequestHandlerClass=handler_for(quiz);url=f'http://127.0.0.1:{args.port}'
        print(f'Crop Quiz: {url}\nAnswers: {quiz.log}\nCtrl+C to stop.',flush=True)
        if args.open:webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:print('\nStopped. Answers remain on disk.')
    finally:server.server_close()


if __name__=='__main__':main()
