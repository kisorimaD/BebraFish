"""Run the local 2D synthesis lab with python -m calibration_lab."""
import argparse
import hashlib
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser
import zipfile

import numpy as np
from PIL import Image
import tifffile

from training.plane20.fit_median import fit_profile, prepare_median, ROOT
from training.plane20.generate_median import generate_pair, relight_pair
import annotate_somas
import annotate_regions
from .preview import PREVIEW_NAMES, comparison_previews
from .metrics import write_metrics
from .light import arrays, open_manifest, open_run

WEB = Path(__file__).resolve().parent
METRICS_LOCK = threading.Lock()
METRICS_MTIME = max((WEB/name).stat().st_mtime for name in ('metrics.py', 'spatial_metrics.py'))
OUTPUT = ROOT/'outputs/calibration_lab'
ASSETS = {
    'soma': {'index.html', 'style.css', 'app.js', 'geometry.mjs', 'assets/frame.bin', 'assets/dataset.json'},
    'region': {'index.html', 'style.css', 'app.js', 'flood.mjs', 'crops.mjs', 'fill-worker.js',
               'assets/image.bin', 'assets/dataset.json'},
}


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def prepare_job(folder, source, *, source_name=None):
    median = folder/'reference.tiff'
    metadata = prepare_median(source, median, diagnostics=folder/'median-diagnostics.npz')
    if source_name is not None:
        metadata["source"] = source_name
    for kind, tool in [('soma', annotate_somas), ('region', annotate_regions)]:
        web = folder/kind
        web.mkdir()
        for name in ASSETS[kind]:
            if '/' not in name:
                shutil.copyfile(tool.WEB/name, web/name)
        if kind == 'soma':
            _, doc = tool.prepare(median, 0, folder, web_dir=web)
        else:
            _, doc = tool.prepare(median, folder, web_dir=web)
        doc.rename(folder/f'{kind}.json')
    image = tifffile.imread(median)
    lo, hi = np.percentile(image, [1, 99.5])
    preview = np.clip((image.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)
    Image.fromarray(preview).save(folder/'reference.png')
    write_json(folder/'source.json', metadata)


def generation(folder, attempt, crops, count, seed, update, *, grow_somas=False):
    image = tifffile.imread(folder/'reference.tiff')
    profile = attempt/'profile.npz'
    soma = json.loads((folder/'soma.json').read_text())
    region = json.loads((folder/'region.json').read_text())
    for name, doc in (('soma', soma), ('region', region)):
        write_json(attempt/f'{name}.json', doc)
    source = json.loads((folder/'source.json').read_text())
    report = fit_profile(image, soma, region, crops, profile,
                         intensity_noise=source.get('intensity_noise'), grow_somas=grow_somas)
    metadata = dict(calibration=report, seed=seed, source=source)

    def scenes():
        for index in range(count):
            scene_seed = int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0])
            yield generate_pair(('scenes', index, scene_seed, attempt), profile)
    return finish_generation(image, attempt, crops, metadata, scenes(), update)


def update_profile_light(profile, path, settings):
    config = json.loads(str(profile['median_config'])) | settings
    np.savez_compressed(path, **(profile | {'median_config': np.array(json.dumps(config))}))
    return config


def regenerate_light(folder, source_run, attempt, changes, update):
    image = tifffile.imread(folder/'reference.tiff')
    report = json.loads((source_run/'report.json').read_text())
    with zipfile.ZipFile(source_run/'dataset.zip') as archive:
        profile = arrays(BytesIO(archive.read('profile.npz')))
        if 'reference_background' not in profile:
            profile['reference_background'] = np.where(profile['neuron_support'], 0, image)
        config = update_profile_light(profile, attempt/'profile.npz', changes)
        for name in ('soma.json', 'region.json'):
            if name in archive.namelist():
                (attempt/name).write_bytes(archive.read(name))
        crops = json.loads(archive.read('centers.json'))
        overrides = (report.get('manual_light') or {}).get('settings', {}) | changes
        metadata = report | dict(calibration=report['calibration'] | config,
                                 manual_light=dict(source_run=source_run.name, settings=overrides))

        def scenes():
            for entry in report['scenes']:
                layout = arrays(BytesIO(archive.read(entry['path'])))
                job = (entry['split'], entry['index'], entry['seed'], attempt)
                yield relight_pair(job, attempt/'profile.npz', layout, entry.get('placement', 'packed'))
        return finish_generation(image, attempt, crops, metadata, scenes(), update)


def finish_generation(image, attempt, crops, metadata, scenes, update):
    profile = attempt/'profile.npz'
    (attempt/'scenes').mkdir()
    reports = []
    for count, row in enumerate(scenes, 1):
        reports.append(row)
        update(count)
    with np.load(attempt/reports[0]['path']) as scene, np.load(profile) as fitted:
        synthetic = scene['image_structured']
        tifffile.imwrite(attempt/'example.tiff', synthetic, photometric='minisblack')
        comparison_previews(attempt, image, synthetic, fitted['tissue'], crops)
    write_json(attempt/'report.json', metadata | dict(scenes=reports,
                  profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()))
    write_json(attempt/'centers.json', crops)
    temporary = attempt/'dataset.zip.tmp'
    with zipfile.ZipFile(temporary, 'w') as archive:
        for name in ('profile.npz', 'report.json', 'centers.json', 'example.tiff'):
            archive.write(attempt/name, name)
        for name in ('soma.json', 'region.json'):
            if (attempt/name).is_file():
                archive.write(attempt/name, name)
        for row in reports:
            archive.write(attempt/row['path'], row['path'])
    temporary.replace(attempt/'dataset.zip')
    return dict(cells=[row['cells'] for row in reports], target=metadata['calibration']['count'])


def restore_previews(folder, attempt):
    """Rebuild previews for completed runs saved by older lab versions."""
    reference = tifffile.imread(folder/'reference.tiff')
    synthetic = tifffile.imread(attempt/'example.tiff')
    crops = json.loads((attempt/'centers.json').read_text())
    with np.load(attempt/'profile.npz') as profile:
        comparison_previews(attempt, reference, synthetic, profile['tissue'], crops)


def saved_projects():
    """Recover prepared projects and their latest complete runs from disk."""
    projects = {}
    for folder in OUTPUT.iterdir() if OUTPUT.exists() else ():
        if not re.fullmatch(r'[0-9a-f]{32}', folder.name) or not (folder/'source.json').is_file():
            continue
        try:
            source = json.loads((folder/'source.json').read_text())
            project = dict(stage='ready', name=Path(source['source']).name)
            attempts = sorted(folder.glob('run-*/dataset.zip'),
                              key=lambda file: file.stat().st_mtime, reverse=True)
            for archive in attempts:
                if not zipfile.is_zipfile(archive):
                    continue
                report = json.loads((archive.parent/'report.json').read_text())
                project.update(stage='done', attempt=archive.parent.name,
                               result=dict(cells=[row['cells'] for row in report['scenes']],
                                           target=report['calibration']['count']))
                break
            projects[folder.name] = project
        except (OSError, ValueError, KeyError):
            continue
    return projects


def validate_crops(crops, shape):
    if not isinstance(crops, list) or not 1 <= len(crops) <= 30:
        raise ValueError('Выберите от 1 до 30 кропов для разметки центров.')
    for crop in crops:
        x, y, w, h = (crop[key] for key in ('x', 'y', 'width', 'height'))
        if not all(type(value) is int for value in (x, y, w, h)) or not (
                0 <= x < x+w <= shape[1] and 0 <= y < y+h <= shape[0]):
            raise ValueError('Кроп выходит за границы исходного TIFF.')
        points = crop['points']
        if not isinstance(points, list) or len(points) > w*h:
            raise ValueError('Неверный список центров.')
        for px, py in points:
            if type(px) is not int or type(py) is not int or not (x <= px < x+w and y <= py < y+h):
                raise ValueError('Центр выходит за границы кропа.')


class Handler(BaseHTTPRequestHandler):
    def respond(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def serve(self, file):
        if not file.is_file():
            self.send_error(404)
            return
        with file.open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(file.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile)

    def trusted(self):
        host = f'127.0.0.1:{self.server.server_port}'
        return self.headers.get('Host') == host and self.headers.get('Origin') in (None, 'http://'+host)

    def route(self):
        return urlsplit(self.path).path

    def job_route(self):
        match = re.fullmatch(r'/jobs/([0-9a-f]{32})/(.*)', self.route())
        if match and match[1] in self.server.jobs:
            return self.server.jobs[match[1]], OUTPUT/match[1], match[2]
        return None, None, ''

    def light_route(self):
        route = self.route()
        if route in ('/light', '/light/data') and getattr(self.server, 'light_manifest', None):
            return route.endswith('/data'), self.server.light_manifest
        match = re.fullmatch(r'/jobs/([0-9a-f]{32})/(run-[0-9a-f]{32})/light(/data)?', route)
        if match:
            folder = OUTPUT/match[1]
            attempt = folder/match[2]
            if (attempt/'dataset.zip').is_file():
                return bool(match[3]), (folder, attempt)
        return None, None

    def light_preview(self, source):
        return open_run(*source) if isinstance(source, tuple) else open_manifest(source)

    def do_GET(self):
        if not self.trusted():
            self.send_error(403)
            return
        if self.route() in ('/', '/app.js', '/style.css', '/light.js', '/light.css'):
            self.serve(WEB/('index.html' if self.route() == '/' else self.route()[1:]))
            return
        data, source = self.light_route()
        if source is not None:
            if not data:
                self.serve(WEB/'light.html')
            else:
                try:
                    self.respond(self.light_preview(source).metadata() | dict(can_apply=isinstance(source, tuple)))
                except (OSError, ValueError, KeyError) as error:
                    self.respond(dict(error=str(error)), 400)
            return
        if self.route() == '/projects':
            with self.server.lock:
                projects = [dict(job=token, name=job.get('name', token), stage=job['stage'])
                            for token, job in self.server.jobs.items()]
            self.respond(projects)
            return
        saved = re.fullmatch(r'/jobs/([0-9a-f]{32})/(run-[0-9a-f]{32})/([a-z0-9.-]+)', self.route())
        if saved:
            job_id, run_id, filename = saved.groups()
            folder = OUTPUT/job_id
            attempt = folder/run_id
            if filename in ('dataset.zip', 'report.json', 'example.tiff', 'soma.json') and (attempt/'dataset.zip').is_file():
                self.serve(attempt/filename)
                return
            if filename in (*PREVIEW_NAMES, 'metrics.png') and (attempt/'dataset.zip').is_file():
                try:
                    lock = METRICS_LOCK if filename.endswith('metrics.png') else self.server.lock
                    with lock:
                        cached = attempt/filename
                        if not cached.is_file() or (filename.endswith('metrics.png') and
                                                    cached.stat().st_mtime < METRICS_MTIME):
                            if filename == 'metrics.png':
                                write_metrics(folder, attempt)
                            else:
                                restore_previews(folder, attempt)
                    self.serve(attempt/filename)
                except (OSError, ValueError, KeyError) as error:
                    self.respond(dict(error=str(error)), 500)
                return
        job, folder, route = self.job_route()
        if job is not None:
            if route == 'status':
                with self.server.lock:
                    state = job.copy()
                self.respond(state)
                return
            if route == 'draft':
                path = folder/'draft.json'
                if path.exists():
                    self.respond(json.loads(path.read_text()))
                elif job['stage'] == 'done':
                    attempt = folder/job['attempt']
                    reviewed = json.loads((attempt/'centers.json').read_text())
                    report = json.loads((attempt/'report.json').read_text())
                    self.respond(dict(reviewed=reviewed, count=len(report['scenes']), seed=report['seed'],
                                      grow_somas=report['calibration'].get('grow_somas', False)))
                else:
                    self.respond({})
                return
            if route == 'reference.png':
                self.serve(folder/route)
                return
            kind, _, asset = route.partition('/')
            if kind in ASSETS:
                if asset == 'api/annotations':
                    self.serve(folder/f'{kind}.json')
                    return
                asset = asset or 'index.html'
                if asset in ASSETS[kind]:
                    self.serve(folder/kind/asset)
                    return
            if job['stage'] == 'done' and route in ('dataset.zip', 'report.json', 'example.tiff', *PREVIEW_NAMES):
                self.serve(folder/job['attempt']/route)
                return
        self.send_error(404)

    def do_POST(self):
        if not self.trusted():
            self.send_error(403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if self.route() == '/upload':
                self.upload(length)
                return
            if not 0 < length < 20*1024**2:
                raise ValueError('Неверный размер запроса.')
            data = json.loads(self.rfile.read(length))
            light_data, source = self.light_route()
            if light_data:
                if data.get('apply') is True:
                    self.apply_light(source, data['settings'])
                else:
                    self.respond(self.light_preview(source).render(data))
                return
            if self.route() == '/prepare':
                source = Path(data['path']).expanduser().resolve(strict=True)
                if not source.is_file() or source.suffix.lower() not in ('.tif', '.tiff'):
                    raise ValueError('Укажите путь к TIFF.')
                token = uuid.uuid4().hex
                folder = OUTPUT/token
                folder.mkdir(parents=True)
                job = dict(stage='preparing', name=source.name)
                with self.server.lock:
                    self.server.jobs[token] = job
                self.background(job, 'ready', lambda: prepare_job(folder, source))
                self.respond(dict(job=token), 202)
                return
            job, folder, route = self.job_route()
            if job is None:
                self.send_error(404)
                return
            with self.server.lock:
                if route == 'draft':
                    meta = json.loads((folder/'soma/assets/dataset.json').read_text())
                    validate_crops(data['crops'], meta['shape_yx'])
                    path = folder/'draft.json'
                    updated = data.get('updated')
                    if type(updated) is not int:
                        raise ValueError('Неверная версия черновика.')
                    if not path.exists() or updated >= json.loads(path.read_text())['updated']:
                        write_json(path, data)
                    self.respond(dict(saved=True))
                    return
                if job['stage'] not in ('ready', 'done', 'fit_error'):
                    self.respond(dict(error='Дождитесь завершения текущей операции.'), 409)
                    return
                if route in ('soma/api/annotations', 'region/api/annotations'):
                    kind = route.split('/')[0]
                    meta = json.loads((folder/kind/'assets/dataset.json').read_text())
                    validator = annotate_somas.validate if kind == 'soma' else annotate_regions.validate_active
                    validator(data, meta)
                    target = folder/f'{kind}.json'
                    write_json(target, data)
                    self.respond(dict(saved=True, path=str(target)))
                    return
                if route == 'generate':
                    crops, count, seed = data['crops'], data['count'], data['seed']
                    meta = json.loads((folder/'soma/assets/dataset.json').read_text())
                    grow_somas = data.get('grow_somas', False)
                    if type(grow_somas) is not bool:
                        raise ValueError('Режим роста должен быть логическим значением.')
                    if type(count) is not int or not 1 <= count <= 256:
                        raise ValueError('Число сцен должно быть от 1 до 256.')
                    if type(seed) is not int or not 0 <= seed < 2**32:
                        raise ValueError('Seed должен быть целым числом от 0 до 4294967295.')
                    validate_crops(crops, meta['shape_yx'])
                    self.start_generation(job, folder, count, lambda attempt, update: generation(
                        folder, attempt, crops, count, seed, update, grow_somas=grow_somas))
                    self.respond(dict(started=True), 202)
                    return
            self.send_error(404)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.respond(dict(error=str(error)), 400)

    def apply_light(self, source, supplied):
        if not isinstance(source, tuple):
            raise ValueError('Новый прогон можно запустить только из проекта калибровки.')
        folder, source_run = source
        preview = self.light_preview(source)
        changes = {key: value for key, value in preview.parse_settings(supplied).items()
                   if value != preview.defaults[key]}
        if not changes:
            raise ValueError('Измените хотя бы один параметр свечения.')
        report = json.loads((source_run/'report.json').read_text())
        with self.server.lock:
            job = self.server.jobs[folder.name]
            if job['stage'] not in ('done', 'fit_error'):
                self.respond(dict(error='Дождитесь завершения текущей операции.'), 409)
                return
            self.start_generation(job, folder, len(report['scenes']), lambda attempt, update:
                                  regenerate_light(folder, source_run, attempt, changes, update))
        self.respond(dict(project=folder.name), 202)

    def start_generation(self, job, folder, count, run):
        attempt = folder/f'run-{uuid.uuid4().hex}'
        attempt.mkdir()
        previous = job.get('attempt') if job['stage'] == 'done' else job.get('previous_attempt')
        job.update(stage='generating', attempt=attempt.name, previous_attempt=previous,
                   completed=0, count=count)
        job.pop('error', None)

        def update(completed):
            with self.server.lock:
                job['completed'] = completed
        self.background(job, 'done', lambda: run(attempt, update))

    def upload(self, length):
        name = Path(unquote(self.headers.get('X-Filename', ''))).name
        suffix = Path(name).suffix.lower()
        if length <= 0 or suffix not in ('.tif', '.tiff'):
            raise ValueError('Выберите файл TIFF.')
        token = uuid.uuid4().hex
        folder = OUTPUT/token
        folder.mkdir(parents=True)
        incoming = folder/('upload'+suffix)
        try:
            with incoming.open('wb') as stream:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024**2))
                    if not chunk:
                        raise ValueError('Загрузка TIFF прервалась.')
                    stream.write(chunk)
                    remaining -= len(chunk)
        except (OSError, ValueError):
            incoming.unlink(missing_ok=True)
            folder.rmdir()
            raise
        job = dict(stage='preparing', name=name)
        with self.server.lock:
            self.server.jobs[token] = job

        def prepare_uploaded():
            try:
                return prepare_job(folder, incoming, source_name=name)
            finally:
                incoming.unlink(missing_ok=True)

        self.background(job, 'ready', prepare_uploaded)
        self.respond(dict(job=token), 202)

    def background(self, job, success, run):
        lock = self.server.lock
        def work():
            try:
                result = run()
                with lock:
                    job.update(stage=success, result=result)
            except Exception as error:
                with lock:
                    job.update(stage='fit_error' if success == 'done' else 'prepare_error', error=str(error))
        threading.Thread(target=work, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--light-source', type=Path, help='Local preview manifest at /light')
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.lock = threading.Lock()
        server.jobs = saved_projects()
        server.light_manifest = args.light_source.resolve(strict=True) if args.light_source else None
        page = '/light' if server.light_manifest else ''
        url = f'http://127.0.0.1:{server.server_port}{page}'
        print(url, flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
