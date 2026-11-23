"""Full-ellipse StarDist targets. Coordinates and ray order are native YX."""
from pathlib import Path
import hashlib
import inspect
import json
import numpy as np


def normalize(image):
    lo, hi = np.percentile(image, [5, 95])
    return np.clip((image.astype(np.float32)-lo)/max(hi-lo, 1e-6), 0, 1).astype(np.float32)


def targets(scene, origin, size=128, rays=32, halo=16):
    """One owner per sample pixel, but rays reach its FULL ellipse boundary.

    Overlaps choose the object with greatest normalized interior depth. This
    retains complete contours in supervision rather than segmenting fragments.
    A single dense head still cannot represent two objects at the same pixel.
    """
    y0, x0 = origin
    centers = scene['centers'] - [y0, x0]
    a, b = scene['major_axis_px']/2, scene['minor_axis_px']/2
    angles = np.radians(scene['orientation_deg'])
    owner = np.full((size, size), -1, np.int32)
    prob = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    near = np.flatnonzero((centers[:, 0]+a >= 0) & (centers[:, 0]-a < size) &
                          (centers[:, 1]+a >= 0) & (centers[:, 1]-a < size))
    for k in near:
        c, s = np.cos(angles[k]), np.sin(angles[k])
        dy, dx = yy-centers[k, 0], xx-centers[k, 1]
        rho = np.sqrt(((dx*c+dy*s)/a[k])**2 + ((-dx*s+dy*c)/b[k])**2)
        p = np.maximum(1-rho, 0)
        use = p > prob
        owner[use], prob[use] = k, p[use]
    dist = np.zeros((size, size, rays+1), np.float32)
    y, x = np.nonzero(owner >= 0)
    k = owner[y, x]
    theta = np.arange(rays)*2*np.pi/rays
    # StarDist: ray 0 points along +X; row=sin(theta), column=cos(theta).
    c, s = np.cos(angles[k])[:, None], np.sin(angles[k])[:, None]
    dx, dy = (x-centers[k, 1])[:, None], (y-centers[k, 0])[:, None]
    u, v = (dx*c+dy*s)/a[k, None], (-dx*s+dy*c)/b[k, None]
    du = (np.cos(theta)*c+np.sin(theta)*s)/a[k, None]
    dv = (-np.cos(theta)*s+np.sin(theta)*c)/b[k, None]
    A, B, C = du*du+dv*dv, u*du+v*dv, u*u+v*v-1
    dist[y, x, :-1] = (-B+np.sqrt(np.maximum(B*B-A*C, 0)))/A
    dist[..., -1] = prob
    valid = np.zeros((size, size), bool)
    valid[halo:size-halo, halo:size-halo] = True
    dist[~valid, -1] = 0
    prob[~valid] = -1  # official StarDist probability loss ignores negative GT
    return prob[..., None], dist


def target_cache_config(directory, rays=32):
    """Bind cached targets to both the scene manifest and the exact target rule."""
    return dict(schema='plane20-full-ellipse-v1', rays=rays,
        dataset_manifest_sha256=hashlib.sha256((Path(directory)/'manifest.jsonl').read_bytes()).hexdigest(),
        target_function_sha256=hashlib.sha256(inspect.getsource(targets).encode()).hexdigest())


class CropSource:
    """Deterministic scene-disjoint crops; negative tissue/background included."""
    def __init__(self, directory, split, seed, size=128, halo=16, rays=32, use_precomputed=None, keep_scenes=False):
        self.paths = sorted((Path(directory)/split).glob('*.npz'))
        if not self.paths:
            raise ValueError(f'No scenes in {directory}/{split}')
        self.seed, self.size, self.halo, self.rays = seed, size, halo, rays
        self.cache = {}
        self.keep_scenes = keep_scenes
        self.target_cache = Path(directory)/'target_cache'
        ready = (self.target_cache/'COMPLETE').is_file()
        self.precomputed = ready if use_precomputed is None else use_precomputed
        if self.precomputed:
            if not ready:
                raise ValueError('Missing complete target cache; run training.plane20.precompute first')
            actual = json.loads((self.target_cache/'config.json').read_text())
            if actual != target_cache_config(directory, rays):
                raise ValueError('Target cache does not match the dataset, rays or target implementation')
        self.split = split
        self.maps = {}

    def crop(self, index):
        """Return the common normalized crop, scene and native origin without targets."""
        image, scene, origin, _ = self.raw_crop(index)
        return normalize(image), scene, origin

    def raw_crop(self, index):
        """Return the unnormalized crop, scene, native origin and scene path."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        path = self.paths[int(rng.integers(len(self.paths)))]
        if path not in self.cache:
            if not (self.precomputed or self.keep_scenes) and len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            with np.load(path) as z:
                self.cache[path] = {k: z[k] for k in
                    ('image', 'centers', 'major_axis_px', 'minor_axis_px', 'orientation_deg')}
                if 'sampling_centers' in z:
                    self.cache[path]['sampling_centers'] = z['sampling_centers']
                if 'image_structured' in z:
                    structured = z['image_structured']
                    if structured.shape != self.cache[path]['image'].shape:
                        raise ValueError(f'Render shape mismatch in {path}')
                    self.cache[path]['image_structured'] = structured
        scene = self.cache[path]
        h, w = scene['image'].shape
        # 80% tissue-oriented samples, 20% uniformly sampled including background.
        for _ in range(32):
            y, x = int(rng.integers(h-self.size+1)), int(rng.integers(w-self.size+1))
            centers = scene.get('sampling_centers', scene['centers']) - [y, x]
            n = np.sum(np.all((centers >= self.halo) & (centers < self.size-self.halo), axis=1))
            if index % 5 == 0 or n >= 8:
                break
        # Alternating sample indices balance the two styles without sharing
        # anatomy across splits or duplicating target caches.
        render = 'image_structured' if index % 2 and 'image_structured' in scene else 'image'
        image = scene[render][y:y+self.size, x:x+self.size]
        return image, scene, (y, x), path

    def sample(self, index):
        image, scene, (y, x) = self.crop(index)
        # crop() has already loaded the scene into the same deterministic cache.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        path = self.paths[int(rng.integers(len(self.paths)))]
        if self.precomputed:
            if path not in self.maps:
                base = self.target_cache/self.split/path.stem
                self.maps[path] = (np.load(base/'prob.npy', mmap_mode='r'),
                                   np.load(base/'dist.npy', mmap_mode='r'))
            prob_map, dist_map = self.maps[path]
            p = prob_map[y:y+self.size, x:x+self.size].copy()[..., None]
            d = np.empty((self.size, self.size, self.rays+1), np.float32)
            d[..., :-1] = dist_map[y:y+self.size, x:x+self.size]
            d[..., -1] = p[..., 0]
            valid = np.zeros((self.size, self.size), bool)
            valid[self.halo:self.size-self.halo, self.halo:self.size-self.halo] = True
            d[~valid, -1] = 0
            p[~valid] = -1
        else:
            p, d = targets(scene, (y, x), self.size, self.rays, self.halo)
        return image[..., None], (p, d)
