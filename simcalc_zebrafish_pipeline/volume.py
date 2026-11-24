"""Static Plane20-style scenes in physical ZYX coordinates."""
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from .median_photometry import (
    cell_amplitudes,
    median_background,
    median_cell_light,
    median_texture,
    smooth_gain_field,
    structured_amount,
    structured_layer,
)
from .soma_shell_model import (
    arc_log_gain,
    gaussian_shell,
    sample_angular_coefficients,
    shell_patch,
)

TEXTURE_STEP_UM = 5.
GAIN_STEP_UM = 4.
PROJECTED_Z_SIGMA_FRACTION = .85

CELL_DTYPE = np.dtype([
    ('cell_id', '<u4'), ('center_zyx_um', '<f8', (3,)),
    ('semiaxes_zyx_um', '<f8', (3,)), ('angle_rad', '<f8'),
    ('amplitude_adu', '<f8'), ('shell_peak_radius', '<f8'), ('shell_width', '<f8'),
    ('angular_coefficients', '<f8', (4, 2)), ('axial_coefficients', '<f8', (2,)),
])


@dataclass(frozen=True)
class Acquisition:
    n_planes: int = 31
    z_start_um: float = 0.
    z_step_um: float = 10.

    def depths(self):
        if (isinstance(self.n_planes, bool)
                or not isinstance(self.n_planes, (int, np.integer)) or self.n_planes < 1):
            raise ValueError('n_planes must be a positive integer')
        if not np.isfinite(self.z_start_um) or not np.isfinite(self.z_step_um) or self.z_step_um <= 0:
            raise ValueError('z_start_um must be finite and z_step_um finite and positive')
        # Match decimal depths exactly across acquisition grids.
        start, step = Decimal(str(self.z_start_um)), Decimal(str(self.z_step_um))
        z = np.fromiter((float(start + k * step) for k in range(self.n_planes)),
                        dtype=np.float64, count=self.n_planes)
        if not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
            raise ValueError('Z positions must be finite and strictly increasing')
        return z


@dataclass(frozen=True)
class Optics:
    sigma_zyx_um: tuple[float, float, float] = (1.65, .35, .35)
    axial_integration_um: float = .4

    def quadrature(self):
        sigma = np.asarray(self.sigma_zyx_um)
        step = self.axial_integration_um
        if (sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0)
                or not np.isfinite(step) or step <= 0 or step > sigma[0] / 2):
            raise ValueError('PSF sigmas must be positive and finite, with axial integration step at most sigma_z / 2')
        radius = int(np.ceil(4 * sigma[0] / step))
        offsets = np.arange(-radius, radius + 1) * step
        weights = np.exp(-.5 * (offsets / sigma[0])**2)
        return offsets, weights / weights.sum()


@dataclass(frozen=True)
class Profile:
    name: str = 'nominal'
    shape_yx: tuple[int, int] = (288, 1728)
    pixel_yx_um: tuple[float, float] = (.576, .576)
    depth_range_um: tuple[float, float] = (0., 300.)
    depth_margin_um: float = 16.
    density_per_um3: float = .00058
    min_center_distance_um: float = 4.2
    diameter_xy_um: float = 7.8
    diameter_z_um: float = 7.
    shell_peak_radius: float = .78
    shell_width: float = .19
    amplitude_adu: float = 115.
    texture_band_rms_adu: float = 0.
    background_floor_adu: float = 0.
    median_style: str = 'baseline'
    emission_model: str = 'ellipsoid'
    acquisition: Acquisition = Acquisition()
    optics: Optics = Optics()

    def depths(self, acquisition=None):
        z = (acquisition or self.acquisition).depths()
        if z[0] < self.depth_range_um[0] or z[-1] > self.depth_range_um[1]:
            raise ValueError(f'Requested Z interval [{z[0]}, {z[-1]}] µm lies outside {self.depth_range_um} µm')
        return z

    def validate(self):
        if (len(self.shape_yx) != 2 or any(isinstance(n, bool)
                or not isinstance(n, (int, np.integer)) or n < 1 for n in self.shape_yx)):
            raise ValueError('shape_yx must contain two positive integers')
        if len(self.pixel_yx_um) != 2 or len(self.depth_range_um) != 2:
            raise ValueError('Expected two XY spacings and two Z bounds')
        if self.median_style not in ('baseline', 'structured'):
            raise ValueError('median_style must be baseline or structured')
        if self.emission_model not in ('ellipsoid', 'projected'):
            raise ValueError('emission_model must be ellipsoid or projected')
        positive = (*self.pixel_yx_um, self.depth_margin_um, self.min_center_distance_um,
                    self.diameter_xy_um, self.diameter_z_um, self.shell_peak_radius, self.shell_width)
        if (not np.isfinite(positive).all() or min(positive) <= 0
                or not np.isfinite([self.density_per_um3, self.amplitude_adu,
                                    self.texture_band_rms_adu, self.background_floor_adu]).all()
                or min(self.density_per_um3, self.amplitude_adu,
                       self.texture_band_rms_adu, self.background_floor_adu) < 0
                or not np.isfinite(self.depth_range_um).all()
                or self.depth_range_um[0] >= self.depth_range_um[1]):
            raise ValueError('Invalid profile dimensions, physical parameters or depth range')
        self.acquisition.depths()
        self.optics.quadrature()


PROFILES = {
    'nominal': Profile(),
    'demo': Profile(name='demo', shape_yx=(128, 128), depth_range_um=(0., 40.),
                    acquisition=Acquisition(5)),
}


@dataclass(frozen=True)
class TissueMaps:
    """Image-derived priors on the profile's native TIFF grid."""
    depths_um: np.ndarray
    support: np.ndarray
    density: np.ndarray
    background_adu: np.ndarray

    def sample(self, field, z, y_px, x_px):
        z, y, x = np.broadcast_arrays(np.interp(z, self.depths_um, np.arange(len(self.depths_um))),
                                       y_px, x_px)
        return map_coordinates(field, np.stack([z, y, x]), order=1, mode='nearest')


@dataclass(frozen=True)
class Scene:
    profile: Profile
    seed: int
    cells: np.ndarray
    phases: np.ndarray
    tissue_maps: TissueMaps | None = None
    texture: np.ndarray | None = None
    focus: np.ndarray | None = None
    placement: np.ndarray | None = None

    def _support(self, z, y, x):
        height, width = np.asarray(self.profile.shape_yx) * self.profile.pixel_yx_um
        p = self.phases
        lower = .13 + .055 * np.sin(x / 32 + z / 26 + p[0]) + .025 * np.sin(x / 13 - z / 18 + p[1])
        upper = .89 + .05 * np.sin(x / 42 - z / 31 + p[2])
        edge = np.minimum(y - height * lower, height * upper - y)
        x_edge = np.minimum(x, width - x)
        return np.clip(np.minimum(edge, x_edge) / 3. + .5, 0, 1)

    def fields(self, z, y, x):
        """Continuous tissue membership and relative center density."""
        if self.tissue_maps is not None:
            dy, dx = self.profile.pixel_yx_um
            coords = (z, y / dy - .5, x / dx - .5)
            support = self.tissue_maps.sample(self.tissue_maps.support, *coords)
            density = self.tissue_maps.sample(self.tissue_maps.density, *coords)
            return support, density
        density = .68 + .25 * np.sin(x / 17 + y / 29 + z / 21 + self.phases[3])**2
        return self._support(z, y, x), density

    def background(self, z, y, x):
        """Diffuse background in ADU (already observed for TIFF-derived maps)."""
        if self.tissue_maps is not None:
            dy, dx = self.profile.pixel_yx_um
            return self.tissue_maps.sample(self.tissue_maps.background_adu, z, y / dy - .5, x / dx - .5)
        support = self._support(z, y, x)
        p = self.phases
        texture = (14 * np.sin(x / 14 + y / 17 + z / 25 + p[4])
                   + 11 * np.cos(x / 22 - y / 11 + z / 18 + p[5])
                   + 5 * np.sin(x / 4 - y / 5 - z / 9 + p[6]))
        return 128 + support * (52 + texture)

    def texture_at(self, depth):
        position = (depth - self.profile.depth_range_um[0]) / TEXTURE_STEP_UM
        i = min(int(position), len(self.texture) - 2)
        weight = position - i
        return (1 - weight) * self.texture[i] + weight * self.texture[i + 1]

    def median_plane(self, depth, y, x, cells):
        profile = self.profile
        support, density = self.fields(depth, y, x)
        tissue = support >= .8
        start = profile.depth_range_um[0] - profile.depth_margin_um
        coords = np.stack(np.broadcast_arrays((depth - start) / GAIN_STEP_UM,
                                               y / GAIN_STEP_UM, x / GAIN_STEP_UM))
        focus = map_coordinates(self.focus, coords, order=1, mode='nearest')
        mixture = 1 / (1 + np.exp(-1.5 * focus))
        distance = distance_transform_edt(np.pad(tissue, 1))[1:-1, 1:-1]
        cells = median_cell_light(cells, mixture, distance, 1., 1.)
        background = median_background(self.background(depth, y, x), cells, profile.background_floor_adu)
        if profile.median_style == 'baseline':
            return background + cells
        placement = map_coordinates(self.placement, coords, order=1, mode='nearest')
        amount = structured_amount(placement, tissue)
        subdued = np.clip((.65 - density) / .45, 0, 1) * support
        texture = self.texture_at(depth) * support * (.35 + 1.8 * amount) if self.texture is not None else 0.
        return structured_layer(background, cells, amount, texture, subdued, .30, .78,
                                profile.background_floor_adu)


def _sample_texture(profile, rng):
    start, stop = profile.depth_range_um
    n = int(np.ceil((stop - start) / TEXTURE_STEP_UM)) + 1
    noise = rng.standard_normal((n, *profile.shape_yx), dtype=np.float32)
    field = median_texture(noise, (.55, 4.2, 4.2), highpass_sigma=(0, 9, 9))
    band = gaussian_filter(field, (0, 3, 3)) - gaussian_filter(field, (0, 7, 7))
    field *= profile.texture_band_rms_adu / max(float(band.std()), 1e-6)
    field.flags.writeable = False
    return field


def sample_scene(profile: Profile, seed: int, *, tissue_maps: TissueMaps | None = None) -> Scene:
    """Sample anatomy and all soma properties once, independently of acquisition."""
    profile.validate()
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, 7)
    cells = np.empty(0, dtype=CELL_DTYPE)
    scene = Scene(profile, seed, cells, phases, tissue_maps)
    height, width = np.asarray(profile.shape_yx) * profile.pixel_yx_um
    z_min, z_max = profile.depth_range_um
    low = z_min - profile.depth_margin_um
    high = z_max + profile.depth_margin_um
    # Oversample to compensate for the small loss from 3D center exclusion.
    count = rng.poisson((high - low) * height * width * profile.density_per_um3 * 1.15)
    candidates = rng.uniform((low, 0, 0), (high, height, width), size=(count, 3))
    support, density = scene.fields(*candidates.T)
    centers = candidates[(support >= .8) & (rng.random(count) < density)]
    if len(centers):
        neighbors = cKDTree(centers).query_ball_point(centers, profile.min_center_distance_um)
        allowed = np.ones(len(centers), dtype=bool)
        keep = []
        for i, near in enumerate(neighbors):
            if allowed[i]:
                keep.append(i)
                allowed[near] = False
        centers = centers[keep]

    count = len(centers)
    cells = np.empty(count, dtype=CELL_DTYPE)
    cells['cell_id'] = np.arange(1, count + 1, dtype=np.uint32)
    cells['center_zyx_um'] = centers
    radius = profile.diameter_xy_um / 2 * np.clip(rng.lognormal(0, .14, count), .65, 1.5)
    stretch = np.sqrt(1 + np.clip(rng.gamma(2, .15, count), 0, 1.1))
    axial = profile.diameter_z_um / 2 * np.clip(rng.lognormal(0, .20, count), .55, 1.6)
    cells['semiaxes_zyx_um'] = np.column_stack((axial, radius / stretch, radius * stretch))
    cells['angle_rad'] = rng.uniform(0, np.pi, count)
    gain_shape = np.ceil(np.array([high - low, height, width]) / GAIN_STEP_UM).astype(int) + 1
    xy_length = 27.93 * np.asarray(profile.pixel_yx_um)
    gain = smooth_gain_field(gain_shape, np.r_[12., xy_length] / GAIN_STEP_UM, rng)
    gain_centers = (centers - [low, 0, 0]) / GAIN_STEP_UM
    cells['amplitude_adu'] = profile.amplitude_adu * cell_amplitudes(gain, gain_centers, .55, .35, rng)
    cells['shell_peak_radius'] = profile.shell_peak_radius
    cells['shell_width'] = profile.shell_width
    cells['angular_coefficients'] = .5 * sample_angular_coefficients(count, rng)
    cells['axial_coefficients'] = rng.normal(0, .22, (count, 2))
    cells.flags.writeable = phases.flags.writeable = False
    texture = _sample_texture(profile, rng) if profile.texture_band_rms_adu > 0 else None
    focus = placement = None
    if tissue_maps is not None:
        focus = smooth_gain_field(gain_shape, np.r_[14., 40. * np.asarray(profile.pixel_yx_um)] / GAIN_STEP_UM, rng)
        focus.flags.writeable = False
        if profile.median_style == 'structured':
            placement = smooth_gain_field(gain_shape, np.r_[10., 35. * np.asarray(profile.pixel_yx_um)] / GAIN_STEP_UM, rng)
            placement.flags.writeable = False
    return Scene(profile, seed, cells, phases, tissue_maps, texture, focus, placement)


@dataclass
class Stack:
    z_positions_um: np.ndarray
    clean_adu: np.ndarray
    instance_offsets: np.ndarray
    instance_voxel_index: np.ndarray
    cell_peak_adu: np.ndarray
    optics: Optics


def _pixel_slice(center, radius, pitch, size):
    start = max(0, int(np.floor((center - radius) / pitch - .5)))
    stop = min(size, int(np.ceil((center + radius) / pitch - .5)) + 1)
    return slice(start, stop)


def _projected_light(cell, relative_z, xy_um, sigma_yx, offsets, weights, signal, peaks):
    _, cy, cx = cell['center_zyx_um']
    az, ay, ax = cell['semiaxes_zyx_um']
    sigma_z = PROJECTED_Z_SIGMA_FRACTION * az
    active = np.flatnonzero(np.abs(relative_z) <= 4 * sigma_z + offsets[-1])
    if not len(active):
        return
    dy, dx = xy_um
    sy, sx = sigma_yx
    origin, kernel, _ = shell_patch(
        (cy / dy - .5, cx / dx - .5), 2 * ax, 2 * ay, np.degrees(cell['angle_rad']),
        True, 0., (sy / dy, sx / dx), peak_radius=cell['shell_peak_radius'],
        width=cell['shell_width'], angular_coefficients=cell['angular_coefficients'], pixel_yx=xy_um,
    )
    start = np.maximum(origin, 0)
    stop = np.minimum(origin + kernel.shape, signal.shape[1:])
    target = tuple(slice(a, b) for a, b in zip(start, stop))
    patch = kernel[tuple(slice(a - o, b - o) for a, b, o in zip(start, stop, origin))]
    delta = relative_z[active, None] + offsets
    emission = np.where(np.abs(delta) <= 4 * sigma_z, np.exp(-.5 * (delta / sigma_z)**2), 0)
    axial = np.einsum('ij,j->i', emission, weights)
    for k, scale in zip(active, axial):
        contribution = (cell['amplitude_adu'] * scale * patch).astype(np.float32)
        signal[k][target] += contribution
        peaks[k] = contribution.max()


def _soma_patch(cell, z, xy_um, shape, optics, offsets, weights, signal, peaks, *, projected):
    cz, cy, cx = cell['center_zyx_um']
    az, ay, ax = cell['semiaxes_zyx_um']
    peak, width = cell['shell_peak_radius'], cell['shell_width']
    cutoff = peak + 4 * width
    cos, sin = np.cos(cell['angle_rad']), np.sin(cell['angle_rad'])
    sy, sx = optics.sigma_zyx_um[1:]
    dy, dx = xy_um
    height, width_px = shape
    # Include emission outside the image that can blur back into the field.
    pad_y, pad_x = int(np.ceil(4 * sy / dy)), int(np.ceil(4 * sx / dx))
    ry, rx = np.hypot(ax * sin, ay * cos), np.hypot(ax * cos, ay * sin)
    reach = max(1., cutoff)
    ys = _pixel_slice(cy, reach * ry + pad_y * dy, dy, height)
    xs = _pixel_slice(cx, reach * rx + pad_x * dx, dx, width_px)
    if ys.start >= ys.stop or xs.start >= xs.stop:
        return []
    y_indices = np.arange(ys.start - pad_y, ys.stop + pad_y)
    x_indices = np.arange(xs.start - pad_x, xs.stop + pad_x)
    y = ((y_indices + .5) * dy - cy)[:, None]
    x = (x_indices + .5) * dx - cx
    u, v = (x * cos + y * sin) / ax, (-x * sin + y * cos) / ay
    radial_xy2 = u**2 + v**2
    inner = (slice(pad_y, pad_y + ys.stop - ys.start), slice(pad_x, pad_x + xs.stop - xs.start))
    h = (z - cz) / az
    voxel_parts = []
    for k in np.flatnonzero(np.abs(h) <= 1):
        iy, ix = np.nonzero(radial_xy2[inner] + h[k]**2 <= 1)
        voxel_parts.append(((k * height + ys.start + iy) * width_px + xs.start + ix).astype(np.int64))
    if projected:
        _projected_light(cell, z - cz, xy_um, (sy, sx), offsets, weights, signal, peaks)
        return voxel_parts

    radial_xy = np.sqrt(radial_xy2)
    log_gain = arc_log_gain(u, v, ax, ay, cell['angular_coefficients'])
    if not np.any(log_gain):
        flux_scale = 1.
    else:
        equatorial = gaussian_shell(radial_xy, peak, width)
        flux_scale = equatorial.sum() / max(float(np.sum(equatorial * np.exp(log_gain))), 1e-300)
    linear, quadratic = cell['axial_coefficients']
    for k in np.flatnonzero(np.abs(z - cz) <= az * cutoff + offsets[-1]):
        w = ((z[k] - cz + offsets) / az)[:, None, None]
        rho = np.sqrt(radial_xy2 + w**2)
        shell = gaussian_shell(rho, peak, width)
        safe_rho = np.maximum(rho, 1e-6)
        direction_z = w / safe_rho
        pattern = linear * direction_z + quadratic * (direction_z**2 - 1/3)
        # Use the same equatorial arcs as Plane20 and fade them at the poles.
        pattern += log_gain * (radial_xy / safe_rho)**2
        axial = np.einsum('i,ijk->jk', weights, np.where(rho <= cutoff, shell * np.exp(pattern), 0))
        blurred = gaussian_filter(axial, (sy / dy, sx / dx), mode='constant')
        contribution = (cell['amplitude_adu'] * flux_scale * blurred[inner]).astype(np.float32)
        signal[k, ys, xs] += contribution
        peaks[k] = contribution.max()
    return voxel_parts


def render_stack(scene: Scene, acquisition: Acquisition | None = None, *, optics: Optics | None = None) -> Stack:
    """Render slices and full instance masks on an independently chosen Z grid."""
    p = scene.profile
    z = p.depths(acquisition)
    optics = optics or p.optics
    offsets, weights = optics.quadrature()
    projected = p.emission_model == 'projected'
    reach = 4 * PROJECTED_Z_SIGMA_FRACTION if projected else p.shell_peak_radius + 4 * p.shell_width
    axial_reach = .8 * p.diameter_z_um * max(1., reach)
    if p.depth_margin_um < axial_reach + offsets[-1]:
        raise ValueError('Scene depth margin is too small for the soma emission and axial PSF')
    height, width = p.shape_yx
    dy, dx = p.pixel_yx_um
    y, x = (np.arange(height)[:, None] + .5) * dy, (np.arange(width) + .5) * dx
    signal = np.zeros((len(z), height, width), dtype=np.float32)
    for k, depth in enumerate(z):
        if scene.tissue_maps is None:
            diffuse = np.zeros((height, width))
            for offset, weight in zip(offsets, weights):
                diffuse += weight * scene.background(depth + offset, y, x)
            sy, sx = optics.sigma_zyx_um[1:]
            signal[k] = gaussian_filter(diffuse, (sy / dy, sx / dx), mode='nearest')
    instance_offsets = np.zeros(len(scene.cells) + 1, dtype=np.int64)
    peaks = np.zeros((len(scene.cells), len(z)), dtype=np.float32)
    parts = []
    for i, cell in enumerate(scene.cells):
        voxels = _soma_patch(cell, z, p.pixel_yx_um, p.shape_yx, optics, offsets, weights, signal, peaks[i],
                            projected=projected)
        parts.extend(voxels)
        instance_offsets[i + 1] = instance_offsets[i] + sum(len(part) for part in voxels)
    for k, depth in enumerate(z):
        if scene.tissue_maps is not None:
            signal[k] = scene.median_plane(depth, y, x, signal[k])
        elif scene.texture is not None:
            support, density = scene.fields(depth, y, x)
            signal[k] += scene.texture_at(depth) * support * (.45 + .8 * density)
    indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return Stack(z, signal, instance_offsets, indices, peaks, optics)
