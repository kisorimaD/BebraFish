"""Fluorescence preview from saved soma-growth meshes."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from .soma_growth import Growth, rasterize


@dataclass(frozen=True)
class GrowthOptics:
    internal_pitch_um: float = .2
    psf_sigma_zyx_um: tuple[float, float, float] = (1.65, .3456, .3456)
    shell_inset_um: float = .55
    shell_sigma_um: float = .24
    amplitude_adu: float = 85.
    background_adu: float = 120.
    interior_fraction: float = 0.

    def validate(self):
        sigma = np.asarray(self.psf_sigma_zyx_um)
        if sigma.shape != (3,):
            raise ValueError('Expected three PSF widths in ZYX order')
        parameters = (*sigma, self.internal_pitch_um, self.shell_sigma_um,
                      self.shell_inset_um, self.amplitude_adu, self.background_adu, self.interior_fraction)
        if not np.isfinite(parameters).all() or min(parameters[:5]) <= 0 or min(parameters[5:]) < 0:
            raise ValueError('Expected positive mesh-render pitches and widths, and nonnegative ADU and inset')
        if self.interior_fraction > 1.:
            raise ValueError('Interior emission must be between zero and the shell peak')


def _emit_shells(masks, optics, cell_gain):
    pitch = optics.internal_pitch_um
    # Leave room for Gaussian tails outside the tissue before sampling the camera.
    shell_pad = int(np.ceil(4 * optics.shell_sigma_um / pitch)) + 1
    pad = np.ceil(4 * np.asarray(optics.psf_sigma_zyx_um) / pitch).astype(int) + shell_pad
    volume = np.zeros(np.asarray(masks.shape_zyx) + 2 * pad, dtype=np.float32)
    for lo, hi, gain in zip(masks.instance_offsets[:-1], masks.instance_offsets[1:], cell_gain):
        if lo == hi:
            continue
        points = np.asarray(np.unravel_index(masks.instance_voxel_index[lo:hi], masks.shape_zyx))
        points += pad[:, None]
        start, stop = points.min(axis=1) - shell_pad, points.max(axis=1) + shell_pad + 1
        inside = np.zeros(stop - start, dtype=bool)
        inside[tuple(points - start[:, None])] = True
        inner = distance_transform_edt(inside)
        outer = distance_transform_edt(~inside)
        signed_um = np.where(inside, inner - .5, .5 - outer) * pitch
        delta = (signed_um - optics.shell_inset_um) / optics.shell_sigma_um
        light = np.where(np.abs(delta) <= 4., np.exp(-.5 * delta**2), 0.)
        if optics.interior_fraction:
            light += optics.interior_fraction * np.clip(delta, 0., 1.)
        volume[tuple(slice(a, b) for a, b in zip(start, stop))] += gain * light.astype(np.float32)
    return volume, pad


def sample_fluorescence(internal, z_positions_um, pixel_yx_um, shape_yx, optics, cell_gain=None):
    """Sample blurred emission from instance masks on the internal optical grid."""
    optics.validate()
    if cell_gain is None:
        cell_gain = np.ones(len(internal.cell_id), dtype=np.float32)
    cell_gain = np.asarray(cell_gain, dtype=np.float32)
    if (cell_gain.shape != internal.cell_id.shape or not np.isfinite(cell_gain).all()
            or np.any(cell_gain < 0)):
        raise ValueError('Expected one finite nonnegative emission gain per soma')
    pitch = optics.internal_pitch_um
    emission, pad = _emit_shells(internal, optics, cell_gain)
    blurred = gaussian_filter(emission, np.asarray(optics.psf_sigma_zyx_um) / pitch, mode='constant')
    height, width = shape_yx
    dy, dx = pixel_yx_um
    coords = np.ix_(z_positions_um, (np.arange(height) + .5) * dy, (np.arange(width) + .5) * dx)
    sample = [axis / pitch - .5 + offset for axis, offset in zip(coords, pad)]
    blurred = map_coordinates(blurred, np.stack(np.broadcast_arrays(*sample)), order=1, prefilter=False)
    return optics.background_adu + optics.amplitude_adu * blurred


def render_growth(growth: Growth, z_positions_um, pixel_yx_um=(.4, .4), *, stage=-1,
                  optics=GrowthOptics(), cell_gain=None):
    """Return fluorescence and exact geometric masks on the requested acquisition grid."""
    optics.validate()
    masks = rasterize(growth, z_positions_um, pixel_yx_um, stage=stage)
    pitch = optics.internal_pitch_um
    n = int(np.floor(growth.tissue_zyx_um[0] / pitch + .5))
    fixed_z = (np.arange(n) + .5) * pitch
    internal = rasterize(growth, fixed_z, (pitch, pitch), stage=stage)
    light = sample_fluorescence(internal, masks.z_positions_um, masks.pixel_yx_um,
                                masks.shape_zyx[1:], optics, cell_gain)
    return light, masks
