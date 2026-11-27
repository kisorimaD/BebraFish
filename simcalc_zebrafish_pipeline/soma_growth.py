"""Experimental contact growth for closed, star-shaped soma meshes in µm."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Somata:
    centers_zyx_um: np.ndarray
    target_volume_um3: np.ndarray
    aspect_yx: np.ndarray
    axial_ratio: np.ndarray
    angles_xyz_rad: np.ndarray
    stiffness: np.ndarray
    cell_id: np.ndarray

    def free_axes_zyx_um(self):
        """The XY equivalent radius is the geometric mean of the XY semiaxes."""
        radius = (3 * self.target_volume_um3 / (4 * np.pi * self.axial_ratio)) ** (1 / 3)
        stretch = np.sqrt(self.aspect_yx)
        return np.column_stack((radius * self.axial_ratio, radius / stretch, radius * stretch))


@dataclass(frozen=True)
class GrowthConfig:
    steps: int = 8
    start_scale: float = .35
    mobility: float = .35
    max_shift_um: float = .7
    clearance_um: float = .15
    relaxation_steps: int = 12
    subdivisions: int = 2


@dataclass(frozen=True)
class Growth:
    somata: Somata
    tissue_zyx_um: np.ndarray
    faces: np.ndarray
    scale: np.ndarray
    centers_zyx_um: np.ndarray
    vertices_zyx_um: np.ndarray
    volumes_um3: np.ndarray


@dataclass(frozen=True)
class Masks:
    cell_id: np.ndarray
    shape_zyx: tuple[int, int, int]
    z_positions_um: np.ndarray
    pixel_yx_um: tuple[float, float]
    instance_offsets: np.ndarray
    instance_voxel_index: np.ndarray

    def occupancy(self):
        counts = np.bincount(self.instance_voxel_index, minlength=np.prod(self.shape_zyx))
        return counts.reshape(self.shape_zyx)


def _icosphere(subdivisions):
    phi = (1 + np.sqrt(5)) / 2
    points = []
    for axis in range(3):
        points.extend(np.roll([0, a, b * phi], axis) for a in (-1, 1) for b in (-1, 1))
    points = [p / np.linalg.norm(p) for p in points]
    faces = ConvexHull(points).simplices.tolist()
    for _ in range(subdivisions):
        midpoints = {}

        def midpoint(i, j):
            edge = tuple(sorted((i, j)))
            if edge not in midpoints:
                p = points[i] + points[j]
                midpoints[edge] = len(points)
                points.append(p / np.linalg.norm(p))
            return midpoints[edge]

        refined = []
        for i, j, k in faces:
            a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            refined.extend(((i, a, c), (j, b, a), (k, c, b), (a, b, c)))
        faces = refined
    points, faces = np.asarray(points), np.asarray(faces)
    inward = np.linalg.det(points[faces]) < 0
    faces[inward] = faces[inward, ::-1]
    return points, faces


def _neighbors(faces, n_vertices):
    connected = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        connected[a].update((b, c))
        connected[b].update((a, c))
        connected[c].update((a, b))
    width = max(map(len, connected))
    index, weights = np.zeros((n_vertices, width), int), np.zeros((n_vertices, width))
    for i, ring in enumerate(connected):
        index[i, :len(ring)] = sorted(ring)
        weights[i, :len(ring)] = 1 / len(ring)
    return index, weights


def mesh_volumes(relative_vertices, faces):
    return np.linalg.det(relative_vertices[..., faces, :]).sum(axis=-1) / 6


def _free_shapes(somata, unit, faces):
    xyz_rotation = Rotation.from_euler('xyz', somata.angles_xyz_rad).as_matrix()
    zyx_rotation = xyz_rotation[:, ::-1, ::-1]
    axes = somata.free_axes_zyx_um()
    transforms = zyx_rotation * axes[:, None, :]
    free = np.einsum('nij,vj->nvi', transforms, unit)
    # Match the target using the actual polygonal volume, including mesh resolution.
    correction = np.cbrt(somata.target_volume_um3 / mesh_volumes(free, faces))
    free *= correction[:, None, None]
    transforms *= correction[:, None, None]
    return free, transforms


def _pair_geometry(centers, transforms, pairs):
    i, j = pairs
    delta = centers[j] - centers[i]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance < 1e-8):
        raise ValueError('Soma centers must remain distinct')
    normal = delta / distance[:, None]
    ri = np.linalg.norm(np.einsum('ij,ijk->ik', normal, transforms[i]), axis=1)
    rj = np.linalg.norm(np.einsum('ij,ijk->ik', normal, transforms[j]), axis=1)
    return distance, normal, ri, rj


def _move_centers(centers, initial, transforms, tissue, pairs, config, deform):
    extent = np.linalg.norm(transforms, axis=2)
    if np.any(2 * extent > tissue):
        raise ValueError('A grown free soma does not fit inside the tissue box')
    # Project onto the intersection of wall clearance and the allowed
    # per-axis displacement interval. Its diagonal is at most max_shift_um.
    shift = config.max_shift_um / np.sqrt(3.)
    travel_min = np.maximum(1e-6, initial - shift)
    travel_max = np.minimum(tissue - 1e-6, initial + shift)
    lower = np.maximum(extent, travel_min)
    upper = np.minimum(tissue - extent, travel_max)
    if deform:
        # If a rigid wall clearance cannot be reached, use the closest
        # reachable center and let the surface take up the wall contact.
        lower = np.clip(lower, travel_min, travel_max)
        upper = np.clip(upper, travel_min, travel_max)
    if np.any(lower > upper):
        raise ValueError('Initial centers have insufficient wall clearance for the configured displacement')
    i, j = pairs
    for _ in range(config.relaxation_steps):
        distance, normal, ri, rj = _pair_geometry(centers, transforms, pairs)
        penetration = np.maximum(ri + rj + config.clearance_um - distance, 0.)
        push = penetration[:, None] * normal
        update = np.zeros_like(centers)
        np.add.at(update, i, -push)
        np.add.at(update, j, push)
        centers = np.clip(centers + config.mobility * .25 * update, lower, upper)
    return centers


def _near_pairs(centers, reach, config):
    margin = 2 * config.max_shift_um + config.clearance_um
    pairs = cKDTree(centers).query_pairs(2 * reach.max() + margin, output_type='ndarray')
    i, j = pairs.T
    distance = np.linalg.norm(centers[j] - centers[i], axis=1)
    return pairs[distance <= reach[i] + reach[j] + margin].T


def _contact_caps(centers, free, transforms, tissue, pairs, reach, clearance_um, pressure_offset):
    """Upper bounds for the multiplier of each preferred surface ray."""
    caps = np.full(free.shape[:2], np.inf)
    for axis in range(3):
        position = free[:, :, axis]
        pos = np.divide(tissue[axis] - centers[:, axis, None], position,
                        out=np.full_like(position, np.inf), where=position > 0)
        neg = np.divide(-centers[:, axis, None], position,
                        out=np.full_like(position, np.inf), where=position < 0)
        caps = np.minimum(caps, np.minimum(pos, neg))
    distance, normals, ri, rj = _pair_geometry(centers, transforms, pairs)
    near = distance <= reach[pairs[0]] + reach[pairs[1]] + clearance_um
    left, right = pairs[:, near]
    span = distance[near] - np.minimum(clearance_um, .2 * distance[near])
    split = (span + ri[near] - rj[near]) / 2 + pressure_offset[left] - pressure_offset[right]
    split = np.clip(split, .08 * span, .92 * span)
    normals = normals[near]
    # A shared plane constrains the whole facet, not just its vertices.
    for indices, directions, limits in ((left, normals, split), (right, -normals, span - split)):
        for start in range(0, len(indices), 2048):
            section = slice(start, start + 2048)
            subset = indices[section]
            along = np.einsum('pvc,pc->pv', free[subset], directions[section])
            bound = np.divide(limits[section, None], along, out=np.full_like(along, np.inf), where=along > 0)
            np.minimum.at(caps, subset, bound)
    return caps


def _grow_surfaces(free, faces, caps, target, stiffness, neighbor_index, weights, scale):
    meshes = free * scale
    affected = np.any(caps < scale, axis=1)
    if not affected.any():
        return meshes
    free, caps, target, stiffness = free[affected], caps[affected], target[affected], stiffness[affected]
    tetra_weights = np.linalg.det(free[:, faces]) / 6
    smoothing = (stiffness / (2 + stiffness))[:, None]

    def pressure(multiplier):
        ratios = np.minimum(multiplier[:, None], caps)
        for _ in range(2):
            mean = (ratios[:, neighbor_index] * weights).sum(axis=2)
            ratios = np.minimum((1 - smoothing) * ratios + smoothing * mean, caps)
        volume = np.sum(tetra_weights * np.prod(ratios[:, faces], axis=2), axis=1)
        return ratios, volume

    low, high = np.zeros(len(target)), scale * (1 + .8 / (1 + stiffness))
    for _ in range(24):
        mid = (low + high) / 2
        ratios, volume = pressure(mid)
        under = volume < target
        low = np.where(under, mid, low)
        high = np.where(under, high, mid)
    meshes[affected] = free * ratios[:, :, None]
    return meshes


def _relax_surfaces(somata, centers, free, transforms, faces, tissue, pairs, reach,
                    config, neighbor_index, weights, scale):
    target = somata.target_volume_um3 * scale**3
    pressure_offset = np.zeros(len(centers))
    for _ in range(config.relaxation_steps):
        caps = _contact_caps(centers, free, transforms * scale, tissue, pairs,
                             reach * scale, config.clearance_um, pressure_offset)
        meshes = _grow_surfaces(free, faces, caps, target, somata.stiffness,
                                neighbor_index, weights, scale)
        deficit = np.maximum(0., 1 - mesh_volumes(meshes, faces) / target)
        if deficit.max() < 5e-4:
            break
        # A compressed soma can claim space from a neighbour whose free
        # sides can expand. Each separating plane remains shared by the pair.
        pressure_offset += .55 * np.cbrt(target) * deficit / (1 + somata.stiffness)
    return meshes


def grow(somata: Somata, tissue_zyx_um, *, deform: bool, config=GrowthConfig()) -> Growth:
    """Grow an identical seed catalog with rigid centers-only or deformable contact."""
    tissue = np.asarray(tissue_zyx_um, dtype=float)
    initial = np.asarray(somata.centers_zyx_um, dtype=float)
    n = len(somata.cell_id)
    positive = (somata.target_volume_um3, somata.aspect_yx, somata.axial_ratio, somata.stiffness)
    if (n < 1 or initial.shape != (n, 3) or tissue.shape != (3,) or not np.isfinite(tissue).all()
            or np.any(tissue <= 0) or not np.isfinite(initial).all()
            or np.any((initial <= 0) | (initial >= tissue))):
        raise ValueError('Expected a positive tissue box and soma centers strictly inside it')
    if (any(np.shape(a) != (n,) or not np.isfinite(a).all() or np.any(np.asarray(a) <= 0)
            for a in positive) or np.shape(somata.angles_xyz_rad) != (n, 3)
            or not np.isfinite(somata.angles_xyz_rad).all() or np.any(somata.cell_id <= 0)
            or len(np.unique(somata.cell_id)) != n):
        raise ValueError('Invalid per-soma geometry, stiffness or IDs')
    if (config.steps < 1 or not 0 < config.start_scale < 1 or not 0 <= config.mobility <= 1
            or not np.isfinite([config.max_shift_um, config.clearance_um]).all()
            or min(config.max_shift_um, config.clearance_um) < 0 or config.relaxation_steps < 1
            or not 0 <= config.subdivisions <= 3):
        raise ValueError('Invalid growth or relaxation parameters')

    unit, faces = _icosphere(config.subdivisions)
    free, transforms = _free_shapes(somata, unit, faces)
    reach = np.linalg.norm(transforms, axis=1).max(axis=1) * (1 + .8 / (1 + somata.stiffness))
    pairs = _near_pairs(initial, reach, config)
    neighbor_index, weights = _neighbors(faces, len(unit))
    centers = initial.copy()
    scales = np.linspace(config.start_scale, 1., config.steps + 1)
    center_history, vertex_history, volume_history = [], [], []
    for scale in scales:
        centers = _move_centers(centers, initial, transforms * scale, tissue, pairs, config, deform)
        meshes = free * scale
        if deform:
            meshes = _relax_surfaces(somata, centers, free, transforms, faces, tissue, pairs, reach,
                                     config, neighbor_index, weights, scale)
        center_history.append(centers)
        vertex_history.append(centers[:, None, :] + meshes)
        volume_history.append(mesh_volumes(meshes, faces))
    return Growth(somata, tissue, faces, scales, np.asarray(center_history),
                  np.asarray(vertex_history), np.asarray(volume_history))


def grow_to_shared_boundaries(growth, radius_limit, *, clearance_um=.008, subdivisions=2, smoothing=0.):
    """Fill the space around relaxed centers until neighboring surfaces or walls stop growth."""
    return fill_shared_boundaries(growth.somata, growth.tissue_zyx_um, growth.centers_zyx_um[-1],
                                  radius_limit, clearance_um=clearance_um,
                                  subdivisions=subdivisions, smoothing=smoothing)


def fill_shared_boundaries(somata, tissue, centers, radius_limit, *,
                           clearance_um=.008, subdivisions=2, smoothing=0.):
    """Fill between already placed centers, with one radius limit per soma."""
    radius_limit = np.asarray(radius_limit, dtype=float)
    if (radius_limit.shape != somata.cell_id.shape or not np.isfinite(radius_limit).all()
            or np.any(radius_limit <= 0) or not np.isfinite(clearance_um) or clearance_um <= 0
            or not 0 <= subdivisions <= 3 or not 0 <= smoothing <= 1):
        raise ValueError('Expected positive per-soma radius limits, clearance and valid mesh resolution')
    unit, faces = _icosphere(subdivisions)
    free, transforms = _free_shapes(somata, unit, faces)
    reach = np.linalg.norm(transforms, axis=1).max(axis=1) * radius_limit
    contact_config = GrowthConfig(max_shift_um=0., clearance_um=clearance_um)
    pairs = _near_pairs(centers, reach, contact_config)
    caps = _contact_caps(centers, free, transforms, tissue, pairs, reach,
                         clearance_um, np.zeros(len(centers)))
    scales = np.array([.35, .65, 1.])
    vertices, volumes = [], []
    if smoothing:
        neighbors, weights = _neighbors(faces, len(unit))
        free_radii = np.linalg.norm(free, axis=2)
    for scale in scales:
        ratios = np.minimum(caps, radius_limit[:, None] * scale)
        if smoothing:
            radii = free_radii * ratios
            mean = (radii[:, neighbors] * weights).sum(axis=2)
            # Round only the protrusions so smoothing cannot cross a shared boundary.
            ratios -= smoothing * np.maximum(0., radii - mean) / free_radii
        meshes = free * ratios[:, :, None]
        volumes.append(mesh_volumes(meshes, faces))
        vertices.append(centers[:, None, :] + meshes)
    return Growth(somata, tissue, faces, scales, np.broadcast_to(centers, (len(scales), *centers.shape)),
                  np.asarray(vertices), np.asarray(volumes))


def rasterize(growth: Growth, z_positions_um, pixel_yx_um=(.4, .4), *, stage=-1) -> Masks:
    """Sample closed meshes at pixel centers, preserving overlaps and instance IDs."""
    z = np.asarray(z_positions_um, dtype=float)
    dy, dx = pixel_yx_um
    if (z.ndim != 1 or not len(z) or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
            or z[0] < 0 or z[-1] > growth.tissue_zyx_um[0]
            or not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0):
        raise ValueError('Expected increasing depths inside the tissue and positive pixel pitches')
    height, width = np.floor(growth.tissue_zyx_um[1:] / [dy, dx] + .5).astype(int)
    y, x = (np.arange(height) + .5) * dy, (np.arange(width) + .5) * dx
    if not len(y) or not len(x):
        raise ValueError('Pixel pitches are too large for the tissue box')
    axes = (z, y, x)
    offsets = [0]
    indices = []
    for center, vertices in zip(growth.centers_zyx_um[stage], growth.vertices_zyx_um[stage]):
        lo = [axis.searchsorted(v - 1e-9) for axis, v in zip(axes, vertices.min(axis=0))]
        hi = [axis.searchsorted(v + 1e-9, side='right') for axis, v in zip(axes, vertices.max(axis=0))]
        if any(a == b for a, b in zip(lo, hi)):
            offsets.append(offsets[-1])
            continue
        subaxes = [axis[a:b] for axis, a, b in zip(axes, lo, hi)]
        occupied = np.zeros(tuple(b - a for a, b in zip(lo, hi)), dtype=bool)
        triangles = vertices[growth.faces]
        inverse = np.linalg.inv((triangles - center).transpose(0, 2, 1))
        # A star mesh is the union of tetrahedra from its center to each face.
        # Restrict each test to that tetrahedron's raster bounding box.
        for tri, inv in zip(triangles, inverse):
            low = np.minimum(tri.min(axis=0), center) - 1e-9
            high = np.maximum(tri.max(axis=0), center) + 1e-9
            start = [axis.searchsorted(v) for axis, v in zip(subaxes, low)]
            stop = [axis.searchsorted(v, side='right') for axis, v in zip(subaxes, high)]
            box = tuple(slice(a, b) for a, b in zip(start, stop))
            if any(a == b for a, b in zip(start, stop)):
                continue
            zz, yy, xx = np.ix_(*(axis[s] for axis, s in zip(subaxes, box)))
            coordinates = [inv[i, 0] * (zz - center[0]) + inv[i, 1] * (yy - center[1])
                           + inv[i, 2] * (xx - center[2]) for i in range(3)]
            a, b, c = coordinates
            occupied[box] |= (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9) & (a + b + c <= 1 + 1e-9)
        iz, iy, ix = np.nonzero(occupied)
        index = ((iz + lo[0]) * height + iy + lo[1]) * width + ix + lo[2]
        indices.append(index.astype(np.int64))
        offsets.append(offsets[-1] + len(index))
    return Masks(growth.somata.cell_id, (len(z), int(height), int(width)), z, (dy, dx),
                 np.asarray(offsets, dtype=np.int64), np.concatenate(indices) if indices else np.empty(0, np.int64))


def save_growth(path, growth: Growth):
    s = growth.somata
    np.savez_compressed(
        path, format_version=np.int64(1), tissue_zyx_um=growth.tissue_zyx_um,
        initial_centers_zyx_um=s.centers_zyx_um, cell_id=s.cell_id, target_volume_um3=s.target_volume_um3,
        aspect_yx=s.aspect_yx, axial_ratio=s.axial_ratio, angles_xyz_rad=s.angles_xyz_rad, stiffness=s.stiffness,
        faces=growth.faces, scale=growth.scale, stage_target_um3=growth.scale[:, None]**3 * s.target_volume_um3,
        centers_zyx_um=growth.centers_zyx_um, vertices_zyx_um=growth.vertices_zyx_um, volumes_um3=growth.volumes_um3,
    )


def load_growth(path) -> Growth:
    with np.load(path, allow_pickle=False) as source:
        if int(source['format_version']) != 1:
            raise ValueError('Unknown soma-growth scene format')
        somata = Somata(source['initial_centers_zyx_um'], source['target_volume_um3'], source['aspect_yx'],
                        source['axial_ratio'], source['angles_xyz_rad'], source['stiffness'], source['cell_id'])
        return Growth(somata, source['tissue_zyx_um'], source['faces'], source['scale'],
                      source['centers_zyx_um'], source['vertices_zyx_um'], source['volumes_um3'])


def save_masks(path, masks: Masks):
    np.savez_compressed(path, format_version=np.int64(1), cell_id=masks.cell_id,
                        shape_zyx=masks.shape_zyx, z_positions_um=masks.z_positions_um,
                        pixel_yx_um=masks.pixel_yx_um, instance_offsets=masks.instance_offsets,
                        instance_voxel_index=masks.instance_voxel_index)
