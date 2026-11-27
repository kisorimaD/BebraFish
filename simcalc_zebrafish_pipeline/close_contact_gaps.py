"""Close gaps in densely packed regions with a bounded 3D nearest-center partition."""
import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .soma_growth import Masks, load_growth, save_masks


@dataclass(frozen=True)
class DenseGapFill:
    density_zyx: np.ndarray
    pixel_yx_um: tuple[float, float]
    z_first_um: float
    z_step_um: float
    min_density: float = .65
    max_distance_um: float = 3.5


def load_gap_fill(path):
    with np.load(path, allow_pickle=False) as data:
        return DenseGapFill(data['density_zyx'], tuple(data['pixel_yx_um']), float(data['z_first_um']),
                            float(data['z_step_um']), float(data['min_density']),
                            float(data['max_distance_um']))


def close_dense_gaps(scene, masks, support, *, yx_origin_um=(0., 0.)):
    """Keep the mesh labels and extend them into nearby vacant points in dense tissue."""
    if not np.array_equal(masks.cell_id, scene.somata.cell_id):
        raise ValueError('Mask IDs must match the base scene')
    centers = scene.centers_zyx_um[-1]
    tree = cKDTree(centers)
    count = masks.occupancy()
    if count.max() > 1:
        raise ValueError('Gap closure requires a scene with nonoverlapping meshes')
    pixel = np.asarray(masks.pixel_yx_um)
    support_pitch = np.asarray(support.pixel_yx_um)
    shape_yx = masks.shape_zyx[1:]
    new_pixels, new_indices = [], []
    for k, depth in enumerate(masks.z_positions_um):
        y, x = np.nonzero(count[k] == 0)
        local_yx = (np.column_stack((y, x)) + .5) * pixel
        support_yx = (local_yx + yx_origin_um) / support_pitch - .5
        support_z = np.full(len(y), (depth - support.z_first_um) / support.z_step_um)
        density = map_coordinates(support.density_zyx, (support_z, *support_yx.T), order=1, mode='nearest')
        dense = density >= support.min_density
        points = np.column_stack((np.full(np.count_nonzero(dense), depth), local_yx[dense]))
        distance, index = tree.query(points, distance_upper_bound=support.max_distance_um)
        reachable = np.isfinite(distance)
        flat = np.ravel_multi_index((y[dense], x[dense]), shape_yx)
        new_pixels.append(flat[reachable] + k * np.prod(shape_yx))
        new_indices.append(index[reachable])
    owners = np.concatenate(new_indices)
    if not owners.size:
        return masks
    pixels = np.r_[masks.instance_voxel_index, np.concatenate(new_pixels)]
    owners = np.r_[np.repeat(np.arange(len(centers)), np.diff(masks.instance_offsets)), owners]
    offsets = np.r_[0, np.bincount(owners, minlength=len(centers)).cumsum()]
    return replace(masks, instance_offsets=offsets, instance_voxel_index=pixels[np.lexsort((pixels, owners))])


def run(source, output):
    from .generate_zenodo import _local_fill_map, read_reference

    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    info = json.loads((source / 'summary.json').read_text())
    _, reference, pitch, step, _, _ = read_reference(Path(info['reference']))
    support = DenseGapFill(_local_fill_map(reference), pitch, step / 2, step)
    scene_path = source / 'scene.npz'
    scene = load_growth(scene_path)
    with np.load(source / 'instances.npz', allow_pickle=False) as data:
        masks = Masks(data['cell_id'], tuple(data['shape_zyx']), data['z_positions_um'],
                      tuple(data['pixel_yx_um']), data['instance_offsets'], data['instance_voxel_index'])
    filled = close_dense_gaps(scene, masks, support)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'geometry'
        pending.mkdir()
        save_masks(pending / 'instances.npz', filled)
        np.savez_compressed(pending / 'dense_gap_fill.npz', **vars(support))
        report = dict(scene=str(scene_path), source_masks=str(source / 'instances.npz'),
                      source_reference=info['reference'], pixel_yx_um=pitch,
                      z_positions_um=filled.z_positions_um.tolist(), min_density=support.min_density,
                      max_distance_um=support.max_distance_um)
        (pending / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        pending.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing Zenodo scene and masks directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
