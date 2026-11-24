"""Render a static Plane20-style volume and write OME-TIFF and instance targets."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from .volume import (
    PROFILES,
    PROJECTED_Z_SIGMA_FRACTION,
    Acquisition,
    render_stack,
    sample_scene,
)


def write_ome(path, image, acquisition, pixel_yx_um):
    dy, dx = pixel_yx_um
    units = {'PhysicalSizeXUnit': 'µm', 'PhysicalSizeYUnit': 'µm', 'PhysicalSizeZUnit': 'µm'}
    depths = acquisition.depths()
    tifffile.imwrite(
        path, image, ome=True, photometric='minisblack',
        metadata=dict(axes='ZYX', PhysicalSizeX=dx, PhysicalSizeY=dy,
                      PhysicalSizeZ=acquisition.z_step_um,
                      Plane={'PositionZ': depths.tolist(), 'PositionZUnit': ['µm'] * len(depths)}, **units),
    )


def write_stack(output, scene, acquisition, stack, *, provenance=None):
    """Publish a complete render in a new directory."""
    profile = scene.profile
    depths = profile.depths(acquisition)
    if not np.array_equal(depths, stack.z_positions_um) or stack.clean_adu.shape != (len(depths), *profile.shape_yx):
        raise ValueError('Rendered stack does not match the requested acquisition and profile')
    metadata = {
        'format_version': 1, 'seed': int(scene.seed), 'profile': asdict(profile), 'axes': 'ZYX',
        'units': {'length': 'µm', 'intensity': 'ADU', 'angle': 'rad'},
        'acquisition': asdict(acquisition), 'optics': asdict(stack.optics),
        'geometry_model': 'ellipsoids with full section masks',
        'emission_model': ('Plane20 shell_patch in XY with a Gaussian axial emission profile'
                           if profile.emission_model == 'projected' else '3D Gaussian radial shell'),
        'background_model': (f'Plane20 median {profile.median_style} compositor'
                             if scene.tissue_maps is not None
                          else 'procedural diffuse field convolved with the PSF'),
        'provenance': provenance or {'xy_scale': 'user acquisition pitch', 'anatomy': 'procedural'},
    }
    if profile.emission_model == 'projected':
        metadata['emission_sigma_z_per_semiaxis'] = PROJECTED_Z_SIGMA_FRACTION
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / 'stack'
        pending.mkdir()
        image = np.rint(np.clip(stack.clean_adu, 0, 65535)).astype(np.uint16)
        write_ome(pending / 'image.ome.tif', image, acquisition, profile.pixel_yx_um)
        np.save(pending / 'clean_adu.npy', stack.clean_adu, allow_pickle=False)
        np.savez_compressed(
            pending / 'instances.npz', cells=scene.cells, z_positions_um=stack.z_positions_um,
            shape_zyx=np.asarray(stack.clean_adu.shape, dtype=np.int64),
            instance_offsets=stack.instance_offsets, instance_voxel_index=stack.instance_voxel_index,
            cell_peak_adu=stack.cell_peak_adu,
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
        pending.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES) + ('real18-22',), default='nominal')
    parser.add_argument('--reference', type=Path, help='Reference NPZ from volume_reference for real18-22')
    parser.add_argument('--median-style', choices=('baseline', 'structured'),
                        help='Plane20 compositor for real18-22 (default: structured)')
    parser.add_argument('--emission-model', choices=('ellipsoid', 'projected'),
                        help='3D radial shell or the Plane20 XY kernel with an axial emission profile')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-planes', type=int)
    parser.add_argument('--z-start-um', type=float)
    parser.add_argument('--z-step-um', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.profile == 'real18-22') != (args.reference is not None):
        parser.error('--reference is required with real18-22 and only applies to that profile')
    if args.median_style is not None and args.profile != 'real18-22':
        parser.error('--median-style only applies to real18-22')
    if args.emission_model is not None and args.profile != 'real18-22':
        parser.error('--emission-model only applies to real18-22')
    try:
        if args.profile == 'real18-22':
            from .volume_reference import read_reference_profile

            profile, maps, provenance = read_reference_profile(args.reference)
            if args.median_style is not None:
                profile = replace(profile, median_style=args.median_style)
        else:
            profile, maps, provenance = PROFILES[args.profile], None, None
        if args.emission_model is not None:
            profile = replace(profile, emission_model=args.emission_model)
        defaults = profile.acquisition
        acquisition = Acquisition(
            defaults.n_planes if args.n_planes is None else args.n_planes,
            defaults.z_start_um if args.z_start_um is None else args.z_start_um,
            defaults.z_step_um if args.z_step_um is None else args.z_step_um,
        )
        profile.depths(acquisition)
        if args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        scene = sample_scene(profile, args.seed, tissue_maps=maps)
        stack = render_stack(scene, acquisition)
        write_stack(args.output, scene, acquisition, stack, provenance=provenance)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'{stack.clean_adu.shape} ZYX, {len(scene.cells)} somas, '
          f'{len(stack.instance_voxel_index)} instance voxels → {args.output}')


if __name__ == '__main__':
    main()
