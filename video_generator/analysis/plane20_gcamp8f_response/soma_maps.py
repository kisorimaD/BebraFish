"""Reconstruct per-soma optical contributions from a saved synthetic scene."""

import json

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix

from training.plane20.morphology import sample_angular_coefficients, shell_patch
from training.plane20.photometry import cell_amplitudes, effective_blur, smooth_gain_field


def soma_maps(scene, profile):
    config = json.loads(str(profile['median_config']))
    shape = scene['image'].shape
    scene_rng = np.random.default_rng(int(scene['seed']))
    _, _, _, spatial, brightness, angular_seed, focus, _ = scene_rng.integers(2**32,size=8)
    centers = scene['centers']
    amplitudes = cell_amplitudes(smooth_gain_field(shape,27.93,spatial),centers,.55,.35,brightness)
    angular = .5*sample_angular_coefficients(len(centers),angular_seed)
    mixture = 1/(1+np.exp(-1.5*smooth_gain_field(shape,40,focus)))
    distance = distance_transform_edt(profile['neuron_support'])
    gain = profile['normalizer']*np.clip(distance/2,0,1)
    gain *= config['amplitude']*.36/(config['width']*config['peak'])
    values, columns, offsets = [], [], [0]
    for cell, center in enumerate(centers):
        origin, kernel, _ = shell_patch(
            center,scene['major_axis_px'][cell],scene['minor_axis_px'][cell],
            scene['orientation_deg'][cell],True,0,.6,peak_radius=config['peak'],
            width=config['width'],angular_coefficients=angular[cell],
        )
        end = origin+kernel.shape
        low, high = np.maximum(0,origin-8), np.minimum(shape,end+8)
        first, last = np.maximum(0,origin), np.minimum(shape,end)
        roi = tuple(slice(a,b) for a,b in zip(low,high))
        patch = tuple(slice(a,b) for a,b in zip(first-origin,last-origin))
        local = tuple(slice(a,b) for a,b in zip(first-low,last-low))
        padded = np.zeros(tuple(high-low),np.float32)
        padded[local] = kernel[patch]*amplitudes[cell]
        light = effective_blur(padded,.2,.8,mixture[roi])*gain[roi]
        y,x = np.nonzero(light)
        values.append(light[y,x].astype(np.float32))
        columns.append((y+low[0])*shape[1]+x+low[1])
        offsets.append(offsets[-1]+len(y))
    return csr_matrix((np.concatenate(values),np.concatenate(columns),offsets),
                      shape=(len(centers),int(np.prod(shape))))
