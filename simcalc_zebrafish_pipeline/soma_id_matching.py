"""Group touching soma regions with a single ID map for all acquisition grids."""
from dataclasses import replace
import heapq

import numpy as np
from scipy.ndimage import gaussian_filter

from .zenodo_contrast import _shared_edges, label_image


def _sparse_removals(plane, reference, dense, visible, areas, count_budget, reduction):
    """Remove sparse cells where the simulated occupied area is locally too high."""
    occupied = plane > 0
    target = reference > 0
    budget = int(reduction * (occupied.sum() - target.sum()))
    protected = np.unique(plane[dense])
    eligible = ~np.isin(visible, protected)
    excess = gaussian_filter(occupied.astype(np.float32) - target, 30.)
    scores = np.bincount(plane.ravel(), weights=excess.ravel())[visible] / areas
    kept = np.flatnonzero(eligible & (scores > .02))
    order = kept[np.argsort(-scores[kept], kind='stable')]
    removed = []
    for index in order:
        if budget <= 0 or len(removed) >= count_budget:
            break
        if areas[index] <= budget:
            removed.append(int(visible[index]))
            budget -= int(areas[index])
    return removed


def fit_id_map(masks, target_counts, *, reference=None, dense=None, reduce_excess=0.):
    """Match counts by merging touching regions, optionally removing excess sparse cells."""
    if masks.occupancy().max() > 1:
        raise ValueError('Count matching requires nonoverlapping masks')
    labels = label_image(masks.shape_zyx, masks.instance_voxel_index, masks.instance_offsets, masks.cell_id)
    target_counts = np.asarray(target_counts)
    if target_counts.shape != (len(labels),) or np.any(target_counts <= 0):
        raise ValueError('Expected one positive target count per plane')
    if reduce_excess and (not 0 < reduce_excess <= 1 or reference is None or dense is None
                          or reference.shape != labels.shape or dense.shape != labels.shape):
        raise ValueError('Sparse-cell removal requires reference and dense masks on the acquisition grid')
    ids = np.asarray(masks.cell_id)
    if np.any(np.diff(ids.astype(np.int64)) <= 0):
        raise ValueError('Expected strictly increasing soma IDs')
    owners = ids.copy()
    # A region sampled on two calibration planes would couple their count constraints.
    observed = np.concatenate([np.unique(frame[frame > 0]) for frame in labels])
    if len(np.unique(observed)) != len(observed):
        raise ValueError('Count matching expects disjoint visible IDs across calibration planes')
    for i, (plane, target) in enumerate(zip(labels, target_counts)):
        visible, areas = np.unique(plane[plane > 0], return_counts=True)
        n_merges = len(visible) - int(target)
        if n_merges < 0:
            raise ValueError('A calibration plane has fewer somata than its target')
        area = dict(zip(map(int, visible), map(int, areas)))
        if reduce_excess:
            removed = _sparse_removals(plane, reference[i], dense[i], visible, areas, n_merges, reduce_excess)
            owners[np.searchsorted(ids, removed)] = 0
            for cell in removed:
                area.pop(cell)
            n_merges -= len(removed)
        neighbors = {cell: {} for cell in area}
        edges, lengths = _shared_edges(plane)
        for (a, b), length in zip(edges.tolist(), lengths.tolist()):
            if a in area and b in area:
                neighbors[a][b] = neighbors[b][a] = length
        queue = []

        def queue_pair(a, b, length):
            a, b = min(a, b), max(a, b)
            cost = area[a] + area[b] + 30. / np.sqrt(length)
            heapq.heappush(queue, (cost, a, b, area[a], area[b]))

        for (a, b), length in zip(edges.tolist(), lengths.tolist()):
            if a in area and b in area:
                queue_pair(a, b, length)
        parents = {}
        for _ in range(n_merges):
            while queue:
                _, a, b, old_a, old_b = heapq.heappop(queue)
                if area.get(a) == old_a and area.get(b) == old_b and b in neighbors[a]:
                    break
            else:
                raise ValueError('Too few touching regions to reach the requested count')
            if area[a] < area[b]:
                a, b = b, a
            parents[b] = a
            area[a] += area.pop(b)
            neighbors[a].pop(b)
            for other, length in neighbors.pop(b).items():
                if other == a:
                    continue
                neighbors[other].pop(b)
                merged = neighbors[a].get(other, 0) + length
                neighbors[a][other] = neighbors[other][a] = merged
            for other, length in neighbors[a].items():
                queue_pair(a, other, length)
        for cell in parents:
            owner = parents[cell]
            while owner in parents:
                owner = parents[owner]
            owners[np.searchsorted(ids, cell)] = owner
    return owners


def remap_ids(masks, source_ids, owner_ids):
    """Apply a fixed, many-to-one ID map to sparse masks."""
    source_ids, owner_ids = np.asarray(source_ids), np.asarray(owner_ids)
    if source_ids.ndim != 1 or source_ids.shape != owner_ids.shape:
        raise ValueError('Soma ID map must contain one owner for each source ID')
    positions = np.searchsorted(source_ids, masks.cell_id)
    if (np.any(positions >= len(source_ids)) or
            not np.array_equal(source_ids[positions], masks.cell_id)):
        raise ValueError('Mask IDs must be present in the saved ID map')
    owners, inverse = np.unique(owner_ids[positions], return_inverse=True)
    pixel_owners = np.repeat(inverse, np.diff(masks.instance_offsets))
    pixels = masks.instance_voxel_index
    if owners.size and owners[0] == 0:
        retained = pixel_owners > 0
        pixels, pixel_owners = pixels[retained], pixel_owners[retained] - 1
        owners = owners[1:]
    offsets = np.r_[0, np.bincount(pixel_owners, minlength=len(owners)).cumsum()]
    return replace(masks, cell_id=owners, instance_offsets=offsets,
                   instance_voxel_index=pixels[np.lexsort((pixels, pixel_owners))])
