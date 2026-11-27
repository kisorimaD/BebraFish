"""Match local rim-to-core contrast while retaining the simulated cell layout."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, uniform_filter


def label_image(shape, indices, offsets, ids):
    labels = np.zeros(tuple(shape), dtype=np.uint32)
    labels.flat[indices] = np.repeat(ids, np.diff(offsets))
    return labels


def _shared_edges(labels):
    edges = []
    for left, right in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        contact = (left != right) & (left > 0) & (right > 0)
        edges.append(np.sort(np.column_stack((left[contact], right[contact])), axis=1))
    return np.unique(np.concatenate(edges), axis=0, return_counts=True)


def _soma_distances(labels, pixel):
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    inside = distance_transform_edt((labels > 0) & ~edge, sampling=pixel)
    return edge, inside.astype(np.float32)


def _rim_and_core(labels, inside):
    occupied = labels > 0
    return occupied & (inside <= .4), occupied & (inside >= 1.15)


def _local_contrast(values, rim_and_core, window):
    means = []
    for mask in rim_and_core:
        fraction = uniform_filter(mask.astype(np.float32), window, mode='nearest')
        signal = uniform_filter(values * mask, window, mode='nearest')
        means.append(signal / np.maximum(fraction, .007))
    return means[0] - means[1]


def _local_rim_highlight(values, rim, pitch):
    stride = np.maximum(1, np.rint(10.08 / pitch).astype(int))
    axes = [np.arange(0, length + spacing - 1, spacing) for length, spacing in zip(values.shape, stride)]
    estimate = np.empty(tuple(map(len, axes)), dtype=np.float32)
    reference = np.percentile(values[rim], 95) if rim.any() else 0.
    for i, y in enumerate(axes[0]):
        for j, x in enumerate(axes[1]):
            neighborhood = np.s_[max(0, y-stride[0]):y+stride[0]+1, max(0, x-stride[1]):x+stride[1]+1]
            samples = values[neighborhood][rim[neighborhood]]
            estimate[i, j] = np.percentile(samples, 95) if samples.size >= 40 else reference
    y, x = np.ogrid[:values.shape[0], :values.shape[1]]
    coordinates = np.stack(np.broadcast_arrays(y / stride[0], x / stride[1]))
    return map_coordinates(estimate, coordinates, order=1, mode='nearest')


def match_reference_contrast(reference, reference_labels, image, synthetic_labels, pixel, borders):
    """Calibrate on local contrast statistics, then apply changes to synthetic shapes."""
    if (image.ndim != 3 or len(borders) != len(image) or reference.shape != reference_labels.shape
            or reference.shape != image.shape or image.shape != synthetic_labels.shape):
        raise ValueError('Contrast calibration requires images and labels in the same ZYX grid')
    pitch = np.asarray(pixel)
    window = tuple(2 * np.rint(10.08 / pitch).astype(int) + 1)

    def band(values):
        values = values.astype(np.float32, copy=False)
        return values - gaussian_filter(values, 3.15 / pitch)

    corrected = np.empty_like(image)
    for i, (original, source, frame, synthetic, border) in enumerate(
            zip(reference, reference_labels, image, synthetic_labels, borders)):
        _, source_inside = _soma_distances(source, pixel)
        edge, synthetic_inside = _soma_distances(synthetic, pixel)
        source_masks = _rim_and_core(source, source_inside)
        synthetic_masks = _rim_and_core(synthetic, synthetic_inside)
        source_band = band(original)
        target = _local_contrast(source_band, source_masks, window)
        observed = _local_contrast(band(frame), synthetic_masks, window)

        distance_to_rim = distance_transform_edt(~edge, sampling=pixel).astype(np.float32)
        rim_weight = gaussian_filter(np.exp(-.5 * (distance_to_rim / .32)**2), .189 / pitch)
        core_weight = np.clip((synthetic_inside - .38) / .86, 0., 1.)
        core_weight = core_weight**2 * (3. - 2. * core_weight)
        correction = band(core_weight - .65 * rim_weight)
        response = -_local_contrast(band(correction), synthetic_masks, window)
        gain = np.clip((observed - target) / np.maximum(response, .20), -25., 125.)
        tissue = gaussian_filter((source > 0).astype(np.float32), 5.355 / pitch)
        support = np.clip((tissue - .13) / .30, 0., 1.)
        gain *= support
        adjusted = frame + correction * gaussian_filter(gain, 3.78 / pitch)
        highlight_limit = _local_rim_highlight(source_band, source_masks[0], pitch)
        excess = np.maximum(band(adjusted) - highlight_limit, 0.)
        # Extra compression acts only on bright peaks close to synthetic rims.
        proximity = np.exp(-.5 * (distance_to_rim / .44)**2)
        adjusted -= .85 * excess * proximity * support
        corrected[i] = np.rint(np.clip(adjusted, 0, 65535)).astype(np.uint16)
        if border:
            corrected[i, :border] = frame[:border]
            corrected[i, -border:] = frame[-border:]
            corrected[i, :, :border] = frame[:, :border]
            corrected[i, :, -border:] = frame[:, -border:]
    return corrected


def _core_contacts(labels, inside, dense):
    core = (inside >= .65) & (labels > 0) & dense
    cores = np.bincount(labels[core], minlength=int(labels.max()) + 1)
    edges, lengths = _shared_edges(labels)
    keep = (lengths >= 3) & np.all(cores[edges] >= 8, axis=1)
    return core, cores, edges[keep], lengths[keep]


def _core_means(frame, labels, core, cores, envelope_sigma):
    detail = frame - gaussian_filter(frame, envelope_sigma)
    total = np.bincount(labels[core], weights=detail[core], minlength=len(cores))
    return total / np.maximum(cores, 1)


def _scaled_error(measured, target):
    return np.sum(np.log((measured + 1.) / (target + 1.))**2)


def _neighbor_adjustment(frame, labels, inside, core, count, edges, lengths, pitch):
    simulated = _core_means(frame, labels, core, count, 7.56 / pitch)
    a, b = edges.T
    weighted = (np.bincount(a, simulated[b] * lengths, minlength=len(count))
                + np.bincount(b, simulated[a] * lengths, minlength=len(count)))
    contact = (np.bincount(a, lengths, minlength=len(count))
               + np.bincount(b, lengths, minlength=len(count)))
    offsets = np.where(contact > 0, weighted / np.maximum(contact, 1) - simulated, 0.)
    transition = np.clip((inside - .15) / .75, 0., 1.)
    adjustment = gaussian_filter((offsets[labels] * transition).astype(np.float32), .20 / pitch)
    return simulated, adjustment


def _contact_positions(labels, core, count, edges):
    coords = np.nonzero(core)
    weights = [np.bincount(labels[core], weights=axis, minlength=len(count)) for axis in coords]
    centers = np.column_stack(weights) / np.maximum(count[:, None], 1)
    return centers[edges].mean(axis=1)


def refine_local_neighbor_contrast(reference, reference_labels, image, synthetic_labels, pixel, borders, dense):
    """Reduce excess variation between neighbors using regional reference statistics."""
    pitch = np.asarray(pixel)
    stride = np.maximum(1, np.rint(12.6 / pitch).astype(int))
    half_window = 20.16 / pitch
    corrected = image.copy()
    for i, (real, source, raw, labels, border, area) in enumerate(
            zip(reference, reference_labels, image, synthetic_labels, borders, dense)):
        real, frame = real.astype(np.float32), raw.astype(np.float32)
        _, source_inside = _soma_distances(source, pixel)
        _, inside = _soma_distances(labels, pixel)
        source_core, source_count, source_edges, _ = _core_contacts(source, source_inside, area)
        core, count, edges, lengths = _core_contacts(labels, inside, area)
        if not len(source_edges) or not len(edges):
            continue
        positions = _contact_positions(labels, core, count, edges)
        ref_positions = _contact_positions(source, source_core, source_count, source_edges)
        actual = _core_means(real, source, source_core, source_count, 7.56 / pitch)
        simulated, adjustment = _neighbor_adjustment(frame, labels, inside, core, count, edges, lengths, pitch)
        response = _core_means(adjustment, labels, core, count, 7.56 / pitch)
        a, b = edges.T
        initial, delta = simulated[a] - simulated[b], response[a] - response[b]
        ref_delta = np.abs(actual[source_edges[:, 0]] - actual[source_edges[:, 1]])
        axes = [np.arange(0, length + spacing - 1, spacing) for length, spacing in zip(frame.shape, stride)]
        amounts = np.zeros(tuple(map(len, axes)), dtype=np.float32)
        trials = np.linspace(0., .85, 10)
        for iy, y in enumerate(axes[0]):
            for ix, x in enumerate(axes[1]):
                point = (y, x)
                local = np.all(np.abs(positions - point) <= half_window, axis=1)
                ref_local = np.all(np.abs(ref_positions - point) <= half_window, axis=1)
                if min(local.sum(), ref_local.sum()) < 20:
                    continue
                target = np.percentile(ref_delta[ref_local], [50, 90])
                observed = np.abs(initial[local] + trials[:, None] * delta[local])
                quantiles = np.percentile(observed, [50, 90], axis=1).T
                error = np.log((quantiles + 1.) / (target + 1.))**2 @ [1., .35]
                amounts[iy, ix] = trials[error.argmin()]
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        coordinates = np.stack(np.broadcast_arrays(y / stride[0], x / stride[1]))
        local_weight = map_coordinates(amounts, coordinates, order=1, mode='nearest')
        frame += local_weight * adjustment
        corrected[i] = np.rint(np.clip(frame, 0, 65535)).astype(np.uint16)
        if border:
            corrected[i, :border] = raw[:border]
            corrected[i, -border:] = raw[-border:]
            corrected[i, :, :border] = raw[:, :border]
            corrected[i, :, -border:] = raw[:, -border:]
    return corrected


def refine_reference_texture(reference, reference_labels, image, synthetic_labels, pixel, borders, dense):
    """Fit the contrast between adjacent cell interiors and the gradient at their rims."""
    pitch = np.asarray(pixel)
    corrected = np.empty_like(image)
    settings = []
    for i, (real, source, raw, labels, border, area) in enumerate(
            zip(reference, reference_labels, image, synthetic_labels, borders, dense)):
        real, frame = real.astype(np.float32), raw.astype(np.float32)
        _, source_inside = _soma_distances(source, pixel)
        _, inside = _soma_distances(labels, pixel)
        source_core, source_count, source_edges, _ = _core_contacts(source, source_inside, area)
        core, count, edges, lengths = _core_contacts(labels, inside, area)
        envelope_sigma = 7.56 / pitch
        weight = 0.
        if len(source_edges) and len(edges):
            actual = _core_means(real, source, source_core, source_count, envelope_sigma)
            simulated, adjustment = _neighbor_adjustment(frame, labels, inside, core, count, edges, lengths, pitch)
            a, b = edges.T
            response = _core_means(adjustment, labels, core, count, envelope_sigma)
            target = np.percentile(np.abs(actual[source_edges[:, 0]] - actual[source_edges[:, 1]]), [50, 90])
            weight = min(np.linspace(0., 1.2, 13), key=lambda amount: _scaled_error(
                np.percentile(np.abs((simulated[a] - simulated[b]) + amount * (response[a] - response[b])),
                              [50, 90]), target))
            frame += weight * adjustment
        rim = _rim_and_core(labels, inside)[0] & area
        source_rim = _rim_and_core(source, source_inside)[0] & area
        blur_um = 0.
        if rim.any() and source_rim.any():
            goal = np.percentile(np.hypot(*np.gradient(real))[source_rim], [50, 90])
            best_error = np.inf
            for sigma in (0., .14, .20, .26, .33, .40):
                candidate = gaussian_filter(frame, sigma / pitch)
                gradient = np.percentile(np.hypot(*np.gradient(candidate))[rim], [50, 90])
                error = _scaled_error(gradient, goal)
                if error < best_error:
                    best_error, blur_um, best = error, sigma, candidate
            frame = best
        corrected[i] = np.rint(np.clip(frame, 0, 65535)).astype(np.uint16)
        if border:
            corrected[i, :border] = raw[:border]
            corrected[i, -border:] = raw[-border:]
            corrected[i, :, :border] = raw[:, :border]
            corrected[i, :, -border:] = raw[:, -border:]
        settings.append(dict(neighbor_blend=float(weight), extra_blur_um=blur_um))
    return corrected, settings
