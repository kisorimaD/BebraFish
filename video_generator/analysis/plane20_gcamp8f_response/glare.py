"""Trade excess static background light for somatic brightness."""
import numpy as np


def redistribute_glare(base, reference, reference_masks, tissue, maps, fraction, display_white,
                       *, return_layers=False):
    if not 0 <= fraction <= 1:
        raise ValueError('Glare reduction must be between zero and one.')
    coverage = np.bincount(reference_masks, minlength=base.size).reshape(base.shape)
    background = reference[(coverage == 0) & tissue]
    threshold = float(np.quantile(background, .90))
    ceiling = float(np.quantile(reference, .9999))
    soma_light = np.asarray(maps.sum(axis=0)).reshape(base.shape)
    diffuse = base - soma_light
    # The thresholds are uniform in ADU. The first term lowers bright diffuse
    # emission and the second limits persistent extreme values in the scene.
    removed = fraction * np.maximum(diffuse - threshold, 0)
    removed += fraction * np.maximum(base - removed - ceiling, 0)
    total = float(np.sum(removed, dtype=np.float64))

    # Give each soma the same additional integrated brightness while keeping
    # the waveform maps unchanged. Bound template weights at saturated pixels.
    template = maps.copy()
    limit = np.quantile(template.data, .999)
    np.minimum(template.data, limit, out=template.data)
    per_soma = np.asarray(template.sum(axis=1)).ravel()
    active = per_soma > 0
    extra = np.zeros(len(per_soma), dtype=np.float64)
    extra[active] = total / (np.count_nonzero(active) * per_soma[active])
    added = np.asarray(template.T @ extra).reshape(base.shape)
    adjusted = (base - removed + added).astype(np.float32)
    np.testing.assert_allclose(np.mean(adjusted, dtype=np.float64),
                               np.mean(base, dtype=np.float64), atol=1e-5)
    stats = dict(reduction_fraction=fraction, diffuse_threshold_adu=threshold,
                 extreme_brightness_threshold_adu=ceiling,
                 removed_sum_adu=total, returned_to_somas_sum_adu=float(added.sum()),
                 mean_frame_adu=[float(x.mean(dtype=np.float64)) for x in (base, adjusted)],
                 diffuse_p95_adu=np.quantile(np.stack((diffuse, diffuse-removed)), .95, axis=(1, 2)).tolist(),
                 extreme_pixel_count=[int(np.count_nonzero(x > ceiling)) for x in (base, adjusted)],
                 static_pixels_above_display_white=[int(np.count_nonzero(x >= display_white))
                                                    for x in (base, adjusted)])
    if return_layers:
        return adjusted, stats, removed, added
    return adjusted, stats
