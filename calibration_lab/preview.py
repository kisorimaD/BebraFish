"""Browser previews of a real reference and its first synthetic scene."""
import numpy as np
from PIL import Image

PREVIEW_NAMES = ('scene.png', 'crops.png', 'scene-p5-p95.png', 'crops-p5-p95.png')


def display(image, limits):
    lo, hi = limits
    scaled = (image.astype(np.float32) - lo) * (255 / max(hi-lo, 1))
    return Image.fromarray(np.rint(np.clip(scaled, 0, 255)).astype(np.uint8))


def crop_windows(tissue, reviewed):
    """Prefer reviewed regions and fill out the preview from across the field."""
    height, width = tissue.shape
    size = min(64, height, width)
    windows = []

    def add(y, x):
        y, x = max(0, min(y, height-size)), max(0, min(x, width-size))
        if all(abs(y-py) >= size/2 or abs(x-px) >= size/2 for py, px in windows):
            windows.append((y, x))

    for crop in reviewed:
        add(crop['y']+(crop['height']-size)//2, crop['x']+(crop['width']-size)//2)
        if len(windows) == 4:
            break

    if len(windows) < 4:
        ys = sorted({*range(0, height-size+1, size), height-size})
        xs = sorted({*range(0, width-size+1, size), width-size})
        candidates = [(y, x) for y in ys for x in xs]
        candidates.sort(key=lambda cell: -tissue[cell[0]:cell[0]+size, cell[1]:cell[1]+size].mean())
        for y, x in candidates:
            add(y, x)
            if len(windows) == 4:
                break
    return [np.s_[y:y+size, x:x+size] for y, x in windows]


def comparison_previews(output, reference, synthetic, tissue, reviewed):
    common = np.percentile(reference, [1, 99.5])
    windows = crop_windows(tissue, reviewed)
    for p5_p95 in (False, True):
        def render(image):
            return display(image, np.percentile(image, [5, 95]) if p5_p95 else common)

        suffix = '-p5-p95' if p5_p95 else ''
        # Match the horizontal orientation of the annotation editors.
        height, width = reference.T.shape
        full = Image.new('L', (2*width+8, height), 48)
        for column, image in enumerate((reference, synthetic)):
            full.paste(render(image.T), (column*(width+8), 0))
        full.save(output/f'scene{suffix}.png')

        size = reference[windows[0]].shape[0]
        gap = 5
        samples = Image.new('L', (2*size+gap, len(windows)*(size+gap)-gap), 48)
        for row, window in enumerate(windows):
            for column, image in enumerate((reference, synthetic)):
                samples.paste(render(image[window].T), (column*(size+gap), row*(size+gap)))
        samples.save(output/f'crops{suffix}.png')
