"""Train a residual U-Net on paired Plane20 scenes and evaluate the three pipelines."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from csbdeep.internals.blocks import unet_block

from .center_inference import evaluate
from .data import CropSource


def input_scale(image):
    low, high = np.percentile(image, [5, 95])
    return np.float32(low), np.float32(max(high - low, 1.0))


def restore_batch(model, images):
    """Denoise a batch of raw YX crops and return restored intensities in ADU."""
    low, high = np.percentile(images, [5, 95], axis=(1, 2), keepdims=True)
    scale = np.maximum(high-low, 1.0).astype(np.float32)
    low = low.astype(np.float32)
    inputs = (images.astype(np.float32)-low)/scale
    return model(inputs[..., None], training=False).numpy()[..., 0]*scale+low


class PairedBatches(tf.keras.utils.Sequence):
    def __init__(self, directory, split, seed, steps, batch_size):
        self.source = CropSource(directory / 'normal_noisy', split, seed,
                                 use_precomputed=False, keep_scenes=True)
        self.clean_dir = directory / 'normal_clean' / split
        if {p.name for p in self.source.paths} != {p.name for p in self.clean_dir.glob('*.npz')}:
            raise ValueError(f'Noisy and clean scene names differ in {split}')
        self.clean_images = {}
        self.steps, self.batch_size = steps, batch_size
        self.training = split == 'train'
        self.epoch = 0

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        start = (self.epoch * self.steps + index) * self.batch_size
        inputs = np.empty((self.batch_size, 128, 128, 1), np.float32)
        targets = np.empty_like(inputs)
        for batch_index in range(self.batch_size):
            noisy, scene, (y, x), path = self.source.raw_crop(start + batch_index)
            if path.name not in self.clean_images:
                with np.load(self.clean_dir / path.name) as clean:
                    image = clean['image']
                    if (image.shape != scene['image'].shape
                            or not np.array_equal(clean['centers'], scene['centers'])):
                        raise ValueError(f'Noisy and clean geometry differ for {path}')
                    self.clean_images[path.name] = image
            clean_patch = self.clean_images[path.name][y:y+128, x:x+128]
            low, scale = input_scale(noisy)
            inputs[batch_index, ..., 0] = (noisy.astype(np.float32) - low) / scale
            targets[batch_index, ..., 0] = (clean_patch.astype(np.float32) - low) / scale
        return inputs, targets

    def on_epoch_end(self):
        if self.training:
            self.epoch += 1


def build_model():
    image = tf.keras.Input((128, 128, 1))
    features = unet_block(2, 16, (3, 3), n_conv_per_depth=2, pool=(2, 2))(image)
    correction = tf.keras.layers.Conv2D(1, 1, kernel_initializer='zeros')(features)
    return tf.keras.Model(image, tf.keras.layers.Add()([image, correction]), name='plane20_denoiser')


def compare(args, model):
    def restore(image):
        return restore_batch(model, image[None])[0]

    results = {}
    for name, detector, restoration in (
        ('noisy_to_clean_detector', 'normal_clean', None),
        ('denoised_to_clean_detector', 'normal_clean', restore),
        ('noisy_to_noisy_detector', 'normal_noisy', None),
    ):
        evaluation = argparse.Namespace(
            run=args.detectors / detector, data=args.data / 'normal_noisy',
            output=args.output / f'{name}.json', split='test',
            seed=991, start_index=32, crops=args.eval_crops, threshold=.3, mask_threshold=.5,
        )
        report = evaluate(evaluation, restore=restoration)
        results[name] = {key: report[key] for key in
                         ('center_detection', 'predicted_centers', 'oracle_centers')}
    (args.output / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--detectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--val-steps', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-crops', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.val_steps, args.batch_size, args.eval_crops) < 1:
        parser.error('Training and evaluation counts must be positive')
    for variant in ('normal_clean', 'normal_noisy'):
        for filename in ('model_config.json', 'weights_best.h5'):
            path = args.detectors / variant / filename
            if not path.is_file():
                parser.error(f'Missing detector file: {path}')
    if not list((args.data / 'normal_noisy' / 'test').glob('*.npz')):
        parser.error('Missing noisy test scenes')
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != 1:
        parser.error('Select one TensorFlow GPU with CUDA_VISIBLE_DEVICES')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(args.seed)

    train = PairedBatches(args.data, 'train', args.seed, args.steps, args.batch_size)
    val = PairedBatches(args.data, 'val', args.seed + 1, args.val_steps, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | dict(
        input='128x128, noisy crop P5-P95 affine scale, minimum range 1 ADU, unclipped',
        target='Paired clean crop on the same noisy intensity scale',
        loss='MAE', optimizer='Adam with learning rate 0.0003',
    )
    (args.output / 'config.json').write_text(json.dumps(config, default=str, indent=2) + '\n')
    checkpoint = args.output / 'weights_best.h5'
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='mae')
    model.fit(train, validation_data=val, epochs=args.epochs, shuffle=False, workers=0, verbose=2,
              callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                         tf.keras.callbacks.ModelCheckpoint(str(checkpoint), save_best_only=True,
                                                            save_weights_only=True),
                         tf.keras.callbacks.CSVLogger(str(args.output / 'history.csv'))])
    model.load_weights(str(checkpoint))
    compare(args, model)


if __name__ == '__main__':
    main()
