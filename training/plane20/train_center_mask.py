"""Train independent center-conditioned masks; no single-owner pixel targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '8')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import numpy as np
import tensorflow as tf
from .center_data import InstanceSource, batch_samples, ROI_SIZE
from .center_model import build_model
from .data import normalize


class InstanceBatches(tf.keras.utils.Sequence):
    def __init__(self, source, steps, batch_size, epoch=0, fixed=False, repeat_one=False,
                 restore=None):
        self.source, self.steps, self.batch_size = source, steps, batch_size
        self.epoch, self.fixed, self.repeat_one = epoch, fixed, repeat_one
        self.cached = None
        self.restore = restore

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        if self.repeat_one and self.cached is not None:
            return self.cached
        start = 0 if self.repeat_one else ((0 if self.fixed else self.epoch)*self.steps+index)*self.batch_size
        value = batch_samples([self.source.sample(start+i) for i in range(self.batch_size)],
                              mask_slots=self.batch_size*self.source.max_masks)
        if self.restore is not None:
            restored = self.restore(value['images'][..., 0])
            value['images'] = np.stack([normalize(image) for image in restored])[..., None]
        if self.repeat_one:
            self.cached = value
        return value

    def on_epoch_end(self):
        self.epoch += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--val-steps', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--base-filters', type=int, choices=(32, 64), default=32)
    p.add_argument('--max-masks', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--gpu-memory-mb', type=int, default=0, help='Per-process GPU memory cap; 0 uses memory growth')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-weights', type=Path, help='Initialize the segmenter from an H5 checkpoint')
    p.add_argument('--denoiser-weights', type=Path, help='Apply this frozen denoiser to raw train and val crops')
    p.add_argument('--overfit-one-batch', action='store_true', help='Diagnostic only: repeat one cached training batch')
    args = p.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if min(args.steps, args.val_steps, args.batch_size, args.max_masks, args.epochs) < 1:
        p.error('counts must be positive')
    if not (args.data/'COMPLETE').is_file():
        p.error('Dataset is incomplete')
    if args.resume and args.init_weights is not None:
        p.error('--init-weights initializes a new run; use only --resume to restore segmenter weights')
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != 1:
        raise RuntimeError('Select exactly one A100 with CUDA_VISIBLE_DEVICES; independent ROIs use global batch indices')
    if args.gpu_memory_mb < 0:
        p.error('--gpu-memory-mb must be nonnegative')
    if args.gpu_memory_mb:
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_mb)])
    else:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    config = dict(schema='plane20-center-mask-v1', base_filters=args.base_filters, roi_size=ROI_SIZE,
                  input_size=128, halo=16, center_sigma=.7, mask_seed_jitter=.35, mask_seed_jitter_clip=.75,
                  losses='modified focal center + offset L1 + mask weighted BCE(pos_weight=4) + per-instance Dice',
                  dataset_manifest_sha256=hashlib.sha256((args.data/'manifest.jsonl').read_bytes()).hexdigest())
    restore = None
    if args.denoiser_weights is not None:
        from .train_denoiser import build_model as build_denoiser, restore_batch

        denoiser = build_denoiser()
        denoiser.load_weights(str(args.denoiser_weights))
        denoiser.trainable = False
        config['denoiser_weights_sha256'] = hashlib.sha256(args.denoiser_weights.read_bytes()).hexdigest()

        def restore(images):
            return restore_batch(denoiser, images)

    if args.resume and json.loads((args.output/'model_config.json').read_text()) != config:
        raise ValueError('Resume configuration or dataset differs from saved run')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'model_config.json').write_text(json.dumps(config, indent=2)+'\n')
    model = build_model(args.base_filters)
    if args.init_weights is not None:
        model.load_weights(str(args.init_weights))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=5.))
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, epoch=epoch_var, best=best_var)
    manager = tf.train.CheckpointManager(checkpoint, str(args.output/'state'), max_to_keep=2)
    if args.resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError('No optimizer checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    initial_epoch = int(epoch_var.numpy())
    train = InstanceBatches(InstanceSource(args.data, 'train', args.seed, args.max_masks, jitter=.35,
                                           raw_images=restore is not None),
                            args.steps, args.batch_size, initial_epoch, repeat_one=args.overfit_one_batch,
                            restore=restore)
    val = InstanceBatches(InstanceSource(args.data, 'val', args.seed+1, args.max_masks, jitter=0,
                                         raw_images=restore is not None),
                          args.val_steps, args.batch_size, fixed=True, restore=restore)
    metadata = vars(args) | dict(config=config, parameters=model.count_params(), tensorflow=tf.__version__,
        samples_per_epoch=args.steps*args.batch_size, validation_samples=args.val_steps*args.batch_size,
        validation_note='Mask losses use GT centers. End-to-end detection/segmentation must be evaluated separately.',
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (args.output/f'run_epoch_{initial_epoch:04d}.json').write_text(json.dumps(metadata, default=str, indent=2)+'\n')

    class Save(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            if not all(np.isfinite(v) for v in logs.values()):
                raise RuntimeError(f'Non-finite metrics: {logs}')
            if logs['val_loss'] < float(best_var.numpy()):
                best_var.assign(logs['val_loss']); self.model.save_weights(str(args.output/'weights_best.h5'))
            self.model.save_weights(str(args.output/'weights_last.h5'))
            epoch_var.assign(epoch+1); manager.save(checkpoint_number=epoch+1)
            with (args.output/'history.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch+1, time=time.time(), **logs))+'\n')

    print(json.dumps(metadata, default=str), flush=True)
    model.fit(train, validation_data=val, epochs=args.epochs, initial_epoch=initial_epoch,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=8, min_lr=1e-6), Save()],
        workers=1, use_multiprocessing=False, max_queue_size=2, verbose=2)


if __name__ == '__main__':
    main()
