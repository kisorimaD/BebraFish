"""Official StarDist U-Net, trained against complete synthetic ellipses."""
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
from stardist.models import Config2D, StarDist2D
from .data import CropSource


class Batches(tf.keras.utils.Sequence):
    def __init__(self, source, steps, batch_size, epoch=0, fixed=False):
        self.source, self.steps, self.batch_size = source, steps, batch_size
        self.epoch, self.fixed = epoch, fixed

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        start = ((0 if self.fixed else self.epoch)*self.steps+index)*self.batch_size
        rows = [self.source.sample(start+i) for i in range(self.batch_size)]
        return np.stack([r[0] for r in rows]), tuple(np.stack([r[1][k] for r in rows]) for k in (0, 1))

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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--base-filters', type=int, choices=(32, 64), default=None,
                   help='U-Net base width; new runs default to 32, resume uses saved config')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if not (args.data/'COMPLETE').exists():
        p.error('Dataset generation is incomplete (missing COMPLETE marker)')
    train_source = CropSource(args.data, 'train', args.seed, use_precomputed=True)
    val_source = CropSource(args.data, 'val', args.seed+1, use_precomputed=True)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('No TensorFlow CUDA GPU detected')
    if args.batch_size % len(gpus):
        p.error('--batch-size must be divisible by the number of visible GPUs')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    if not args.resume and args.output.exists():
        raise FileExistsError(f'{args.output} already exists; use a new run or --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config2D(n_rays=32, grid=(1, 1), n_channel_in=1,
        unet_n_depth=3, unet_n_filter_base=args.base_filters or 32, unet_n_conv_per_depth=2,
        train_patch_size=(128, 128), train_batch_size=args.batch_size,
        train_learning_rate=args.learning_rate, train_background_reg=0,
        train_loss_weights=(1., .2), train_tensorboard=False, use_gpu=False)
    # use_gpu=False disables only optional OpenCL target computation, not CUDA training.
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        model = StarDist2D(None if args.resume else config, name='model', basedir=str(args.output))
        if args.base_filters is not None and model.config.unet_n_filter_base != args.base_filters:
            raise ValueError(f'Requested width {args.base_filters} differs from saved model width '
                             f'{model.config.unet_n_filter_base}; use a new run directory')
        model.prepare_for_training()
        epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        best_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False)
        state = tf.train.Checkpoint(model=model.keras_model, optimizer=model.keras_model.optimizer,
                                    epoch=epoch_var, best=best_var)
        manager = tf.train.CheckpointManager(state, str(args.output/'state'), max_to_keep=2)
        if args.resume:
            if not manager.latest_checkpoint:
                raise FileNotFoundError('No resumable optimizer checkpoint')
            state.restore(manager.latest_checkpoint).expect_partial()
    start_epoch = int(epoch_var.numpy())
    train = Batches(train_source, args.steps, args.batch_size, start_epoch)
    val = Batches(val_source, args.val_steps, args.batch_size, fixed=True)
    metadata = vars(args) | dict(gpus=[g.name for g in gpus], tensorflow=tf.__version__,
        replicas=strategy.num_replicas_in_sync, normalization='per-input-crop P5-P95 clipped [0,1]',
        model_base_filters=model.config.unet_n_filter_base,
        model_parameters=model.keras_model.count_params(),
        targets='Full analytical ellipse rays; max normalized-depth pixel ownership; halo16 ignored; all generated cells supervised',
        dataset_config=json.loads((args.data/'config.json').read_text()),
        dataset_manifest_sha256=hashlib.sha256((args.data/'manifest.jsonl').read_bytes()).hexdigest(),
        target_cache=json.loads((args.data/'target_cache/config.json').read_text()),
        training_source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in Path(__file__).parent.glob('*.py')})
    (args.output/f'run_epoch_{start_epoch:04d}.json').write_text(json.dumps(metadata, default=str, indent=2)+'\n')

    class Save(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            if not all(np.isfinite(v) for v in logs.values()):
                raise RuntimeError(f'Non-finite training metrics: {logs}')
            if logs['val_loss'] < float(best_var.numpy()):
                best_var.assign(logs['val_loss'])
                self.model.save_weights(str(args.output/'model/weights_best.h5'))
            self.model.save_weights(str(args.output/'model/weights_last.h5'))
            epoch_var.assign(epoch+1)
            manager.save(checkpoint_number=epoch+1)
            with (args.output/'history.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch+1, time=time.time(), **logs))+'\n')

    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=8, min_lr=1e-6), Save()]
    print(json.dumps(metadata, default=str), flush=True)
    model.keras_model.fit(train, validation_data=val, epochs=args.epochs,
        initial_epoch=start_epoch, callbacks=callbacks, workers=1, use_multiprocessing=False,
        max_queue_size=2, verbose=2)


if __name__ == '__main__':
    main()
