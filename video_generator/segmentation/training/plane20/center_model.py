"""U-Net features -> center heatmap/offsets -> independent center-conditioned masks."""
import tensorflow as tf
from .center_data import ROI_SIZE


def gather_rois(tensor, points, batch_indices, size=ROI_SIZE):
    """Exact native-pixel crops. Integer anchors need no bilinear RoI resampling."""
    anchor = tf.cast(tf.floor(points+.5), tf.int32)
    grid = tf.range(size)-size//2
    gy, gx = tf.meshgrid(grid, grid, indexing='ij')
    xy = anchor[:, None, None, :]+tf.stack((gy, gx), axis=-1)[None]
    shape = tf.shape(tensor)
    valid = tf.reduce_all((xy >= 0) & (xy < shape[1:3]), axis=-1)
    xy = tf.maximum(0, tf.minimum(xy, shape[1:3]-1))
    bi = tf.broadcast_to(batch_indices[:, None, None, None], tf.concat((tf.shape(xy)[:-1], [1]), axis=0))
    values = tf.gather_nd(tensor, tf.concat((bi, xy), axis=-1))
    return values*tf.cast(valid[..., None], values.dtype)


class CenterMask(tf.keras.Model):
    def __init__(self, base_filters=32):
        super().__init__(name='center_mask')
        from csbdeep.internals.blocks import unet_block
        image = tf.keras.Input((128, 128, 1))
        features = unet_block(3, base_filters, (3, 3), n_conv_per_depth=2, pool=(2, 2))(image)
        center = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(features)
        center = tf.keras.layers.Conv2D(1, 1, bias_initializer=tf.keras.initializers.Constant(-2.944439),
                                       name='center_logits')(center)
        offset = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(features)
        offset = tf.keras.layers.Conv2D(2, 1, name='center_offsets')(offset)
        self.encoder = tf.keras.Model(image, (features, center, offset), name='unet_detector')
        self.mask_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 1, activation='relu'),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, 1)], name='conditional_mask')
        self.trackers = {k: tf.keras.metrics.Mean(name=k) for k in
            ('loss', 'center_loss', 'offset_loss', 'mask_bce', 'mask_dice_loss', 'center_gt_score')}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def decode_masks(self, features, images, points, batch_indices, training=False):
        anchor = tf.floor(points+.5)
        roi = gather_rois(tf.concat((features, images), axis=-1), points, batch_indices)
        grid = tf.cast(tf.range(ROI_SIZE), tf.float32)-ROI_SIZE//2
        gy, gx = tf.meshgrid(grid, grid, indexing='ij')
        relative = tf.stack((gy, gx), axis=-1)[None]+(anchor-points)[:, None, None, :]
        conditioned = tf.concat((roi, relative/(ROI_SIZE/2)), axis=-1)
        return self.mask_head(conditioned, training=training)

    def call(self, data, training=False):
        features, heat, offsets = self.encoder(data['images'], training=training)
        masks = self.decode_masks(features, data['images'], data['points'], data['batch_indices'], training)
        return heat, offsets, masks

    def losses_for(self, data, predictions):
        heat_logits, offsets, mask_logits = predictions
        probability = tf.clip_by_value(tf.sigmoid(heat_logits), 1e-6, 1-1e-6)
        target, valid = data['heatmap'], data['valid_region']
        positive = tf.cast(target == 1, tf.float32)*valid
        negative = tf.cast(target < 1, tf.float32)*valid
        center_loss = -tf.reduce_sum(positive*(1-probability)**2*tf.math.log(probability)
            + negative*(1-target)**4*probability**2*tf.math.log(1-probability))/tf.maximum(tf.reduce_sum(positive), 1.)
        gathered = tf.gather_nd(offsets, data['offset_indices'])
        offset_loss = tf.reduce_sum(tf.abs(gathered-data['offset_targets']))/tf.maximum(tf.cast(tf.size(gathered), tf.float32), 1.)
        y, p = data['masks'], tf.sigmoid(mask_logits)
        bce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, mask_logits, pos_weight=4.), axis=(1, 2, 3))
        dice = 1-(2*tf.reduce_sum(p*y, axis=(1, 2, 3))+1e-5)/(tf.reduce_sum(p+y, axis=(1, 2, 3))+1e-5)
        weight = data['mask_valid'];denom = tf.maximum(tf.reduce_sum(weight), 1.)
        mask_bce, mask_dice = tf.reduce_sum(weight*bce)/denom, tf.reduce_sum(weight*dice)/denom
        score = tf.reduce_sum(probability*positive)/tf.maximum(tf.reduce_sum(positive), 1.)
        return dict(loss=center_loss+offset_loss+mask_bce+mask_dice,
                    center_loss=center_loss, offset_loss=offset_loss,
                    mask_bce=mask_bce, mask_dice_loss=mask_dice, center_gt_score=score)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            losses = self.losses_for(data, self(data, training=True))
        gradients = tape.gradient(losses['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for k, v in losses.items():
            self.trackers[k].update_state(v)
        return {k: v.result() for k, v in self.trackers.items()}

    def test_step(self, data):
        losses = self.losses_for(data, self(data, training=False))
        for k, v in losses.items():
            self.trackers[k].update_state(v)
        return {k: v.result() for k, v in self.trackers.items()}


def build_model(base_filters=32):
    model = CenterMask(base_filters)
    model(dict(images=tf.zeros((1, 128, 128, 1)), points=tf.constant([[64., 64.]]), batch_indices=tf.constant([0])))
    return model
