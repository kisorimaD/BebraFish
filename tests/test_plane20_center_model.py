"""TensorFlow-only native pixel alignment and overlapping ROI gradient checks."""
import unittest
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None


@unittest.skipIf(tf is None, 'TensorFlow is installed in the server training environment')
class ROITests(unittest.TestCase):
    def test_integer_roi_alignment_and_subpixel_anchor(self):
        from training.plane20.center_model import gather_rois
        with tf.device('/CPU:0'):
            image = np.arange(2*16*16).reshape(2, 16, 16, 1).astype(np.float32)
            result = gather_rois(tf.constant(image), tf.constant([[8.2, 7.8], [5.1, 6.4]]), tf.constant([0, 1]), size=4).numpy()
            np.testing.assert_array_equal(result[0], image[0, 6:10, 6:10])
            np.testing.assert_array_equal(result[1], image[1, 3:7, 4:8])

    def test_overlapping_rois_add_gradients_and_outside_is_zero(self):
        from training.plane20.center_model import gather_rois
        with tf.device('/CPU:0'):
            image = tf.Variable(tf.ones((1, 8, 8, 1)))
            with tf.GradientTape() as tape:
                values = gather_rois(image, tf.constant([[4., 4.], [4., 5.]]), tf.constant([0, 0]), size=4)
                loss = tf.reduce_sum(values)
            gradient = tape.gradient(loss, image).numpy()[0, ..., 0]
            expected = np.zeros((8, 8));expected[2:6, 2:6] += 1;expected[2:6, 3:7] += 1
            np.testing.assert_array_equal(gradient, expected)
            outside = gather_rois(image, tf.constant([[0., 0.]]), tf.constant([0]), size=4).numpy()
            self.assertEqual(float(outside.sum()), 4.)


if __name__ == '__main__':
    unittest.main()
