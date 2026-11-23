"""Metrics distinguish affine amplitude changes from loss of fine structure."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fit_contrast_sharpness import crop_features,contrast_signal,threshold_test,auc


class ContrastSharpnessTests(unittest.TestCase):
    def test_normalized_shape_metrics_are_invariant_to_affine_brightness(self):
        y,x=np.indices((96,96));image=100+20*np.cos(2*np.pi*x/8)+10*np.cos(2*np.pi*y/4)
        f=crop_features(image,image,image,[(0,0)],size=96)
        changed=image*2.7+80;g=crop_features(changed,changed,changed,[(0,0)],size=96)
        np.testing.assert_allclose(g[0,0],f[0,0]*2.7,rtol=1e-10)
        np.testing.assert_allclose(g[0,2:],f[0,2:],rtol=1e-10,atol=1e-10)

    def test_blur_reduces_amplitude_normalized_fine_structure(self):
        y,x=np.indices((96,96));image=100+20*np.cos(2*np.pi*x/8)+20*np.cos(2*np.pi*y/4)
        before=crop_features(image,image,image,[(0,0)],size=96)[0]
        blurred=gaussian_filter(image,1.5)
        after=crop_features(blurred,blurred,blurred,[(0,0)],size=96)[0]
        self.assertLess(after[2],before[2]);self.assertLess(after[3],before[3]*.2)

    def test_contrast_correction_preserves_energy_positivity_and_empty_mask(self):
        rng=np.random.default_rng(42);signal=rng.uniform(0,100,(64,64)).astype(np.float32);signal[:,:8]=0
        np.testing.assert_array_equal(contrast_signal(signal,1),signal)
        changed=contrast_signal(signal,1.8)
        self.assertTrue((changed>=0).all());self.assertFalse(changed[:,:8].any())
        self.assertAlmostEqual(changed.sum(dtype=float)/signal.sum(dtype=float),1,places=6)
        self.assertGreater(changed.std(),signal.std())

    def test_discriminator_threshold_is_fitted_on_train_and_auc_handles_ties(self):
        real=np.array([5.,6.,7.,8.]);synthetic=np.array([1.,2.,3.,4.]);train=np.array([True,False,True,False])
        result=threshold_test(real,synthetic,train)
        self.assertEqual(result['test_balanced_accuracy'],1);self.assertEqual(result['test_auc'],1)
        self.assertEqual(auc(np.ones(5),np.ones(5)),.5)


if __name__=='__main__':unittest.main()
