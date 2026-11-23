"""Axis conventions and noise correlation for the exploratory realism trial."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from try_heterogeneous_scene import correlated_noise,effective_blur
from diagnose_scene_realism import directional_curves,correlation_length,neighbor_spread


class RealismTests(unittest.TestCase):
    def test_vertical_blur_is_native_x_and_preserves_flux(self):
        signal=np.zeros((129,129));signal[64,64]=100
        result=effective_blur(signal,.4,1.8,np.full(signal.shape,.5))
        y,x=np.indices(signal.shape)
        vx=np.sum(result*(x-64)**2)/result.sum();vy=np.sum(result*(y-64)**2)/result.sum()
        self.assertGreater(vx,vy*4)
        self.assertAlmostEqual(result.sum(),100,places=7)
        self.assertTrue((result>=0).all())
        curves=directional_curves(result,np.ones(signal.shape,bool))
        self.assertGreater(correlation_length(curves[1]),correlation_length(curves[0]))

    def test_correlated_noise_unit_variance_and_requested_correlation(self):
        noise,fraction=correlated_noise((768,768),123)
        self.assertGreater(fraction,0);self.assertLess(fraction,1)
        self.assertAlmostEqual(noise.std(),1,delta=.01)
        for a,b in [(noise[:-1],noise[1:]),(noise[:,:-1],noise[:,1:])]:
            self.assertAlmostEqual(np.corrcoef(a.ravel(),b.ravel())[0,1],.265,delta=.015)
        same,_=correlated_noise(noise.shape,123)
        np.testing.assert_array_equal(noise,same)

    def test_neighbor_variance_keeps_signed_contrasts_without_log_clamping(self):
        values=np.array([-3.,10.,20.]);centers=np.array([[0,0],[0,1],[0,2]])
        result=neighbor_spread(values,centers)
        self.assertEqual(result['pairs'],3);self.assertEqual(result['positive_log_pairs'],1)
        self.assertAlmostEqual(result['raw_difference_variance_adu2'],np.var([-13.,-23.,-10.]))


if __name__=='__main__':unittest.main()
