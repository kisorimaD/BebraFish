"""Noise transfer follows synthetic illumination without losing the read floor."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from refine_medium_noise import adapted_noise_sigma


class MediumNoiseTests(unittest.TestCase):
    def test_equal_illumination_reproduces_observed_sigma_and_dimming_reduces_noise(self):
        real=np.full((24,24),180.);observed=np.full(real.shape,20.)
        same,_=adapted_noise_sigma(real,real,observed,100.,25.,.5,.01)
        np.testing.assert_allclose(same,20.,atol=1e-5)
        dim,_=adapted_noise_sigma(np.full_like(real,140.),real,observed,100.,25.,.5,.01)
        self.assertTrue((dim<same).all());self.assertTrue((dim>=5).all())
        dark,_=adapted_noise_sigma(np.full_like(real,100.),real,observed,100.,25.,.5,.01)
        np.testing.assert_array_equal(dark,5.)

    def test_excess_reduction_keeps_photon_and_read_components(self):
        real=np.full((24,24),180.);observed=np.full(real.shape,20.);clean=np.full_like(real,140.)
        full,alpha=adapted_noise_sigma(clean,real,observed,100.,25.,.5,.01,excess_scale=1.)
        reduced,_=adapted_noise_sigma(clean,real,observed,100.,25.,.5,.01,excess_scale=.25)
        no_excess,_=adapted_noise_sigma(clean,real,observed,100.,25.,.5,.01,excess_scale=0.)
        self.assertTrue((full>reduced).all());self.assertTrue((reduced>no_excess).all());self.assertTrue((no_excess>5).all())
        np.testing.assert_allclose(no_excess**2,25+alpha*.5*40,rtol=1e-6)
        with self.assertRaises(ValueError):adapted_noise_sigma(clean,real,observed,100.,25.,.5,.01,excess_scale=-1.)


if __name__=='__main__':unittest.main()
