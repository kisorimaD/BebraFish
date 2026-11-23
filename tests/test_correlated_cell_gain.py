"""Correlated gain is per-cell, preserves contours and has a matched control."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from try_correlated_cell_gain import smooth_gain_field,cell_amplitudes,fit_variogram
from empirical_morphology import build_empirical_scene


class CorrelatedGainTests(unittest.TestCase):
    def test_field_is_smooth_and_reproducible(self):
        field=smooth_gain_field((512,512),16,123)
        np.testing.assert_array_equal(field,smooth_gain_field(field.shape,16,123))
        self.assertAlmostEqual(field.mean(),0,places=10);self.assertAlmostEqual(field.std(),1,places=10)
        nearby=np.corrcoef(field[:,:-8].ravel(),field[:,8:].ravel())[0,1]
        distant=np.corrcoef(field[:,:-80].ravel(),field[:,80:].ravel())[0,1]
        self.assertGreater(nearby,.7);self.assertLess(abs(distant),.3)
        centers=np.array([[100,100],[100,101],[200,200]],float)
        a=cell_amplitudes(field,centers,.5,0,1)
        self.assertTrue((a>0).all());self.assertAlmostEqual(a.mean(),1)
        self.assertLess(abs(np.log(a[0]/a[1])),.1)
        shuffled=a[np.array([2,0,1])];np.testing.assert_array_equal(np.sort(a),np.sort(shuffled))

    def test_explicit_amplitudes_scale_whole_emission_without_changing_contours(self):
        shape=(32,32);ones=np.ones(shape,np.float32)
        profile=dict(tissue=ones.astype(bool),tissue_field=ones,cell_gain=ones)
        settings=dict(amplitude_cv=.2,ring_fraction=1.,ring_center_fill=0.,blur_px=.6)
        centers=np.array([[10.,10.],[22.,22.]])
        geometry=dict(major_axis_px=np.array([7.,7.]),minor_axis_px=np.array([6.,6.]),orientation_deg=np.zeros(2))
        kwargs=dict(profile=profile,settings=settings,centers=centers,bank=None,seed=2,geometry=geometry)
        a=build_empirical_scene(**kwargs,amplitudes=np.ones(2));b=build_empirical_scene(**kwargs,amplitudes=np.full(2,3.))
        np.testing.assert_allclose(b['signal'],a['signal']*3,rtol=2e-7,atol=1e-7)
        np.testing.assert_array_equal(a['instance_offsets'],b['instance_offsets']);np.testing.assert_array_equal(a['instance_pixel_index'],b['instance_pixel_index'])
        for bad in [[1],[-1,1],[float('nan'),1]]:
            with self.assertRaises(ValueError):build_empirical_scene(**kwargs,amplitudes=bad)

    def test_variogram_fit_recovers_known_model(self):
        d=np.array([5.,10.,15.,22.,30.,42.]);sigma=.5;jitter=.12;length=18
        gamma=jitter**2+sigma**2*(1-np.exp(-d*d/(2*length**2)))
        rows=[dict(distance=float(x),gamma=float(v),count=20,hi=float(x+2)) for x,v in zip(d,gamma)]
        result=fit_variogram(rows)
        self.assertAlmostEqual(result['correlation_length_px'],length,places=3)
        self.assertAlmostEqual(result['correlated_log_sigma'],sigma,places=4)
        self.assertAlmostEqual(result['independent_log_sigma'],jitter,places=4)


if __name__=='__main__':unittest.main()
