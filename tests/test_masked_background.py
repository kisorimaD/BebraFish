"""Background extension uses only measured pixels and preserves simple fields."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from generate_masked_background import interpolate_background


class BackgroundTests(unittest.TestCase):
    def test_unknown_intensities_do_not_affect_constant_extension(self):
        mask=np.zeros((45,39),bool);mask[:,:7]=True;mask[:,-7:]=True
        image=np.full(mask.shape,123.)
        result,_=interpolate_background(image,mask,sigma=2,step=3)
        np.testing.assert_allclose(result,123,atol=1e-4)
        image[~mask]=999999
        changed,_=interpolate_background(image,mask,sigma=2,step=3)
        np.testing.assert_array_equal(result,changed)

    def test_gradient_is_bounded_monotonic_and_finite(self):
        mask=np.zeros((45,39),bool);mask[:,:7]=True;mask[:,-7:]=True
        image=np.broadcast_to(np.linspace(100,180,39),mask.shape).copy()
        result,_=interpolate_background(image,mask,sigma=2,step=3)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(),100)
        self.assertLessEqual(result.max(),180)
        self.assertTrue((np.diff(result[22])>=-.001).all())
        with self.assertRaises(ValueError):interpolate_background(image,np.zeros_like(mask))


if __name__=='__main__':unittest.main()
