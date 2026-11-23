"""Existing mid-shell contours sample centered image bands without axis errors."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inspect_shell_angles import perimeter_strip


class ShellAngleTests(unittest.TestCase):
    def test_constant_band_and_linear_image_have_expected_values(self):
        angle=np.linspace(0,2*np.pi,64,endpoint=False)
        polygon=np.column_stack([24.5+8*np.cos(angle),24.5+6*np.sin(angle)])
        constant=np.full((50,50),137.)
        contour,profile,strip=perimeter_strip(constant,polygon)
        np.testing.assert_allclose(profile,137,atol=1e-10)
        np.testing.assert_allclose(strip,137,atol=1e-10)
        y,x=np.indices(constant.shape);image=100+2*x+3*y
        contour,profile,strip=perimeter_strip(image,polygon)
        expected=100+2*(contour[:,0]-.5)+3*(contour[:,1]-.5)
        np.testing.assert_allclose(profile,expected,atol=1e-10)
        self.assertEqual(profile.shape,(128,));self.assertEqual(strip.shape,(128,13))


if __name__=='__main__':unittest.main()
