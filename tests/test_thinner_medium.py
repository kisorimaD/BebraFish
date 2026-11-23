"""Thinning changes radial width, preserving intrinsic power and mid-shell GT."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from try_thinner_medium import thinner_shell_patch,OLD_WIDTH,NEW_WIDTH
from soma_shell_model import shell_patch,sample_angular_coefficients


class ThinnerMediumTests(unittest.TestCase):
    def test_flux_and_full_mask_preserved_with_medium_angular_modulation(self):
        coeff=sample_angular_coefficients(1,123)[0]*.5
        args=([50.2,50.4],11.,8.,31.,True,0.,.6)
        a,k,m=shell_patch(*args,width=OLD_WIDTH,angular_coefficients=coeff)
        b,q,n=thinner_shell_patch(*args,angular_coefficients=coeff)
        self.assertAlmostEqual(q.sum(dtype=float)/k.sum(dtype=float),1.,places=6)
        self.assertTrue(np.isfinite(q).all());self.assertGreaterEqual(q.min(),0)
        masks=[]
        for origin,mask in [(a,m),(b,n)]:
            canvas=np.zeros((120,120),bool);canvas[origin[0]:origin[0]+mask.shape[0],origin[1]:origin[1]+mask.shape[1]]=mask;masks.append(canvas)
        np.testing.assert_array_equal(*masks)

    def test_unblurred_radial_half_maximum_width_reduced_fifteen_percent(self):
        widths=[]
        for renderer in [lambda *a:shell_patch(*a,width=OLD_WIDTH),thinner_shell_patch]:
            origin,kernel,_=renderer([80.,80.],40.,30.,0.,True,0.,0.)
            profile=kernel[int(80-origin[0]),int(80-origin[1]):]
            peak=int(np.argmax(profile));half=profile[peak]/2
            left=np.interp(half,profile[:peak+1],np.arange(peak+1))
            tail=profile[peak:];right=np.interp(half,tail[::-1],np.arange(peak,len(profile))[::-1])
            widths.append(right-left)
        self.assertAlmostEqual(NEW_WIDTH/OLD_WIDTH,.85)
        self.assertAlmostEqual(widths[1]/widths[0],.85,delta=.01)


if __name__=='__main__':unittest.main()
