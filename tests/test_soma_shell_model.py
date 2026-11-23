"""Parametric morphology and photometric fits preserve annotated geometry."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from soma_shell_model import sample_morphology,shell_patch,radial_profiles,normalize_profiles,sample_angular_coefficients,angular_log_modulation


class ShellTests(unittest.TestCase):
    def test_parametric_marginals_reproducible_and_positive(self):
        report=dict(diameter_fit_comparison=[dict(name='normal',parameters=[8.13,1.43])],
            aspect_ratio_excess_fit_comparison=[dict(name='gamma',parameters=[2.337,0.,.0984])],
            orientation=dict(axial_mean_deg=3.76,vonmises_doubled_kappa=.765))
        a=sample_morphology(report,50000,123);b=sample_morphology(report,50000,123)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertTrue(np.all(a['minor_axis_px']>0));self.assertTrue(np.all(a['aspect_ratio']>=1))
        self.assertAlmostEqual(a['diameter_px'].mean(),8.13,delta=.03)
        self.assertAlmostEqual(a['aspect_ratio'].mean(),1+2.337*.0984,delta=.005)
        self.assertGreater(len(np.unique(a['diameter_px'])),49000)

    def test_shell_width_and_core_fill_do_not_change_mask(self):
        masks=[]
        for width,fill,blur in [(.04,0.,.6),(.28,.2,1.)]:
            origin,kernel,mask=shell_patch([40.2,40.3],10.,8.,37.,True,fill,blur,peak_radius=1.,width=width)
            full=np.zeros((100,100),bool);full[origin[0]:origin[0]+mask.shape[0],origin[1]:origin[1]+mask.shape[1]]=mask
            masks.append(full);self.assertTrue(np.isfinite(kernel).all());self.assertGreaterEqual(kernel.min(),0)
        np.testing.assert_array_equal(*masks)

    def test_angular_modulation_preserves_geometry_flux_and_zero_control(self):
        coefficients=sample_angular_coefficients(3,123)
        np.testing.assert_array_equal(coefficients,sample_angular_coefficients(3,123))
        for fill in [0.,.2]:
            args=([35.2,36.4],11.,7.,28.,True,fill,.6)
            origin,base,mask=shell_patch(*args,width=.36)
            o,zero,m=shell_patch(*args,width=.36,angular_coefficients=np.zeros((4,2)))
            np.testing.assert_array_equal(base,zero);np.testing.assert_array_equal(origin,o);np.testing.assert_array_equal(mask,m)
            for coeff in coefficients:
                o,modulated,m=shell_patch(*args,width=.36,angular_coefficients=coeff*.75)
                np.testing.assert_array_equal(origin,o);np.testing.assert_array_equal(mask,m)
                self.assertTrue(np.isfinite(modulated).all());self.assertGreaterEqual(modulated.min(),0)
                self.assertAlmostEqual(float(modulated.sum()),float(base.sum()),delta=float(base.sum())*1e-6)
                self.assertGreater(float(np.max(abs(modulated-base))),.05)

    def test_angular_log_field_is_periodic_and_smooth(self):
        coefficients=sample_angular_coefficients(1,19)[0]
        phases=np.array([0.,1e-5,2*np.pi-1e-5,2*np.pi])
        value=angular_log_modulation(phases,coefficients)
        self.assertAlmostEqual(value[0],value[-1],places=10)
        self.assertLess(abs(value[1]-value[-2]),1e-3)
        with self.assertRaises(ValueError):angular_log_modulation(phases,np.zeros((4,3)))

    def test_radial_peak_at_annotation_contour_and_pixel_edge_convention(self):
        center=np.array([50.2,50.3]);major,minor=20.,16.
        origin,kernel,_=shell_patch(center,major,minor,27.,True,0.,.0,peak_radius=1.,width=.12)
        image=np.zeros((110,110));image[origin[0]:origin[0]+kernel.shape[0],origin[1]:origin[1]+kernel.shape[1]]=kernel
        r,p=radial_profiles(image,[center],[major],[minor],[27.],n_angles=128)
        self.assertAlmostEqual(r[np.argmax(p[0])],1.,delta=.05)
        r2,p2=radial_profiles(image,[center+.5],[major],[minor],[27.],n_angles=128,coordinate_offset=.5)
        np.testing.assert_allclose(p,p2)
        normal,core,contrast=normalize_profiles(r,p)
        self.assertGreater(contrast[0],.8);self.assertLess(core[0],.01)


if __name__=='__main__':unittest.main()
