"""Empirical geometry must survive ring/solid choice and arbitrary rotation."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import ellipse_patch, build_empirical_scene


class EmpiricalMorphologyTests(unittest.TestCase):
    def test_mask_independent_of_fluorescence_profile(self):
        masks=[]
        for ring,fill,blur in [(False,0,.6),(True,0,.6),(True,.5,1.)]:
            origin,_,mask=ellipse_patch([40.3,40.2],10.,6.,31.,ring,fill,blur)
            full=np.zeros((80,80),bool)
            full[origin[0]:origin[0]+mask.shape[0],origin[1]:origin[1]+mask.shape[1]]=mask
            masks.append(full)
        np.testing.assert_array_equal(masks[0],masks[1])
        np.testing.assert_array_equal(masks[0],masks[2])

    def test_arbitrary_angle_and_axis_lengths(self):
        origin,_,mask=ellipse_patch([60.3,60.2],30,12,23.,True,.25,.6)
        yy,xx=np.nonzero(mask)
        values,vectors=np.linalg.eigh(np.cov(np.stack([xx,yy]),ddof=0)+np.eye(2)/12)
        axes=4*np.sqrt(values[::-1])
        angle=np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))%180
        self.assertLess(abs(axes[0]-30),.3)
        self.assertLess(abs(axes[1]-12),.3)
        self.assertLess(abs(angle-23),1)

    def test_uniform_empirical_geometry_reproducible_and_full_overlaps_retained(self):
        shape=(80,80)
        profile=dict(tissue=np.ones(shape,bool),tissue_field=np.ones(shape),cell_gain=np.ones(shape))
        bank=dict(major_axis_px=np.array([9.,12.]),minor_axis_px=np.array([7.,6.]),
                  orientation_deg=np.array([15.,61.]),numbers=np.array([1,2]))
        settings=dict(amplitude_cv=.2,ring_fraction=.5,ring_center_fill=.25,blur_px=.6)
        centers=np.array([[30.,30.],[33.,32.],[50.,50.]])
        a=build_empirical_scene(profile,settings,centers,bank,123)
        b=build_empirical_scene(profile,settings,centers,bank,123)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(a['major_axis_px'],bank['major_axis_px'][a['empirical_indices']])
        union=np.zeros(shape[0]*shape[1],bool);union[a['instance_pixel_index']]=True
        np.testing.assert_array_equal(union.reshape(shape),a['labels']>0)
        self.assertGreater(len(a['instance_pixel_index']),union.sum())


if __name__=='__main__':unittest.main()
