"""Thin the accepted medium shell by 15%, with matched per-cell emitted flux.

Freezes background, 28px gain field, angular modulation, geometry, optical blur
and the exact accepted noise realization. Only intrinsic radial width changes.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from soma_shell_model import shell_patch,radial_profiles,normalize_profiles
from try_heterogeneous_scene import effective_blur,correlated_noise


OLD_WIDTH=.36
NEW_WIDTH=OLD_WIDTH*.85


def thinner_shell_patch(center,major,minor,angle,ring,fill,blur,*,angular_coefficients=None):
    args=(center,major,minor,angle,ring,fill,blur)
    options=dict(peak_radius=1.,angular_coefficients=angular_coefficients)
    _,reference,_=shell_patch(*args,width=OLD_WIDTH,**options)
    origin,kernel,mask=shell_patch(*args,width=NEW_WIDTH,**options)
    # Preserve photon budget rather than making cells dimmer when shrinking.
    kernel=(kernel.astype(float)*(reference.sum(dtype=float)/kernel.sum(dtype=float))).astype(np.float32)
    return origin,kernel,mask


def profile_summary(image,centers,geometry,indices):
    c=centers[indices];g=[geometry[k][indices] for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    radii,curves=radial_profiles(image,c,*g)
    norm,core,contrast=normalize_profiles(radii,curves)
    return dict(median_core_adu=float(np.median(core)),median_shell_minus_core_adu=float(np.median(contrast)),
        core=core,contrast=contrast,radii=radii,curves=curves)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_medium_thinner');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/plane20_medium_noise'
    with np.load(source/'synthetic.npz') as z:old={k:z[k] for k in z.files}
    source_report=json.loads((source/'report.json').read_text())
    with np.load(ROOT/'outputs/plane20_angular_shells/synthetic.npz') as z:normalizer=z['normalizer'].copy()
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:allowed=z['neuron_support'].copy();exclusion=z['exclusion_mask'].copy()
    with np.load(ROOT/'outputs/plane20_heterogeneous_trial_v2/synthetic.npz') as z:mixture=z['blur_mixture'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_realism_diagnostics_v2/measurements.npz') as z:context=z['synthetic_indices'].copy()
    settings=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())['selected']|dict(ring_fraction=1.,ring_center_fill=0.)
    geometry={k:old[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']}
    ones=np.ones(allowed.shape,np.float32);profile=dict(tissue=allowed,tissue_field=ones,cell_gain=ones)
    scene=build_empirical_scene(profile,settings,old['centers'],None,20261118,geometry=geometry,amplitudes=old['amplitudes'],
        angular_coefficients=old['angular_coefficients'],patch_renderer=thinner_shell_patch)
    feather=np.clip(distance_transform_edt(allowed)/2,0,1)
    cells=(effective_blur(scene['signal'],.2,.8,mixture)*normalizer*feather).astype(np.float32)
    clean=old['background']+cells
    unit,_=correlated_noise(clean.shape,source_report['seed']);noise=unit*old['noise_sigma']
    np.testing.assert_array_equal(np.rint(np.clip(old['clean']+noise,0,65535)).astype(np.uint16),old['image'])
    image=np.rint(np.clip(clean+noise,0,65535)).astype(np.uint16)
    np.testing.assert_array_equal(scene['instance_offsets'],old['instance_offsets']);np.testing.assert_array_equal(scene['instance_pixel_index'],old['instance_pixel_index'])
    assert not cells[exclusion].any() and np.isfinite(clean).all()
    tifffile.imwrite(args.output/'synthetic.tiff',image);tifffile.imwrite(args.output/'synthetic_clean.tiff',clean);tifffile.imwrite(args.output/'cell_signal.tiff',cells)
    np.savez_compressed(args.output/'synthetic.npz',image=image,clean=clean,cell_signal=cells,background=old['background'],noise_sigma=old['noise_sigma'],
        labels=old['labels'],centers=old['centers'],instance_offsets=old['instance_offsets'],instance_pixel_index=old['instance_pixel_index'],
        amplitudes=old['amplitudes'],angular_coefficients=old['angular_coefficients'],**geometry)
    before=profile_summary(old['clean'],old['centers'],geometry,context);after=profile_summary(clean,old['centers'],geometry,context)
    np.savez_compressed(args.output/'paired_profiles.npz',radii=before['radii'],before=before['curves'],after=after['curves'],indices=context)
    changes=dict(before_median_core_adu=before['median_core_adu'],after_median_core_adu=after['median_core_adu'],
        paired_median_core_change_adu=float(np.median(after['core']-before['core'])),
        before_median_shell_minus_core_adu=before['median_shell_minus_core_adu'],after_median_shell_minus_core_adu=after['median_shell_minus_core_adu'],
        paired_median_shell_minus_core_change_adu=float(np.median(after['contrast']-before['contrast'])))
    report=dict(source=str(source),old_width=OLD_WIDTH,new_width=NEW_WIDTH,intrinsic_width_multiplier=.85,
        shell_strength=.5,gain_field_length_px=27.932934624762108,flux='Per-cell integral after base optical blur matched to prior .36-width kernel before summation.',
        noise='Exact accepted noise map and random realization frozen for comparison.',paired_context_somas=len(context),profile_changes=changes,
        final_scene_cell_energy_ratio=float(cells.sum(dtype=float)/old['cell_signal'].sum(dtype=float)),
        checks=dict(unchanged_complete_GT=True,same_background=True,same_amplitudes=True,same_angular_coefficients=True,same_noise=True,finite=True,no_signal_in_exclusion=True),
        limitations=['15% applies to intrinsic Gaussian radial width; fixed PSF blur makes visible thickness reduction smaller and context dependent.',
            'Dark centers receive light from neighboring cells, so this is not a guarantee of dark nuclei everywhere.',
            'Local optical mixing, gain and boundary clipping can slightly alter total final-scene energy despite per-cell flux matching.',
            'Paired profile comparison uses 98 existing synthetic context somas, not a new unbiased biological measurement.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5]);rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(5,4,figsize=(15,17),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,old['clean'],clean,old['image'],image]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Before, clean','15% thinner, clean','Before, noisy','15% thinner, noisy'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(17,10),constrained_layout=True)
    for ax,im,title in zip(axs,[real,old['image'],image],['Real frame 500','Medium + updated noise; previous shell','Same scene and noise; intrinsic shell width reduced 15%']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_axis_off()
    fig.savefig(args.output/'full_comparison.png',dpi=145);plt.close(fig)
    # Single-cell radial section: separate brightness difference from edge width.
    args0=([30.,30.],10.,8.,0.,True,0.,settings['blur_px'])
    origin,reference,_=shell_patch(*args0,width=OLD_WIDTH)
    origin_new,narrow,_=thinner_shell_patch(*args0)
    x=np.arange(81);y=np.full_like(x,30);baseline=np.zeros((81,81));thinner=np.zeros_like(baseline)
    for canvas,o,k in [(baseline,origin,reference),(thinner,origin_new,narrow)]:canvas[o[0]:o[0]+k.shape[0],o[1]:o[1]+k.shape[1]]=k
    fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,im,title in [(axs[0],baseline,'Before (isolated soma)'),(axs[1],thinner,'15% thinner, same total emission')]:
        ax.imshow(im[18:43,18:43].T,cmap='gray',vmin=0,vmax=1.3);ax.set_title(title);ax.set_axis_off()
    axs[2].plot(x-30,baseline[y,x],label='Before');axs[2].plot(x-30,thinner[y,x],label='Thinner');axs[2].set(xlim=(-10,10),xlabel='Native X from center (px)',ylabel='Emission (same photon budget)',title='Radial section, before scene mixing');axs[2].legend()
    fig.savefig(args.output/'radial_section.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
