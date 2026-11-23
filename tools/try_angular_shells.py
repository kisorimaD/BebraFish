"""Controlled trial of independent smooth periodic emission along each soma.

Uses the accepted 28px gain field unchanged; preserves individual shell energy
before image-field multiplication, and never changes complete mid-shell GT.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter1d
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from soma_shell_model import shell_patch,sample_angular_coefficients,angular_log_modulation
from try_heterogeneous_scene import effective_blur,correlated_noise
from inspect_shell_angles import perimeter_strip
from diagnose_scene_realism import summary


def measured_profiles(image,centers,geometry,indices):
    profiles=[]
    for i in indices:
        theta=np.linspace(0,2*np.pi,128,endpoint=False);angle=np.radians(geometry['orientation_deg'][i])
        u=geometry['major_axis_px'][i]/2*np.cos(theta);v=geometry['minor_axis_px'][i]/2*np.sin(theta)
        x=centers[i,1]+.5+u*np.cos(angle)-v*np.sin(angle)
        y=centers[i,0]+.5+u*np.sin(angle)+v*np.cos(angle)
        contour,profile,_=perimeter_strip(image,np.column_stack([x,y]))
        perimeter=np.linalg.norm(contour-np.roll(contour,-1,axis=0),axis=1).sum()
        profiles.append(gaussian_filter1d(profile,.4/(perimeter/len(profile)),mode='wrap'))
    return np.asarray(profiles)


def profile_stats(profiles):
    deviations=profiles-profiles.mean(axis=1,keepdims=True)
    power=abs(np.fft.rfft(deviations,axis=1))**2
    return dict(angular_std_adu=summary(profiles.std(axis=1)),
        p90_minus_p10_adu=summary(np.percentile(profiles,90,axis=1)-np.percentile(profiles,10,axis=1)),
        low_modes_power_fraction=summary(power[:,1:3].sum(axis=1)/np.maximum(power.sum(axis=1),1e-8)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_angular_shells');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/plane20_correlated_gain_larger'
    with np.load(source/'synthetic.npz') as z:base={k:z[k] for k in z.files}
    source_report=json.loads((source/'report.json').read_text())
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:allowed=z['neuron_support'].copy();exclusion=z['exclusion_mask'].copy()
    with np.load(ROOT/'outputs/plane20_heterogeneous_trial_v2/synthetic.npz') as z:mixture=z['blur_mixture'].copy();noise_sigma=z['noise_sigma'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_realism_diagnostics_v2/measurements.npz') as z:context=z['synthetic_indices'].copy()
    with np.load(ROOT/'outputs/plane20_shell_angles/angular_profiles.npz') as z:real_profiles=z['mean_profiles'].copy()
    geometry={k:base[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']}
    settings=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())['selected']|dict(ring_fraction=1.,ring_center_fill=0.)
    profile=dict(tissue=allowed,tissue_field=np.ones(allowed.shape,np.float32),cell_gain=np.ones(allowed.shape,np.float32))
    coefficients=sample_angular_coefficients(len(base['centers']),20261119)
    feather=np.clip(distance_transform_edt(allowed)/2,0,1)
    noise,_=correlated_noise(allowed.shape,source_report['seed']+3);noise*=noise_sigma
    strengths=[0.,.25,.50,.75];clean_images=[];images=[];signals=[];profiles=[];metrics={};checks={}
    for strength in strengths:
        scene=build_empirical_scene(profile,settings,base['centers'],None,source_report['seed'],geometry=geometry,
            amplitudes=base['amplitudes'],angular_coefficients=coefficients*strength,
            patch_renderer=partial(shell_patch,peak_radius=1.,width=.36))
        cells=(effective_blur(scene['signal'],.2,.8,mixture)*base['normalizer']*feather).astype(np.float32)
        np.testing.assert_array_equal(scene['instance_offsets'],base['instance_offsets'])
        np.testing.assert_array_equal(scene['instance_pixel_index'],base['instance_pixel_index'])
        assert not cells[exclusion].any() and np.isfinite(cells).all() and (cells>=0).all()
        if strength==0:
            difference=float(np.max(abs(cells-base['cell_signal'])))
            np.testing.assert_allclose(cells,base['cell_signal'],rtol=3e-7,atol=5e-5)
            checks['zero_strength_reproduction_max_error_adu']=difference
            cells=base['cell_signal'].copy()
        clean=base['background']+cells
        image=np.rint(np.clip(clean+noise,0,65535)).astype(np.uint16)
        if strength==0:np.testing.assert_array_equal(image,base['image'])
        curves=measured_profiles(clean,base['centers'],geometry,context)
        profiles.append(curves);metrics[str(strength)]=profile_stats(curves)
        metrics[str(strength)]['total_cell_signal_ratio_to_baseline']=float(cells.sum(dtype=float)/base['cell_signal'].sum(dtype=float))
        clean_images.append(clean);images.append(image);signals.append(cells)
        tag=f'{strength:.2f}'.replace('.','p')
        tifffile.imwrite(args.output/f'strength_{tag}.tiff',image);tifffile.imwrite(args.output/f'strength_{tag}_clean.tiff',clean)
        print(json.dumps(dict(strength=strength,metrics=metrics[str(strength)])),flush=True)
    # Check per-cell energy independently of scene overlap and coarse fields.
    flux_errors=[]
    for i in range(len(base['centers'])):
        params=[base['centers'][i],geometry['major_axis_px'][i],geometry['minor_axis_px'][i],geometry['orientation_deg'][i],True,0.,settings['blur_px']]
        _,baseline,_=shell_patch(*params,peak_radius=1.,width=.36)
        _,modulated,_=shell_patch(*params,peak_radius=1.,width=.36,angular_coefficients=coefficients[i]*max(strengths))
        flux_errors.append(abs(float(modulated.sum(dtype=float)/baseline.sum(dtype=float))-1))
    checks.update(max_relative_intrinsic_flux_error=float(max(flux_errors)),unchanged_complete_GT=True,
        no_signal_in_exclusion=True,same_gain_amplitudes=True,same_noise=True,finite=True)
    assert max(flux_errors)<1e-5
    np.savez_compressed(args.output/'synthetic.npz',images=np.array(images),clean=np.array(clean_images),cell_signal=np.array(signals),
        strengths=np.array(strengths),background=base['background'],labels=base['labels'],centers=base['centers'],
        instance_offsets=base['instance_offsets'],instance_pixel_index=base['instance_pixel_index'],
        amplitudes=base['amplitudes'],angular_coefficients_unit=coefficients,normalizer=base['normalizer'],**geometry)
    np.savez_compressed(args.output/'angular_profiles.npz',strengths=np.array(strengths),synthetic_profiles=np.array(profiles),real_mean_profiles=real_profiles,synthetic_indices=context)
    report=dict(source=str(source),source_seed=source_report['seed'],angular_seed=20261119,cell_count=len(base['centers']),
        gain_field_length_px=source_report['generated_field_length_px'],strengths=strengths,
        model='Four cosine/sine modes in equal-arc-length perimeter phase, independent coefficients per soma, log-emission exponentiated and shell energy renormalized before optical blur.',
        relative_mode_weights=[1.,.75,.5,.35],metrics=metrics,real_observed_profile_statistics=profile_stats(real_profiles),checks=checks,
        limitations=['Independent per-cell peripheral patterns are an exploratory assumption; spatial correlations between peripheral patterns are not modeled.',
            'Strengths are a visual sensitivity series, not parameters fitted to uncontaminated intrinsic membrane emission.',
            'Observed profile statistics include neighboring cells and projection/blur; comparing selected real somas to random synthetic context somas is biased.',
            'Intrinsic patch flux is conserved. Spatially varying image gain, optical mixing and exclusion clipping may slightly alter final scene energy.',
            'Only emission changes; geometry, shell thickness, local PSF and global amplitude correlations are unchanged.',
            'The existing mid-shell labels remain anatomical contours even where an arc becomes faint.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5]);rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(5,4,figsize=(15,17),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real]+images):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel((['Real','Uniform shell','Weak: 0.25','Medium: 0.50','Strong: 0.75'])[row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(5,1,figsize=(17,16),constrained_layout=True)
    for ax,im,title in zip(axs,[real]+images,['Real frame 500','Uniform shell; 28px gain field','Weak perimeter modulation: 0.25','Medium perimeter modulation: 0.50','Strong perimeter modulation: 0.75']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_axis_off()
    fig.savefig(args.output/'full_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(4,6,figsize=(14,9),constrained_layout=True)
    # Deterministic spread of cells, not chosen for best-looking modulation.
    selected=np.linspace(0,len(base['centers'])-1,6).round().astype(int)
    for col,i in enumerate(selected):
        for row,strength in enumerate(strengths):
            _,kernel,_=shell_patch([30.,30.],geometry['major_axis_px'][i],geometry['minor_axis_px'][i],geometry['orientation_deg'][i],True,0.,settings['blur_px'],peak_radius=1.,width=.36,angular_coefficients=coefficients[i]*strength)
            ax=axs[row,col];ax.imshow(kernel.T,cmap='gray',vmin=0,vmax=2.);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Cell #{i+1}')
            if col==0:ax.set_ylabel(f'Strength {strength:.2f}')
    fig.savefig(args.output/'isolated_shells.png',dpi=150);plt.close(fig)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
