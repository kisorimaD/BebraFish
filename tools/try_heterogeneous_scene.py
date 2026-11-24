"""Controlled photometry / effective anisotropic blur / correlated-noise trial.

All variants preserve the established soma contours and user exclusion mask.
Effective blur is fitted to texture statistics; it does not identify optical PSF.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from soma_shell_model import shell_patch
from diagnose_scene_realism import texture_tiles,summary,soma_light,neighbor_spread


from median_photometry import effective_blur


def correlated_noise(shape,seed,sigma=.65,rho=.265):
    rng=np.random.default_rng(seed)
    impulse=np.zeros((17,17));impulse[8,8]=1
    kernel=gaussian_filter(impulse,sigma)
    energy=float(np.sum(kernel**2))
    neighbor=float(np.sum(kernel[:-1]*kernel[1:])/energy)
    fraction=float(np.clip(rho/neighbor,0,1))
    colored=gaussian_filter(rng.normal(size=shape),sigma)/np.sqrt(energy)
    white=rng.normal(size=shape)
    return np.sqrt(1-fraction)*white+np.sqrt(fraction)*colored,fraction


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_heterogeneous_trial');args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    seed=20261117
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:old={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:mean=z['mean'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_realism_diagnostics/measurements.npz') as z:
        noise_sigma=z['smooth_noise_sigma'].copy();context=z['synthetic_indices'].copy();ac=z['real_centers'].copy()
    fit=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())
    settings=fit['selected']|dict(ring_fraction=1.,ring_center_fill=0.)
    geometry={k:old[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']}
    ones=np.ones(mean.shape,np.float32);allowed=old['neuron_support'];valid=distance_transform_edt(allowed)>4
    profile=dict(tissue=allowed,tissue_field=ones,cell_gain=ones)
    feather=np.clip(distance_transform_edt(allowed)/2,0,1)
    weight=gaussian_filter(allowed.astype(float),16.)
    envelope=gaussian_filter(np.where(allowed,np.maximum(mean-old['background'],0.),0.),16.)/np.maximum(weight,1e-6)
    def scale(signal):
        local=gaussian_filter(signal,16.)/np.maximum(weight,1e-6)
        return (signal*np.divide(envelope,np.maximum(local,1e-6),out=np.zeros_like(local),where=local>1e-5)*feather).astype(np.float32)
    target=texture_tiles(mean,valid)
    train=np.array([(r['y']//192)%2==0 for r in target]);validation=~train
    def features(rows,which):
        return np.array([np.median([r[k] for r,ok in zip(rows,which) if ok]) for k in ['highpass_std','horizontal_length_px','vertical_length_px']])
    targets={name:features(target,m) for name,m in [('train',train),('validation',validation)]}
    def metrics(cells):
        rows=texture_tiles(old['background']+cells,valid)
        out={}
        for name,mask in [('train',train),('validation',validation)]:
            v=features(rows,mask);t=targets[name]
            out[name]=dict(highpass_std=float(v[0]),horizontal_length_px=float(v[1]),vertical_length_px=float(v[2]),
                vertical_over_horizontal=float(v[2]/v[1]),objective=float(np.mean(np.log(v/t)**2)))
        return out
    candidates=[];raw_candidates=[];best=None
    print('Fitting positive cell amplitudes and intrinsic shell width',flush=True)
    for log_sigma in [.65,1.,1.35]:
        for width in [.20,.28,.36]:
            scene=build_empirical_scene(profile,settings|dict(amplitude_cv=log_sigma),old['centers'],None,seed,geometry=geometry,
                patch_renderer=partial(shell_patch,peak_radius=1.,width=width))
            cells=scale(scene['signal']);m=metrics(cells)
            row=dict(log_amplitude_sigma=log_sigma,shell_width=width,metrics=m)
            candidates.append(row);raw_candidates.append((row,scene['signal'].copy(),cells.copy()))
            # Keep all candidates for a joint fit: blur also changes contrast.
            loss=abs(np.log(m['train']['highpass_std']/targets['train'][0]))
            if best is None or loss<best[0]:best=(loss,row,scene['signal'].copy(),cells.copy())
            print(json.dumps(row),flush=True)
    _,selected,raw,photometry=best
    # Random smooth spatial field, independent of real image texture. This trial
    # explores variable focus; its per-location values are not inferred from data.
    rng=np.random.default_rng(seed+5);field=gaussian_filter(rng.normal(size=mean.shape),40.)
    field=(field-field.mean())/max(field.std(),1e-8);mixture=1/(1+np.exp(-field*1.5))
    blur_candidates=[];best=None;isotropic=None;directional=None
    print('Joint fit of amplitudes, shell width and effective directional blur',flush=True)
    for photo_row,raw_signal,photo_cells in raw_candidates:
        for horizontal,vertical in [(0.,0.),(.4,.4),(.8,.8),(1.2,1.2),(.2,.8),(.4,1.2),(.6,1.6),(.8,1.6),(.8,2.),(1.2,2.)]:
            cells=scale(effective_blur(raw_signal,horizontal,vertical,mixture));m=metrics(cells)
            row=dict(log_amplitude_sigma=photo_row['log_amplitude_sigma'],shell_width=photo_row['shell_width'],horizontal_sigma=horizontal,vertical_sigma=vertical,metrics=m);blur_candidates.append(row)
            loss=m['train']['objective']
            entry=(loss,row,cells.copy(),photo_row,raw_signal,photo_cells)
            if best is None or loss<best[0]:best=entry
            if horizontal==vertical and (isotropic is None or loss<isotropic[0]):isotropic=entry
            if horizontal!=vertical and (directional is None or loss<directional[0]):directional=entry
        print(json.dumps(dict(completed_amplitude=photo_row['log_amplitude_sigma'],width=photo_row['shell_width'],best_train_objective=best[0])),flush=True)
    _,blur_selected,blurred,selected,raw,photometry=best
    unit_noise,fraction=correlated_noise(mean.shape,seed+9)
    effective_noise=unit_noise*noise_sigma
    clean=old['background']+blurred
    noisy=np.rint(np.clip(clean+effective_noise,0,65535)).astype(np.uint16)
    versions=dict(photometry_clean=old['background']+photometry,isotropic_control_clean=old['background']+isotropic[2],directional_control_clean=old['background']+directional[2],
        heterogeneous_clean=clean,heterogeneous=noisy,background=old['background'])
    for name,im in versions.items():tifffile.imwrite(args.output/(name+'.tiff'),im)
    np.savez_compressed(args.output/'synthetic.npz',image=noisy,clean=clean,cell_signal=blurred,background=old['background'],
        labels=old['labels'],centers=old['centers'],instance_offsets=old['instance_offsets'],instance_pixel_index=old['instance_pixel_index'],
        noise_sigma=noise_sigma,blur_mixture=mixture.astype(np.float32),**geometry)
    assert not blurred[old['exclusion_mask']].any()
    np.testing.assert_array_equal(old['labels'],tifffile.imread(ROOT/'outputs/plane20_masked_background/annotation_labels.tiff'))
    annotations=json.loads((ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json').read_text())
    labels=[a for a in annotations['annotations'] if a['confidence']=='confident' and not a['truncated']]
    ra=[np.array([a['measurements'][k] for a in labels]) for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    light={};spreads={}
    for name,im in [('real',real),('previous',old['image']),('photometry',versions['photometry_clean']),('new_clean',clean),('new',noisy)]:
        c=ac if name=='real' else old['centers'][context]
        g=ra if name=='real' else [geometry[k][context] for k in ['major_axis_px','minor_axis_px','orientation_deg']]
        v=soma_light(im,c,*g);light[name]={k:summary(vv) for k,vv in v.items()};spreads[name]=neighbor_spread(v['shell'],c)
    report=dict(seed=seed,cell_count=len(old['centers']),selection_bias_correction=False,
        split='64px tiles stepped 48px; alternate native Y bands of 192px train/validation. Target is temporal mean; not an independent recording.',
        tile_counts=dict(train=int(train.sum()),validation=int(validation.sum())),target_features={k:v.tolist() for k,v in targets.items()},
        amplitude_candidates=candidates,selected_photometry=selected,blur_candidates=blur_candidates,selected_blur=blur_selected,
        isotropic_control=isotropic[1],directional_control=directional[1],baseline_metrics=metrics(old['cell_signal']),light=light,neighbor_shell_spreads=spreads,
        correlated_noise=dict(gaussian_sigma_px=.65,colored_variance_fraction=fraction,target_adjacent_correlation=.265,
            sigma_source='8px smoothed robust temporal pair-difference sigma from 25 pairs, includes biological activity and residual motion.'),
        cautions=['This is a controlled exploratory trial, not a new validated SimCalc calibration.',
            'Effective blur combines shape, projection, motion and optical blur; fitted sigma is not a measured optical PSF or proof of astigmatism.',
            'Variable-focus field is random smooth augmentation; it is not an estimated map of real defocus.',
            'Narrower intrinsic shell and blur are jointly confounded.',
            '98 real annotated somas are selected for visibility; light summaries are not unbiased population parameters.',
            'Noise variance map contains motion and calcium dynamics and must not be interpreted as pure camera variance.',
            'Same foreground support and complete mid-shell GT masks; optical tails are still suppressed in exclusion regions.'])
    report['interpretation']=dict(anisotropic_blur_validation='Directional fit wins training but loses validation slightly to isotropic control; do not treat fitted directional sigma as established optical calibration.',
        visibility='Texture fits use all valid tiles; real soma summaries are visibility-selected and synthetic context somas are random.')
    np.testing.assert_allclose(clean,old['background']+blurred,rtol=0,atol=0)
    report['checks']=dict(unchanged_complete_instance_masks=True,exact_additive_identity=True,no_cell_signal_in_exclusion=True,finite=bool(np.isfinite(clean).all()))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5])
    fig,axs=plt.subplots(5,1,figsize=(17,16),constrained_layout=True)
    for ax,im,title in zip(axs,[real,old['image'],versions['photometry_clean'],clean,noisy],['Real frame 500','Previous synthetic','Broader cell amplitudes + adjusted shell (no noise)','Also effective directional, spatially varying blur (no noise)','Also spatially varying, correlated effective noise']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_axis_off()
    fig.savefig(args.output/'full_ablation.png',dpi=145);plt.close(fig)
    rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(4,4,figsize=(15,14),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,old['image'],clean,noisy]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Previous','New clean','New + correlated noise'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate([rois[0],rois[2],rois[3]]):
        for row,im in enumerate([mean,versions['isotropic_control_clean'],clean]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real temporal mean','Isotropic control','Directional trial'][row])
    fig.savefig(args.output/'blur_controls.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(selected=selected,blur=blur_selected,isotropic=isotropic[1],light=light),indent=2),flush=True)


if __name__=='__main__':main()
