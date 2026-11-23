"""Contrast vs sharpness controls with spatially held-out crop statistics.

Structural features use independent real half-means and synthetic clean signal,
not noisy Laplacian variance. The metrics characterize texture, not optical PSF.
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
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.stats import rankdata
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from crop_quiz import valid_origins
from try_thinner_medium import thinner_shell_patch,profile_summary
from try_heterogeneous_scene import effective_blur,correlated_noise


def contrast_signal(signal,gain,sigma=8.):
    """Boost emission deviations around its slow local mean, keeping positivity.

    This is an effective photometry correction, not identified cellular optics.
    Global cell-energy normalization retains overall brightness after clipping.
    """
    if not np.isfinite(gain) or gain<1:raise ValueError('Contrast gain must be finite and >=1')
    if gain==1:return signal.copy()
    smooth=gaussian_filter(signal.astype(float),sigma)
    result=np.maximum(smooth+gain*(signal-smooth),0.)
    result*=signal.sum(dtype=float)/max(result.sum(dtype=float),1e-12)
    return result.astype(np.float32)


def structure_maps(a,b):
    a=gaussian_filter(a.astype(float),1.)-gaussian_filter(a.astype(float),8.)
    b=gaussian_filter(b.astype(float),1.)-gaussian_filter(b.astype(float),8.)
    ay,ax=np.gradient(a);by,bx=np.gradient(b)
    return a,b,ay*by+ax*bx


def crop_features(image,a,b,coords,size=96):
    aa,bb,energy=structure_maps(a,b);rows=[]
    window=np.outer(np.hanning(size),np.hanning(size))
    freq=np.fft.fftfreq(size);radius=np.hypot(freq[:,None],freq[None,:]);middle=(radius>=.06)&(radius<.15);high=(radius>=.15)&(radius<.30)
    for y,x in coords:
        roi=(slice(y,y+size),slice(x,x+size));im=image[roi].astype(float);A=aa[roi];B=bb[roi]
        q=np.percentile(im,[5,50,95]);variance=np.mean((A-A.mean())*(B-B.mean()))
        # Cross-spectrum suppresses uncorrelated noise between real half-means.
        fa=np.fft.fft2((A-A.mean())*window);fb=np.fft.fft2((B-B.mean())*window);power=np.real(fa*np.conj(fb))
        rows.append([q[2]-q[0],q[1],np.sqrt(max(float(energy[roi].mean()/max(variance,1e-8)),0.)),
            float(power[high].sum()/max(power[middle].sum(),1e-8))])
    return np.asarray(rows)


def auc(real,synthetic):
    scores=np.r_[real,synthetic];ranks=rankdata(scores)
    return float((ranks[:len(real)].sum()-len(real)*(len(real)+1)/2)/(len(real)*len(synthetic)))


def threshold_test(real,synthetic,train):
    r,s=real[train],synthetic[train];values=np.unique(np.r_[r,s]);thresholds=(values[:-1]+values[1:])/2
    if not len(thresholds):return dict(test_auc=.5,test_balanced_accuracy=.5,threshold=None)
    best=None
    for sign in [1,-1]:
        for t in thresholds:
            score=.5*(np.mean(sign*r>sign*t)+np.mean(sign*s<=sign*t))
            if best is None or score>best[0]:best=(float(score),sign,float(t))
    score,sign,t=best;test=~train
    return dict(threshold=t,higher_means_real=sign==1,train_balanced_accuracy=score,
        test_balanced_accuracy=float(.5*(np.mean(sign*real[test]>sign*t)+np.mean(sign*synthetic[test]<=sign*t))),
        test_auc=auc(sign*real[test],sign*synthetic[test]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_contrast_sharpness');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with np.load(ROOT/'outputs/plane20_medium_thinner/synthetic.npz') as z:base={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:mask=z['exclusion_mask'].copy();allowed=z['neuron_support'].copy()
    with np.load(ROOT/'outputs/plane20_angular_shells/synthetic.npz') as z:normalizer=z['normalizer'].copy()
    with np.load(ROOT/'outputs/plane20_heterogeneous_trial_v2/synthetic.npz') as z:mixture=z['blur_mixture'].copy()
    with tifffile.TiffFile(ROOT/'data/Plane_Aligned_20.tiff') as tif:frames=np.array([tif.pages[i].asarray() for i in range(492,524)],np.float32)
    real_a=frames[::2].mean(0);real_b=frames[1::2].mean(0)
    eligible,shape=valid_origins(mask,96,96);rng=np.random.default_rng(99)
    coords=np.column_stack(np.unravel_index(eligible,shape))
    coords=coords[coords[:,0]//224==(coords[:,0]+95)//224]
    coords=coords[rng.choice(len(coords),min(600,len(coords)),replace=False)]
    train=(coords[:,0]//224)%2==0
    if not train.any() or not (~train).any():raise ValueError('No eligible positions in one spatial partition')
    features={'real':crop_features(real,real_a,real_b,coords)}
    unit,_=correlated_noise(real.shape,20261121);noise=unit*base['noise_sigma']
    np.testing.assert_array_equal(np.rint(np.clip(base['clean']+noise,0,65535)).astype(np.uint16),base['image'])
    candidates=[];candidate_signals=[]
    for gain in [1.,1.2,1.4,1.6,1.8,2.]:
        cells=contrast_signal(base['cell_signal'],gain);clean=base['background']+cells;image=np.rint(np.clip(clean+noise,0,65535)).astype(np.uint16)
        f=crop_features(image,clean,clean,coords);features[f'gain_{gain}']=f
        # Fit amplitudes only. Sharpness is examined independently afterwards.
        quantiles=[25,50,75]
        loss=float(np.mean(np.log(np.percentile(f[train,0],quantiles)/np.percentile(features['real'][train,0],quantiles))**2))
        row=dict(gain=gain,train_range_loss=loss,median_range=float(np.median(f[:,0])));candidates.append(row);candidate_signals.append(cells)
        print(json.dumps(row),flush=True)
    chosen=int(np.argmin([r['train_range_loss'] for r in candidates]));gain=candidates[chosen]['gain'];contrast_cells=candidate_signals[chosen]
    # Controlled physical sharpening: halve added scene PSF and reduce base PSF.
    # Width .306, angular pattern and whole-cell gains remain unchanged.
    settings=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())['selected']|dict(ring_fraction=1.,ring_center_fill=0.,blur_px=.35)
    geometry={k:base[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']}
    ones=np.ones(real.shape,np.float32);profile=dict(tissue=allowed,tissue_field=ones,cell_gain=ones)
    scene=build_empirical_scene(profile,settings,base['centers'],None,20261118,geometry=geometry,amplitudes=base['amplitudes'],
        angular_coefficients=base['angular_coefficients'],patch_renderer=thinner_shell_patch)
    np.testing.assert_array_equal(scene['instance_offsets'],base['instance_offsets']);np.testing.assert_array_equal(scene['instance_pixel_index'],base['instance_pixel_index'])
    sharper=(effective_blur(scene['signal'],.1,.4,mixture)*normalizer*np.clip(distance_transform_edt(allowed)/2,0,1)).astype(np.float32)
    variants={'baseline':base['cell_signal'],'contrast_only':contrast_cells,'sharper_only':sharper,'contrast_and_sharper':contrast_signal(sharper,gain)}
    results={};images={};cleans={}
    names=['range_p95_p05','median_brightness','normalized_gradient','high_mid_cross_power_ratio']
    for name,cells in variants.items():
        clean=base['background']+cells;image=np.rint(np.clip(clean+noise,0,65535)).astype(np.uint16)
        images[name]=image;cleans[name]=clean;f=crop_features(image,clean,clean,coords);features[name]=f
        results[name]={key:dict(real_median=float(np.median(features['real'][~train,j])),synthetic_median=float(np.median(f[~train,j])),
            discriminator=threshold_test(features['real'][:,j],f[:,j],train)) for j,key in enumerate(names)}
        tifffile.imwrite(args.output/(name+'.tiff'),image)
        print(name,json.dumps(results[name]),flush=True)
    # Prefer contrast-only: physical sharpening worsens normalized texture scale
    # if it is already finer than real data. Keep both controls for inspection.
    selected='contrast_only';cells=variants[selected];clean=cleans[selected];image=images[selected]
    assert not cells[mask].any() and np.isfinite(clean).all()
    np.testing.assert_allclose(cells.sum(dtype=float),base['cell_signal'].sum(dtype=float),rtol=1e-6)
    tifffile.imwrite(args.output/'synthetic.tiff',image);tifffile.imwrite(args.output/'synthetic_clean.tiff',clean)
    np.savez_compressed(args.output/'synthetic.npz',image=image,clean=clean,cell_signal=cells,background=base['background'],noise_sigma=base['noise_sigma'],
        labels=base['labels'],centers=base['centers'],instance_offsets=base['instance_offsets'],instance_pixel_index=base['instance_pixel_index'],
        amplitudes=base['amplitudes'],angular_coefficients=base['angular_coefficients'],**geometry)
    np.savez_compressed(args.output/'crop_metrics.npz',coords_yx=coords,train=train,feature_names=np.array(names),**features)
    with np.load(ROOT/'outputs/plane20_realism_diagnostics_v2/measurements.npz') as z:context=z['synthetic_indices'].copy()
    before_profiles=profile_summary(base['clean'],base['centers'],geometry,context)
    after_profiles=profile_summary(clean,base['centers'],geometry,context)
    paired_dark_centers=dict(before_median_core_adu=before_profiles['median_core_adu'],after_median_core_adu=after_profiles['median_core_adu'],
        before_median_shell_minus_core_adu=before_profiles['median_shell_minus_core_adu'],after_median_shell_minus_core_adu=after_profiles['median_shell_minus_core_adu'],
        median_paired_contrast_change_adu=float(np.median(after_profiles['contrast']-before_profiles['contrast'])))
    report=dict(paired_dark_centers=paired_dark_centers,selected=selected,selected_contrast_gain=gain,candidates=candidates,metrics=results,
        sample=dict(crop_size=96,positions=len(coords),train=int(train.sum()),test=int((~train).sum()),split='Alternating 224px native-Y bands, crops wholly inside one band. Same sampled positions in each class; train and test crop pixels do not overlap.'),
        real_structure='Independent even/odd half-means of frames492:524; cross-gradient and cross-spectrum reduce independent noise bias.',
        feature_definitions=dict(range_p95_p05='P95-P05 in raw ADU, single exposure, noise included.',
            normalized_gradient='sqrt(mean(grad(A) dot grad(B))/cov(A,B)), A/B=G1(I)-G8(I); invariant to common positive affine intensity changes, unit 1/px.',
            high_mid_cross_power_ratio='Cross-spectral power 0.15:0.30 cyc/px divided by 0.06:0.15 cyc/px, Hann-windowed, no image-noise power for synthetic clean.'),
        contrast_model='Snew=max(0,G8(S)+gain*(S-G8(S))), globally energy normalized. Applied only to positive cellular emission before background/noise.',
        sharpness_control='Base PSF .6 -> .35px; extra horizontal/vertical sigmas .2/.8 -> .1/.4px. Effective control, not optical calibration.',
        checks=dict(no_mask_leakage=True,unchanged_complete_GT=True,same_background_and_noise=True,global_cell_energy_preserved=True),
        limitations=['Normalized texture metrics still depend on cell size, packing, angular patterns and projected structures. They do not identify optical resolution or PSF alone.',
            'Real half-means may retain motion and calcium changes; spatial train/test uses one recording and one synthetic scene.',
            'Crops overlap within partitions; discriminator performance has few independent anatomical regions and is not a population estimate.',
            'Contrast mapping is an effective model correction, not a measured biological emission law.',
            'Fixed noise isolates the signal changes; a full sensor calibration is not performed.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5]);rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(5,4,figsize=(15,17),constrained_layout=True)
    display=[real,images['baseline'],image,images['sharper_only'],images['contrast_and_sharper']]
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate(display):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Before','Contrast only (selected)','Reduced PSF blur','Contrast + reduced PSF'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for j,key,ax in zip([0,2,3],['Raw contrast range','Amplitude-normalized gradient','Cross-spectrum high/mid ratio'],axs):
        for name,color in [('real','#242424'),('baseline','#ce9a58'),('contrast_only','#479f83'),('contrast_and_sharper','#9775bd')]:
            ax.hist(features[name][~train,j],bins=20,density=True,histtype='step',lw=1.6,label=name,color=color)
        ax.set_title(key);ax.legend(fontsize=7)
    fig.savefig(args.output/'metric_distributions.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(selected=selected,gain=gain,sample=report['sample']),indent=2))


if __name__=='__main__':main()
