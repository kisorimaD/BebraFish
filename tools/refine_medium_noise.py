"""Make effective noise respond to actual synthetic illumination.

Keeps accepted medium angular modulation, 28px gain field and all clean signal
unchanged. The temporal noise map is not assumed to be pure camera noise.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

ROOT=Path(__file__).resolve().parents[1]
from try_heterogeneous_scene import correlated_noise


def adapted_noise_sigma(clean,real_mean,observed_sigma,offset,read_variance,shot_gain,excess_fraction,*,excess_scale=1.,smooth_sigma=8.):
    """Transfer spatial excess above read floor using a signal-dependent model.

    alpha(x) is an effective spatial variance multiplier, fitted at real mean.
    sigma_new² = read + alpha*(gain*Snew + excess_scale*excess*Snew²).
    The last term may include activity/motion; its scale is an augmentation
    choice rather than an identified detector parameter.
    """
    if clean.shape!=real_mean.shape or clean.shape!=observed_sigma.shape:raise ValueError('Noise maps must have the same shape')
    if any(not np.isfinite(v) or v<0 for v in [read_variance,shot_gain,excess_fraction,excess_scale,smooth_sigma]):raise ValueError('Noise parameters must be finite and nonnegative')
    signal_old=np.maximum(gaussian_filter(np.asarray(real_mean,float),smooth_sigma)-offset,0.)
    signal_new=np.maximum(gaussian_filter(np.asarray(clean,float),smooth_sigma)-offset,0.)
    variance_old=shot_gain*signal_old+excess_fraction*signal_old**2
    alpha=np.divide(np.maximum(observed_sigma**2-read_variance,0.),np.maximum(variance_old,1e-6))
    # Prevent poorly identified nearly dark locations from amplifying a new scene.
    alpha=np.clip(alpha,0.,4.)
    variance=read_variance+alpha*(shot_gain*signal_new+excess_scale*excess_fraction*signal_new**2)
    return np.sqrt(variance).astype(np.float32),alpha.astype(np.float32)


def roi_metrics(image,clean,cells,sigma,roi):
    y0,y1,x0,x1=roi;s=(slice(y0,y1),slice(x0,x1))
    signal_hp=(cells-gaussian_filter(cells,3.))[s]
    noise=(image.astype(float)-clean)[s]
    return dict(signal_highpass_std_adu=float(signal_hp.std()),noise_std_adu=float(noise.std()),
        signal_highpass_std_over_noise_std=float(signal_hp.std()/max(noise.std(),1e-6)),
        noise_sigma_p10_median_p90=np.percentile(sigma[s],[10,50,90]).astype(float).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_medium_noise');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/plane20_angular_shells'
    with np.load(source/'synthetic.npz') as z:data={k:z[k] for k in z.files}
    i=int(np.flatnonzero(data['strengths']==.5)[0]);clean=data['clean'][i].copy();cells=data['cell_signal'][i].copy();previous=data['images'][i].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:mean=z['mean'].copy()
    with np.load(ROOT/'outputs/plane20_trial/profile.npz') as z:offset=z['background'].copy()
    with np.load(ROOT/'outputs/plane20_heterogeneous_trial_v2/synthetic.npz') as z:old_sigma=z['noise_sigma'].copy()
    settings=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())['selected']
    seed=json.loads((source/'report.json').read_text())['source_seed']+3
    unit,fraction=correlated_noise(clean.shape,seed)
    np.testing.assert_array_equal(np.rint(np.clip(clean+unit*old_sigma,0,65535)).astype(np.uint16),previous)
    rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    variants={'previous':previous};sigmas={'previous':old_sigma};metrics={};alphas=None
    scales={'illumination_adapted':1.,'reduced_excess':.25,'shot_and_read_only':0.}
    for name,scale in scales.items():
        sigma,alphas=adapted_noise_sigma(clean,mean,old_sigma,offset,settings['read_variance_adu2'],settings['shot_gain_adu'],settings['excess_variance_fraction'],excess_scale=scale)
        image=np.rint(np.clip(clean+unit*sigma,0,65535)).astype(np.uint16)
        variants[name]=image;sigmas[name]=sigma
        tifffile.imwrite(args.output/(name+'.tiff'),image);tifffile.imwrite(args.output/(name+'_sigma.tiff'),sigma)
    for name,image in variants.items():metrics[name]={str(roi):roi_metrics(image,clean,cells,sigmas[name],roi) for roi in rois}
    selected='reduced_excess'
    tifffile.imwrite(args.output/'synthetic.tiff',variants[selected]);tifffile.imwrite(args.output/'synthetic_clean.tiff',clean)
    tifffile.imwrite(args.output/'noise_sigma.tiff',sigmas[selected])
    np.savez_compressed(args.output/'synthetic.npz',image=variants[selected],clean=clean,cell_signal=cells,
        background=data['background'],noise_sigma=sigmas[selected],effective_spatial_variance_multiplier=alphas,
        labels=data['labels'],centers=data['centers'],instance_offsets=data['instance_offsets'],instance_pixel_index=data['instance_pixel_index'],
        amplitudes=data['amplitudes'],angular_coefficients=data['angular_coefficients_unit']*.5,
        **{k:data[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']})
    report=dict(source=str(source),shell_strength=.5,gain_field_length_px=27.932934624762108,selected=selected,
        selected_excess_variance_scale=scales[selected],seed=seed,noise_structure='Same unit-variance correlated noise realization, neighbor correlation ~0.265, as before.',
        method='sigma² = read_variance + alpha(x)*(shot_gain*Ssynthetic + scale*excess_fraction*Ssynthetic²); S is 8px smoothed illumination minus old detector baseline.',
        variance_parameters={k:settings[k] for k in ['read_variance_adu2','shot_gain_adu','excess_variance_fraction']},
        noise_colored_fraction=fraction,variants=scales,metrics=metrics,
        diagnosis='Real-location temporal variance stayed unchanged while random gain suppressed synthetic fluorescence locally; left crop median cell amplitude ~0.52.',
        checks=dict(exact_clean_preservation=True,exact_GT_preservation=True,same_unit_noise_realization=True,read_noise_floor_preserved=True,finite=bool(np.isfinite(sigmas[selected]).all())),
        limitations=['This is an effective noise augmentation, not newly measured camera calibration.',
            'Temporal MAD contains activity/motion, so the excess-variance term is not identifiable as detector noise.',
            'Excess scale 0.25 is a visual choice to reveal structure; real temporal variance is not a target it should now match.',
            'The clean left-crop cell texture remains lower contrast than real data. Noise correction cannot fully resolve that difference.',
            'Spatial variance multiplier is inherited and clipped to [0,4]; its optical/biological origin is not identified.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5]);display=[real,previous,variants['illumination_adapted'],variants[selected],clean]
    names=['Real frame 500','Medium: previous fixed noise','Noise follows synthetic illumination','Also reduce excess variance to 25% (selected)','Same synthetic medium, without noise']
    fig,axs=plt.subplots(5,4,figsize=(15,17),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate(display):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Previous','Adapted','Reduced excess','Clean'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    y0,y1,x0,x1=rois[0]
    fig,axs=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
    for ax,im,name in zip(axs,display,names):
        ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(name,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(args.output/'left_crop.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(17,10),constrained_layout=True)
    for ax,im,name in zip(axs,[real,previous,variants[selected]],['Real frame 500','Medium: previous fixed noise','Medium: illumination-aware noise, reduced excess variance']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(name);ax.set_axis_off()
    fig.savefig(args.output/'full_comparison.png',dpi=145);plt.close(fig)
    print(json.dumps({name:rows[str(rois[0])] for name,rows in metrics.items()},indent=2))


if __name__=='__main__':main()
