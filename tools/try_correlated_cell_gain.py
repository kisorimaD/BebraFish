"""Spatially correlated per-cell log-gain, with a matched independent control.

Gain scales entire soma emission, not background or per-pixel shell texture.
The photometric normalizer is frozen using a neutral unit-amplitude scene.
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
from scipy.ndimage import gaussian_filter,map_coordinates,distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from soma_shell_model import shell_patch
from diagnose_scene_realism import soma_light
from try_heterogeneous_scene import effective_blur,correlated_noise

BINS=np.array([0,8,12,18,25,35,50,75,110.])


def smooth_gain_field(shape,length_px,seed):
    """Gaussian log-field with approximate covariance exp(-r²/(2*length²))."""
    rng=np.random.default_rng(seed);padding=int(np.ceil(4*length_px))
    padded=rng.normal(size=(shape[0]+2*padding,shape[1]+2*padding))
    field=gaussian_filter(padded,length_px/np.sqrt(2),mode='constant')[padding:padding+shape[0],padding:padding+shape[1]]
    return (field-field.mean())/max(field.std(),1e-12)


def cell_amplitudes(field,centers,field_sigma,jitter_sigma,seed):
    rng=np.random.default_rng(seed)
    log_gain=field_sigma*map_coordinates(field,centers.T,order=1,mode='nearest')+jitter_sigma*rng.normal(size=len(centers))
    amplitudes=np.exp(log_gain-.5*(field_sigma**2+jitter_sigma**2))
    # Global normalization only; does not divide away local group brightness.
    return amplitudes/amplitudes.mean()


def variogram(values,centers):
    distance=pdist(centers);gamma=.5*pdist(np.asarray(values)[:,None],'sqeuclidean')
    which=np.digitize(distance,BINS)-1
    rows=[]
    for i in range(len(BINS)-1):
        use=which==i
        rows.append(dict(lo=float(BINS[i]),hi=float(BINS[i+1]),distance=float(np.mean(distance[use])) if use.any() else float((BINS[i]+BINS[i+1])/2),
            count=int(use.sum()),gamma=float(np.mean(gamma[use])) if use.any() else None))
    return rows


def fit_variogram(rows):
    use=[r for r in rows if r['count']>=5 and r['hi']<=50]
    d=np.array([r['distance'] for r in use]);y=np.array([r['gamma'] for r in use]);weights=np.sqrt(np.minimum([r['count'] for r in use],50))
    def model(p,d):return p[0]**2+p[1]**2*(-np.expm1(-d*d/(2*p[2]**2)))
    fit=least_squares(lambda p:(model(p,d)-y)*weights,[.12,.45,20.],bounds=([0,0,3],[1.5,2,80]))
    nugget,sigma,length=map(float,fit.x)
    return dict(independent_log_sigma=nugget,correlated_log_sigma=sigma,correlation_length_px=length,
        definition='gamma(r)=jitter_sigma² + field_sigma²*(1-exp(-r²/(2*length²))), fitted only through 50px.',
        caution='Observed shell brightness, not isolated intrinsic soma gain; overlap, geometry and selection remain confounded.')


def within_crop_permutation(values,centers,groups,repetitions=2000):
    distance=pdist(centers);near=(distance>=8)&(distance<18);far=(distance>=25)&(distance<50)
    def statistic(v):
        gamma=.5*pdist(v[:,None],'sqeuclidean')
        return float(gamma[far].mean()-gamma[near].mean())
    rng=np.random.default_rng(44);observed=statistic(values);null=[]
    for _ in range(repetitions):
        permuted=values.copy()
        for group in np.unique(groups):
            indices=np.flatnonzero(groups==group);permuted[indices]=rng.permutation(values[indices])
        null.append(statistic(permuted))
    return dict(statistic='Mean gamma(25:50px) minus mean gamma(8:18px), gains permuted only within annotation crop.',
        observed=observed,null_95=np.percentile(null,[2.5,97.5]).tolist(),
        exploratory_one_sided_p=float((1+sum(np.array(null)>=observed))/(repetitions+1)),
        groups=int(len(np.unique(groups))),repetitions=repetitions,
        caution='Not a definitive hypothesis test: annotation selection and heterogeneous regions limit exchangeability.')


def coarse_trend(mean,background,allowed,sigma=64.):
    weights=gaussian_filter(allowed.astype(float),sigma)
    return gaussian_filter(np.where(allowed,np.maximum(mean-background,0.),0.),sigma)/np.maximum(weights,1e-6)


def log_shell_gain(image,centers,geometry,background,trend):
    shell=soma_light(image,centers,*geometry)['shell']
    bg=map_coordinates(background,centers.T,order=1)
    local=map_coordinates(trend,centers.T,order=1)
    return np.log(np.maximum(shell-bg,1.)/np.maximum(local,1.))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_correlated_gain')
    p.add_argument('--length-multiplier',type=float,default=1.)
    p.add_argument('--field-log-sigma',type=float,help='Fix gain parameters instead of refitting; requires --jitter-log-sigma')
    p.add_argument('--jitter-log-sigma',type=float)
    args=p.parse_args()
    if not np.isfinite(args.length_multiplier) or not 0<args.length_multiplier<=8:p.error('Length multiplier must be finite and in (0,8]')
    if (args.field_log_sigma is None)!=(args.jitter_log_sigma is None):p.error('Supply both gain sigmas together')
    if args.field_log_sigma is not None and any(not np.isfinite(v) or v<0 for v in [args.field_log_sigma,args.jitter_log_sigma]):p.error('Gain sigmas must be finite and nonnegative')
    args.output.mkdir(parents=True,exist_ok=False)
    seed=20261118
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:old={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_heterogeneous_trial_v2/synthetic.npz') as z:previous={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:mean=z['mean'].copy();q90=z['q90'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_realism_diagnostics_v2/measurements.npz') as z:ac=z['real_centers'].copy();indices=z['synthetic_indices'].copy()
    annotations=json.loads((ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json').read_text())
    labels=[a for a in annotations['annotations'] if a['confidence']=='confident' and not a['truncated']]
    real_geometry=[np.array([a['measurements'][k] for a in labels]) for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    geometry={k:old[k] for k in ['major_axis_px','minor_axis_px','orientation_deg']}
    synthetic_geometry=[geometry[k][indices] for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    centers=old['centers'];sc=centers[indices];allowed=old['neuron_support'];bg=old['background'];trend=coarse_trend(mean,bg,allowed)
    observed={name:log_shell_gain(im,ac,real_geometry,bg,trend) for name,im in [('real_frame',real),('real_mean',mean),('real_q90',q90)]}
    observed['previous']=log_shell_gain(previous['image'],sc,synthetic_geometry,bg,trend)
    curves={name:variogram(value,sc if name=='previous' else ac) for name,value in observed.items()}
    fitted=fit_variogram(curves['real_mean']);length=fitted['correlation_length_px']*args.length_multiplier
    conditional=within_crop_permutation(observed['real_mean'],ac,np.array([a['crop_id'] for a in labels]))
    sensitivity={}
    for sigma in [32,64,96]:
        alternative=coarse_trend(mean,bg,allowed,sigma)
        sensitivity[str(sigma)]=fit_variogram(variogram(log_shell_gain(mean,ac,real_geometry,bg,alternative),ac))
    field=smooth_gain_field(mean.shape,length,seed)
    fit=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text());settings=fit['selected']|dict(ring_fraction=1.,ring_center_fill=0.)
    ones=np.ones(mean.shape,np.float32);profile=dict(tissue=allowed,tissue_field=ones,cell_gain=ones)
    renderer=partial(shell_patch,peak_radius=1.,width=.36)
    scene_args=dict(profile=profile,settings=settings,centers=centers,bank=None,seed=seed,geometry=geometry,patch_renderer=renderer)
    neutral=build_empirical_scene(**scene_args,amplitudes=np.ones(len(centers)))
    mixture=previous['blur_mixture']
    def optics(signal):return effective_blur(signal,.2,.8,mixture)
    weight=gaussian_filter(allowed.astype(float),16.)
    reference_mean=gaussian_filter(optics(neutral['signal']),16.)/np.maximum(weight,1e-6)
    normalizer=trend/np.maximum(reference_mean,1e-6)
    feather=np.clip(distance_transform_edt(allowed)/2.,0,1)
    def cells_for(amplitudes):
        scene=build_empirical_scene(**scene_args,amplitudes=amplitudes)
        return (optics(scene['signal'])*normalizer*feather).astype(np.float32)
    # Fit the observable variogram, not unobserved biological gain parameters.
    # Common field/noise seed keeps candidate comparisons controlled.
    target=np.array([r['gamma'] for r in curves['real_mean']]);use=np.array([r['hi']<=50 and r['count']>=5 for r in curves['real_mean']])
    weights=np.sqrt(np.minimum([r['count'] for r in curves['real_mean']],50))[use]
    rows=[];best=None
    for field_sigma in ([args.field_log_sigma] if args.field_log_sigma is not None else [.35,.55,.75]):
        for jitter_sigma in ([args.jitter_log_sigma] if args.jitter_log_sigma is not None else [.10,.20,.35]):
            amplitudes=cell_amplitudes(field,centers,field_sigma,jitter_sigma,seed+1)
            cells=cells_for(amplitudes);gain=log_shell_gain(bg+cells,sc,synthetic_geometry,bg,trend);curve=variogram(gain,sc)
            predicted=np.array([r['gamma'] for r in curve]);loss=float(np.mean((np.log(np.maximum(predicted[use],1e-4)/np.maximum(target[use],1e-4))*weights)**2)/np.mean(weights**2))
            row=dict(field_log_sigma=field_sigma,jitter_log_sigma=jitter_sigma,loss=loss);rows.append(row)
            if best is None or loss<best[0]:best=(loss,row,amplitudes,cells,gain,curve)
            print(json.dumps(row),flush=True)
    _,selected,amplitudes,cells,gain,curve=best
    # Shuffle exactly the same amplitudes: identical marginal distribution and
    # overall cell-power sum, different spatial correlation.
    independent=amplitudes[np.random.default_rng(seed+2).permutation(len(centers))]
    iid_cells=cells_for(independent)
    curves['correlated_clean']=curve;curves['shuffled_clean']=variogram(log_shell_gain(bg+iid_cells,sc,synthetic_geometry,bg,trend),sc)
    noise,_=correlated_noise(mean.shape,seed+3);noise*=previous['noise_sigma']
    clean=bg+cells;iid_clean=bg+iid_cells
    image=np.rint(np.clip(clean+noise,0,65535)).astype(np.uint16);iid_image=np.rint(np.clip(iid_clean+noise,0,65535)).astype(np.uint16)
    curves['correlated']=variogram(log_shell_gain(image,sc,synthetic_geometry,bg,trend),sc)
    curves['shuffled']=variogram(log_shell_gain(iid_image,sc,synthetic_geometry,bg,trend),sc)
    np.testing.assert_array_equal(np.sort(amplitudes),np.sort(independent))
    assert not cells[old['exclusion_mask']].any()
    np.testing.assert_array_equal(clean,bg+cells)
    # Diagnostic: demonstrate how the old local normalizer suppresses the field.
    neutral_cells=optics(neutral['signal'])*normalizer*feather
    old_local=gaussian_filter(cells,16.)/np.maximum(weight,1e-6)
    erased=cells*trend/np.maximum(old_local,1e-6)
    curves['renormalized_correlated']=variogram(log_shell_gain(bg+erased,sc,synthetic_geometry,bg,trend),sc)
    for name,im in dict(correlated=image,correlated_clean=clean,shuffled=iid_image,shuffled_clean=iid_clean,
        gain_field=np.exp(selected['field_log_sigma']*field).astype(np.float32),background=bg).items():tifffile.imwrite(args.output/(name+'.tiff'),im)
    np.savez_compressed(args.output/'synthetic.npz',image=image,clean=clean,cell_signal=cells,background=bg,shuffled_image=iid_image,shuffled_clean=iid_clean,
        labels=old['labels'],centers=centers,instance_offsets=old['instance_offsets'],instance_pixel_index=old['instance_pixel_index'],
        amplitudes=amplitudes,shuffled_amplitudes=independent,log_field=field.astype(np.float32),normalizer=normalizer.astype(np.float32),**geometry)
    report=dict(seed=seed,cell_count=len(centers),generated_field_length_px=length,length_multiplier=args.length_multiplier,gain_parameters_fixed=args.field_log_sigma is not None,observable_fit=fitted,within_crop_permutation=conditional,trend_scale_sensitivity=sensitivity,selected=selected,candidates=rows,variograms=curves,
        log_gain_variance={k:float(np.var(v)) for k,v in observed.items()},
        method='Per-cell multiplier exp(field_sigma*G(center)+jitter_sigma*epsilon), normalized only globally. Gaussian random field, continuous across mask; per-cell amplitude held constant over the whole soma.',
        photometry='Coarse real mean minus background on sigma64 sets envelope. Local sigma16 normalizer calculated from neutral unit-amplitude scene only, before assigning gains.',
        control='Exact permutation of the same per-cell gains; same geometry, background, blur, and noise realization.',
        checks=dict(same_amplitude_histogram=True,no_cell_signal_in_exclusion=True,unchanged_complete_GT=True,finite=bool(np.isfinite(clean).all())),
        limitations=['98 annotated visible somas are selected; nearby pairs are dependent and shortest-distance bin contains only 9 real pairs.',
            'The rising observed variogram supports spatial dependence, but anatomical populations, local illumination, activity and optical blur are not separated.',
            'Fitting and evaluation use one frame/recording and one random field realization; selected parameters are exploratory, not independently validated.',
            'Observable correlation length is not the exact biological gain length; cell overlap and fluorescence extraction affect it.',
            'Soma shells are still individually uniform; angular shell variation is not introduced here.',
            'The two variants can have different global rendered intensity despite identical cell-gain histograms because local overlap and normalizer vary with position.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5])
    fig,axs=plt.subplots(4,1,figsize=(17,13),constrained_layout=True)
    for ax,im,title in zip(axs,[real,previous['image'],iid_image,image],['Real frame 500','Previous independent wide gains','Matched shuffled gains; frozen normalization','Locally correlated gains + small independent jitter']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_axis_off()
    fig.savefig(args.output/'full_comparison.png',dpi=145);plt.close(fig)
    rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(4,4,figsize=(15,14),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,previous['image'],iid_image,image]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Previous','Same gains shuffled','Correlated gains'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name,color in [('real_frame','#222222'),('real_mean','#4987b0'),('previous','#cc9049'),('shuffled','#ab78b1'),('correlated','#43987e'),('renormalized_correlated','#d06b71')]:
        curve=curves[name];axs[0].plot([r['distance'] for r in curve],[r['gamma'] for r in curve],'-o',ms=3,label=name,color=color)
    axs[0].set(xlim=(0,80),xlabel='Cell separation (px)',ylabel='0.5 * mean squared log-brightness difference');axs[0].legend(fontsize=7)
    axs[1].hist(amplitudes,bins=40,histtype='step',density=True,label='Correlated');axs[1].hist(independent,bins=40,histtype='step',density=True,ls='--',label='Shuffled: exact same histogram');axs[1].set(xlabel='Intrinsic per-cell gain',ylabel='Density');axs[1].legend(fontsize=8)
    fig.savefig(args.output/'correlation_diagnostics.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,4),constrained_layout=True);artist=ax.imshow(np.exp(selected['field_log_sigma']*field).T,cmap='magma');fig.colorbar(artist,ax=ax,label='Correlated gain (independent cell jitter not shown)');ax.scatter(centers[:,0],centers[:,1],s=.1,color='cyan',alpha=.1);ax.set_title('Generated smooth gain field, sampled once at each soma center');fig.savefig(args.output/'gain_field.png',dpi=145);plt.close(fig)
    print(json.dumps(dict(observable_fit=fitted,selected=selected),indent=2))


if __name__=='__main__':main()
