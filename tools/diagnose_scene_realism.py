"""Compare local soma photometry, directional texture and temporal noise."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates,distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from soma_shell_model import radial_profiles,normalize_profiles


def soma_light(image,centers,major,minor,angle):
    r,curves=radial_profiles(image,centers,major,minor,angle)
    _,core,contrast=normalize_profiles(r,curves)
    shell=np.max(curves[:,(r>=.65)&(r<=1.35)],axis=1)
    return dict(core=core,contrast=contrast,shell=shell)


def neighbor_spread(values,centers,radius=30):
    pairs=cKDTree(centers).query_pairs(radius,output_type='ndarray')
    v=np.asarray(values);raw=v[pairs[:,0]]-v[pairs[:,1]]
    positive=(v[pairs[:,0]]>=5)&(v[pairs[:,1]]>=5)
    delta=np.log(v[pairs[positive,0]])-np.log(v[pairs[positive,1]])
    return dict(pairs=int(len(pairs)),raw_difference_variance_adu2=float(np.var(raw)),
        positive_log_pairs=int(positive.sum()),pair_log_difference_variance=float(np.var(delta)),
        pair_abs_log_difference_median=float(np.median(abs(delta))),
        implied_independent_lognormal_sigma=float(np.sqrt(np.mean(delta**2)/2)))


def directional_curves(image,valid,maxlag=8):
    hp=np.asarray(image,float)-gaussian_filter(np.asarray(image,float),6.)
    curves=[]
    for axis in (0,1):
        curve=[]
        for lag in range(maxlag+1):
            a=[slice(None),slice(None)];b=a.copy()
            if lag:a[axis]=slice(None,-lag);b[axis]=slice(lag,None)
            a,b=tuple(a),tuple(b);mask=valid[a]&valid[b]
            curve.append(float(np.mean(hp[a][mask]*hp[b][mask])))
        curve=np.array(curve);curves.append(curve/max(curve[0],1e-8))
    return np.asarray(curves)


def correlation_length(curve):
    below=np.flatnonzero(curve<=.5)
    if not len(below):return float(len(curve)-1)
    i=below[0]
    return float(i-1+(.5-curve[i-1])/(curve[i]-curve[i-1])) if i else 0.


def texture_tiles(image,valid,step=48,size=64):
    rows=[]
    for y in range(0,image.shape[0]-size+1,step):
        for x in range(0,image.shape[1]-size+1,step):
            m=valid[y:y+size,x:x+size]
            if m.mean()<.75:continue
            im=image[y:y+size,x:x+size]
            curves=directional_curves(im,m)
            ly,lx=map(correlation_length,curves)
            rows.append(dict(y=y,x=x,horizontal_length_px=ly,vertical_length_px=lx,
                vertical_over_horizontal=lx/max(ly,.01),mean_adu=float(im[m].mean()),
                highpass_std=float(np.std((im-gaussian_filter(im.astype(float),6.))[m]))))
    return rows


def summary(v):
    v=np.asarray(v);return dict(zip(['p10','median','p90'],map(float,np.percentile(v,[10,50,90]))))


def paired_anisotropy(rows_a,rows_b):
    delta=np.array([a['vertical_over_horizontal']-b['vertical_over_horizontal'] for a,b in zip(rows_a,rows_b)])
    blocks=np.array([a['y']//144 for a in rows_a]);ids=np.unique(blocks);rng=np.random.default_rng(12);medians=[]
    for _ in range(2000):
        indices=np.concatenate([np.flatnonzero(blocks==j) for j in rng.choice(ids,len(ids),replace=True)])
        medians.append(np.median(delta[indices]))
    return dict(median=float(np.median(delta)),block_bootstrap_95=np.percentile(medians,[2.5,97.5]).tolist(),blocks=len(ids),
        caution='Only 7 longitudinal blocks; shape and optical effects remain confounded.')


def noise_map(movie_path):
    # Spread pairs across recording; consecutive differences contain camera noise
    # and any fast activity/motion. Robust MAD suppresses rare excursions.
    starts=np.arange(0,980,40)
    with tifffile.TiffFile(movie_path) as tif:
        differences=np.array([(tif.pages[int(i+1)].asarray().astype(np.float32)-tif.pages[int(i)].asarray())/np.sqrt(2.) for i in starts])
    center=np.median(differences,axis=0)
    sigma=1.482602218*np.median(abs(differences-center),axis=0)
    return sigma,differences,starts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_realism_diagnostics');args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    with np.load(ROOT/'outputs/plane20_masked_background/synthetic.npz') as z:syn={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:mean=z['mean'].copy();q90=z['q90'].copy();median=z['median'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    annotations=json.loads((ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json').read_text())
    labels=[a for a in annotations['annotations'] if a['confidence']=='confident' and not a['truncated']]
    ac=np.array([[a['measurements']['centroid_y']-.5,a['measurements']['centroid_x']-.5] for a in labels])
    axes=[np.array([a['measurements'][k] for a in labels]) for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    # One synthetic soma per annotation: nearest spatial context, without reuse.
    tree=cKDTree(syn['centers']);near=tree.query(ac,k=20)[1];chosen=[]
    for candidates in near:
        chosen.append(next(int(i) for i in candidates if i not in chosen))
    chosen=np.asarray(chosen);sc=syn['centers'][chosen]
    values={};spreads={}
    for name,im in [('real_frame',real),('real_mean',mean),('real_q90',q90),('synthetic',syn['image']),('synthetic_clean',syn['clean'])]:
        actual=name.startswith('real');indices=chosen
        centers=ac if actual else sc
        geom=axes if actual else [syn[k][indices] for k in ['major_axis_px','minor_axis_px','orientation_deg']]
        values[name]=soma_light(im,centers,*geom)
        spreads[name]={k:neighbor_spread(values[name][k],centers) for k in ['contrast','shell']}
    valid=distance_transform_edt(syn['neuron_support'])>4
    textures={name:texture_tiles(im,valid) for name,im in [('real_frame',real),('real_mean',mean),('real_median',median),('real_q90',q90),('synthetic',syn['image']),('synthetic_clean',syn['clean'])]}
    curves={name:directional_curves(im,valid) for name,im in [('real_mean',mean),('real_q90',q90),('synthetic_clean',syn['clean'])]}
    sigma,differences,starts=noise_map(ROOT/'data/Plane_Aligned_20.tiff')
    sigma_smooth=np.sqrt(gaussian_filter(sigma**2,8.))
    # Noise correlation in safely empty areas; an upper bound if motion remains.
    empty=distance_transform_edt(syn['exclusion_mask'])>8
    correlations=[]
    for axis in (0,1):
        a=[slice(None)]*3;b=a.copy();a[axis+1]=slice(None,-1);b[axis+1]=slice(1,None);a,b=tuple(a),tuple(b)
        ma=[slice(None)]*2;mb=ma.copy();ma[axis]=slice(None,-1);mb[axis]=slice(1,None);m=empty[tuple(ma)]&empty[tuple(mb)]
        av=differences[a][:,m].ravel();bv=differences[b][:,m].ravel();limit=np.percentile(abs(av),99)
        good=(abs(av)<limit)&(abs(bv)<limit);correlations.append(float(np.corrcoef(av[good],bv[good])[0,1]))
    report=dict(axis_convention='Displayed vertical = native X = array axis 1; displayed horizontal = native Y = array axis 0.',
        annotated_somas=len(labels),comparison='Same number of synthetic somas from nearest spatial contexts; real annotations selected for visibility, so not unbiased.',
        light={name:{k:summary(v) for k,v in data.items()} for name,data in values.items()},neighbor_spreads=spreads,
        directional_texture={name:dict(tiles=len(rows),**{k:summary([r[k] for r in rows]) for k in ['horizontal_length_px','vertical_length_px','vertical_over_horizontal','highpass_std']}) for name,rows in textures.items()},
        noise=dict(pair_starts=starts.tolist(),method='Temporal MAD of 25 dispersed adjacent-frame differences / sqrt(2); includes motion/activity.',
            empty_sigma_adu=summary(sigma_smooth[empty]),neuron_sigma_adu=summary(sigma_smooth[valid]),
            adjacent_pixel_correlation_empty=dict(horizontal_native_y=correlations[0],vertical_native_x=correlations[1])),
        cautions=['Texture anisotropy includes real cell shape, alignment, projected processes, blur and motion; not an isolated PSF measurement.',
            'A single plane without point-source calibration cannot establish astigmatism.',
            'Temporal mean may broaden due to residual motion; compare median, q90 and single frames.',
            'Neighbor variance combines cell amplitudes, geometry, overlap, annotation selection, noise and activity.'])
    report['paired_anisotropy_difference']=paired_anisotropy(textures['real_mean'],textures['synthetic_clean'])
    report['ratios']={k:spreads['real_frame'][k]['raw_difference_variance_adu2']/spreads['synthetic'][k]['raw_difference_variance_adu2'] for k in ['contrast','shell']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'measurements.npz',noise_sigma=sigma,smooth_noise_sigma=sigma_smooth,real_centers=ac,synthetic_indices=chosen,**{name+'_'+k:v for name,vs in values.items() for k,v in vs.items()},**{name+'_curves':v for name,v in curves.items()})
    for name,rows in textures.items():
        with (args.output/(name+'_tiles.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for name,color in [('real_frame','#333333'),('real_mean','#528ab9'),('synthetic','#cf934b'),('synthetic_clean','#6fbd98')]:
        axs[0,0].hist(values[name]['contrast'],bins=np.linspace(0,180,30),density=True,histtype='step',lw=2,label=name,color=color)
        v=values[name]['shell'];centers=ac if name.startswith('real') else sc;pairs=cKDTree(centers).query_pairs(30,output_type='ndarray');delta=np.log(v[pairs[:,0]])-np.log(v[pairs[:,1]])
        axs[0,1].hist(delta,bins=np.linspace(-3,3,30),density=True,histtype='step',lw=2,label=name,color=color)
    axs[0,0].set(xlabel='Shell minus core (ADU)',ylabel='Density');axs[0,1].set(xlabel='log shell brightness difference, cells <=30px apart',ylabel='Density')
    for name,color in [('real_mean','#333333'),('real_q90','#528ab9'),('synthetic_clean','#cf934b')]:
        for axis,style in [(0,'--'),(1,'-')]:axs[1,0].plot(curves[name][axis],style,color=color,label=name+(' horizontal' if axis==0 else ' vertical'))
    axs[1,0].axhline(.5,color='gray',lw=.7);axs[1,0].set(xlabel='Lag (px)',ylabel='High-pass spatial correlation')
    for name,color in [('real_mean','#333333'),('real_q90','#528ab9'),('synthetic_clean','#cf934b')]:
        axs[1,1].hist([r['vertical_over_horizontal'] for r in textures[name]],bins=np.linspace(.4,3,25),density=True,histtype='step',lw=2,label=name,color=color)
    axs[1,1].set(xlabel='Vertical/horizontal correlation half-width per tile',ylabel='Density')
    for ax in axs.flat:ax.legend(fontsize=7)
    fig.savefig(args.output/'photometry_and_anisotropy.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(16,10),constrained_layout=True)
    for ax,im,title,cmap in [(axs[0],mean,'Movie temporal mean','gray'),(axs[1],sigma_smooth,'Effective temporal noise sigma, ADU (includes activity and motion)','magma'),(axs[2],syn['background'],'Estimated background','magma')]:
        artist=ax.imshow(im.T,cmap=cmap);fig.colorbar(artist,ax=ax,shrink=.7);ax.set_title(title)
    fig.savefig(args.output/'noise_map.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
