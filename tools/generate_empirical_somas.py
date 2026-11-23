"""Generate a Plane20 trial using uniformly sampled annotated ellipse geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from compare_packing import render_centers, contact_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,default=ROOT/'outputs/plane20_trial')
    p.add_argument('--shapes',type=Path,default=ROOT/'outputs/soma_analysis/empirical_shapes.npz')
    p.add_argument('--packing',type=Path,default=ROOT/'outputs/plane20_packing/variant_4.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_empirical')
    p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    fit=json.loads((args.fit/'report.json').read_text());settings=fit['selected'].copy()
    with np.load(args.fit/'profile.npz') as z:profile={k:z[k] for k in z.files}
    with np.load(args.shapes) as z:bank={k:z[k] for k in z.files}
    with np.load(args.fit/'frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(args.packing) as z:
        centers=z['centers'].copy();previous=z['image'].copy()
    settings['cell_count']=len(centers)
    scene=build_empirical_scene(profile,settings,centers,bank,args.seed)
    result=render_centers(profile,settings,centers,args.seed,scene=scene)
    data={**result,**{k:v for k,v in scene.items() if k not in ['signal','labels']},
          'real_frame':real,'shape_yx':np.array(real.shape),'seed':np.array(args.seed)}
    np.savez_compressed(args.output/'synthetic.npz',**data)
    tifffile.imwrite(args.output/'synthetic.tiff',result['image'])
    tifffile.imwrite(args.output/'labels.tiff',result['labels'])
    interior=binary_erosion(profile['tissue'],iterations=15)
    metrics=contact_metrics(result,interior,real.shape)
    for label,im in [('real',real),('previous',previous),('empirical',result['image'])]:
        metrics[label+'_mean_tissue_adu']=float(im[interior].mean())
    metrics['intrinsic_contrast_adu']=float(np.std((result['clean']-gaussian_filter(result['clean'],4))[profile['tissue']]))
    metrics['sampled_median_major_px']=float(np.median(scene['major_axis_px']))
    metrics['sampled_median_minor_px']=float(np.median(scene['minor_axis_px']))
    metrics['sampled_median_ratio']=float(np.median(scene['major_axis_px']/scene['minor_axis_px']))
    metrics['sampled_median_diameter_px']=float(np.median(np.sqrt(scene['major_axis_px']*scene['minor_axis_px'])))
    offsets=result['instance_offsets'];pixels=result['instance_pixel_index']
    diameters=[];ratios=[]
    for i in range(len(centers)):
        yy,xx=np.unravel_index(pixels[offsets[i]:offsets[i+1]],real.shape)
        if not len(xx):raise ValueError('Empty instance footprint')
        cov=np.cov(np.stack([xx,yy]),ddof=0)+np.eye(2)/12
        eig=np.linalg.eigvalsh(cov)
        diameters.append(2*np.sqrt(len(xx)/np.pi));ratios.append(np.sqrt(eig[1]/eig[0]))
    metrics['raster_median_ratio']=float(np.median(ratios))
    report=dict(seed=args.seed,cell_count=len(centers),shape_yx=list(real.shape),settings=settings,
        empirical_bank_sha256=hashlib.sha256(args.shapes.read_bytes()).hexdigest(),
        empirical_bank_size=len(bank['numbers']),selection_bias_correction=False,
        inherited_geometry_fields_not_used=['sigma_px','size_cv','eccentricity'],
        sampling='Uniform joint empirical bootstrap of major/minor axes and orientation from 98 complete annotated somas.',
        geometry='Explicit filled ellipse, independent of solid/ring fluorescence type. Full rotated covariance for emission; solid/ring widths scaled separately to same outer half-maximum ellipse.',
        packing=str(args.packing),metrics=metrics,
        limitations=['One-frame exploratory synthesis, no independent validation.',
            'Cell count and tissue/density maps inherited; interhemispheric exclusion region not corrected.',
            'Annotations may favor visible somas; no selection-bias correction applied.',
            'Global empirical orientation distribution, no location-conditioned morphology or tissue deformation.',
            'Same 50% ring fraction, noise and optical blur as prior trial; photometry not re-estimated.',
            'Subpixel continuous ellipse parameters and raster mask moments differ due to pixelization and clipping.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5])
    fig,axs=plt.subplots(3,1,figsize=(17,12),constrained_layout=True)
    for ax,im,title in zip(axs,[real,previous,result['image']],
        ['Real frame 500',f'Previous dense synthetic: nearly round, {len(centers)} cells',f'Empirical soma geometry: {len(centers)} cells, no selection correction']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(args.output/'full_comparison.png',dpi=160);plt.close(fig)
    # Systematically distributed regions, not just the bright calibration crop.
    rois=[(250,410,90,250),(600,760,100,260),(930,1090,90,250),(1150,1310,90,250)]
    fig,axs=plt.subplots(3,len(rois),figsize=(15,11),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,previous,result['image']]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y={y0}:{y1}, X={x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Previous','Empirical'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for values,name,color in [(bank['equivalent_diameter_px'],'Annotated polygons','#d3955d'),(diameters,'Synthetic raster ellipses','#4c889c')]:
        axs[0].hist(values,bins=np.linspace(3,14,30),histtype='step',density=True,label=name,lw=2,color=color)
    for values,name,color in [(bank['major_axis_px']/bank['minor_axis_px'],'Annotated','#d3955d'),(ratios,'Synthetic raster','#4c889c')]:
        axs[1].hist(values,bins=np.linspace(1,1.9,30),histtype='step',density=True,label=name,lw=2,color=color)
    axs[0].set(xlabel='Equivalent diameter (px)',ylabel='Density');axs[1].set(xlabel='Major / minor',ylabel='Density')
    for ax in axs:ax.legend()
    fig.savefig(args.output/'geometry_check.png',dpi=170);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
