"""Estimate masked movie background and add parametric soma fluorescence.

Experimental Plane20 path; never changes annotations or frozen calibration.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from annotate_regions import load_image,validate,compute_mask
from empirical_morphology import build_empirical_scene
from soma_shell_model import sample_morphology,shell_patch
from calibrated_simcalc import detector


def interpolate_background(image,mask,sigma=8.,step=4):
    """Masked normalized smoothing, then discrete harmonic extension.

    A coarse grid limits cost; reflection at image edges gives a zero normal
    derivative. No unmasked pixel intensity is used by the estimator.
    """
    if image.shape!=mask.shape or not mask.any():raise ValueError('Nonempty same-shape mask required')
    weight=gaussian_filter(mask.astype(float),sigma)
    smooth=gaussian_filter(np.where(mask,image,0.).astype(float),sigma)/np.maximum(weight,1e-12)
    ys=np.unique(np.r_[np.arange(0,image.shape[0],step),image.shape[0]-1])
    xs=np.unique(np.r_[np.arange(0,image.shape[1],step),image.shape[1]-1])
    anchors=mask[np.ix_(ys,xs)]&(weight[np.ix_(ys,xs)]>.15)
    values=smooth[np.ix_(ys,xs)].copy()
    if not anchors.any():raise ValueError('Mask too small for interpolation grid')
    unknown=~anchors;index=np.full(anchors.shape,-1,int);index[unknown]=np.arange(unknown.sum())
    row=[];col=[];data=[];rhs=np.zeros(unknown.sum())
    for y,x in np.argwhere(unknown):
        i=index[y,x];degree=0
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny,nx=y+dy,x+dx
            if not(0<=ny<len(ys) and 0<=nx<len(xs)):continue
            # Nonuniform final edge interval: finite-volume conductance.
            conductance=1./abs((ys[ny]-ys[y]) if dy else (xs[nx]-xs[x]))
            degree+=conductance
            if anchors[ny,nx]:rhs[i]+=conductance*values[ny,nx]
            else:row.append(i);col.append(index[ny,nx]);data.append(-conductance)
        row.append(i);col.append(i);data.append(degree)
    if unknown.any():
        matrix=coo_matrix((data,(row,col)),shape=(unknown.sum(),unknown.sum())).tocsr()
        values[unknown]=spsolve(matrix,rhs)
    gy,gx=np.meshgrid(np.interp(np.arange(image.shape[0]),ys,np.arange(len(ys))),np.interp(np.arange(image.shape[1]),xs,np.arange(len(xs))),indexing='ij')
    field=map_coordinates(values,[gy,gx],order=1,mode='nearest')
    # Use the smooth observed field in known regions, harmonic field elsewhere.
    # Coarse-grid interpolation already makes their shared boundary continuous.
    return field.astype(np.float32),dict(smoothing_sigma_px=sigma,grid_step_px=step,anchor_count=int(anchors.sum()))


def complete_contours(centers,geometry,allowed,blur,peak,width):
    """Reject whole intersecting instances, never cut soma masks at the edge."""
    keep=[]
    for i,center in enumerate(centers):
        origin,kernel,footprint=shell_patch(center,geometry['major_axis_px'][i],geometry['minor_axis_px'][i],geometry['orientation_deg'][i],True,0.,blur,peak_radius=peak,width=width)
        y,x=np.nonzero(footprint);y+=origin[0];x+=origin[1]
        if len(y) and np.all((y>=0)&(y<allowed.shape[0])&(x>=0)&(x<allowed.shape[1])) and allowed[y,x].all():keep.append(i)
    return np.asarray(keep,int)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=ROOT/'annotations/regions_q90_72560cae13ef.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_masked_background')
    p.add_argument('--seed',type=int,default=20261116)
    args=p.parse_args()
    raw=args.annotations.read_bytes();doc=json.loads(raw);q90=load_image(Path(doc['dataset']['source_path']))
    validate(doc,dict(image_sha256=hashlib.sha256(q90.tobytes()).hexdigest(),shape_yx=list(q90.shape)))
    exclusion=compute_mask(q90,doc)
    if not exclusion.any() or exclusion.all():raise ValueError('Need both selected background and unselected pixels')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'points.json').write_bytes(raw)
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:
        mean=z['mean'].astype(np.float32);assert np.array_equal(q90,z['q90'])
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    with np.load(ROOT/'outputs/plane20_trial/profile.npz') as z:profile={k:z[k] for k in z.files}
    fit=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text())
    shell_report=json.loads((ROOT/'outputs/plane20_shell_fit_v2/report.json').read_text())
    morph_report=json.loads((ROOT/'outputs/soma_analysis/report.json').read_text())
    settings=fit['selected']|dict(ring_fraction=1.,ring_center_fill=0.)
    shell=shell_report['selected_shell']
    background,bginfo=interpolate_background(mean,exclusion)
    distance=distance_transform_edt(~exclusion).astype(np.float32)
    # Spatial holdout: separated 24px blocks from the known-background region.
    yy,xx=np.indices(mean.shape);heldout=exclusion&(((yy//24)*7+xx//24)%11==0)
    test_background,_=interpolate_background(mean,exclusion&~heldout)
    errors=test_background[heldout]-mean[heldout]
    # Retain the established relaxed layout; remove cells conflicting with the new annotation.
    with np.load(ROOT/'outputs/plane20_shell_fit_v2/synthetic.npz') as z:
        centers=z['centers'].copy()
    original_count=len(centers)
    geometry=sample_morphology(morph_report,len(centers),args.seed)
    allowed=profile['tissue'].astype(bool)&~exclusion
    keep=complete_contours(centers,geometry,allowed,settings['blur_px'],shell['peak_radius'],shell['width'])
    centers=centers[keep];geometry={k:v[keep] for k,v in geometry.items()}
    if not len(keep):raise ValueError('No complete soma fits the mask')
    # Geometry and calibrated shell emission are reused. Remove the old fitted
    # fluorescence/background fields before constructing this additive scene.
    ones=np.ones(mean.shape,np.float32)
    scene_profile=profile|dict(tissue=allowed,tissue_field=ones,cell_gain=ones)
    scene=build_empirical_scene(scene_profile,settings,centers,None,args.seed,geometry=geometry,
        patch_renderer=partial(shell_patch,peak_radius=shell['peak_radius'],width=shell['width']))
    # Only coarse average residual light sets amplitude. No real cellular texture
    # is copied: the 16px field is divided by the synthetic 16px mean emission.
    weights=gaussian_filter(allowed.astype(float),16.)
    residual=np.maximum(mean-background,0.)
    envelope=gaussian_filter(np.where(allowed,residual,0.),16.)/np.maximum(weights,1e-6)
    local=gaussian_filter(scene['signal'],16.)/np.maximum(weights,1e-6)
    factor=np.divide(envelope,np.maximum(local,1e-6),out=np.zeros_like(envelope),where=local>1e-5)
    cells=(scene['signal']*factor).astype(np.float32)
    # Mid-shell contours are strictly excluded. Optical emission outside the
    # contour is softly attenuated over 2 px and exactly zero in selected areas.
    feather=np.clip(distance_transform_edt(allowed)/2.,0,1)
    cells*=feather
    clean=background+cells
    noisy=detector(clean,profile['background'],settings['shot_gain_adu'],settings['read_variance_adu2'],np.random.default_rng(args.seed+1),settings['excess_variance_fraction'])
    synthetic=np.rint(np.clip(noisy,0,65535)).astype(np.uint16)
    labels=scene['labels'];assert not labels[exclusion].any();assert not cells[exclusion].any()
    assert np.isfinite(clean).all() and np.all(cells>=0)
    for name,array in dict(background=background,cell_signal=cells,synthetic_clean=clean,synthetic=synthetic,
        annotation_labels=labels,exclusion_mask=exclusion.astype(np.uint8)*255,neuron_support=allowed.astype(np.uint8)*255,
        distance_to_background=distance).items():tifffile.imwrite(args.output/(name+'.tiff'),array)
    np.savez_compressed(args.output/'synthetic.npz',image=synthetic,clean=clean,background=background,cell_signal=cells,
        exclusion_mask=exclusion,neuron_support=allowed,distance_to_background=distance,centers=centers,
        labels=labels,instance_offsets=scene['instance_offsets'],instance_pixel_index=scene['instance_pixel_index'],
        **geometry,seed=np.asarray(args.seed))
    near=allowed&(distance<=30);far=allowed&(distance>60)
    report=dict(seed=args.seed,annotation_path=str(args.annotations),annotation_sha256=hashlib.sha256(raw).hexdigest(),
        image_sha256=doc['dataset']['image_sha256'],shape_yx=list(mean.shape),mask_points=len(doc['points']),
        background_pixels=int(exclusion.sum()),background_fraction=float(exclusion.mean()),
        background_source='Exact temporal mean of all 1000 frames, measured only inside annotated q90 exclusion mask.',
        interpolation='Masked normalized Gaussian smoothing then harmonic (Laplace) extension with zero-normal-gradient image edges; bilinear coarse-grid reconstruction.',
        **bginfo,background_range_adu=[float(background.min()),float(background.max())],
        spatial_holdout=dict(definition='24px tiles, 1/11 pattern, withheld before all smoothing/interpolation',pixels=int(heldout.sum()),
            rmse_adu=float(np.sqrt(np.mean(errors**2))),median_absolute_error_adu=float(np.median(abs(errors))),p90_absolute_error_adu=float(np.percentile(abs(errors),90))),
        original_cell_count=original_count,cell_count=len(centers),rejected_cells=int(original_count-len(centers)),
        near_edge_neuron_pixels=int(near.sum()),far_interior_neuron_pixels=int(far.sum()),max_extrapolation_distance_px=float(distance[allowed].max()),
        shell_parameters=shell,selection_bias_correction=False,cell_light='Nonnegative fitted shell signal; 16px smooth residual temporal mean sets local amplitude; exactly zero within exclusion.',
        composition='clean = background + cell_signal; effective inherited detector noise applied once afterwards.',
        checks=dict(nonempty_complete_instances=int((np.diff(scene['instance_offsets'])>0).sum()),
            fully_occluded_in_ownership_labels=int(len(centers)-len(np.unique(labels[labels>0]))),instance_pixels_in_exclusion=int(exclusion.ravel()[scene['instance_pixel_index']].sum()),
            cell_signal_in_exclusion_max=float(cells[exclusion].max()),all_finite=bool(np.isfinite(clean).all())),
        limitations=['An unselected pixel is not proven neuron tissue: candidate support also intersects the previous coarse tissue map.',
            'Interior background cannot be identified from edges alone; harmonic extrapolation suppresses unsupported internal gradients.',
            'Intensity-threshold-selected background can underestimate diffuse light near bright cell boundaries; movie mean reduces noise, not selection bias.',
            'Cell layout/density prior inherited, whole intersecting somas removed rather than deformed; the packing near boundaries is not refitted.',
            'Local cellular brightness uses real smooth residual mean; temporal dynamics and per-cell spatial gain distributions are not fitted.',
            'Detector noise inherited; q90 is an aggregate whereas the result is one simulated exposure.',
            'Emission tails are attenuated at the mask edge; this is a hard anatomical exclusion approximation, not a PSF model across the boundary.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lo,hi=np.percentile(real,[1,99.5])
    fig,axs=plt.subplots(5,1,figsize=(17,16),constrained_layout=True)
    panels=[(q90,'Real q90 + annotated background boundary'),(background,'Estimated diffuse background (temporal mean, ADU)'),
        (cells,'Synthetic soma fluorescence only (ADU)'),(synthetic,f'Background + {len(centers)} synthetic somas + detector noise'),(real,'Real frame 500 (same display range)')]
    for i,(ax,(im,title)) in enumerate(zip(axs,panels)):
        ax.imshow(im.T,cmap='gray',vmin=0 if i==2 else lo,vmax=hi-lo if i==2 else hi)
        if i==0:ax.contour(exclusion.T,levels=[.5],colors=['#5fe3c3'],linewidths=.5)
        ax.set_title(title);ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(args.output/'full_comparison.png',dpi=145);plt.close(fig)
    rois=[(320,480,90,250),(620,800,100,280),(970,1130,100,260),(1170,1330,80,240)]
    fig,axs=plt.subplots(4,4,figsize=(15,14),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,background,cells,synthetic]):
            ax=axs[row,col];roi=im[y0:y1,x0:x1];ax.imshow(roi.T,cmap='gray',vmin=0 if row==2 else lo,vmax=hi-lo if row==2 else hi)
            ax.contour(exclusion[y0:y1,x0:x1].T,levels=[.5],colors=['#5fe3c3'],linewidths=.5)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y {y0}:{y1}, X {x0}:{x1}')
            if col==0:ax.set_ylabel(['Real frame 500','Background','Synthetic cells','Sum + noise'][row])
    fig.savefig(args.output/'edge_crops.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(16,7),constrained_layout=True)
    for ax,im,title,cmap in [(axs[0],background,'Interpolated background, ADU','magma'),(axs[1],distance,'Distance from observed background, px (not an error estimate)','viridis')]:
        artist=ax.imshow(im.T,cmap=cmap);fig.colorbar(artist,ax=ax,shrink=.7);ax.set_title(title)
        ax.contour(exclusion.T,levels=[.5],colors=['cyan'],linewidths=.35)
    fig.savefig(args.output/'background_diagnostics.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(16,10),constrained_layout=True)
    for ax,im,title in zip(axs,[real,clean,synthetic],['Real frame 500','Synthetic cells + interpolated background (no noise)','Same composition + inherited detector noise']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_axis_off()
    fig.savefig(args.output/'result.png',dpi=140);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['background_fraction','background_range_adu','spatial_holdout','cell_count','max_extrapolation_distance_px','checks']},indent=2))


if __name__=='__main__':main()
