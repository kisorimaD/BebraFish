"""Fit observable mid-shell light profiles; generate parametric morphology.

Geometry parameters come from annotation analysis, not empirical resampling.
The shell is an effective image model, not deconvolved membrane thickness.
"""
import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from empirical_morphology import build_empirical_scene
from soma_shell_model import sample_morphology,shell_patch,radial_profiles,normalize_profiles,profile_statistics
from compare_packing import render_centers,contact_metrics


def median_stats(rows):
    return {k:dict(p10=float(np.nanpercentile([r[k] for r in rows],10)),
                   median=float(np.nanmedian([r[k] for r in rows])),
                   p90=float(np.nanpercentile([r[k] for r in rows],90)),
                   n_finite=int(np.isfinite([r[k] for r in rows]).sum())) for k in rows[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_shell_fit')
    p.add_argument('--seed',type=int,default=20261115)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    annotations_path=ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json'
    annotations=json.loads(annotations_path.read_text())
    labels=[a for a in annotations['annotations'] if a['confidence']=='confident' and not a['truncated']]
    morph_report=json.loads((ROOT/'outputs/soma_analysis/report.json').read_text())
    assert hashlib.sha256(annotations_path.read_bytes()).hexdigest()==morph_report['input_annotation_sha256']
    fit=json.loads((ROOT/'outputs/plane20_trial/report.json').read_text());settings=fit['selected'].copy()
    with np.load(ROOT/'outputs/plane20_trial/profile.npz') as z:profile={k:z[k] for k in z.files}
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    assert hashlib.sha256(real.astype('<u2',copy=False).tobytes()).hexdigest()==annotations['dataset']['frame_sha256']
    with np.load(ROOT/'outputs/plane20_empirical/synthetic.npz') as z:previous={k:z[k] for k in z.files}
    centers=previous['centers'];settings['cell_count']=len(centers)
    geometry=sample_morphology(morph_report,len(centers),args.seed)
    ac=np.array([[a['measurements']['centroid_y'],a['measurements']['centroid_x']] for a in labels])
    aa=np.array([a['measurements']['major_axis_px'] for a in labels]);ab=np.array([a['measurements']['minor_axis_px'] for a in labels]);at=np.array([a['measurements']['orientation_deg'] for a in labels])
    radii,real_curves=radial_profiles(real,ac,aa,ab,at,coordinate_offset=.5)
    real_normalized,real_core,real_contrast=normalize_profiles(radii,real_curves)
    # Held-out spatial band, rather than random neighboring cells.
    validation=(ac[:,0]>=740)&(ac[:,0]<900)
    train=~validation
    tree=cKDTree(centers)
    edge=distance_transform_edt(profile['tissue'])
    cy,cx=np.rint(centers).astype(int).T
    def context_indices(mask):
        near=tree.query(ac[mask]-.5,k=8)[1].ravel();near=np.unique(near)
        return near[edge[cy[near],cx[near]]>12]
    eval_train=context_indices(train);eval_val=context_indices(validation)
    # Avoid shared synthetic evaluators across the separated spatial bands.
    eval_val=eval_val[~np.isin(eval_val,eval_train)]
    target_train=np.median(real_normalized[train],axis=0);target_val=np.median(real_normalized[validation],axis=0)
    use_r=(radii>=.1)&(radii<=1.6)
    real_statistics=profile_statistics(radii,real_curves,aa,ab)
    with (args.output/'annotated_light_statistics.csv').open('w') as f:
        rows=[dict(number=a['number'],split='validation' if v else 'train',**s) for a,v,s in zip(labels,validation,real_statistics)]
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    def extract(result,indices):
        _,curves=radial_profiles(result['image'],centers[indices],geometry['major_axis_px'][indices],geometry['minor_axis_px'][indices],geometry['orientation_deg'][indices],radii)
        norm,core,contrast=normalize_profiles(radii,curves)
        good=contrast>5
        return curves[good],norm[good],indices[good],contrast[good]
    def evaluate(result):
        tr,nt,it,ct=extract(result,eval_train);va,nv,iv,cv=extract(result,eval_val)
        pt,pv=np.median(nt,axis=0),np.median(nv,axis=0)
        loss=float(np.sqrt(np.mean((pt[use_r]-target_train[use_r])**2)))
        # Weak scale term complements profile shape, without post-hoc contrast remap.
        contrast_loss=float(abs(np.log(np.median(ct)/np.median(real_contrast[train]))))
        return dict(train_profile_rmse=loss,validation_profile_rmse=float(np.sqrt(np.mean((pv[use_r]-target_val[use_r])**2))),
                    train_contrast_log_error=contrast_loss,objective=loss+.08*contrast_loss,
                    n_train_profiles=len(it),n_validation_profiles=len(iv)),(pt,pv,tr,va,it,iv)
    # Control: parametric morphology, previous broad SimCalc emission.
    control_scene=build_empirical_scene(profile,settings,centers,None,args.seed,geometry=geometry)
    control=render_centers(profile,settings,centers,args.seed,scene=control_scene)
    control_metrics,control_curves=evaluate(control)
    candidates=[];best=None;best_scene=None;best_result=None;best_curves=None
    def candidate(peak,width,fill):
        nonlocal best,best_scene,best_result,best_curves
        start=time.monotonic()
        params=dict(peak_radius=peak,width=width,core_fill=fill)
        options=settings|dict(ring_fraction=1.,ring_center_fill=fill)
        scene=build_empirical_scene(profile,options,centers,None,args.seed,geometry=geometry,
                patch_renderer=partial(shell_patch,peak_radius=peak,width=width))
        result=render_centers(profile,options,centers,args.seed,scene=scene)
        values,curves=evaluate(result);row=params|values
        candidates.append(row)
        if best is None or row['objective']<best['objective']:
            best=row;best_scene=scene;best_result=result;best_curves=curves
        print(json.dumps(row|dict(seconds=round(time.monotonic()-start,2))),flush=True)
    for peak in [.9,1.,1.1]:
        for width in [.04,.10,.18,.28]:candidate(peak,width,0.)
    selected_peak,selected_width=best['peak_radius'],best['width']
    for fill in [.10,.20]:candidate(selected_peak,selected_width,fill)
    # A short local refinement, driven exclusively by training objective.
    peak,width,fill=best['peak_radius'],best['width'],best['core_fill']
    for dp,dw in [(-.05,0),(.05,0),(0,-.03),(0,.03)]:
        if width+dw>.005:candidate(peak+dp,width+dw,fill)
    # Bracket the width optimum; do not leave it at the initial grid edge.
    peak,fill=best['peak_radius'],best['core_fill']
    for width in [.34,.36,.38,.42,.5]:candidate(peak,width,fill)
    peak,width,fill=best['peak_radius'],best['width'],best['core_fill']
    for dp in [-.05,.05]:candidate(peak+dp,width,fill)
    _,_,train_curves,val_curves,train_idx,val_idx=best_curves
    control_data=control_curves
    curve_indices=np.r_[train_idx,val_idx];synthetic_curves=np.concatenate([train_curves,val_curves])
    syn_statistics=profile_statistics(radii,synthetic_curves,geometry['major_axis_px'][curve_indices],geometry['minor_axis_px'][curve_indices])
    control_idx=np.r_[control_data[4],control_data[5]];control_profiles=np.concatenate([control_data[2],control_data[3]])
    control_statistics=profile_statistics(radii,control_profiles,geometry['major_axis_px'][control_idx],geometry['minor_axis_px'][control_idx])
    normalized_syn,_,_=normalize_profiles(radii,synthetic_curves)
    normalized_control,_,_=normalize_profiles(radii,control_profiles)
    # Compare the previous empirical image at its own ellipse coordinates.
    _,old_curves=radial_profiles(previous['image'],centers[eval_train],previous['major_axis_px'][eval_train],previous['minor_axis_px'][eval_train],previous['orientation_deg'][eval_train],radii)
    old_norm,_,old_contrast=normalize_profiles(radii,old_curves);old_norm=old_norm[old_contrast>5]
    report=dict(seed=args.seed,n_annotations=len(labels),n_light_train=int(train.sum()),n_light_validation=int(validation.sum()),
        validation_definition='Native Y in [740,900) held out for light fitting; geometry distributions and coarse photometry use earlier data from entire frame, so not an independent-recording test.',
        geometry_sampling='Positive Normal diameter, shifted Gamma aspect ratio, doubled-angle von Mises orientation; parametric independent marginals, no empirical resampling or selection correction.',
        diameter_normal_parameters=next(m['parameters'] for m in morph_report['diameter_fit_comparison'] if m['name']=='normal'),
        ratio_minus_one_gamma_parameters=next(m['parameters'] for m in morph_report['aspect_ratio_excess_fit_comparison'] if m['name']=='gamma'),
        angle_vonmises_doubled=morph_report['orientation'],selection_bias_correction=False,
        annotation_sha256=hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        annotation_boundary='User-confirmed mid-shell contour, not external whole-soma surface. labels follow this contour and include the dark center.',
        selected_shell=best,candidates=candidates,control=control_metrics,
        real_light_statistics=median_stats(real_statistics),control_light_statistics=median_stats(control_statistics),new_light_statistics=median_stats(syn_statistics),
        psf_blur_fixed_px=settings['blur_px'],ring_fraction=1.,
        reference_profile_sha256=hashlib.sha256((ROOT/'outputs/plane20_trial/profile.npz').read_bytes()).hexdigest(),
        reference_detector={k:settings[k] for k in ['shot_gain_adu','read_variance_adu2','excess_variance_fraction']},
        limitations=['One-plane selected visible somas; no correction for selection bias.',
            'Gaussian shell sigma is pre-blur effective model width, not biological cytoplasm thickness.',
            'Angular median suppresses bright neighbors but does not remove unresolved neuropil or overlap; outer baseline and width remain confounded.',
            'Half-prominence width depends on outer troughs and neighbors; reported only for profiles with a local peak.',
            'Optical blur fixed at inherited 0.6px; intrinsic shell width and PSF are not separately identified.',
            'All fitted annotations have brighter peripheral profiles than their centers; ring fraction set to 1 for this selected population.',
            'Global distributions omit size/aspect dependence and spatially varying orientation.',
            'Cell locations, interhemispheric exclusion and tissue deformations unchanged.'])
    interior=binary_erosion(profile['tissue'],iterations=15)
    report['scene_metrics']=contact_metrics(best_result,interior,real.shape)
    report['scene_metrics']['intrinsic_highpass_std_adu']=float(np.std((best_result['clean']-gaussian_filter(best_result['clean'],4))[profile['tissue']]))
    report['scene_metrics']['mean_tissue_adu']=float(best_result['image'][interior].mean())
    report['scene_metrics']['sampled_diameter_median_px']=float(np.median(geometry['diameter_px']))
    report['scene_metrics']['sampled_axis_ratio_median']=float(np.median(geometry['aspect_ratio']))
    report['scene_metrics']['cell_texture_fraction']=float(best_result['cell_texture_fraction'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    final={**best_result,**{k:v for k,v in best_scene.items() if k not in ['signal','labels']},**geometry,
        'annotation_labels':best_result['labels'],'shape_yx':np.array(real.shape),'seed':np.array(args.seed),
        'metadata_json':np.asarray(json.dumps(report))}
    np.savez_compressed(args.output/'synthetic.npz',**final)
    np.savez_compressed(args.output/'parametric_broad_control.npz',**control,**geometry)
    np.savez_compressed(args.output/'light_profiles.npz',radii=radii,real_profiles=real_curves,
        train=train,validation=validation,new_profiles=synthetic_curves,new_profile_indices=curve_indices,
        old_profiles=old_curves,control_profiles=control_profiles)
    tifffile.imwrite(args.output/'synthetic.tiff',best_result['image'])
    tifffile.imwrite(args.output/'annotation_labels.tiff',best_result['labels'])
    lo,hi=np.percentile(real,[1,99.5])
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axs[0,0]
    for curves,title,color in [(real_normalized,'Real annotated','#242424'),(old_norm,'Previous empirical + broad light','#b389b0'),(normalized_control,'Parametric + broad control','#d39155'),(normalized_syn,'Parametric + fitted shell','#3c9e87')]:
        med=np.median(curves,axis=0);ax.plot(radii,med,label=title,color=color)
    ax.axvline(1,color='gray',ls='--',lw=1);ax.set(xlabel='Elliptical radius / annotation mid-shell radius',ylabel='(I - core) / (shell peak - core)',title='Observed angular-median radial profiles');ax.legend(fontsize=8)
    ax=axs[0,1]
    for values,title,color in [(train,'Real train','#333333'),(validation,'Real held-out spatial band','#777777')]:
        cur=real_normalized[values];med=np.median(cur,axis=0);low,high=np.percentile(cur,[25,75],axis=0);ax.plot(radii,med,label=title,color=color);ax.fill_between(radii,low,high,color=color,alpha=.1)
    ax.plot(radii,best_curves[0],label='Fitted synthetic train contexts',color='#3c9e87');ax.plot(radii,best_curves[1],label='Synthetic held-out contexts',color='#679ed4',ls='--');ax.set(xlabel='Normalized radius',ylabel='Normalized brightness',title='Spatial check of light profiles');ax.legend(fontsize=8)
    ax=axs[1,0]
    for rows,title,color in [(real_statistics,'Real','#333333'),(control_statistics,'Broad control','#d39155'),(syn_statistics,'Fitted shell','#3c9e87')]:
        vals=np.array([r['peak_radius'] for r in rows]);ax.hist(vals,bins=np.linspace(.2,1.5,25),density=True,histtype='step',lw=2,label=title,color=color)
    ax.set(xlabel='Peak radius / mid-shell contour radius',ylabel='Density',title='Location of fluorescence maximum');ax.legend(fontsize=8)
    ax=axs[1,1]
    for rows,title,color in [(real_statistics,'Real','#333333'),(control_statistics,'Broad control','#d39155'),(syn_statistics,'Fitted shell','#3c9e87')]:
        vals=np.array([r['half_prominence_width_px'] for r in rows]);vals=vals[np.isfinite(vals)];ax.hist(vals,bins=np.linspace(0,5,25),density=True,histtype='step',lw=2,label=title,color=color)
    ax.set(xlabel='Apparent half-prominence radial width (px)',ylabel='Density',title='Effective width; outer neighbors affect this statistic');ax.legend(fontsize=8)
    fig.savefig(args.output/'light_profile_fit.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(17,12),constrained_layout=True)
    for ax,im,title in zip(axs,[real,previous['image'],best_result['image']],['Real frame 500','Previous empirical geometry + broad fluorescence','Parametric morphology + fitted peripheral shell']):
        ax.imshow(im.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(title);ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(args.output/'full_comparison.png',dpi=160);plt.close(fig)
    rois=[(300,420,110,230),(410,530,150,270),(760,880,100,220),(1160,1280,120,240)]
    fig,axs=plt.subplots(4,4,figsize=(14,13),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(rois):
        for row,im in enumerate([real,previous['image'],control['image'],best_result['image']]):
            ax=axs[row,col];ax.imshow(im[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y={y0}:{y1}, X={x0}:{x1}')
            if col==0:ax.set_ylabel(['Real','Previous empirical','Parametric broad','Parametric shell'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=160);plt.close(fig)
    # Paired true annotation examples show brightness measured at each radius.
    order=np.argsort(real_contrast);chosen=order[np.linspace(0,len(order)-1,12).round().astype(int)]
    fig,axs=plt.subplots(2,6,figsize=(15,6),constrained_layout=True)
    for ax,i in zip(axs.flat,chosen):
        cy0,cx0=ac[i];x0,x1=int(cx0-12),int(cx0+13);y0,y1=int(cy0-12),int(cy0+13)
        ax.imshow(real[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi,extent=[y0,y1,x1,x0],interpolation='nearest')
        poly=np.array(labels[i]['points_xy']);poly=np.vstack([poly,poly[0]]);ax.plot(poly[:,1],poly[:,0],color='#9de3bc',lw=.7)
        ax.set_title(f"#{labels[i]['number']}  shell-core={real_contrast[i]:.0f} ADU",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Annotation contour passes approximately through the middle of the luminous shell')
    fig.savefig(args.output/'annotated_shells.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(selected=best,control=control_metrics,real_statistics=report['real_light_statistics'],new_statistics=report['new_light_statistics'],scene=report['scene_metrics']),indent=2),flush=True)


if __name__=='__main__':main()
