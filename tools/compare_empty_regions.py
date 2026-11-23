"""Compare ray escape and diameter-closing diagnostics on movie q90.

Outputs candidate DARK SEEDS, not a validated no-neuron segmentation.
Does not change the simulation's tissue/support profiles.
"""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.ndimage import gaussian_filter,shift,map_coordinates,distance_transform_edt
from skimage.morphology import diameter_closing
from skimage.feature import peak_local_max
import tifffile

ROOT=Path(__file__).resolve().parents[1]


def ray_escape(image,length=25,directions=48):
    """Min over directions of the max intensity rise along a straight ray.

    Out-of-image rays are invalid. Float32 smoothed q90 is expected.
    """
    local=gaussian_filter(image,20)
    tolerance=np.maximum(8.,.12*np.maximum(local-100.,0))
    barrier=np.full(image.shape,np.inf,np.float32)
    passes=np.zeros(image.shape,np.float32);valid=np.zeros(image.shape,np.float32)
    for theta in np.arange(directions)*2*np.pi/directions:
        maximum=image.copy()
        for radius in range(1,length+1):
            sampled=shift(image,(-radius*np.cos(theta),-radius*np.sin(theta)),order=1,
                          mode='constant',cval=np.inf,prefilter=False)
            np.maximum(maximum,sampled,out=maximum)
        rise=maximum-image
        np.minimum(barrier,rise,out=barrier)
        valid+=np.isfinite(rise)
        passes+=rise<=tolerance
    fraction=passes/np.maximum(valid,1)
    return barrier,fraction,tolerance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/empty_region_ideas')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:q90=z['q90'].copy()
    with np.load(ROOT/'outputs/plane20_trial/profile.npz') as z:old_support=z['tissue'].copy()
    image=gaussian_filter(q90,.7)
    start=time.monotonic();barrier,fraction,tolerance=ray_escape(image)
    print('Rays completed',round(time.monotonic()-start,1),flush=True)
    # An intentionally loose, explicit brightness gate for this first comparison.
    # Score maps are saved ungated so these assumptions can be reviewed separately.
    dark=(image<220)&old_support
    ray_seed=dark&(barrier<=tolerance)&(fraction>=.06)
    morphology={}
    for diameter in [12,16,20]:
        closed=diameter_closing(image,diameter_threshold=diameter,connectivity=2)
        depth=closed-image
        # Filled nucleus has a large depth. A broad dark area survives closing.
        # Both brightness and depth are needed: bright intercellular pixels also
        # have a small depth and must not be interpreted as evidence of emptiness.
        seed=dark&(closed<190)&(depth<8)
        morphology[diameter]=dict(closed=closed,depth=depth,seed=seed)
    # Regional safety margin: small dark minima are candidate somas, not GT.
    peaks=peak_local_max(morphology[16]['depth'],min_distance=3,threshold_abs=5.,
                         exclude_border=5,labels=old_support.astype(np.uint8))
    peak_mask=np.zeros(image.shape,bool);peak_mask[tuple(peaks.T)]=True
    distance=distance_transform_edt(~peak_mask)
    regional_seed=dark&(fraction>=.15)&(distance>12)
    data=dict(nuclear_candidate_centers_yx=peaks,distance_to_nuclear_candidate_px=distance,regional_dark_seeds=regional_seed,q90=q90,smoothed_q90=image,ray_barrier_adu=barrier,
        ray_escape_fraction=fraction,ray_tolerance_adu=tolerance,
        brightness_gate=dark,ray_dark_seeds=ray_seed,previous_support=old_support)
    for diameter,m in morphology.items():
        data.update({f'diameter{diameter}_closed':m['closed'],f'diameter{diameter}_small_dark_depth':m['depth'],f'diameter{diameter}_dark_seeds':m['seed']})
    np.savez_compressed(args.output/'diagnostics.npz',**data)
    tifffile.imwrite(args.output/'ray_escape_fraction.tiff',fraction)
    tifffile.imwrite(args.output/'small_dark_depth_d16.tiff',morphology[16]['depth'])
    ann=json.loads((ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json').read_text())
    cells=[a for a in ann['annotations'] if a['confidence']=='confident' and not a['truncated']]
    coords=np.array([[a['measurements']['centroid_y']-.5,a['measurements']['centroid_x']-.5] for a in cells])
    def at_center(a):return map_coordinates(a.astype(float),coords.T,order=0,mode='nearest')
    controls={}
    for name,seed in [('ray',ray_seed)]+[(f'diameter{d}',m['seed']) for d,m in morphology.items()]+[('regional_hybrid',regional_seed)]:
        flags=at_center(seed)>0
        controls[name]=dict(known_cell_centers_flagged_empty=int(flags.sum()),n_known_cells=len(cells),
            annotation_numbers_flagged=[a['number'] for a,flag in zip(cells,flags) if flag],
            fraction_of_previous_support_seeded=float(seed[old_support].mean()))
    examples={}
    rois={'central_gap':(650,730,150,225),'cell_rich':(360,460,150,225),'right_cells':(950,1050,130,230)}
    for name,(y0,y1,x0,x1) in rois.items():
        sl=np.s_[y0:y1,x0:x1]
        examples[name]=dict(roi_y0_y1_x0_x1=[y0,y1,x0,x1],ray_fraction_seeded=float(ray_seed[sl].mean()),
            morphology16_fraction_seeded=float(morphology[16]['seed'][sl].mean()),regional_fraction_seeded=float(regional_seed[sl].mean()),
            median_ray_barrier_adu=float(np.median(barrier[sl])),median_small_dark_depth_adu=float(np.median(morphology[16]['depth'][sl])))
    report=dict(ray=dict(length_px=25,directions=48,pre_smoothing_sigma_px=.7,
            tolerance='max(8 ADU, 0.12*(Gaussian20(image)-100 ADU))',minimum_passing_valid_direction_fraction=.06),
        morphology=dict(diameter_thresholds_px=[12,16,20],connectivity=2,
            seed_criterion='smoothed q90<220, closed image<190, closing difference<8 ADU, within previous support'),
        regional_hybrid=dict(candidate_core_depth_threshold_adu=5.,peak_min_distance_px=3,n_candidate_cores=len(peaks),protect_radius_px=12,minimum_passing_ray_fraction=.15,
            criterion='brightness gate AND >=15% escaping rays AND farther than 12px from candidate nuclear minimum'),
        annotation_center_controls=controls,illustrative_regions=examples,
        limitations=['No manual no-neuron GT exists; illustrative gap is visual evidence, not a validated negative population.',
            'Known annotated centers are selected visible somas, so false-empty rates on them do not estimate whole-tissue sensitivity.',
            'Brightness gates and tolerances are exploratory thresholds, not calibrated posterior probabilities.',
            'Ray seeds can follow thin corridors or miss curved passages; connected morphology can join minima through faint bridges.',
            'Bright neuron-free diffuse regions can fail the dark gate and remain unknown.',
            'Regional hybrid was chosen after inspecting annotation-center errors; it has no independent validation set.',
            'Only dark seed maps are produced, not smoothed final exclusion zones or a modified simulator mask.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(4,1,figsize=(17,13),constrained_layout=True)
    axs[0].imshow(q90.T,cmap='gray',vmin=105,vmax=427);axs[0].set_title('Q90 over all 1000 frames')
    show=np.where(old_support,fraction,np.nan)
    im=axs[1].imshow(show.T,cmap='viridis',vmin=0,vmax=1);axs[1].set_title('Fraction of valid 25px rays staying within local brightness tolerance');fig.colorbar(im,ax=axs[1],fraction=.015,pad=.01)
    im=axs[2].imshow(np.where(old_support,morphology[16]['depth'],np.nan).T,cmap='magma',vmin=0,vmax=50);axs[2].set_title('Diameter closing (16px) minus q90: high response = small enclosed dark minimum');fig.colorbar(im,ax=axs[2],fraction=.015,pad=.01)
    axs[3].imshow(q90.T,cmap='gray',vmin=105,vmax=427)
    color=np.zeros((*q90.T.shape,4));color[ray_seed.T]=[.15,.85,.95,.48];color[morphology[16]['seed'].T]=[1.,.62,.15,.48];color[(ray_seed&morphology[16]['seed']).T]=[.4,1.,.35,.55]
    axs[3].imshow(color);axs[3].set_title('DARK SEEDS only: cyan=rays, orange=morphology, green=both; no final absence mask')
    for ax in axs:ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(args.output/'method_comparison.png',dpi=160);plt.close(fig)
    crop_rois=[(300,500,100,300),(600,800,90,290),(900,1100,90,290)]
    fig,axs=plt.subplots(4,3,figsize=(13,15),constrained_layout=True)
    for col,(y0,y1,x0,x1) in enumerate(crop_rois):
        for row in range(4):
            ax=axs[row,col];ax.imshow(q90[y0:y1,x0:x1].T,cmap='gray',vmin=105,vmax=427)
            if row:
                seed=([None,ray_seed,morphology[16]['seed'],regional_seed][row])[y0:y1,x0:x1].T
                overlay=np.zeros((*seed.shape,4));overlay[seed]=([.15,.85,.95,.6] if row==1 else [1.,.62,.15,.6] if row==2 else [1.,.25,.25,.6]);ax.imshow(overlay)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Y={y0}:{y1}, X={x0}:{x1}')
            if col==0:ax.set_ylabel(['Q90','Ray seeds','Morphology seeds','Regional hybrid'][row])
    fig.savefig(args.output/'crop_comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(17,7),constrained_layout=True)
    for ax in axs:ax.imshow(q90.T,cmap='gray',vmin=105,vmax=427);ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    axs[0].scatter(peaks[:,0],peaks[:,1],s=3,color='#60b5cf',alpha=.7);axs[0].set_title('Candidate nuclear minima, not a neuron count')
    overlay=np.zeros((*q90.T.shape,4));overlay[regional_seed.T]=[1.,.25,.25,.55];axs[1].imshow(overlay);axs[1].set_title('Conservative regional DARK SEEDS, red; preliminary, not an approved no-neuron mask')
    fig.savefig(args.output/'regional_candidate.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(controls=controls,examples=examples),indent=2),flush=True)


if __name__=='__main__':main()
