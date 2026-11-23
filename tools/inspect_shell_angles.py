"""Feasibility: peripheral light profiles using existing mid-shell polygons.

Observed angular variation includes neighbors, blur and annotation errors;
this does not estimate an intrinsic angular emission distribution.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simcalc_zebrafish_pipeline'))
from analyze_somas import smooth,resample
from soma_shell_model import radial_profiles,normalize_profiles


def perimeter_strip(image,points,n=128):
    # Annotation coordinates describe pixel edges; array samples are centers.
    contour,_=resample(smooth(points,.5,preserve_area=True),n)
    tangent=np.roll(contour,-1,axis=0)-np.roll(contour,1,axis=0)
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    offsets=np.linspace(-1.5,1.5,13)
    sample=contour[:,None,:]+normal[:,None,:]*offsets[None,:,None]-.5
    strip=map_coordinates(image.astype(float),[sample[:,:,1].ravel(),sample[:,:,0].ravel()],order=1,mode='nearest').reshape(n,len(offsets))
    # Integrate a 3px band instead of taking a noise-biased radial maximum.
    return contour,strip.mean(axis=1),strip


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_shell_angles');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    annotations=json.loads((ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json').read_text())
    labels=[a for a in annotations['annotations'] if a['confidence']=='confident' and not a['truncated']]
    with np.load(ROOT/'outputs/plane20_movie_diagnostics/aggregates.npz') as z:mean=z['mean'].copy()
    with np.load(ROOT/'outputs/plane20_trial/frame_fit.npz') as z:real=z['real_frame'].copy()
    centers=np.array([[a['measurements']['centroid_y'],a['measurements']['centroid_x']] for a in labels])
    geom=[np.array([a['measurements'][k] for a in labels]) for k in ['major_axis_px','minor_axis_px','orientation_deg']]
    r,radial=radial_profiles(mean,centers,*geom,coordinate_offset=.5)
    _,core,contrast=normalize_profiles(r,radial)
    contours=[];profiles=[];profiles_frame=[];strips=[];rows=[]
    for i,a in enumerate(labels):
        contour,profile,strip=perimeter_strip(mean,a['points_xy']);_,frame,_=perimeter_strip(real,a['points_xy'])
        perimeter=np.linalg.norm(contour-np.roll(contour,-1,axis=0),axis=1).sum()
        profile=gaussian_filter1d(profile,.4/(perimeter/len(profile)),mode='wrap')
        emission=profile-core[i];positive=np.maximum(emission,1)
        coeff=np.fft.rfft(profile-profile.mean());power=abs(coeff)**2
        rows.append(dict(number=a['number'],mean_band_adu=float(profile.mean()),core_adu=float(core[i]),
            angular_std_adu=float(profile.std()),p90_minus_p10_adu=float(np.percentile(profile,90)-np.percentile(profile,10)),
            contrast_normalized_angular_std=float(profile.std()/max(contrast[i],1)),
            relative_shell_light_cv=float(profile.std()/max(emission.mean(),1)),
            power_fraction_modes_1_2=float(power[1:3].sum()/max(power.sum(),1e-8)),
            power_fraction_modes_3_6=float(power[3:7].sum()/max(power.sum(),1e-8))))
        contours.append(contour);profiles.append(profile);profiles_frame.append(frame);strips.append(strip)
    np.savez_compressed(args.output/'angular_profiles.npz',numbers=np.array([a['number'] for a in labels]),
        contours_xy=np.array(contours),mean_profiles=np.array(profiles),frame500_profiles=np.array(profiles_frame),mean_strips=np.array(strips),core_adu=core)
    report=dict(n_somas=len(labels),method='Smooth existing polygons by 0.5px, equal-arc resample 128 points, integrate intensities over ±1.5px normal strip; angular smoothing 0.4px of arc.',
        annotation_target='Previously confirmed mid-shell contour. No new annotation required for this feasibility measurement.',
        summary={k:dict(zip(['p10','median','p90'],map(float,np.percentile([row[k] for row in rows],[10,50,90])))) for k in ['angular_std_adu','p90_minus_p10_adu','contrast_normalized_angular_std','power_fraction_modes_1_2','power_fraction_modes_3_6']},
        per_soma=rows,limitations=['Measured light includes neighboring somas and projected structures, not solely intrinsic shell variation.',
            'Mid-shell contour errors and polygon smoothing can change extracted profiles; radial strip reduces but does not remove this sensitivity.',
            'Mean movie reduces noise but also mixes activity and residual motion; profile in frame500 is saved for comparison.',
            'Harmonic modes use arc-length fraction, not geometrical angle; they are a coarse measure of complexity.',
            'Examples are systematically selected by angular-variation quantiles, not random or population-unbiased.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    scores=np.array([row['contrast_normalized_angular_std'] for row in rows]);order=np.argsort(scores);chosen=order[np.linspace(0,len(order)-1,8).round().astype(int)]
    fig,axs=plt.subplots(4,4,figsize=(14,12),constrained_layout=True)
    lo,hi=np.percentile(real,[1,99.5])
    for j,i in enumerate(chosen):
        row=(j//4)*2;col=j%4;cy,cx=centers[i];y0,y1=int(cy-13),int(cy+14);x0,x1=int(cx-13),int(cx+14)
        ax=axs[row,col];ax.imshow(mean[y0:y1,x0:x1].T,cmap='gray',vmin=lo,vmax=hi,extent=[y0,y1,x1,x0]);c=contours[i]
        ax.scatter(c[:,1],c[:,0],c=np.arange(len(c)),cmap='hsv',s=3);ax.set_title(f"Soma #{labels[i]['number']}");ax.set_xticks([]);ax.set_yticks([])
        ax=axs[row+1,col];t=np.arange(128)/128
        ax.plot(t,profiles_frame[i],color='#777777',alpha=.5,lw=.8,label='Frame 500, raw band');ax.plot(t,profiles[i],color='#318c77',lw=1.6,label='Temporal mean, smoothed band')
        ax.axhline(core[i],color='#c89859',ls='--',lw=.8,label='Core baseline');ax.set(xlabel='Fraction of soma perimeter',ylabel='ADU');ax.legend(fontsize=6)
    fig.savefig(args.output/'existing_contour_profiles.png',dpi=155);plt.close(fig)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
