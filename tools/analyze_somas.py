"""Analyze hand-annotated soma polygons without changing calibration or labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def measure(points):
    p = np.asarray(points, dtype=float)
    origin = p.mean(axis=0); p = p-origin
    x,y=p.T; u,v=np.roll(p,-1,axis=0).T
    c=x*v-u*y; signed=c.sum()/2
    cx=((x+u)*c).sum()/(6*signed); cy=((y+v)*c).sum()/(6*signed)
    xx=((x*x+x*u+u*u)*c).sum()/(12*signed)-cx*cx
    yy=((y*y+y*v+v*v)*c).sum()/(12*signed)-cy*cy
    xy=((2*x*y+x*v+u*y+2*u*v)*c).sum()/(24*signed)-cx*cy
    cov=np.array([[xx,xy],[xy,yy]])
    values,vectors=np.linalg.eigh(cov); major,minor=4*np.sqrt(values[::-1])
    angle=np.arctan2(vectors[1,1],vectors[0,1]); angle=(angle+np.pi/2)%np.pi-np.pi/2
    area=abs(signed); per=np.linalg.norm(p-np.roll(p,-1,axis=0),axis=1).sum()
    return dict(area_px2=float(area),equivalent_diameter_px=float(2*np.sqrt(area/np.pi)),
        major_axis_px=float(major),minor_axis_px=float(minor),aspect_ratio=float(major/minor),
        perimeter_px=float(per),circularity=float(4*np.pi*area/per**2),
        orientation_deg=float(np.degrees(angle)),centroid_x=float(cx+origin[0]),centroid_y=float(cy+origin[1]))


def resample(p, n=256):
    p=np.asarray(p); loop=np.vstack([p,p[0]])
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(loop,axis=0),axis=1))]
    t=np.arange(n)*length[-1]/n
    return np.stack([np.interp(t,length,loop[:,i]) for i in range(2)],axis=1),length[-1]/n


def smooth(p, sigma, preserve_area=False):
    q,step=resample(p)
    q=gaussian_filter1d(q,sigma/step,axis=0,mode='wrap')
    if preserve_area:
        center=np.mean(q,axis=0)
        q=center+(q-center)*np.sqrt(measure(p)['area_px2']/measure(q)['area_px2'])
    return q


def ellipse_iou(points,m):
    p=np.asarray(points); lo=np.floor(p.min(axis=0)-2); hi=np.ceil(p.max(axis=0)+2)
    y,x=np.mgrid[lo[1]:hi[1]:.25,lo[0]:hi[0]:.25]; x=x+.125;y=y+.125
    points_grid=np.stack([x.ravel(),y.ravel()],axis=1)
    polygon=MplPath(np.vstack([p,p[0]])).contains_points(points_grid)
    theta=np.radians(m['orientation_deg']); delta=points_grid-[m['centroid_x'],m['centroid_y']]
    along=delta[:,0]*np.cos(theta)+delta[:,1]*np.sin(theta)
    across=-delta[:,0]*np.sin(theta)+delta[:,1]*np.cos(theta)
    ellipse=(along/(m['major_axis_px']/2))**2+(across/(m['minor_axis_px']/2))**2<=1
    return float(np.count_nonzero(polygon&ellipse)/np.count_nonzero(polygon|ellipse))


def summary(v):
    v=np.asarray(v)
    return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),
                quantiles=dict(zip(['p05','p10','p25','p50','p75','p90','p95'],map(float,np.percentile(v,[5,10,25,50,75,90,95])))),
                minimum=float(v.min()),maximum=float(v.max()))


def quantile(v,q,weights):
    ix=np.argsort(v); v=np.asarray(v)[ix]; w=np.asarray(weights)[ix]
    return float(np.interp(q,(np.cumsum(w)-.5*w)/w.sum(),v))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=ROOT/'annotations/Plane_Aligned_20_frame500_9b7fb43eb0f3.json')
    p.add_argument('--input',type=Path,default=ROOT/'data/Plane_Aligned_20.tiff')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/soma_analysis')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.annotations.read_text())
    frame=tifffile.imread(args.input,key=data['dataset']['frame_index'])
    assert list(frame.shape)==data['dataset']['shape_yx']
    assert hashlib.sha256(frame.astype('<u2',copy=False).tobytes()).hexdigest()==data['dataset']['frame_sha256']
    rows=[]; polygons=[]
    for a in data['annotations']:
        m=measure(a['points_xy']);q=smooth(a['points_xy'],.5);ms=measure(q)
        saved=a.get('measurements',{})
        for k in m:
            if k in saved and k!='orientation_deg':
                assert np.isclose(m[k],saved[k],rtol=1e-5,atol=1e-5),(a['number'],k,m[k],saved[k])
        row=dict(number=a['number'],id=a['id'],crop_id=a['crop_id'],confidence=a['confidence'],
            truncated=a['truncated'],usable=a['confidence']=='confident' and not a['truncated'],
            vertex_count=len(a['points_xy']),**m,
            ellipse_iou=ellipse_iou(a['points_xy'],m),
            smooth_diameter_px=ms['equivalent_diameter_px'],smooth_aspect_ratio=ms['aspect_ratio'])
        rows.append(row);polygons.append(np.asarray(a['points_xy']))
    with (args.output/'cell_measurements.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    use=np.array([r['usable'] for r in rows]); clean=[r for r in rows if r['usable']]
    field=lambda k:np.array([r[k] for r in clean])
    D=field('equivalent_diameter_px');A=field('major_axis_px');B=field('minor_axis_px');Q=A/B
    angle=np.radians(field('orientation_deg'));groups=np.array([r['crop_id'] for r in clean]);unique=np.unique(groups)
    rng=np.random.default_rng(20261113)
    metrics=['area_px2','equivalent_diameter_px','major_axis_px','minor_axis_px','aspect_ratio','ellipse_iou']
    report=dict(n_total=len(rows),n_usable=len(clean),n_truncated=sum(r['truncated'] for r in rows),
        n_annotation_crops=len(unique),frame_index=data['dataset']['frame_index'],
        input_annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        frame_sha256=data['dataset']['frame_sha256'],summaries={k:summary(field(k)) for k in metrics})
    # Compare full sparse footprints; raster labels would lose overlaps.
    sim_path=ROOT/'outputs/plane20_trial/frame_fit.npz'
    simulation=None
    if sim_path.exists():
        with np.load(sim_path) as z:
            offsets=z['instance_offsets'];pixels=z['instance_pixel_index'];shape=z['synthetic_labels'].shape
            from scipy.ndimage import distance_transform_edt
            edge_distance=distance_transform_edt(z['tissue_mask'])
            sd,sq=[],[]
            for k in range(len(offsets)-1):
                yy,xx=np.unravel_index(pixels[offsets[k]:offsets[k+1]],shape)
                if len(xx)<3:continue
                cy,cx=np.mean(yy),np.mean(xx)
                if edge_distance[int(round(cy)),int(round(cx))]<12:continue
                cov=np.cov(np.stack([xx,yy]),ddof=0)+np.eye(2)/12
                eig=np.linalg.eigvalsh(cov)
                sd.append(2*np.sqrt(len(xx)/np.pi));sq.append(np.sqrt(eig[1]/eig[0]))
            simulation=dict(diameter=np.array(sd),ratio=np.array(sq))
        report['previous_trial_full_footprints']=dict(n_interior=len(sd),equivalent_diameter_px=summary(sd),aspect_ratio=summary(sq),
            note='Raster footprint moments include unit-pixel area variance; tissue boundary margin 12px. Trial profile, not frozen Plane16.')
    report['old_kernel_covariance_limit']=dict(eccentricity_parameter=.08,maximum_continuous_axis_ratio=float(np.sqrt(1.08/.92)),
        allowed_axis_angles_deg=[-45,45],meaning='Equal covariance diagonals with random off-diagonal; eccentricity parameter is not geometric ellipse eccentricity.')
    # CIs describe annotation sampling only, not visibility bias or missed cells.
    bootstrap={}
    for k in metrics[:-1]:
        vals=field(k);cell_medians=np.median(rng.choice(vals,(4000,len(vals))),axis=1)
        cluster_medians=[]
        for _ in range(4000):
            sample=np.concatenate([vals[groups==g] for g in rng.choice(unique,len(unique))])
            cluster_medians.append(np.median(sample))
        bootstrap[k]=dict(cell_median_ci95=np.percentile(cell_medians,[2.5,97.5]).tolist(),
            annotation_crop_cluster_median_ci95=np.percentile(cluster_medians,[2.5,97.5]).tolist())
    report['bootstrap']=bootstrap
    report['elongation_fractions']={f'ratio_above_{t}':float(np.mean(Q>t)) for t in [1.08,1.1,1.2,1.3,1.4,1.5]}
    # Model-free shared resampling is the primary morphology proposal.
    S=np.sqrt(A*B); Z=np.stack([np.log(S),np.log(Q-1)],axis=1)
    report['ellipse_generator_fit']=dict(parameterization='z = [log(sqrt(major*minor)), log(major/minor-1)]; z ~ Normal(mu,cov). q=1+exp(z1), major=exp(z0)*sqrt(q), minor=exp(z0)/sqrt(q).',
        mean=Z.mean(axis=0).tolist(),covariance_mle=np.cov(Z,rowvar=False,ddof=0).tolist(),
        correlation=float(np.corrcoef(Z.T)[0,1]),recommendation='Prefer joint empirical resampling of major, minor and orientation; parametric fit is exploratory, not population-corrected.')
    candidates=[]
    for name,dist in [('normal',stats.norm),('lognormal',stats.lognorm),('gamma',stats.gamma)]:
        pars=dist.fit(D) if name=='normal' else dist.fit(D,floc=0)
        ll=float(np.sum(dist.logpdf(D,*pars)));k=2
        aicc=2*k-2*ll+2*k*(k+1)/(len(D)-k-1)
        candidates.append(dict(name=name,parameters=list(map(float,pars)),aicc=aicc))
    report['diameter_fit_comparison']=sorted(candidates,key=lambda r:r['aicc'])
    lm=np.log(D);report['lognormal_diameter_fit']=dict(log_mean=float(lm.mean()),log_sd_mle=float(lm.std()),median=float(np.exp(lm.mean())))
    ratio_candidates=[]
    for name,dist in [('gamma',stats.gamma),('lognormal',stats.lognorm)]:
        pars=dist.fit(Q-1,floc=0);ll=float(np.sum(dist.logpdf(Q-1,*pars)))
        ratio_candidates.append(dict(name=name,parameters=list(map(float,pars)),aicc=4-2*ll+12/(len(Q)-3)))
    report['aspect_ratio_excess_fit_comparison']=sorted(ratio_candidates,key=lambda c:c['aicc'])
    # Axial orientation: theta and theta+pi are identical, so fit doubled angles.
    kappa,mu,_=stats.vonmises.fit(2*angle,fscale=1)
    R=abs(np.mean(np.exp(2j*angle)))
    report['orientation']=dict(reference='degrees from native TIFF +X, modulo 180; +X is vertical in transposed display',
        axial_mean_deg=float(np.degrees(mu)/2),resultant_length=float(R),vonmises_doubled_kappa=float(kappa),
        by_crop={g:dict(n=int(np.sum(groups==g)),axial_mean_deg=float(np.degrees(np.angle(np.mean(np.exp(2j*angle[groups==g]))))/2),
            resultant_length=float(abs(np.mean(np.exp(2j*angle[groups==g]))))) for g in unique})
    report['by_crop']={g:dict(n=int(np.sum(groups==g)),median_diameter_px=float(np.median(D[groups==g])),
        median_aspect_ratio=float(np.median(Q[groups==g]))) for g in unique}
    # Two nearby annotations can denote duplicates; report candidates for review.
    centers=np.stack([field('centroid_x'),field('centroid_y')],axis=1)
    pairs=[]
    for i in range(len(clean)):
        for j in range(i):
            dist=np.linalg.norm(centers[i]-centers[j])
            if dist<min(B[i],B[j])*.35:
                pairs.append(dict(numbers=[clean[i]['number'],clean[j]['number']],center_distance_px=float(dist)))
    report['possible_duplicate_pairs']=pairs
    sensitivity={}
    for sigma in [.25,.5,.75]:
        measured=[measure(smooth(p,sigma)) for p,r in zip(polygons,rows) if r['usable']]
        dd=np.array([m['equivalent_diameter_px'] for m in measured]);qq=np.array([m['aspect_ratio'] for m in measured])
        area_preserving=[measure(smooth(p,sigma,True)) for p,r in zip(polygons,rows) if r['usable']]
        sensitivity[str(sigma)]=dict(median_diameter_px=float(np.median(dd)),median_aspect_ratio=float(np.median(qq)),
            median_relative_diameter_change=float(np.median(dd/D-1)),max_absolute_ratio_change=float(np.max(abs(qq-Q))),
            area_preserving_median_aspect_ratio=float(np.median([m['aspect_ratio'] for m in area_preserving])))
    report['smoothing_sensitivity_sigma_px']=sensitivity
    # These scenarios assume p(selected|D) proportional to D**alpha on the
    # observed range. They cannot reconstruct cells missing from that range.
    scenarios=[]
    for alpha in [0,1,2]:
        weights=D**(-alpha);weights/=weights.sum()
        scenarios.append(dict(alpha=alpha,assumption='Relative selection propensity proportional to D**alpha within observed range',
            weighted_median_diameter_px=quantile(D,.5,weights),weighted_mean_diameter_px=float(weights@D),
            weighted_median_aspect_ratio=quantile(Q,.5,weights),effective_sample_size=float(1/(weights@weights)),
            weighted_log_mean_diameter=float(weights@np.log(D))))
    report['visibility_bias_scenarios']=scenarios
    report['crop_equal_weight_sensitivity']=dict(median_diameter_px=quantile(D,.5,np.array([1/np.sum(groups==g) for g in groups])),
        median_aspect_ratio=quantile(Q,.5,np.array([1/np.sum(groups==g) for g in groups])))
    # Empirical shapes and measurements retain joint relations; no source edits.
    normalized=[]; aligned=[];smoothed_aligned=[]
    for poly,r in zip(polygons,rows):
        if not r['usable']:continue
        q,_=resample(poly);q=q-[r['centroid_x'],r['centroid_y']];theta=np.radians(r['orientation_deg'])
        rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        normalized.append(q/np.sqrt(r['area_px2']));aligned.append(q@rot/np.sqrt(r['area_px2']))
        qs=smooth(poly,.5,True)-[r['centroid_x'],r['centroid_y']];smoothed_aligned.append(qs@rot/np.sqrt(r['area_px2']))
    np.savez_compressed(args.output/'empirical_shapes.npz',numbers=field('number').astype(int),
        major_axis_px=A,minor_axis_px=B,equivalent_diameter_px=D,orientation_deg=field('orientation_deg'),
        normalized_contours_xy=np.array(normalized),aligned_unit_area_contours_xy=np.array(aligned),
        smoothed_aligned_unit_area_contours_xy=np.array(smoothed_aligned),crop_ids=groups)
    report['limitations']=['Selected visible somas from one frame, not representative random population sampling.',
        'Two incomplete cells excluded; no physical pixel size is assumed.',
        'Annotation crop clusters may overlap spatially and are only approximate resampling units.',
        'Ellipse axes are equivalent filled-area second-moment axes, not independent boundary least-squares fits.',
        'Smoothing rounds corners and shrinks uncorrected area; original polygon metrics remain primary.',
        'Bias weights are hypothetical sensitivity scenarios, not an identified correction or Bayesian posterior.',
        'Empirical resampling retains only observed support; missing tiny/dim cells cannot be recovered from these labels.']
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Overview and measurements.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    ax=axs[0,0];ax.hist(D,bins=12,color='#619aaf',alpha=.8,density=True);xx=np.linspace(D.min()*.8,D.max()*1.15,200)
    for c in candidates:
        dist={'normal':stats.norm,'lognormal':stats.lognorm,'gamma':stats.gamma}[c['name']];ax.plot(xx,dist.pdf(xx,*c['parameters']),label=c['name'])
    ax.set(xlabel='Equivalent soma diameter (px)',ylabel='Density',title=f'{len(clean)} complete somas');ax.legend(fontsize=8)
    ax=axs[0,1];ax.hist(Q,bins=12,color='#a581ac');ax.axvspan(1,np.sqrt(1.08/.92),color='gray',alpha=.25,label='Previous covariance range');ax.set(xlabel='Major / minor axis',title='Elongation');ax.legend(fontsize=8)
    ax=axs[0,2];colors=plt.cm.tab10(np.arange(len(unique)));
    for g,c in zip(unique,colors):
        ix=groups==g;ax.scatter(B[ix],A[ix],s=22,color=c,label=g[:8])
    ax.plot([0,13],[0,13],'--',c='gray');ax.set(xlabel='Minor axis (px)',ylabel='Major axis (px)',title='Joint axes, colors = annotation crop',xlim=(3,13),ylim=(3,13));ax.set_aspect('equal')
    ax=axs[1,0];ax.scatter(D,Q,c=field('centroid_y'),cmap='viridis');rho,pval=stats.spearmanr(D,Q);ax.set(xlabel='Equivalent diameter (px)',ylabel='Aspect ratio',title=f'Size vs elongation: Spearman r={rho:.2f}')
    ax=axs[1,1];ax.hist(np.degrees(angle),bins=np.linspace(-90,90,13),color='#79a581');ax.set(xlabel='Angle from native TIFF +X (degrees)',title=f'Axial orientation: R={R:.2f}')
    ax=axs[1,2];inds=np.arange(len(unique));ax.plot(inds,[report['by_crop'][g]['median_diameter_px'] for g in unique],'o-',label='Crop median');ax.axhline(np.median(D),color='gray',ls='--');ax.set_xticks(inds,[f'C{i+1}\nn={np.sum(groups==g)}' for i,g in enumerate(unique)]);ax.set(ylabel='Diameter (px)',title='Between annotation crops')
    fig.savefig(args.output/'morphology_summary.png',dpi=170);plt.close(fig)
    # All annotations, no selection of attractive shapes. Display transpose only.
    fig,ax=plt.subplots(figsize=(17,4.8),constrained_layout=True);lo,hi=np.percentile(frame,[1,99.5]);ax.imshow(frame.T,cmap='gray',vmin=lo,vmax=hi)
    norm=plt.Normalize(Q.min(),Q.max()); cmap=plt.cm.plasma
    for poly,r in zip(polygons,rows):
        q=np.vstack([poly,poly[0]]);ax.plot(q[:,1],q[:,0],color=cmap(norm(r['aspect_ratio'])) if r['usable'] else '#999999',lw=.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,label='Major / minor',shrink=.75);ax.set(xlabel='Native Y (px)',ylabel='Native X (px)',title='All 100 annotations; gray = incomplete')
    fig.savefig(args.output/'annotated_overview.png',dpi=170);plt.close(fig)
    # Spread gallery over the entire aspect-ratio ranking; include every tail.
    order=np.argsort(Q); chosen=order[np.linspace(0,len(order)-1,24).round().astype(int)]
    fig,axs=plt.subplots(4,6,figsize=(14,10),constrained_layout=True)
    number_to_poly={r['number']:p for r,p in zip(rows,polygons)}
    for ax,i in zip(axs.flat,chosen):
        m=clean[i];poly=number_to_poly[m['number']];cx,cy=m['centroid_x'],m['centroid_y'];half=11
        x0,x1=max(0,int(cx-half)),min(frame.shape[1],int(cx+half+1));y0,y1=max(0,int(cy-half)),min(frame.shape[0],int(cy+half+1))
        ax.imshow(frame[y0:y1,x0:x1].T,extent=[y0,y1,x1,x0],cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        pp=np.vstack([poly,poly[0]]);ax.plot(pp[:,1],pp[:,0],c='#71f5b3',lw=1)
        ax.add_patch(Ellipse((cy,cx),width=m['major_axis_px'],height=m['minor_axis_px'],angle=90-m['orientation_deg'],fill=False,edgecolor='#f2a6ff',lw=1,ls='--'))
        ax.set_title(f"#{m['number']}  q={m['aspect_ratio']:.2f}  D={m['equivalent_diameter_px']:.1f}",fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Green: original polygon; dashed purple: filled-area moment ellipse; sorted by elongation')
    fig.savefig(args.output/'ellipse_gallery.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    ax=axs[0];ax.scatter(D,field('smooth_diameter_px'),s=18);ax.plot([4,11],[4,11],'--',color='gray');ax.set(xlabel='Original diameter (px)',ylabel='Smoothed diameter (px)',title='Periodic smoothing, sigma=0.5 px')
    ax=axs[1];ax.plot([r['alpha'] for r in scenarios],[r['weighted_median_diameter_px'] for r in scenarios],'o-');ax.set(xticks=[0,1,2],xlabel='Assumed size-selection strength alpha',ylabel='Weighted median diameter (px)',title='Sensitivity only: not a measured correction')
    fig.savefig(args.output/'sensitivity.png',dpi=170);plt.close(fig)
    if simulation is not None:
        fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,rv,sv,title,xlabel in [(axs[0],D,simulation['diameter'],'Size distribution','Equivalent diameter (px)'),(axs[1],Q,simulation['ratio'],'Shape distribution','Major / minor axis')]:
            bins=np.linspace(min(rv.min(),sv.min())*.98,max(rv.max(),sv.max())*1.02,25)
            ax.hist(sv,bins=bins,density=True,histtype='step',lw=2,label='Previous trial synthetic footprints')
            ax.hist(rv,bins=bins,density=True,alpha=.5,label='98 annotated complete somas')
            ax.set(xlabel=xlabel,ylabel='Density',title=title);ax.legend(fontsize=8)
        fig.savefig(args.output/'synthetic_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['n_usable','summaries','smoothing_sensitivity_sigma_px','visibility_bias_scenarios','ellipse_generator_fit','diameter_fit_comparison','orientation','possible_duplicate_pairs']},indent=2))


if __name__=='__main__':main()
