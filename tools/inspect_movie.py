"""Exact movie aggregates, adjacent-frame phase correlation, and fixed-scale GIFs.

Uses disk-backed movie storage and small spatial tiles; never realigns source.
"""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from threadpoolctl import threadpool_limits
import tifffile

ROOT=Path(__file__).resolve().parents[1]


def exact_quantiles(movie, tile_rows=16):
    """NumPy linear quantiles, exact over ALL frames, bounded working memory."""
    n,h,w=movie.shape
    positions=np.array([.5,.9])*(n-1)
    ks=sorted(set(np.floor(positions).astype(int))|set(np.ceil(positions).astype(int)))
    median=np.empty((h,w),np.float32);q90=np.empty_like(median)
    for y in range(0,h,tile_rows):
        # Partition the copy, never the movie or the input TIFF.
        tile=np.array(movie[:,y:y+tile_rows,:],copy=True)
        tile.partition(ks,axis=0)
        for dest,pos in zip([median,q90],positions):
            lo,hi=int(np.floor(pos)),int(np.ceil(pos));a=pos-lo
            dest[y:y+tile_rows]=(1-a)*tile[lo].astype(np.float32)+a*tile[hi].astype(np.float32)
    return median,q90


@lru_cache(maxsize=8)
def registration_window(shape):
    return np.outer(np.hanning(shape[0]),np.hanning(shape[1])).astype(np.float32)


@lru_cache(maxsize=8)
def frequency_band(shape):
    fy=fft.fftfreq(shape[0])[:,None];fx=fft.fftfreq(shape[1])[None,:]
    r=np.hypot(fy,fx)
    # Explicit zeroing matters: phase normalization cancels soft attenuation.
    return (r>=1/80)&(r<=.25)


def registration_fft(image):
    image=np.asarray(image,np.float32)
    high=image-gaussian_filter(image,6.)
    high-=np.mean(high)
    value=fft.fftn(high*registration_window(image.shape))
    value*=frequency_band(image.shape)
    return value


def pair_shift(reference_fft,moving_fft,normalization='phase'):
    return phase_cross_correlation(reference_fft,moving_fft,space='fourier',
        upsample_factor=20,normalization=normalization)[0]


def peak_to_sidelobe(reference,moving):
    product=reference*np.conj(moving)
    product/=np.maximum(np.abs(product),np.finfo(np.float32).eps)
    corr=np.abs(fft.ifftn(product))
    peak=np.unravel_index(np.argmax(corr),corr.shape)
    ys=(np.arange(peak[0]-5,peak[0]+6)%corr.shape[0]);xs=(np.arange(peak[1]-5,peak[1]+6)%corr.shape[1])
    valid=np.ones(corr.shape,bool);valid[np.ix_(ys,xs)]=False
    bg=corr[valid]
    return float((corr[peak]-bg.mean())/max(bg.std(),1e-10))


def shift_summary(rows,prefix):
    shifts=np.array([[r[prefix+'_dy'],r[prefix+'_dx']] for r in rows]);norm=np.linalg.norm(shifts,axis=1)
    return dict(n=len(rows),median_abs_dy_px=float(np.median(abs(shifts[:,0]))),
        median_abs_dx_px=float(np.median(abs(shifts[:,1]))),
        displacement_norm_quantiles_px=dict(zip(['p50','p90','p95','p99','max'],map(float,np.percentile(norm,[50,90,95,99,100])))),
        fraction_above_0_1_px=float(np.mean(norm>.1)),fraction_above_0_25_px=float(np.mean(norm>.25)),
        fraction_above_0_5_px=float(np.mean(norm>.5)),fraction_above_1_px=float(np.mean(norm>1)))


def write_gif(path,movie,indices,black,white,*,roi=None,max_width=1100,fps=10):
    """Constant mapping and 64-level grayscale palette; no per-frame normalization."""
    font=ImageFont.truetype(findfont('DejaVu Sans'),14)
    frames=[]
    if roi is None:
        h,w=movie.shape[1:];roi=(0,h,0,w)
    y0,y1,x0,x1=roi
    factor=min(max_width/(y1-y0),2.)
    size=(int(round((y1-y0)*factor)),int(round((x1-x0)*factor)))
    for index in indices:
        frame=np.asarray(movie[index,y0:y1,x0:x1],dtype=np.float32).T
        image=Image.fromarray(np.rint(np.clip((frame-black)*255/(white-black),0,255)).astype(np.uint8))
        image=image.resize(size,Image.Resampling.NEAREST)
        panel=Image.new('L',(size[0],size[1]+26),0)
        panel.paste(image,(0,26))
        draw=ImageDraw.Draw(panel)
        draw.text((8,4),f'Frame {index:04d} | fixed {black:.0f}..{white:.0f} ADU | playback {fps} fps',font=font,fill=235)
        # Fixed palette avoids adaptive color/brightness flicker and reduces GIF size.
        codes=np.rint(np.asarray(panel,dtype=float)*63/255).astype(np.uint8)
        panel=Image.fromarray(codes).convert('P')
        levels=np.rint(np.linspace(0,255,64)).astype(np.uint8)
        palette=np.zeros((256,3),dtype=np.uint8);palette[:64]=levels[:,None]
        panel.putpalette(palette.ravel().tolist())
        frames.append(panel)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False,disposal=2)
    return dict(path=str(path),frame_indices=list(map(int,indices)),playback_fps=fps,
        source_fps=None,palette_levels=64,display_range_adu=[float(black),float(white)],roi_y0_y1_x0_x1=list(roi),
        size_xy=[size[0],size[1]+26],bytes=path.stat().st_size)


def save_region_diagnostic(aggregates,output):
    """Contrast and old support are diagnostic evidence, not a replacement mask."""
    med=aggregates['median'];slow=gaussian_filter(med,12)
    band=gaussian_filter(med,.7)-gaussian_filter(med,2.5)
    texture=np.sqrt(gaussian_filter(band**2,8))
    tifffile.imwrite(output/'cell_scale_texture_rms.tiff',texture.astype(np.float32))
    prior=ROOT/'outputs/plane20_trial/profile.npz'
    if not prior.exists():return {}
    with np.load(prior) as p:allowed=p['tissue'].copy();density=p['density'].copy()
    if allowed.shape!=med.shape:return {}
    regions={}
    for name,(y0,y1,x0,x1) in {'candidate_low_soma_gap':(650,730,150,225),'cell_rich_example':(360,460,150,225),'outside_tissue_example':(30,130,30,130)}.items():
        roi=np.s_[y0:y1,x0:x1]
        regions[name]=dict(roi_y0_y1_x0_x1=[y0,y1,x0,x1],median_adu=float(np.median(med[roi])),q90_median_adu=float(np.median(aggregates['q90'][roi])),
            texture_rms_median_adu=float(np.median(texture[roi])),previous_allowed_fraction=float(allowed[roi].mean()),previous_mean_density_weight=float(density[roi].mean()))
    fig,axs=plt.subplots(3,1,figsize=(17,10),constrained_layout=True)
    axs[0].imshow(med.T,cmap='gray',vmin=99,vmax=372);axs[0].contour(allowed.T,levels=[.5],colors=['#e9b667'],linewidths=.7);axs[0].set_title('Median with previous allowed-cell support contour (amber)')
    axs[1].imshow(slow.T,cmap='gray',vmin=99,vmax=300);axs[1].set_title('Slow fluorescence field, Gaussian sigma=12 px: bright diffuse signal does not imply somas')
    shown=axs[2].imshow(texture.T,cmap='magma',vmin=0,vmax=np.percentile(texture,99));axs[2].set_title('Cell-scale texture RMS: local structure diagnostic, not a cell-presence mask');fig.colorbar(shown,ax=axs[2],label='ADU',fraction=.015,pad=.01)
    for ax in axs:ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(output/'region_structure_diagnostic.png',dpi=160);plt.close(fig)
    return regions


def save_outlier_diagnostic(movie,output,index,black,white):
    indices=[max(0,index-1),index,min(len(movie)-1,index+1)]
    frames=[movie[i] for i in indices]
    fig,axs=plt.subplots(3,1,figsize=(17,10),constrained_layout=True)
    for ax,frame,i in zip(axs,frames,indices):
        ax.imshow(frame.T,cmap='gray',vmin=black,vmax=white);ax.set_title(f'Original frame {i} | same ADU scale');ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
    fig.savefig(output/f'outlier_frames_{indices[0]}_{indices[-1]}.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for row,(name,(y0,y1,x0,x1)) in enumerate([('left cells',(300,540,100,280)),('right cells',(920,1160,80,260))]):
        a=frames[0][y0:y1,x0:x1].T;b=frames[1][y0:y1,x0:x1].T
        red=np.clip((a.astype(float)-black)/(white-black),0,1);cyan=np.clip((b.astype(float)-black)/(white-black),0,1)
        for col,img in enumerate([a,b,np.stack([red,cyan,cyan],axis=-1)]):
            ax=axs[row,col];ax.imshow(img,cmap='gray',**(dict(vmin=black,vmax=white) if col<2 else {}));ax.axis('off');ax.set_title(f'{name}: '+[f'frame {indices[0]}',f'frame {index}',f'red={indices[0]}, cyan={index}'][col])
    fig.savefig(output/'outlier_local_overlay.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/Plane_Aligned_20.tiff')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/plane20_movie_diagnostics')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    with tifffile.TiffFile(args.input) as tif:
        n=len(tif.pages);shape=tif.pages[0].shape;dtype=tif.pages[0].dtype
        if len(shape)!=2 or dtype!=np.uint16:raise ValueError('Expected uint16 2D TIFF pages')
        meta=dict(input=str(args.input.resolve()),input_bytes=args.input.stat().st_size,frame_count=n,
            shape_yx=list(shape),dtype=str(dtype),source_fps=None,
            source_physical_pixel_size_um=None,all_frames_used_for_aggregates=True)
        with tempfile.TemporaryDirectory(prefix='movie_work_',dir=args.output) as scratch:
            movie=np.lib.format.open_memmap(Path(scratch)/'movie.npy',mode='w+',dtype=np.uint16,shape=(n,*shape))
            total=np.zeros(shape,np.float64);total2=np.zeros(shape,np.float64);maximum=np.zeros(shape,np.uint16)
            for i,page in enumerate(tif.pages):
                frame=page.asarray()
                if frame.shape!=shape:raise ValueError('Inconsistent page shape')
                movie[i]=frame;total+=frame;total2+=np.square(frame,dtype=np.float64);np.maximum(maximum,frame,out=maximum)
            movie.flush();mean=(total/n).astype(np.float32);std=np.sqrt(np.maximum(total2/n-mean.astype(float)**2,0)).astype(np.float32)
            median,q90=exact_quantiles(movie)
            aggregates=dict(mean=mean,maximum=maximum,q90=q90,median=median,std=std,
                q90_minus_median=q90-median,max_minus_q90=maximum.astype(np.float32)-q90)
            np.savez_compressed(args.output/'aggregates.npz',**aggregates,frame_count=np.array(n))
            for name,image in aggregates.items():tifffile.imwrite(args.output/f'{name}.tiff',image)
            print(f'Exact aggregates completed for all {n} frames ({time.monotonic()-start:.1f}s)',flush=True)
            black,white=np.percentile(median,[1,99.5])
            names=['mean','maximum','q90','median']
            titles=['Mean over all frames','Maximum over all frames','90th percentile (linear, exact)','Median (linear, exact)']
            for individual in [False,True]:
                fig,axs=plt.subplots(4,1,figsize=(17,13),constrained_layout=True)
                for ax,name,title in zip(axs,names,titles):
                    image=aggregates[name];lo,hi=np.percentile(image,[1,99.5]) if individual else (black,white)
                    shown=ax.imshow(image.T,cmap='gray',vmin=lo,vmax=hi);ax.set_title(f'{title} | display {lo:.1f}..{hi:.1f} ADU');ax.set_xlabel('Native Y (px)');ax.set_ylabel('Native X (px)')
                    fig.colorbar(shown,ax=ax,label='ADU',fraction=.015,pad=.01)
                fig.savefig(args.output/('aggregates_individual_scale.png' if individual else 'aggregates_common_scale.png'),dpi=160);plt.close(fig)
            rois=[(300,500,100,300),(600,800,90,290),(900,1100,90,290)]
            fig,axs=plt.subplots(4,3,figsize=(13,13),constrained_layout=True)
            for row,(name,title) in enumerate(zip(names,titles)):
                for col,(y0,y1,x0,x1) in enumerate(rois):
                    ax=axs[row,col];ax.imshow(aggregates[name][y0:y1,x0:x1].T,cmap='gray',vmin=black,vmax=white);ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(f'Y={y0}:{y1}, X={x0}:{x1}')
                    if col==0:ax.set_ylabel(name)
            fig.savefig(args.output/'aggregate_crops.png',dpi=160);plt.close(fig)
            fig,axs=plt.subplots(2,1,figsize=(17,7),constrained_layout=True)
            for ax,name in zip(axs,['q90_minus_median','max_minus_q90']):
                image=aggregates[name];shown=ax.imshow(image.T,cmap='magma',vmin=0,vmax=np.percentile(image,99.5));ax.set_title(name+' (ADU)');fig.colorbar(shown,ax=ax,fraction=.015,pad=.01)
            fig.savefig(args.output/'temporal_range.png',dpi=160);plt.close(fig)
            # Crop coordinates native Y/X; avoid relying on the bright outer contour only.
            regions=dict(global_frame=(0,shape[0],0,shape[1]),
                         left_cells=(250,600,64,272),right_cells=(850,1250,64,272))
            for name,(y0,y1,x0,x1) in regions.items():
                if not (0<=y0<y1<=shape[0] and 0<=x0<x1<=shape[1]):raise ValueError('ROIs are specific to Plane20 shape')
            rows=[];reference_rows=[];previous_ffts=None;first_ffts=None
            # FFT/DFT calls with many tiny thread teams are slower and less predictable.
            with threadpool_limits(limits=1):
                for i in range(n):
                    current={name:registration_fft(movie[i,y0:y1,x0:x1]) for name,(y0,y1,x0,x1) in regions.items()}
                    if first_ffts is None:first_ffts=current.copy()
                    if previous_ffts is not None:
                        row=dict(reference_frame=i-1,moving_frame=i)
                        for name in regions:
                            dy,dx=pair_shift(previous_ffts[name],current[name])
                            row[name+'_dy']=float(dy);row[name+'_dx']=float(dx)
                        dy,dx=pair_shift(previous_ffts['global_frame'],current['global_frame'],normalization=None)
                        row['global_ncc_dy']=float(dy);row['global_ncc_dx']=float(dx)
                        row['global_phase_peak_to_sidelobe']=peak_to_sidelobe(previous_ffts['global_frame'],current['global_frame'])
                        rows.append(row)
                    if i%10==0 or i==n-1:
                        row=dict(reference_frame=0,moving_frame=i)
                        for name in regions:
                            dy,dx=pair_shift(first_ffts[name],current[name]);row[name+'_dy']=float(dy);row[name+'_dx']=float(dx)
                        reference_rows.append(row)
                    previous_ffts=current
                    if i%100==0:print(f'Phase correlation frame {i}/{n} ({time.monotonic()-start:.1f}s)',flush=True)
            for filename,data in [('adjacent_shifts.csv',rows),('reference_shifts.csv',reference_rows)]:
                with (args.output/filename).open('w') as f:
                    w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
            motion={name:shift_summary(rows,name) for name in regions}
            motion['global_ncc']=shift_summary(rows,'global_ncc')
            fixed={name:shift_summary(reference_rows,name) for name in regions}
            pair_norm=np.array([np.hypot(r['global_frame_dy'],r['global_frame_dx']) for r in rows]);rank=np.argsort(pair_norm)[::-1]
            worst=[rows[i]|dict(global_norm_px=float(pair_norm[i])) for i in rank[:15]]
            diff=np.array([[r['global_frame_dy']-r['global_ncc_dy'],r['global_frame_dx']-r['global_ncc_dx']] for r in rows])
            local_diff=np.array([[r['left_cells_dy']-r['right_cells_dy'],r['left_cells_dx']-r['right_cells_dx']] for r in rows])
            meta.update(aggregate_display_range_adu=[float(black),float(white)],
                aggregates_definition='All 1000 native uncorrected frames; exact median and q90 use NumPy linear quantile convention.',
                phase_correlation=dict(regions_y0_y1_x0_x1={k:list(v) for k,v in regions.items()},
                    shift_convention='Shift to apply to moving frame to align it to reference, native (dy,dx), pixels. Plots transpose image only.',
                    preprocessing='6px high-pass + separable Hann window; FFT phase normalized within hard radial frequency band [1/80, 0.25] cycles/pixel. Global unnormalized NCC on same FFTs is a sensitivity check.',
                    upsample_factor=20,nominal_grid_step_px=.05,adjacent=motion,fixed_reference_frame0=fixed,worst_adjacent_pairs=worst,
                    median_phase_vs_ncc_difference_px=float(np.median(np.linalg.norm(diff,axis=1))),
                    p95_phase_vs_ncc_difference_px=float(np.percentile(np.linalg.norm(diff,axis=1),95)),
                    median_left_vs_right_shift_difference_px=float(np.median(np.linalg.norm(local_diff,axis=1))),
                    p95_left_vs_right_shift_difference_px=float(np.percentile(np.linalg.norm(local_diff,axis=1),95))),
                limitations=['Shift grid spacing is not an error bar or proof of absolute registration accuracy.',
                    'Phase correlation estimates translation, not local warps, rotations or changes in cell shape.',
                    'Calcium changes, noise and interpolation affect subpixel estimates, particularly small ROIs.',
                    'Fixed-reference drift is measured directly against frame0, not integrated adjacent estimates.',
                    'No resampling or additional alignment was applied before aggregates or GIFs.',
                    'No source frame interval found; GIF playback speed is chosen for viewing, not asserted acquisition timing.'])
            fig,axs=plt.subplots(4,1,figsize=(15,12),constrained_layout=True)
            ts=np.arange(1,n)
            for axis,label,color in [('dy','native Y','#378ba0'),('dx','native X','#bd7052')]:
                axs[0].plot(ts,[r['global_frame_'+axis] for r in rows],label=label,lw=.8,color=color)
            axs[0].set(ylabel='Adjacent shift (px)',title='Global phase correlation: shift to apply to frame t vs t-1');axs[0].legend()
            for name,color in [('global_frame','#333333'),('left_cells','#4f9e88'),('right_cells','#a77bb2')]:
                vals=np.array([np.hypot(r[name+'_dy'],r[name+'_dx']) for r in rows]);axs[1].plot(ts,vals,lw=.7,label=name,color=color)
            axs[1].set(ylabel='Shift norm (px)',title='Adjacent-frame global and local measurements');axs[1].legend()
            for name,style in [('global_frame','-'),('left_cells','--'),('right_cells',':')]:
                for axis,color in [('dy','#378ba0'),('dx','#bd7052')]:
                    axs[2].plot([r['moving_frame'] for r in reference_rows],[r[name+'_'+axis] for r in reference_rows],ls=style,label=name+' '+axis,color=color)
            axs[2].set(ylabel='Shift vs frame0 (px)',title='Direct fixed-reference check, every 10 frames');axs[2].legend(ncol=3,fontsize=8)
            axs[3].plot(ts,np.linalg.norm(diff,axis=1),lw=.7,label='phase vs global unnormalized NCC');axs[3].plot(ts,np.linalg.norm(local_diff,axis=1),lw=.7,label='left vs right local phase')
            axs[3].set(xlabel='Moving frame index',ylabel='Estimator disagreement (px)');axs[3].legend()
            fig.savefig(args.output/'motion_diagnostics.png',dpi=160);plt.close(fig)
            # Mean brightness is descriptive; no automatic motion/photometry correction.
            frame_mean=np.asarray([np.mean(movie[i],dtype=np.float64) for i in range(n)])
            with (args.output/'brightness_trace.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['frame_index','whole_frame_mean_adu']);w.writerows(enumerate(frame_mean))
            fig,ax=plt.subplots(figsize=(13,3),constrained_layout=True);ax.plot(frame_mean);ax.set(xlabel='Frame index',ylabel='Mean ADU',title='Whole-frame brightness over the film (uncorrected)');fig.savefig(args.output/'brightness_trace.png',dpi=160);plt.close(fig)
            gif_black,gif_white=np.percentile(movie[0],[1,99.5])
            meta['gifs']=[]
            meta['gifs'].append(write_gif(args.output/'movie_overview.gif',movie,range(0,n,10),gif_black,gif_white,max_width=960,fps=10))
            worst_index=int(rows[int(rank[0])]['moving_frame']);clip_start=max(0,min(n-120,worst_index-60));clip_indices=range(clip_start,min(clip_start+120,n))
            meta['gifs'].append(write_gif(args.output/'detail_contiguous.gif',movie,clip_indices,gif_black,gif_white,
                roi=(320,600,90,280),max_width=560,fps=10))
            meta['contiguous_clip_selection']=dict(reason='120 consecutive frames around the largest global adjacent phase estimate',worst_pair_moving_frame=worst_index)
            meta['whole_frame_mean_adu_first_last']=[float(frame_mean[0]),float(frame_mean[-1])]
            meta['whole_frame_mean_first50_last50_adu']=[float(np.mean(frame_mean[:50])),float(np.mean(frame_mean[-50:]))]
            meta['region_examples']=save_region_diagnostic(aggregates,args.output)
            meta['cell_scale_texture_definition']='sqrt(Gaussian_sigma8((Gaussian_sigma0.7(median)-Gaussian_sigma2.5(median))^2)); descriptive structure map, not validated neuron segmentation.'
            save_outlier_diagnostic(movie,args.output,worst_index,gif_black,gif_white)
            meta['elapsed_seconds']=time.monotonic()-start
            (args.output/'report.json').write_text(json.dumps(meta,indent=2)+'\n')
            print(json.dumps(dict(adjacent=motion,fixed=fixed,worst=worst[:3],gifs=[dict(path=g['path'],bytes=g['bytes']) for g in meta['gifs']]),indent=2),flush=True)
            del movie


if __name__=='__main__':main()
