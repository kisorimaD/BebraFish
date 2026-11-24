"""Parametric contour morphology and a narrow elliptical shell emission model."""
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter, map_coordinates


def sample_morphology(report, count, seed):
    """Positive normal diameter, shifted gamma ratio, axial von Mises angle.

    Parameters describe the annotation's mid-shell contour. No selection
    correction. Size/elongation independence is an explicit approximation.
    """
    rng=np.random.default_rng(seed)
    normal=next(m for m in report['diameter_fit_comparison'] if m['name']=='normal')
    mu,sd=normal['parameters']
    gamma=next(m for m in report['aspect_ratio_excess_fit_comparison'] if m['name']=='gamma')
    shape,_,scale=gamma['parameters']
    diameter=stats.truncnorm.rvs(-mu/sd,np.inf,loc=mu,scale=sd,size=count,random_state=rng)
    ratio=1+rng.gamma(shape,scale,size=count)
    orient=report['orientation']
    theta=rng.vonmises(np.radians(2*orient['axial_mean_deg']),orient['vonmises_doubled_kappa'],size=count)/2
    return dict(major_axis_px=diameter*np.sqrt(ratio),minor_axis_px=diameter/np.sqrt(ratio),
                orientation_deg=np.degrees(theta),diameter_px=diameter,aspect_ratio=ratio)


def sample_angular_coefficients(count,seed):
    """Independent smooth per-cell log-emission, four cosine/sine modes.

    Coordinates are fraction of perimeter; expected log-field variance is one.
    Multiply coefficients by desired log-modulation strength before rendering.
    """
    weights=np.array([1.,.75,.5,.35])
    weights/=np.linalg.norm(weights)
    return np.random.default_rng(seed).normal(size=(count,4,2))*weights[None,:,None]


def angular_log_modulation(phase,coefficients):
    coefficients=np.asarray(coefficients,float)
    if coefficients.ndim!=2 or coefficients.shape[1]!=2 or not np.isfinite(coefficients).all():
        raise ValueError('Angular coefficients must be a finite K by 2 array')
    value=np.zeros_like(phase,dtype=float)
    for k,(a,b) in enumerate(coefficients,1):value+=a*np.cos(k*phase)+b*np.sin(k*phase)
    return value


def gaussian_shell(radius, peak_radius, width):
    return np.exp(-.5 * ((radius - peak_radius) / width)**2)


def arc_log_gain(u, v, major, minor, coefficients):
    """Evaluate the Plane20 brightness harmonics at equal ellipse arc length."""
    phi = np.mod(np.arctan2(v, u), 2 * np.pi)
    grid = np.linspace(0, 2 * np.pi, 1025)
    speed = np.hypot(major * np.sin(grid), minor * np.cos(grid))
    arc = np.r_[0, np.cumsum((speed[:-1] + speed[1:]) / 2 * np.diff(grid))]
    phase = np.interp(phi, grid, arc) * (2 * np.pi / arc[-1])
    return angular_log_modulation(phase, coefficients)


def shell_patch(center_yx,major,minor,angle_deg,ring,fill,blur,*,peak_radius=1.,width=.1,
                angular_coefficients=None,pixel_yx=None):
    """Gaussian shell in elliptical radius rho; rho=1 is annotation mid-shell.

    width is pre-blur Gaussian sigma in normalized radius, not physical wall
    thickness. PSF, sampling and neighbors alter measured apparent width.
    With pixel_yx, diameters and pitches use the same physical unit.
    """
    theta=np.radians(angle_deg);c,s=np.cos(theta),np.sin(theta)
    extent=major/2*(peak_radius+4*width)
    if pixel_yx is not None:
        extent/=min(pixel_yx)
    radius=int(np.ceil(extent+3*np.max(blur)))+2
    origin=np.floor(center_yx).astype(int)-radius;side=2*radius+1
    y,x=np.mgrid[:side,:side].astype(float);y-=center_yx[0]-origin[0];x-=center_yx[1]-origin[1]
    if pixel_yx is not None:
        y*=pixel_yx[0];x*=pixel_yx[1]
    rho=np.sqrt(((x*c+y*s)/(major/2))**2+((-x*s+y*c)/(minor/2))**2)
    shell=gaussian_shell(rho,peak_radius,width)
    core=np.exp(-(rho/max(.7*peak_radius,1e-5))**8)
    kernel=(1-fill)*shell+fill*core
    if angular_coefficients is not None:
        coefficients=np.asarray(angular_coefficients,float)
        if coefficients.ndim!=2 or coefficients.shape[1]!=2 or not np.isfinite(coefficients).all():
            raise ValueError('Angular coefficients must be a finite K by 2 array')
        if np.any(coefficients):
            log_gain=arc_log_gain((x*c+y*s)/(major/2),(-x*s+y*c)/(minor/2),major,minor,coefficients)
            # Log subtraction avoids overflow and is cancelled by normalization.
            gain=np.exp(log_gain-log_gain.max())
            # Renormalize shell alone, keeping an optional core component fixed.
            modulated_shell=shell*gain
            modulated_shell*=shell.sum()/max(modulated_shell.sum(),1e-300)
            kernel=(1-fill)*modulated_shell+fill*core
    kernel=gaussian_filter(kernel,blur,mode='constant')
    return origin,kernel.astype(np.float32),rho<=1


def radial_profiles(image, centers_yx, major, minor, angles_deg, radii=None, n_angles=64,
                    coordinate_offset=0.):
    """Angular median at each normalized elliptical radius.

    offset=.5 for pixel-edge annotation coordinates, 0 for generator centers
    in native integer-centered pixel coordinates. Bilinear interpolation.
    """
    radii=np.linspace(0,1.8,73) if radii is None else np.asarray(radii)
    theta=np.arange(n_angles)*2*np.pi/n_angles
    r=radii[None,:,None];alpha=theta[None,None,:]
    major=np.asarray(major)[:,None,None]/2;minor=np.asarray(minor)[:,None,None]/2
    angle=np.radians(angles_deg)[:,None,None]
    u=r*major*np.cos(alpha);v=r*minor*np.sin(alpha)
    x=np.asarray(centers_yx)[:,1,None,None]+u*np.cos(angle)-v*np.sin(angle)-coordinate_offset
    y=np.asarray(centers_yx)[:,0,None,None]+u*np.sin(angle)+v*np.cos(angle)-coordinate_offset
    sampled=map_coordinates(np.asarray(image,dtype=float),[y.ravel(),x.ravel()],order=1,mode='nearest').reshape(x.shape)
    return radii,np.median(sampled,axis=2)


def normalize_profiles(r, curves):
    core=curves[:,r<=.25].mean(axis=1)
    # Evaluate a wide enough shell neighborhood to include annotation errors.
    shell=curves[:,(r>=.55)&(r<=1.35)].max(axis=1)
    contrast=shell-core
    return (curves-core[:,None])/np.maximum(contrast[:,None],1),core,contrast


def profile_statistics(r,curves,major,minor):
    """Observable profiles, not a deconvolved biological wall measurement."""
    from scipy.signal import peak_widths
    normalized,core,contrast=normalize_profiles(r,curves)
    out=[]
    for i,p in enumerate(normalized):
        ix=np.flatnonzero((r>=.55)&(r<=1.35));peak=ix[np.argmax(p[ix])]
        if peak==0 or peak==len(p)-1 or p[peak]<=p[peak-1] or p[peak]<=p[peak+1]:
            width=np.nan
        else:
            width=peak_widths(p,[peak],rel_height=.5)[0][0]*(r[1]-r[0])
        inner=np.flatnonzero(p[:peak]>=.5)
        half=float(r[inner[0]]) if len(inner) else np.nan
        out.append(dict(core_adu=float(core[i]),shell_minus_core_adu=float(contrast[i]),
            peak_radius=float(r[peak]),inner_half_rise_radius=half,
            half_prominence_width_rho=float(width),
            half_prominence_width_px=float(width*np.sqrt(major[i]*minor[i])/2)))
    return out
