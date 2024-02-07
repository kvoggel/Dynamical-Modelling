import numpy as np
from scipy.special import gamma, gamma, gammaincinv, gammainc
from mgefit.mge_fit_1d import mge_fit_1d
from scipy.optimize import newton, brentq
"""
Converting any Sersic profiles into a MGE (Multi-Gaussian expansion profile).
    
Parameters of the input sersic:
sersic_params = {
    're': 8.63 *scale,  # Effective radius multiplied by the scale
    'mag': 16.33,        # Magnitude
    'n': 5.04,           # Sersic index
    'pa': -52.15,        # Position angle
    'q': 0.668           # Axis ratio
}
Returns:
mge-- Array containing a MGE with the 4 columns:
Luminosity [Lsun/pc^2],  sigma ["],  q,  P.A. [degree]
This mge file can be then fed to JAM models
---------------------------------------------------
Written by Karina Voggel, last update 7. Feb 2024
"""
    
# Helper function for computeSersicBn
def func_sersic_bn(bn_arr, nser):
    return gamma(2.0*nser) - 2*gammainc(2.0*nser, bn_arr)*gamma(2.0*nser)

# Compute Sersic b_n given Sersic n
def computeSersicBn(n):
    # The gammaincinv function computes the inverse of the lower incomplete gamma function
    # We use it to find bn such that gammainc(2n, bn) = 0.5
    bn = gammaincinv(2.0 * n, 0.5)
    return bn
# Compute Sersic b_n given Sersic n 
#def computeSersicBn(n):
#    return newton(lambda x: func_sersic_bn(x, n), x0=(2*n - 0.324)>0.1)

# Helper function for sersic_findrange
def func_sersic_accr(R, nser, bn, re, f):
    x = bn*(R/re)**(1.0/nser)
    return gammainc(2.0*nser, x) - f 

# Find the radius that encloses fraction f of the mass, given Sersic parameters, rein and n
def sersic_findrange(f_in, rein, n):
    bn = computeSersicBn(n)
    return brentq(lambda x: func_sersic_accr(x, n, bn, rein, f_in), a=rein/1e5, b=rein*1e5, xtol=f_in/10.0)

# Fit MGE to Sersic
def fitmge2sersic(n_sersic, Re_sersic, q, NSTEP=None, fitbound=None, ngauss=None, frac=None):
    if ngauss is None:
        A,B = np.polyfit(np.log10([0.5,30]), [1,30], 1)
        # Evaluate the polynomial at np.log10(n_sersic)
        xpol=np.log10(n_sersic)
        ypol=np.polyval([A, B], xpol)
        ngauss = int(ypol+1)
        print('Number auf Gaussians', ngauss)
    if NSTEP is None:
        NSTEP = 300
    if frac is None:
        frac = 1e-4

    fitmin = sersic_findrange(frac, Re_sersic, n_sersic)
    fitmax = sersic_findrange(1 - frac, Re_sersic, n_sersic)

    if fitbound is not None:
        fitmin = min(fitmin, fitbound[0])
        fitmax = max(fitmax, fitbound[1])

    R = np.logspace(np.log10(fitmin), np.log10(fitmax), NSTEP)
    bn = computeSersicBn(n_sersic)
    profile = np.exp(-bn*((R/Re_sersic)**(1.0/n_sersic) - 1))

    profile = (1 + np.random.normal(size=NSTEP)*0.001)*profile

    mge = mge_fit_1d(R, profile, ngauss=ngauss)
    lum = mge.sol[0,:] / np.sqrt(2*np.pi*mge.sol[1,:]**2)
    print('lum as calculated in the sersic2mge', lum)
    mge.sol[0,:] = lum

    return mge.sol
    
def mag2flux(mag, zero_pt=21.10, ABwave=None):
    """
    Convert from magnitudes to flux (ergs/s/cm^2/A).
    
    Parameters:
    mag -- scalar or vector of magnitudes
    zero_pt -- scalar giving the zero point level of the magnitude (default 21.10)
    ABwave -- scalar or vector in Angstroms for Oke AB magnitudes conversion
    
    Returns:
    flux -- scalar or vector flux, in erg cm-2 s-1 A-1
    """
    if ABwave is not None:
        flux = 10**(-0.4 * (mag + 2.406 + 5 * np.log10(ABwave)))
    else:
        flux = 10**(-0.4 * (mag + zero_pt))
    return flux


def flux2mag(flux, zero_pt=21.10, ABwave=None):
    """
    Convert from flux (ergs/s/cm^2/A) to magnitudes.

    Parameters:
    flux -- scalar or vector flux, in erg cm^-2 s^-1 A^-1
    zero_pt -- scalar giving the zero point level of the magnitude (default 21.10)
    ABwave -- scalar or vector in Angstroms for Oke AB magnitudes conversion

    Returns:
    mag -- scalar or vector of magnitudes
    """
    if ABwave is not None:
        mag = -2.5 * np.log10(flux) - 5 * np.log10(ABwave) - 2.406
    else:
        mag = -2.5 * np.log10(flux) - zero_pt
    return mag

def sersic2mge(sersic, Msun, A_b=None, fitrange=None, frac=None):
    if fitrange is None:
        fitrange = [0.01, 10]

    mge = fitmge2sersic(sersic['n'], sersic['re'], sersic['q'], fitbound=fitrange, frac=frac)

    zero_pt = 21.0  # arbitrary zero_point

    totflux = np.sum(mge[0, :]*(2 * np.pi) *(mge[1, :]**2)*sersic['q'])
    targetmag = sersic['mag']
    print('totalflux= ', totflux)
    if A_b is not None:
        targetmag -= A_b
		
		#totflux=total( mge[0,*] * (2*!pi) * mge[1,*]^2 * sersic.q)
		#targetmag=sersic.mag
		#if keyword_set(A_B) then targetmag -= a_b
        #normfl= mag2flux( targetmag,zero_pt)
		#mge[0,*] *= normfl/totflux
        #mu=flux2mag(mge[0,*],zero_pt) 
		#mag= flux2mag(mge[0,*] * (2*!pi) * mge[1,*]^2 * sersic.q,zero_pt)
		#L_obs=(64800.0/!pi)^2*10^(0.4*(Msun-mu))
    normfl = mag2flux(targetmag, zero_pt)
    mge[0, :] *= normfl / totflux
    #mu = -2.5 * np.log10(mge[0, :]) + zero_pt
    mu =flux2mag(mge[0,:],zero_pt)
    #mag = -2.5 * np.log10(mge[0, :] * (2 * np.pi) * mge[1, :]**2 * sersic['q']) + zero_pt
    mag= flux2mag(mge[0, :] * (2 * np.pi) * mge[1, :]**2 * sersic['q'], zero_pt)
    L_obs = (64800.0 / np.pi)**2 *(10**(0.4 * (Msun - mu)))

    mge_out = np.zeros((len(mge[1, :]), 4))
    mge_out[:, 0] = L_obs
    mge_out[:, 1] = mge[1, :]  # sigma
    mge_out[:, 2] = sersic['q']
    mge_out[:, 3] = sersic['pa']

    return mge_out


def sersics2mge(sersics, Msun, A_B=None, fitrange=None, frac=None):
    if fitrange is None:
        fitrange = [0.01, 10]

    # Initialize an empty list to store the MGEs for all Sersic profiles
    mge_list = []

    # Loop over each Sersic profile and convert it to MGE
    for sersic in sersics:
        mge = sersic2mge(sersic, Msun, A_b=A_B, fitrange=fitrange, frac=frac)
        mge_list.append(mge)

    # Combine all MGEs into a single array
    mge_combined = np.vstack(mge_list)

    # Optionally, you can print the combined MGE for inspection
    # print(mge_combined)

    return mge_combined
