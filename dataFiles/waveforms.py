from __future__ import division, print_function

from astropy.io import fits
from scipy import *
from numpy import *
from numpy.random import *
from numpy.random import randn
from numpy.fft import *
from matplotlib import *
from matplotlib.pyplot import *
#from scipy.optimize import *
import scipy.optimize as spo
from scipy import stats
from numpy.polynomial.polynomial import polyfit
import six

def allplots():
   """
   A list of all plot commands.
   """

   ssshift, acf2dshift, axacf = plot_diss2(pulse_intensities_bpcorr, chanfreqs, dtime, file)

def smoothcirc(y, nsmo):
  """
  Smooths y over nsmo samples assuming y is a circular array
  nsmo is assumed to be odd
  """
  npoints = size(y)
  ysmo = zeros(npoints)
  for n in range(npoints):
    sum=0.
    for m in range(nsmo):
      ind = n - (nsmo-1)//2 + m
      if ind < 0: ind += npoints
      if ind > npoints-1: ind -= npoints
      sum += y[ind]
    ysmo[n] = sum/nsmo
  return ysmo

def posmax3(y):
  m = y[0]
  index = 0
  for i, x in enumerate(y):
    if x > m:
      m = x
      index = i
  return m, index

def find_hwhm(array):
    """
    Finds half width at half maximum in sample numbers via interpolation.
    Assumes array is symmetric about maximum.
    """
    ninterp=3                   # 3 pt linear interpolation
    # put maximum in center of array to avoid edge effects
    shift = int(size(array)/2. - array.argmax())
    array = roll(array, shift)
    amax=array.max()
    amaxloc=array.argmax()
    xvec=range(size(array))
    half=where(diff(sign(array[amaxloc:]-amax/2.)))[0][0]
    start=amaxloc+half-(ninterp-1)//2
    xinterp=xvec[start:start+ninterp]
    yinterp=array[start:start+ninterp]
    hwhm = interp(amax/2., yinterp[::-1], xinterp[::-1])-amaxloc
    return hwhm


def find_fwhm(array):
    """
    Finds full width at half maximum in sample numbers via interpolation.
    """
    ninterp=3                   # 3 pt linear interpolation
    # put maximum in center of array
    shift = int(size(array)/2. - array.argmax())
    array = roll(array, shift)
    amax=array.max()
    amaxloc=array.argmax()
    xvec=range(size(array))
    half1=where(diff(sign(array[:amaxloc]-amax/2.)))[0][0]
    half2=where(diff(sign(array[amaxloc:]-amax/2.)))[0][0]
    start1=half1-(ninterp-1)//2
    start2=amaxloc+half2-(ninterp-1)//2
    xinterp1=xvec[start1:start1+ninterp]
    yinterp1=array[start1:start1+ninterp]
    xinterp2=xvec[start2:start2+ninterp]
    yinterp2=array[start2:start2+ninterp]
    hwhm_minus = -interp(amax/2., yinterp1, xinterp1)+amaxloc
    hwhm_plus = interp(amax/2., yinterp2[::-1], xinterp2[::-1])-amaxloc
    fwhm = hwhm_minus+hwhm_plus
    return fwhm


def dmshift(fMHz, fref, DM, tbin):
  """
  Calculates the DM shift in sample numbers for frequency fMHz relative
  to frequency fref   

  fMHz, fref 			MHz
  DM 				pc/cc
  tbin = time per sample 	seconds
  """
  dc = 2.41e-16			# dispersion constant for f in Hz, DM in pc/cc
				# NEED TO CHECK THAT THIS DC IS USED IN
				# IN THE COHERENT DEDISPERSION
  dmshift_time = (DM/(dc*1.e12)) * (1./fMHz**2 - 1./fref**2)  
  dmshift_samples = dmshift_time / tbin  
  return dmshift_time, dmshift_samples

def shiftit(y, shift):
  """
  shifts array y by amount shift (in sample numbers) 
  uses shift theorem and FFT
  shift > 0  ==>  lower sample number (earlier)
  modeled after fortran routine shiftit
  """
  yfft = fft(y)
  constant = (shift*2*pi)/float(size(y)) 
  theta = constant*arange(size(y))
  c = cos(theta)
  s = sin(theta)
  work = zeros(size(y), dtype='complex') 
  work.real = c * yfft.real - s * yfft.imag
  work.imag = c * yfft.imag + s * yfft.real
  # enforce hermiticity
  for n in range(size(y)//2):
    work.real[size(y)-n-1] =  work.real[n+1]
    work.imag[size(y)-n-1] = -work.imag[n+1]
  work[size(y)//2] = 0.+0.j
  workifft = ifft(work)
  return workifft.real

def test_dtsnr_expressions():
  """
  Compare predicted TOA error using t and f-domain approaches
     1. Cordes & Shannon 2010:
        Using \Delta t = P/N:

        dtsnr = \frac
		{\Delta t}
		{\rm SNR \left[ \sum_{t=1}^{N-1}(U_t - U_{t-1})^2\right]^{1/2}}

        defining the effective width $W_{\rm eff}$ as (different from CS10):

	W_{\rm eff} = 
		\frac {\Delta t} 
                      {\left[ \sum_{t=1}^{N-1}(U_t - U_{t-1})^2\right]^{1/2}} 

        this becomes

	dtsnr = \frac{W_{\rm eff}}{\rm SNR}

        Better:

        W_{\rm eff} =
                \sqrt{N}\frac {\Delta t}
                      {\left[ \sum_{t=1}^{N-1}(U_t - U_{t-1})^2\right]^{1/2}}

        This is better because it does not depend on N.
 
        and then

	dtsnr = \frac{W_{\rm eff}}{\sqrt{N} \rm SNR}

    2. J. Taylor's (1992) expression:

	dtsnr = 
           \frac 
           {\sqrt{N} P}
           {2\pi \rm SNR
              \left[\sum_{k=1}^{N/2}k^2\vert\tildeS_k\vert^2\right]^{1/2}}

        This corresponds to an effective width (using the "better" definition
        above::

        W_{\rm eff} = 
	\frac           
        {N P}           
        {2\pi\left[\sum_{k=1}^{N/2}k^2\vert\tildeS_k\vert^2\right]^{1/2}}

    This routine shows the equivalence of these expressions.

    Note that SNR in this expression is for the SNR of the profile being
    analyzed.   For the same conditions (bandwidth, total observing time,
    Tsys), SNR will depend on the number of profile bins being used. 
  """
  N=256
  N=64
  N=1024
  N=2048
  N = int(six.moves.input('enter N = # samples across P: '))
  P=1.				# seconds
  dt=P/N			# profile sample interval (sec)
  W = 0.05*P			# FWHM (sec)
  W_bins = W/dt			# bins
  We = W / (2.*sqrt(log(2.)))		# 1/e half width (sec) 
  We_bins = We/dt		# 1/e half width (bins)
  print("W, W_bins, We, We_bins = ", W, W_bins, We, We_bins)
  print()
  snr=15.
  snr=50.
  # expected sigtoa from continuous expression in RD83, CD85, CS10 
  dtsnrc = (2.*pi*log(2.))**(-0.25)*sqrt(W*P/float(N))/snr
  print("dtsnrc = ", dtsnrc, " from continuous expression for Gaussian")
  
  sigmat = 1./snr	# assumes randn gives rms = 1.
  print("snr, sigmat = ", snr, sigmat)
  shift = -3.1			# bins
  x = arange(-N//2.,N//2.)	# bins
  arg = ((x - shift)/We_bins)**2
  template=exp(-arg)
  y = template + randn(size(x))/snr
  plot(x,y)

  # calculate dtsnr using CS10 expression:
  sum1=0.
  for n in range(1,N):
     sum1 += (template[n]-template[n-1])**2 
  int_numerical = sum1 / dt
  weff1 = dt / sqrt(sum1)		# sec
  dtsnr1 = dt / (snr*sqrt(sum1)) 	# sec

  # dtsnr using Taylor FFT report:
  ffttemplate= fft(template)
  sum2 = 0.
  for k in range(1,N//2):
    sum2 += k**2*abs(ffttemplate[k])**2 
  weff2 = (P*sqrt(N))/(2.*pi*sqrt(2.*sum2))	 # sec  
  dtsnr2 = (P*sqrt(N))/(2.*pi*snr*sqrt(2.*sum2)) # sec 

  print("dtsnr1, dtsnr2 = ", dtsnr1, dtsnr2, " sec")
  print("ratio = ", dtsnr2/ dtsnr1)
  print()
  print("ratio dtsnrc/dtsnr1 = ", dtsnrc/dtsnr1)
  print()
  print("W (FWHM) = ", W/dt, " bins ", W, " sec")
  print("We (HW @ 1/e) = ", We_bins, " bins ", We, " sec")
  print("weff1, weff2 = ", weff1, weff2)
  print("ratio weff1/W = ", weff1/W)
  print()
  print("alternative definitions of Weff:")
  print("Weff_alt0 = SNR*dtsnr1 = ", snr*dtsnr1) 
  print("Weff_alt1 = sqrt(N)*SNR*dtsnr1 = ", sqrt(N)*snr*dtsnr1)
  print("Weff_alt2 = N*SNR*dtsnr1 = ", N*snr*dtsnr1)
  print()

  # compare with output from get_toa:

  toa, b, dtoa, db = get_toa(template, y, sigmat) 
  print("toa(bins), b, dtoa(bins), db = ", toa, b, dtoa, db)
  print("toa(sec), b, dtoa(sec), db = ", toa*dt, b, dtoa*dt, db)

  print()
  # numerical vs analytical integrals:
  int_analytical = sqrt(2.*pi*log(2.)) / W
  print("int_analytical = ", int_analytical)
  print("int_numerical = ", int_numerical)

  print("ratio = ", int_analytical / int_numerical)
  print()
  print("Effective widths vs N for W = ", W)
  print("  Nn   Weff1    Weff_alt1    dtsnr1")
  for Nn in (256, 512, 1024, 2048):
    dt = P/Nn
    W_bins = W/dt			# bins
    We = W / (2.*sqrt(log(2.)))		# 1/e half width (sec) 
    We_bins = We/dt		# 1/e half width (bins)
    x = arange(-Nn/2.,Nn/2.)	# bins
    arg = ((x - shift)/We_bins)**2
    template=exp(-arg)
    sum1=0.
    for n in range(1,Nn):
       sum1 += (template[n]-template[n-1])**2 
    int_numerical = sum1 / dt
    weff1 = dt / sqrt(sum1)		# sec
    dtsnr1 = weff1 / snr		# sec
    weff_alt1 = sqrt(Nn)*dt/sqrt(sum1)	# sec
    print(Nn,  weff1, weff_alt1, dtsnr1)
     

  show()
  #savefig('test_shiftit.png')
  six.moves.input('hit return when done with plot')
  close()

  # repeat for 100 realizations, compare numerical error with observed sigma:
  print()
  nrlz=1000
  nrlz=100
  nrlz=5000
  print("Calculating TOA stats for ", nrlz, " realizations:")
  toavec = zeros(nrlz)
  for n in range(nrlz):
     if n%50 == 0: print("   realization ", n)
     y = template + randn(size(x))/snr
     toa, b, dtoa, db = get_toa(template, y, sigmat) 
     toavec[n] = toa
  sigtoa_bins = std(toavec)			# bins
  sigtoa = sigtoa_bins*dt			# sec
 
  toavec_wunits = toavec*dt/W			# toa's in units of W 
  sigtoa_wunits = sigtoa_bins*dt/W		# sigtoa in units of W
  sigtoa_predicted_wunits = dtsnr1/W		# predicted dtoa in W units

  print()
  print(nrlz, " realizations for N = ", N)
  print("sigtoa_bins, sigtoa = ", sigtoa_bins, sigtoa)
  print("ratio sigtoa / sigtoa_predicted(dtsnr2) = ", sigtoa / dtsnr2)

  # use bins of (toa/W)*snr:
  # use bins of sigtoa units: (i.e. divide TOAs by predicted TOAs
  #hmax = 5.*sigtoa_predicted_wunits*snr
  #dtoahist, histbin_edges = histogram(toavec_wunits*snr, bins=50, range=(-hmax, hmax))
  hmax = 5.
  dtoahist, histbin_edges = histogram(toavec*dt/dtsnr1, bins=50, range=(-hmax, hmax))

  plot(histbin_edges[0:size(dtoahist)], dtoahist, drawstyle='steps-post')
  #arg = 0.5*(histbins/(snr*sigtoa_predicted_wunits))**2
  #gpdf = (sqrt(2.*pi)*sigtoa_predicted_wunits*snr)**(-1.)*exp(-arg)
  arg = 0.5*histbin_edges**2
  gpdf = (sqrt(2.*pi))**(-1.)*exp(-arg)
  dx=histbin_edges[1]-histbin_edges[0]
  plot(histbin_edges, gpdf*nrlz*dx)
  tick_params(axis='both', labelsize=14)
  #xlabel(r'$\Delta t_{\rm S/N} \times \rm (S/N) / W$', fontsize=15)
  xlabel(r'$\Delta t / \Delta t_{\rm S/N}$', fontsize=15)
  ylabel(r'$\rm Counts$', fontsize=15)
  #annotate('S/N = %3.0f'%(snr), xy=(0.05,0.9), xycoords='axes fraction', ha='left', fontsize=15)
  a,b,c,d=axis()
  axis(xmin=-hmax, xmax=hmax,ymin=c-0.03*(d-c), ymax=d+0.05*(d-c))
  plotfile='test_dtsnr_expressions_toahist_' + str(nrlz) + '.png'
  show()
  savefig(plotfile)
  six.moves.input('hit return when done')
  close()

  return toavec, dtoahist, histbin_edges

def test_shiftit():
  N=256
  W = 10.
  snr=50.
  shift = -3.1
  x = arange(-N/2.,N/2.)
  arg = ((x - shift)/W)**2
  y = exp(-arg) + randn(size(x))/snr
  plot(x,y)
  show()
  yshift = shiftit(y, 3.1)
  plot(x,yshift)
  savefig('test_shiftit.png')
  six.moves.input('hit return when done with plot')
  return yshift 

def get_toa3(template, profile, sigma_t, dphi_in=0.1, snrthresh=0., nlagsfit=5, norder=2, debug=False):
  """
  Calculates TOA and its error in samples (bins).
  Uses least-squares method in frequency domain, minimizing chi^2.
  Also calculates scale factor for template matching.
  Input: template = template file; if normalized to unity max,
              the scale factor divided by the input sigma_t is
              the peak to rms S/N.
         profile = average profile to process
         sigma_t = off pulse rms in same units as profile.
  Output:     
  tauccf = TOA (bins) based on parabloic interpolation of CCF.
  tauhat = TOA (bins) using Fourier-domain fitting., 
  bhat = best-fit amplitude of pulse. 
  sigma_tau = error on tauhat. 
  sigma_b = error on bhat.
  snr = bhat/sigma_t.
  rho = cross correlation coefficient between template and centered profile.
  """
  # Some initial values:
  snr_coarse = max(profile)/sigma_t
  tauhat = 0.
  bhat = 0.
  sigma_tau = -1.
  sigma_b = -1.
  rho = -2.

  # find coarse estimates for scale factor and tau from CCF maximum 
  #  (quadratically interpolated)
  ccf = correlate(template, profile, 'full')
  lags = arange(-size(profile)+1., size(profile), 1.)
  ccfmaxloc = ccf.argmax()
  ccffit = ccf[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]
  lagfit = lags[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]
#  print lagfit,ccffit,norder,ccfmaxloc,nlagsfit,ccfmaxloc-(nlagsfit-1)/2,ccfmaxloc+(nlagsfit-1)/2+1,type(lagfit) ##EDITED BY MICHAEL
  p = polyfit(lagfit, ccffit, norder)
  ccfhat = p[0] + p[1]*lagfit + p[2]*lagfit**2
  tauccf = p[1]/(2.*p[2])

  if debug:
     plot(template, 'k-', lw=3)
     plot(profile, 'r-', lw=1)
     six.moves.input('hit return')

  # roughly center the pulse to line up with the template:
  ishift = int(-tauccf)
  profile = roll(profile, ishift)
  if debug:
     print("ishift = ", ishift)
     plot(profile, 'g-', lw=1)
     six.moves.input('hit return')
  bccf = sum(template*profile)/sum(template**2) 

  if debug: 
     print("bccf, tauccf = ", bccf, tauccf)

  # Search range for TOA using Fourier-domain method:
  # expect -fwhm/2 < tauhat < fwhm/2  since pulse has been centered

  # fwhm, taumin, taumax currently not used.  But should we do a 
  # windowed TOA calculation? 
  fwhm = find_fwhm(template)		# fwhm in samples (bins)
  taumin = -fwhm/2.
  taumax = fwhm/2.
  
  tfft = fft(template)
  pfft = fft(profile)
  bhat0 = bccf
  tauhat0 = tauccf+ishift
  paramvec0 = array((bhat0, tauhat0))
  
  paramvec = spo.minpack.leastsq(tfresids, paramvec0, args=(tfft, pfft)) 
  bhat = paramvec[0][0]
  tauhat = paramvec[0][1]
  if debug:
     print(paramvec0)
     print(paramvec)
     print("bhat = ", bhat)
     print("tauhat = ", tauhat)
  sigma_tau, sigma_b = toa_errors_additive(tfft, bhat, sigma_t)
  # snr = scale factor / sigma_t:
  snr = (bhat*template.max())/sigma_t
  # rho = correlation coefficient of template and shifted profile:
  profile_shifted = shiftit(profile, +tauhat)	# checked sign: this is correct
  if debug:
     plot(profile_shifted, 'b-', lw=2)
     six.moves.input('hit return')
  # TBA: correct rho for off-pulse noise.  
  # Two possibilities: 
  #     1. subtract noise variance term  from sum(profile_shifted**2) 
  #  or 2. calculate ACF of profile_shifted with a one or two sample lag.
  rho = sum(template*profile_shifted) / sqrt(sum(template**2)*sum(profile_shifted**2))
  tauhat = tauhat - ishift	# account for initial shift
  return tauccf, tauhat, bhat, sigma_tau, sigma_b, snr, rho

def tfresids(params, tfft, pfft):
  """
  """
  b=params[0]
  tau=params[1]
  Nfft = size(pfft)
  Nsum = Nfft//2
  arg=(2.*pi*tau/float(Nfft)) * arange(0., Nfft, 1.)
  phasevec = cos(arg) - 1j*sin(arg)
  #resids = abs(pfft[1:Nsum] - b*tfft[1:Nsum]*phasevec[1:Nsum])
  resids = abs(pfft[1:Nsum] - b*tfft[1:Nsum]*phasevec[1:Nsum])
  return resids

def test_chi2(tfft, pfft):
    b=1.0
    tauvec=arange(-10., 10., 0.02)
    chi2vec=[]
    for tau in tauvec:
       params=array((b, tau))
       chi2vec=append(chi2vec, chi2(params, tfft, pfft))
    plot(tauvec, chi2vec, 'k-')
    return tauvec, chi2vec
    

def chi2(params, tfft, pfft):
  """
  Chi^2 for template fitting in Fourier domain.
  """
  b=params[0]
  tau=params[1]
  Nfft = size(pfft)
  Nsum = Nfft//2
  arg=(2.*pi*tau/float(Nfft)) * arange(0., Nfft, 1.)
  #phasevec = complex(cos(arg), sin(arg))
  phasevec = cos(arg) - 1j*sin(arg)
  summand = abs(pfft[1:Nsum] - b*tfft[1:Nsum]*phasevec[1:Nsum])**2 
  chi2 = sum(summand)
  return chi2


def get_toa_new(template, profile, sigma_t, dphi_in=0.1, snrthresh=0., nlagsfit=5, norder=2, debug=False):
  """
  Calculates TOA and its error in samples (bins).
  Also calculates scale factor for template matching.
  Input: template = template file; if normalized to unity max,
              the scale factor divided by the input sigma_t is
              the peak to rms S/N.
         profile = average profile to process
         sigma_t = off pulse rms in same units as profile.
         
       
  """
  # process only if S/N > snrthresh:
  snr_coarse = max(profile)/sigma_t
  tauhat = 0.
  bhat = 0.
  sigma_tau = -1.
  sigma_b = -1.
  rho = -2.
  #if snr_coarse < snrthresh:
    #return tauhat, bhat, sigma_tau, sigma_b


  # find coarse TOA  estimate from CCF maximum (quadratically interpolated)
  ccf = correlate(template, profile, 'full')
  lags = arange(-size(profile)+1., size(profile), 1.)
  ccfmaxloc = ccf.argmax()
  ccffit = ccf[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]
  lagfit = lags[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]
  p = polyfit(lagfit, ccffit, norder)
  ccfhat = p[0] + p[1]*lagfit + p[2]*lagfit**2
  tauccf = p[1]/(2.*p[2])
  #tauccf -= ishift
  if debug: print("tauccf = ", tauccf)

  if debug:
     plot(template, lw=3)
     plot(profile, lw=1)

  # roughly center the pulse to line up with the template:
  #ishift = int(size(profile)/2.-profile.argmax())
  #ishift = int(size(profile)/2.-tauccf)
  ishift = int(-tauccf)
  profile = roll(profile, ishift)
  if debug:
     print("ishift = ", ishift)
     plot(profile, lw=1)

  # Search range for TOA using Fourier-domain method:
  # expect -fwhm/2 < tauhat < fwhm/2  since pulse has been centered

  fwhm = find_fwhm(template)		# fwhm in samples (bins)
  taumin = -fwhm/2.
  taumax = fwhm/2.
  
  tfft = fft(template)
  pfft = fft(profile)
  # test that interval contains a zero of Zfft: 
  # if not, return default values: 
  Zproduct = Zfft(taumin, tfft, pfft)*Zfft(taumax, tfft, pfft)
  #print "tauwindow,tmaxt, tmaxp, taumin,max, Zproduct = ", tauwindow,tmaxt, tmaxp, taumin, taumax, Zproduct
  if Zproduct > 0.:
      #print "got here2"
      return tauccf, tauhat, bhat, sigma_tau, sigma_b, snr_coarse, rho
  tauhat = spo.zeros.brentq(Zfft, taumin, taumax, args=(tfft, pfft), maxiter=1000)
  bhat = Qfft(tauhat, tfft, pfft).real
  sigma_tau, sigma_b = toa_errors_additive(tfft, bhat, sigma_t)
  # snr = scale factor / sigma_t:
  snr = (bhat*template.max())/sigma_t
  # rho = correlation coefficient of template and shifted profile:
  profile_shifted = shiftit(profile, -tauhat)
  if debug:
     plot(profile_shifted, lw=2)
  rho = sum(template*profile_shifted) / sqrt(sum(template**2)*sum(profile_shifted**2))
  tauhat = tauhat - ishift	# account for initial shift
  return tauccf, tauhat, bhat, sigma_tau, sigma_b, snr, rho

def get_toa(template, profile, sigma_t, dphi_in=0.1, snrthresh=0.):
  """
  Calculates TOA and its error in samples (bins).
  Also calculates scale factor for template matching.
  Input: template = template file; if normalized to unity max,
              the scale factor divided by the input sigma_t is
              the peak to rms S/N.
         profile = average profile to process
         sigma_t = off pulse rms in same units as profile.
         
       
  """
  # process only if S/N > snrthresh:
  snr = max(profile)/sigma_t
  tauhat = 0.
  bhat = 0.
  sigma_tau = -1.
  sigma_b = -1.
  if snr < snrthresh:
    return tauhat, bhat, sigma_tau, sigma_b
    
  # ******** make wider pulse window **********
  # use kwarg to define high S/N window size
  # define search window in accordance with S/N:
  # S/N > 20 		2% of the number of phase bins
  # 15 < S/N <= 20 	4% 
  #  8 < S/N <= 15	6%
  dphi=dphi_in
  #if snr > 20: 
      #dphi = dphi_in
  if 15. < snr and snr <= 20.: 
      dphi = 2.*dphi_in
  if  8. <= snr and snr <= 15.: 
      dphi = 3.*dphi_in
  if dphi > 0.5: dphi=0.5
  nphasebins=size(template)
  tauwindow = int(dphi*nphasebins)
  
  pmax=profile.max()
  tmaxp=profile.argmax()
  tmax=template.max()
  tmaxt=template.argmax()
  taumin = tmaxp-tmaxt - tauwindow/2.
  #taumin = tmaxt - tauwindow/2.

  # if taumin < 0.: taumin += size(template)
  taumax = taumin + tauwindow
  tfft = fft(template)
  pfft = fft(profile)
  # test that interval contains a zero of Zfft: 
  # if not, return default values: 
  Zproduct = Zfft(taumin, tfft, pfft)*Zfft(taumax, tfft, pfft)
  #print "tauwindow,tmaxt, tmaxp, taumin,max, Zproduct = ", tauwindow,tmaxt, tmaxp, taumin, taumax, Zproduct
  if Zproduct > 0.:
      #print "got here2"
      return tauhat, bhat, sigma_tau, sigma_b
  tauhat = spo.zeros.brentq(Zfft, taumin, taumax, args=(tfft, pfft), maxiter=1000)
  bhat = Qfft(tauhat, tfft, pfft).real
  sigma_tau, sigma_b = toa_errors_additive(tfft, bhat, sigma_t)
  return tauhat, bhat, sigma_tau, sigma_b

def Zfft(tau, tfft, pfft):
  """
  defines function whose zero crossing is an estimate of the TOA, tauhat
  input:
	tau = trial TOA
        tfft = fft of template profile
	pfft = fft of pulse profile whose TOA is wanted
  output:
	Zfft = sum (P*conj(T)*exp(+2pi*i*tau))
	where P,T are FTs of profile and template
  method: 
	Taylor algorithm (Taylor 1992)
  """
  Nfft = size(pfft)
  phasefac = numpy.zeros(size(pfft), dtype=complex)
  Nsum=size(pfft)//2
  for n in range(size(pfft)):
    arg = 2.*pi*n*tau/float(Nfft)
    phasefac[n] = complex(cos(arg), sin(arg)) 
  summand= range(1,Nsum)*imag(pfft[1:Nsum]*conj(tfft[1:Nsum])*phasefac[1:Nsum]) 
  Z = sum(summand)
  return Z				# real

def Qfft(tau, tfft, pfft):
  """
  function whose real part is the scale factor and
  imaginary part has a zero crossing at TOA estimate
  input:
	tau = trial TOA
        tfft = fft of template profile
	pfft = fft of pulse profile
  output:
	Q = sum (P*conj(T)*exp(+2pi*i*tau)) / sum(|T|^2)
	where P,T are FTs of profile and template
  """

  Nfft = size(pfft)
  phasefac = numpy.zeros(size(pfft), dtype=complex)
  Nsum=size(pfft)//2
  for n in range(size(pfft)):
    arg = 2.*pi*n*tau/float(Nfft)
    phasefac[n] = complex(cos(arg), sin(arg)) 
  summand= (pfft[1:Nsum]*conj(tfft[1:Nsum])*phasefac[1:Nsum])  
  Q = sum(summand) / sum(abs(tfft[1:Nsum])**2)
  return Q

def Z(tau, template, profile):
  """
  defines function whose zero crossing is an estimate of the TOA, tauhat
  input:
	tau = trial TOA
	profile = pulse profile whose TOA is wanted
        template = template profile
  output:
	Z = sum (P*conj(T)*exp(+2pi*i*tau))
	where P,T are FTs of profile and template
  method: 
	Taylor algorithm (Taylor 1992)
  """
  pfft = fft(profile)
  tfft = fft(template)
  Nfft = size(pfft)
  phasefac = numpy.zeros(size(pfft), dtype=complex)
  Nsum=size(pfft)//2
  for n in range(size(pfft)):
    arg = 2.*pi*n*tau/float(Nfft)
    phasefac[n] = complex(cos(arg), sin(arg)) 
  summand= range(1,Nsum)*imag(pfft[1:Nsum]*conj(tfft[1:Nsum])*phasefac[1:Nsum]) 
  Z = sum(summand)
  return Z				# real

def Q(tau, template, profile):
  """
  function whose real part is the scale factor and
  imaginary part has a zero crossing at TOA estimate
  input:
	tau = trial TOA
	profile = pulse profile whose TOA is wanted
        template = template profile
  output:
	Q = sum (P*conj(T)*exp(+2pi*i*tau)) / sum(|T|^2)
	where P,T are FTs of profile and template
  method: 
	JMC algorithm (1990s)
	tauhat corresponds to \partial_\tau Re(Q) = 0 or Im(Q) = 0.
  """

  pfft = fft(profile)
  tfft = fft(template)
  Nfft = size(pfft)
  phasefac = numpy.zeros(size(pfft), dtype=complex)
  Nsum=size(pfft)//2
  for n in range(size(pfft)):
    arg = 2.*pi*n*tau/float(Nfft)
    phasefac[n] = complex(cos(arg), sin(arg)) 
  summand= (pfft[1:Nsum]*conj(tfft[1:Nsum])*phasefac[1:Nsum])  
  Q = sum(summand) / sum(abs(tfft[1:Nsum])**2)
  return Q

def toa_errors_additive(tfft, b, sigma_t):
  """
  Calculates error in b = scale factor and tau = TOA due to additive noise.

  input:
        fft of template 
        b = fit value for scale factor
	sigma_t = rms additive noise in time domain
  output:
  	sigma_b
        sigma_tau
  """
  Nfft = size(tfft)
  Nsum = Nfft // 2
  kvec = arange(1,Nsum)
  sigma_b = sigma_t*sqrt(float(Nfft) / (2.*sum(abs(tfft[1:Nsum])**2)))
  sigma_tau = (sigma_t*Nfft/(2.*pi*abs(b))) \
		* sqrt(float(Nfft) \
		/ (2.*sum(kvec**2*abs(tfft[1:Nsum])**2)))
  return sigma_tau, sigma_b

def test_stuff():
  t=arange(1024)
  W=10.
  We = W / (2.*log(2.))			# 
  offset = 512.
  dtoa = 3.1
  snr=10.
  snr=100.
  snr=1000.
  snr=float(six.moves.input("enter snr: "))
  dtoa=float(six.moves.input("enter dtoa (bins): "))
  taumin=-10.+dtoa
  taumax=taumin+20.
  template=exp(-((t-offset)/We)**2)
  # create profile with fractional shift and then rotate by integer part of dtoa
  dtoa_frac, dtoa_int = modf(dtoa)
  dtoa_int=int(dtoa_int)
  print(dtoa_frac, dtoa_int)
  #profile_nonoise=exp(-((t-offset-dtoa)/We)**2) 
  profile_nonoise_frac=exp(-((t-offset-dtoa_frac)/We)**2) 
  profile_nonoise=numpy.zeros(size(t))
  for n in range(size(t)):
    index = n-dtoa_int
    if index < 0: index += size(t)
    if index > size(t)-1: index -= size(t)
    profile_nonoise[n] = profile_nonoise_frac[index]
    
  profile = snr*profile_nonoise + randn(size(t))
  subplot(211)
  plot(template)
  subplot(212)
  plot(profile)
  tfft = fft(template)
  pfft = fft(profile)
  sigma_t = 1./snr		# since rms(randn) = 1. and gaussian peak = 1
  tauhat = spo.zeros.brentq(Zfft, taumin, taumax, args=(tfft, pfft))
  bhat = Qfft(tauhat, tfft, pfft).real
  sigma_tau, sigma_b = toa_errors_additive(tfft, bhat, sigma_t)
  print(snr, sigma_t, tauhat, sigma_tau, bhat, sigma_b)
  tauhat2, bhat2, sigma_tau2, sigma_b2 = get_toa(template, profile, sigma_t)
  print(snr, sigma_t, tauhat2, sigma_tau2, bhat2, sigma_b2)
  six.moves.input('hit enter when done')
  close()
  return

def sigtoa_vs_snr():
  """
  plots rms TOA vs S/N
  """
  nrealizations=100
  nrealizations=10
  W = 10.				# FWHM
  We = W / (2.*sqrt(log(2.)))			# 
  sigtoa_predicted_snr1 = W /(2.*pi*log(2.))**(0.25)
  offset = 512.
  dtoa = 3.1
  taumin=-10.
  taumax=10.
  t=arange(1024)
  template=exp(-((t-offset)/We)**2)
  profile_nonoise=exp(-((t-offset-dtoa)/We)**2) 
  snrvec = 10.**arange(0.6, 4.1, 0.3)
  #snrvec = 10.**arange(1.0,1.01, 0.3)
  rmstoavec = numpy.zeros(size(snrvec))
  sigtoa_predicted_vec = numpy.zeros(size(snrvec))
  for n, snr0 in enumerate(snrvec): 
    print("processing snr = ", snr0)
    tauvec=numpy.zeros(nrealizations)
    bvec = numpy.zeros(nrealizations)
    sigtoa_predicted = sigtoa_predicted_snr1 / snr0
    sigtoa_predicted_vec[n] = sigtoa_predicted
    for nrlz in range(nrealizations):
      #snr = snr0 * (1.+0.3*cos(2.*pi*nrlz / nrealizations))
      snr = snr0
      profile=profile_nonoise + randn(size(template))/snr
      tauhat = spo.zeros.brentq(Z, taumin, taumax, args=(template, profile))
      bhat = Q(tauhat, template, profile).real
      tauvec[nrlz] = tauhat
      bvec[nrlz] = bhat
    rmstoavec[n] = std(tauvec)
  plot(snrvec, rmstoavec, 'ko')
  plot(snrvec, sigtoa_predicted_vec, 'bo')
  xscale('log')
  yscale('log')
  xlabel(r'$\rm S/N$', fontsize=18)
  ylabel(r'$\rm \sigma_{\rm TOA}$', fontsize=18)
  show()
  savefig('rmstoa_vs_snr.png')
  six.moves.input('hit return when done')
  close()
  return

def remove_off_pulse_mean(profile):
  """
  Remove off-pulse means from input profile:
  """
  # define window for off-pulse mean
  dphase = 0.1
  nphasebins=size(profile)
  windowsize = int(dphase * nphasebins)
  avepmax, aveploc = posmax3(profile)
  binoff1 = aveploc + 2.*windowsize
  #if binoff1 > nphasebins: binoff1 -= nphasebins
  # find off-pulse statistics
  offvector=zeros(windowsize)
  for nphi in range(windowsize):
    binoff = binoff1 + nphi
    if binoff > nphasebins-1: binoff -= nphasebins
    offvector[nphi] = profile[binoff]
  noff,(offmin,offmax),offmean,offvar,offskew,offkurt=stats.describe(offvector)
  offrms=sqrt(offvar)
  # subtract mean
  profile -= offmean
  return profile, offmean

def fit_across_channels(template, channel_profiles, channel_frequencies, nsubints2sum):
    """
    Calculates slope across frequency by fitting TOAs and shifts profiles

    """ 
    nsubints=channel_profiles.shape[0]
    nsubbands=channel_profiles.shape[1]    
    nphasebins=channel_profiles.shape[2]
    
    # calculate profiles averaged over nsubints2sum integrations:
    print("  calculating profiles:")
    channel_profiles_integrated = zeros(nsubbands*nphasebins)
    channel_profiles_integrated.shape = (nsubbands, nphasebins)    
    for ni in range(nsubints2sum):
      for ns in range(nsubbands):
        channel_profiles_integrated[ns] += channel_profiles[ni, ns]
    channel_profiles_integrated /= nsubints2sum
    toavec=zeros(nsubbands)
    bvec=zeros(nsubbands)
    print("  calculating initial toas vs. channel of time-integrated profiles")
    for ns in range(nsubbands):
      toavec[ns], bvec[ns] = get_toa(template, channel_profiles_integrated[ns],1.)[0:2]
    print("  doing least squares fit across frequency")
    dtoafit=[]
    freqfit=[]
    bweights=[]
    for ns in range(nsubbands):
      if bvec[ns] != 0.: 
          bweights.append(bvec[ns])
          dtoafit.append(toavec[ns])
          freqfit.append(channel_frequencies[ns])
    bweights=array(bweights)
    dtoafit=array(dtoafit)
    freqfit=array(freqfit)
    chanfreqmax=max(channel_frequencies)   # note true max, not max of freqfit
    xfit=(1./freqfit**2 - 1./chanfreqmax**2) 
    # do weighted least-squares fit of straight line: 
    #	dtoa = constant + slope*xfit with bweights as weights
    p = polyfit(xfit, dtoafit, 1, w=bweights)
    xeval=(1./channel_frequencies**2 - 1./chanfreqmax**2) 
    dtoahat = p[0] + p[1]*xeval
    # shift channel profiles
    print("  shifting profiles using fit")
    channel_profiles_fixed = zeros(shape(channel_profiles))
    channel_profiles_integrated = zeros((nsubbands,nphasebins))
    broadband_profiles=zeros((nsubints2sum, nphasebins))
    for ni in range(nsubints2sum):
        for ns in range(nsubbands):
           shift = dtoahat[ns] 
           channel_profiles_fixed[ni,ns] = shiftit(channel_profiles[ni,ns], shift)
           channel_profiles_integrated[ns] += channel_profiles_fixed[ni, ns]
           broadband_profiles[ni] += channel_profiles_fixed[ni, ns]
    channel_profiles_integrated /= nsubints2sum
    broadband_profiles /= nsubbands
    # now do toa analysis of broadband profiles to remove drift in time
    print("  doing least squares fit across subintegrations")
    toa2vec=zeros(nsubints2sum)
    b2vec=zeros(nsubints2sum)
    xfit = zeros(nsubints2sum)
    for ni in range(nsubints2sum):
      toa2vec[ni], b2vec[ni] = get_toa(template, broadband_profiles[ni],1.)[0:2]
      xfit[ni] = float(ni)  
    p = polyfit(xfit, toa2vec, 1, w=b2vec)
    xeval=arange(float(nsubints2sum))
    toa2hat = p[0] + p[1]*xeval

    print("  shifting profiles")
    #channel_profiles_fixed = zeros(shape(channel_profiles)) # fix the fixed ones!
    channel_profiles_integrated = zeros((nsubbands,nphasebins))
    broadband_profiles=zeros((nsubints2sum, nphasebins))
    for ni in range(nsubints2sum):
        for ns in range(nsubbands):
           shift = toa2hat[ni] 
           channel_profiles_fixed[ni,ns] = shiftit(channel_profiles_fixed[ni,ns], shift)
           channel_profiles_integrated[ns] += channel_profiles_fixed[ni, ns]
           broadband_profiles[ni] += channel_profiles_fixed[ni, ns]
    channel_profiles_integrated /= nsubints2sum
    broadband_profiles /= nsubbands
    return channel_profiles_fixed, channel_profiles_integrated, broadband_profiles, toavec, bvec, dtoahat, toa2vec, b2vec, toa2hat

def calc_weighted_sum(w, d):
  """
  Calculates the weighted sum of 1D data vector d with 1d weights vector w
  """
  return sum(w*d)/sum(w)

def calculate_pulse_intensities(template, channel_profiles):
  """
  Uses template as weighting function to calculate pulsed intensities.

  Input:
	template = 1D array
	channel_profiles = sequence of pulse profiles vs. frequency
        (assumed shape is (nsubints, nsubbands, nphasebins)
  Output:
	array of amplitudes
  """ 
  nsubints=channel_profiles.shape[0]
  nsubbands=channel_profiles.shape[1]    
  nphasebins=channel_profiles.shape[2]
  pulse_intensities=zeros((nsubints, nsubbands))
  for ni in range(nsubints):
    for ns in range(nsubbands):
      pulse_intensities[ni, ns] = \
         calc_weighted_sum(template, channel_profiles[ni, ns]) 
  return pulse_intensities

def spike_remove(vector, fracthresh):
  """
  Removes spikes from 1D vector by comparing nearest neighbors

  Input = vector to fix
  fracthresh = threshold above local mean to correct in units of local mean
  (e.g. fracthresh = 0.05 to remove spikes 5% above the local mean)
  """ 
  thresh = 1. + fracthresh
  N=size(vector)
  vectorout=zeros(N)
  nfix=0
  for n in range(2,N-1): 
    vectorout[n] = vector[n]
    m2 = 0.5*(vector[n-1] + vector[n+1])
    if vector[n] > thresh*m2:
      nfix+=1
      vectorout[n] = m2 
      #print "fixing ", n, m2, vector[n], vectorout[n]
  return vectorout, nfix

def pca_core_v1(template, vector_array, plotlabel=''):
  """
  Does PCA on difference vectors 

  Difference vectors are calculated by scaling the input template
  (which can have arbitrary normalization) to minimize the 
  mean-square difference between it and each vector. 
  """
  #nsubints,nphasebins=shape(broadband_profiles)[0:2]
  nvectors,vlength=shape(vector_array)[0:2]
  dvectors = zeros(shape(vector_array)) 
  for ni in range(nvectors):
    scaling = sum(template*vector_array[ni]) / sum(template**2)
    dvectors[ni] = vector_array[ni] - scaling*template
    #print ni, scaling

  # Plot vectors and difference vectors for window selection:
  subplot(121)
  for ni in range(nvectors):
    plot(vector_array[ni] + ni*0.1*max(vector_array[0]))
  axis(xmin=0.025*vlength, xmax=1.025*vlength) 

  subplot(122)
  for ni in range(nvectors):
    plot(3.*dvectors[ni] + ni*0.1*max(vector_array[0]))
  axis(xmin=0.025*vlength, xmax=1.025*vlength) 
  show()

  n1 = 0
  n2 = vlength
  winselect = \
        six.moves.input('select window range for processing? return=no, anything else=yes: ')
  if winselect != '':
     happy = 0
     while not(happy):
       print("select two points using cursor, left-hand point first")
       n1 = int(ginput(1)[0][0])
       print("now select a right-hand point")
       n2 = int(ginput(1)[0][0])
       if n1 < n2:
         print("ok, you have selected samples ", \
          n1, n2, " out of ", vlength, " samples per vector")
         if n2 > vlength-1:
            print("changing n2 to vlength= ", vlength )
            n2 = vlength-1
         happy = 1
       else:
         print("re-do: n1 > n2 ", n1, n2)
  close()
  if winselect:
     dvectors_save = dvectors
     nwinselect = n2-n1+1
     dvectors=zeros((nvectors, nwinselect))
     for ni in range(nvectors):
       dvectors[ni] = dvectors_save[ni,n1:n2+1]

  print("calculating covariance matrix")
  cmatrix = cov(transpose(dvectors))
  print("calculating eigensystem")
  eigenvalues, eigenvectors = linalg.eig(cmatrix)
  sigma_e = sqrt(eigenvalues.real)
  dot_products = dot(transpose(eigenvectors), transpose(dvectors))
  aaa = max(vector_array[0])
  print("pca_core: ",  aaa)
  print(type(vector_array))
  plot_pca_core_v1(vector_array, dvectors, eigenvectors, \
                eigenvalues, dot_products, n1, n2, plotlabel)
  return eigenvalues, eigenvectors, dot_products

def plot_pca_core_v1(vector_array, dvectors, eigenvectors, eigenvalues, dot_products, n1, n2, plotlabel):
  """
  """
  nvectors, vlength = shape(vector_array)
  npoints = shape(eigenvectors)[0]
  print(shape(vector_array))
  print(type(vector_array))
  plotfile='pca_core' + '.png'
  aaa = max(vector_array[0])

  print("aaa = ", aaa)
  print("shape: plot_pca_core = ", shape(vector_array))
 
  subplot(221)
  for ni in range(nvectors):
    plot(vector_array[ni,n1:n2+1] + ni*0.1*aaa)
  annotate('PCA on Vector Array %s'%(plotlabel), xy=(0.5,0.95), xycoords='figure fraction', ha='center', fontsize=12)
  title(r'$ \rm Vectors$', fontsize=12)
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  xmin,xmax,ymin,ymax=axis()
  axis(xmin=-0.05*npoints, xmax=1.05*npoints, ymin=-0.08*(ymax-ymin), ymax=ymax+0.05*(ymax-ymin))
  xlabel(r'$\rm Vector\, Element\,\,$')
  ylabel(r'$\rm Vector \,Number$')

  subplot(222)
  for ni in range(nvectors):
    yplot = 30.*dvectors[ni]
    plot(yplot + ni*0.6*max(vector_array[0]))
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  xmin,xmax,ymin,ymax=axis()
  axis(xmin=-0.05*npoints, xmax=1.05*npoints, ymin=-0.08*(ymax-ymin), ymax=ymax+0.05*(ymax-ymin))
  xlabel(r'$\rm Vector \, Element\,\,$')
  ylabel(r'$\rm  Vector\, Number$')
  title(r'$ \rm Difference \, Vectors$', fontsize=12)

  neig2plot=5
  subplot(223)
  yoffset=2.0*max(abs(transpose(eigenvectors)[0]))
  for ni in range(neig2plot):
    yplot = transpose(eigenvectors)[ni]+ni*yoffset 
    plot(yplot) 
    annotate(r'$\rm ev\, %d$'%(ni),xy=(0., (ni+0.1)*yoffset), xycoords='data',ha='left', fontsize=10)
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  xmin,xmax,ymin,ymax=axis()
  axis(xmin=-0.08*npoints, xmax=1.05*npoints, ymin=ymin-0.1*(ymax-ymin), ymax=ymax+0.05*(ymax-ymin))
  xlabel(r'$\rm Vector\, Element\,\,$')
  ylabel(r'$\rm  Eigenvectors$')

  subplot(224)
  yi = range(nvectors)
  plot(dot_products[0], yi)
  xmin,xmax,ymin,ymax=axis()
  axis(ymin=ymin-0.1*(ymax-ymin), ymax=ymax+0.05*(ymax-ymin))
  xlabel(r'$\rm Dot\, Product: \,\, ev0 \cdot\, \delta (Vector)$')
  xticks(fontsize=10)
  yticks(fontsize=10)
  ylabel(r'$\rm  Vector\,\, Number$')
  show()

  savefig(plotfile)
  return 


def broadband_pca(template, broadband_profiles, filenamein):
  """
  Does PCA on difference profiles and plots results

  difference profiles are calculated by scaling the template
  to minimize the mean-square difference between it and the profile
  """
  #global files
  #global dataset # file name from calling program
  #print "file = ", dataset
  #print files
  
  plotfile='broadband_pca_' + filenamein + '.png'
  nsubints,nphasebins=shape(broadband_profiles)[0:2]
  phi = arange(nphasebins) / float(nphasebins) - 0.5
  phi_select = arange(nphasebins) / float(nphasebins) - 0.5
  dprofiles = zeros(shape(broadband_profiles)) 
  for ni in range(nsubints):
    scaling = sum(template*broadband_profiles[ni]) / sum(template**2)
    dprofiles[ni] = broadband_profiles[ni] - scaling*template
    #print ni, scaling

  # Plot profiles and difference profiles for phase window selection:
  subplot(121)
  for ni in range(nsubints):
    plot(broadband_profiles[ni] + ni*0.1*max(broadband_profiles[0]))
  axis(xmin=0.025*nphasebins, xmax=1.025*nphasebins) 

  subplot(122)
  for ni in range(nsubints):
    plot(3.*dprofiles[ni] + ni*0.1*max(broadband_profiles[0]))
  axis(xmin=0.025*nphasebins, xmax=1.025*nphasebins) 
  show()

  nphi1 = 0
  nphi2 = nphasebins
  phase_select = \
        six.moves.input('select pulse phase range? return=no, anything else=yes: ')
  if phase_select != '':
     happy = 0
     while not(happy):
       print("select two points using cursor, lower phase first")
       nphi1 = int(ginput(1)[0][0])
       print("now select a higher phase")
       nphi2 = int(ginput(1)[0][0])
       if nphi1 < nphi2:
         print("ok, you have selected samples ", \
          nphi1, nphi2, " out of ", nphasebins, " phase bins")
         if nphi2 > nphasebins-1:
            print("changing nphi2 to nphasebins = ", nphasebins)
            nphi2 = nphasebins-1
         happy = 1
       else:
         print("re-do: nphi1 > nphi2 ", nphi1, nphi2)
  close()
  if phase_select:
     dprofiles_save = dprofiles
     nphi_select = nphi2-nphi1+1
     phi_select = phi[nphi1:nphi2+1]
     dprofiles=zeros((nsubints, nphi_select))
     for ni in range(nsubints):
       dprofiles[ni] = dprofiles_save[ni,nphi1:nphi2+1]

  print("calculating covariance matrix")
  cmatrix = cov(transpose(dprofiles))
  print("calculating eigensystem")
  eigenvalues, eigenvectors = linalg.eig(cmatrix)
  sigma_e = sqrt(eigenvalues.real)
  dot_products = dot(transpose(eigenvectors), transpose(dprofiles))
 
  subplot(221)
  for ni in range(nsubints):
    plot(phi_select, broadband_profiles[ni,nphi1:nphi2+1] + ni*0.1*max(broadband_profiles[0]))
  #axis(xmin=0.025*nphasebins, xmax=1.025*nphasebins) 
  #axis(xmin=-0.5, xmax=0.5)
  annotate('PCA on Broadband Profiles for file %s'%(filenamein), xy=(0.5,0.95), xycoords='figure fraction', ha='center', fontsize=12)
  title(r'$ \rm Profiles$', fontsize=12)
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  xmin,xmax,ymin,ymax=axis()
  phimin=phi_select[0]
  phimax=phi_select[size(phi_select)-1]
  axis(xmin=0.95*phimin, xmax= 1.05*phimax, ymin=-0.05*(ymax-ymin), ymax=1.05*(ymax-ymin))
  xlabel(r'$\rm Pulse\, Phase\,\,(cycles)$')
  ylabel(r'$\rm Profile\,Number$')

  subplot(222)
  for ni in range(nsubints):
    yplot = 30.*dprofiles[ni]
    plot(phi_select, yplot + ni*0.2*max(broadband_profiles[0]))
    #plot(phi_select, yplot + ni)
  #axis(xmin=0.025*nphasebins, xmax=1.025*nphasebins) 
  #axis(xmin=-0.5, xmax=0.5)
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  #tick_params(axis='x', fontsize=12)
  xmin,xmax,ymin,ymax=axis()
  #axis(ymin=-0.05*(ymax-ymin), ymax=1.05*(ymax-ymin))
  axis(xmin=0.95*phimin, xmax= 1.05*phimax, ymin=-0.05*(ymax-ymin), ymax=1.05*(ymax-ymin))
  xlabel(r'$\rm Pulse\, Phase\,\,(cycles)$')
  ylabel(r'$\rm  Profile\, Number$')
  title(r'$ \rm Difference \, Profiles$', fontsize=12)

  neig2plot=5
  subplot(223)
  #yoffset=2.5*max(abs(sigma_e[0]*transpose(eigenvectors)[0]))
  yoffset=max(abs(transpose(eigenvectors)[0]))
  for ni in range(neig2plot):
    #yplot = sigma_e[ni]*transpose(eigenvectors)[ni]+ni*yoffset 
    yplot = transpose(eigenvectors)[ni]+ni*yoffset 
    plot(phi_select, yplot) 
    annotate(r'$\rm ev\, %d$'%(ni),xy=(phi_select[0], (ni+0.1)*yoffset), xycoords='data',ha='left', fontsize=10)
  #axis(xmin=0.025*nphasebins, xmax=1.025*nphasebins) 
  #axis(xmin=-0.5, xmax=0.5)
  tick_params(axis='y', labelleft='off')
  xticks(fontsize=10)
  xmin,xmax,ymin,ymax=axis()
  #axis(ymin=ymin-0.05*(ymax-ymin), ymax=1.05*(ymax-ymin))
  axis(xmin=0.95*phimin, xmax= 1.05*phimax, ymin=ymin-0.05*(ymax-ymin), ymax=ymax+0.08*(ymax-ymin))
  xlabel(r'$\rm Pulse\, Phase\,\,(cycles)$')
  ylabel(r'$\rm  Eigenvectors$')

  subplot(224)
  yi = range(nsubints)
  plot(dot_products[0], yi)
  xmin,xmax,ymin,ymax=axis()
  axis(ymin=-0.05*(ymax-ymin), ymax=1.05*(ymax-ymin))
  #xlabel(r'$\rm \Delta TOA \,\,(bins)$')
  xlabel(r'$\rm Dot\, Product: \,\, ev0 \cdot\, \delta (Profile)$')
  xticks(fontsize=10)
  yticks(fontsize=10)
  ylabel(r'$\rm  Profile\,\, Number$')

  savefig(plotfile)
  return eigenvalues, eigenvectors, dot_products

def corr_toa_dots(toavec, dotprod_array, ntests):
    """ 
    calculates correlation coefficients between TOAs and 
    dot products between eigenvectors and data vectors  

    toavec = 1D vector of TOAs corresponding to data vectors and dot products
    dotprod_array = array of dot products between data vectors and eigenvectors
    ntests = number of eigenvectors/dotprods to test 

    note dotprod_array can be complex so need to take real part
    """
    print("\nPearson and Spearman Correlations between TOAs and PCA dot products:")
    print("  Ev     rho_p     rho_s       prob_p          prob_s")
    for nt in range(ntests):
      ccp, pp = stats.pearsonr(toavec, dotprod_array[nt].real)
      ccs, ps = stats.spearmanr(toavec, dotprod_array[nt].real)
      print('   %d    %6.3f    %6.3f    %e    %e  '%(nt, ccp, ccs, pp, ps))

    return

def sigtoa_vs_inttime(template, profile_array, noff1, noff2):
    """
    Calculates and plots rms TOA for increasing integration time.

    Input:
        template = vector containing template
        profile_array = array of profiles
        noff1,noff2 = indices of off-pulse points to use

    Calculates TOAs and then rms
    Then sum profile pairs and repeat
    Repeat process until ~4 profiles are included in set

    Need to check get_toa calculation; for B1937 data it gives
    a much smaller TOA error than is actually seen (factor of 5)
    """
    rmsnoise_dum = 1.		        # dummy value
    nprofiles, nphase = shape(profile_array)
    nlevels = int(log2(nprofiles))-1
    print(nprofiles, nphase)
    print(nlevels)
    levels=2.**arange(nlevels)		# number of profiles averaged
    rmstoavec=zeros(nlevels)
    rmstoavec_predicted=zeros(nlevels)
    for nl in range(nlevels):
      nprofs_nl = nprofiles // 2**nl
      print("nl, nprofs_nl = ", nl, nprofs_nl)
      toavec=zeros(nprofs_nl)
      sigtoavec=zeros(nprofs_nl)
      if nl ==0:
         profarray = profile_array
         profarray_previous = profarray
      else: 
         profarray=zeros((nprofs_nl, nphase))
         for npl in range(nprofs_nl):
             profarray[npl] = \
                  0.5*(profarray_previous[2*npl]+profarray_previous[2*npl+1])
         profarray_previous = profarray
      for np in range(nprofs_nl):
         rmsnoise = std(profarray[np,noff1:noff2])
         toavec[np], bb, sigtoavec[np], sigbb = \
                 get_toa(template, profarray[np], rmsnoise)
         print("    np, rmsnoise, S/N, toa, toaerr = ", \
                 np, rmsnoise, bb/rmsnoise,toavec[np], sigtoavec[np])
      rmstoavec[nl] = std(toavec) 
      rmstoavec_predicted[nl] = sqrt(sum(sigtoavec**2)/size(sigtoavec))
      print("        nl, rmstoa, rmstoa_predicted = ", \
                     nl, rmstoavec[nl], rmstoavec_predicted[nl])
 
    return levels, rmstoavec

def quadrant_toa_correlation(template, channel_profiles, pulse_intensities, outlab=''):
    """
    Averages channel_profiles in quadrants across the frequency range.
    Calculates TOAs vs subintegration and computes correlation coefficient 

    Input:
       template
       channel_profiles
       pulse_intensities = on-pulse intensities (bandpass corrected)
       (e.g. pulse_intensities_bpcorr from main program)

    Output and Usage:
    quadrant_profiles, quadrant_grandave_profiles, qtoa_array, qtoaerr_array, qb_array, qberr_array, qsnr_array, cc_matrix, ccprob_matrix = quadrant_toa_correlation(template, channel_profiles, pulse_intensities_bpcorr)
   
    output file has cc_matrix and other brief stats

    """
    outfilename='quadrant_toa_correlation_' + outlab + '.out'
    nsubints, nsubbands, nphasebins = shape(channel_profiles) 
    chanmin = int(six.moves.input('enter min channel to use: '))
    chanmax = int(six.moves.input('enter max channel to use: '))
    noff1 = int(six.moves.input('enter first off-pulse sample: '))
    noff2 = int(six.moves.input('enter second off-pulse sample: '))
    chan0 = chanmin
    chan1 = chanmin+int(0.25*(chanmax-chanmin))
    chan2 = chan1 +int(0.25*(chanmax-chanmin))
    chan3 = chan2 +int(0.25*(chanmax-chanmin))
    chanvec = (chan0, chan1, chan2, chan3, chanmax)

    print("chan0,1,2,3 = ", chan0, chan1, chan2, chan3)
    
    quadrant_profiles = zeros((nsubints, 4, nphasebins))
    quadrant_grandave_profiles = zeros((4, nphasebins))
    qtoa_array = zeros((nsubints, 4))
    qtoaerr_array = zeros((nsubints, 4))
    qb_array = zeros((nsubints, 4))
    qberr_array = zeros((nsubints, 4))
    qsnr_array = zeros((nsubints, 4))
    for nq in range(0,4):
      print("processing quadrant ", nq)
      c1=chanvec[nq]
      c2=chanvec[nq+1]
      for ni in range(nsubints):
        nsum = 0
        for ns in range(c1,c2):
           nsum += 1
           quadrant_profiles[ni, nq] += channel_profiles[ni, ns]
        quadrant_profiles[ni, nq] /=float(nsum) 
        quadrant_grandave_profiles[nq] += quadrant_profiles[ni, nq]
        rmsnoise=std(quadrant_profiles[ni,nq,noff1:noff2])
        toa, b, sigtoa, sigb  = \
                 get_toa(template, quadrant_profiles[ni,nq], rmsnoise)
        #print "   ni,nq, toa, b, sigtoa, sigb ", ni,nq,toa,b,sigtoa,sigb       
        qtoa_array[ni,nq] = toa
        qtoaerr_array[ni,nq] = sigtoa
        qb_array[ni,nq] = b
        qberr_array[ni,nq] = sigb
        qsnr_array[ni,nq] = b/rmsnoise
    quadrant_grandave_profiles /= float(nsubints)
    
    cc_matrix=zeros((4,4))
    ccprob_matrix=zeros((4,4))
    # cross correlations between quadrants:
    for nq1 in range(0,4):
        for nq2 in range(0,4):
            ccp, pp = stats.pearsonr(qtoa_array[:,nq1], qtoa_array[:,nq2])
            #cc ccp, pp = stats.spearmanr(toavec, dotprod_array[nt].real)
            cc_matrix[nq1,nq2] = ccp
            ccprob_matrix[nq1,nq2] = pp
            print('  %d  %d   %6.3f   %e'%(nq1, nq2, ccp, pp))
    fout=open(outfilename, 'w') 
    print("Cross-correlation Matrix:", file=fout)
    print(cc_matrix, file=fout)
    print(file=fout)
    print("Cross-correlation Probability:", file=fout)
    print(ccprob_matrix, file=fout)
    print(file=fout)
    print("Mean S/N for Each Quadrant", file=fout)
    print("  Q    <S/N>", file=fout)
    for q in range(4):
      print('  %d   %6.1f'%(q, mean(qsnr_array[:,q])), file=fout)
    fout.close()
    return quadrant_profiles, quadrant_grandave_profiles, qtoa_array, qtoaerr_array, qb_array, qberr_array, qsnr_array, cc_matrix, ccprob_matrix

def plot_quadrant_toas(qtoa_array, plotfilelab=''):
   """
   Plots dTOAs from four frequency quadrants as scatter plots 
   """
   plotfile='plot_quadrant_' + plotfilelab + '.png'
   dqtoa_array=zeros(shape(qtoa_array))
   #remove means:

   for q in range(4):
       dqtoa_array[:,q] = qtoa_array[:,q] - mean(qtoa_array[:,q])     
       dqmax=max(dqtoa_array[:,q])
       dqmin=min(dqtoa_array[:,q])
       if q ==0: 
           maxtoa=dqmax
           mintoa=dqmin
       if dqmax > maxtoa: maxtoa = dqmax
       if dqmin < mintoa: mintoa = dqmin
   print("min, maxtoa = ", mintoa, maxtoa)

   for q1 in range(4):
      for q2 in range(q1+1,4):
         subplot(3,3,q2+q1*3)
         plot(dqtoa_array[:,q1], dqtoa_array[:,q2], 'o')
         ccp, pp = stats.pearsonr(dqtoa_array[:,q1], dqtoa_array[:,q2])
         axmin = mintoa - 0.15*(maxtoa-mintoa) 
         axmax = maxtoa + 0.15*(maxtoa-mintoa)
         axis(xmin=axmin, xmax=axmax, ymin=axmin, ymax=axmax)
         title(r'$\rho(%d,%d) = %5.2f$'%(q1, q2, ccp))
         plot((axmin, axmax), (axmin, axmax), lw=1)
         subplots_adjust(wspace=0.35, hspace=0.35)
         if q2 == q1+1:
             xlabel(r'$\delta$TOA')
             ylabel(r'$\delta$TOA')
   annotate(r'Cross Correlation of $\delta$TOAs (bins) Between Bandpass Quadrants', xy=(0.5, 0.95), xycoords='figure fraction', ha='center',fontsize=15)
   savefig(plotfile)
   return

def quadrant_toa_ccf(qtoa_array, plotfilelab=''):
   """
   Plots cross correlation functions between the frequency quadrants as scatter plots 
   """
   plotfile='quadrant_ccf_' + plotfilelab + '.png'
   dqtoa_array=zeros(shape(qtoa_array))
   #remove means:

   for q in range(4):
       dqtoa_array[:,q] = qtoa_array[:,q] - mean(qtoa_array[:,q])     
       dqmax=max(dqtoa_array[:,q])
       dqmin=min(dqtoa_array[:,q])
       if q ==0: 
           maxtoa=dqmax
           mintoa=dqmin
       if dqmax > maxtoa: maxtoa = dqmax
       if dqmin < mintoa: mintoa = dqmin
   print("min, maxtoa = ", mintoa, maxtoa)

   for q1 in range(4):
      acf1 = correlate(dqtoa_array[:,q1], dqtoa_array[:,q1], mode='full')
      nlags = size(acf1)
      lags = range(-nlags//2+1, nlags//2+1)	# zero lag @ (nlags-1)/2
      for q2 in range(q1+1,4):
         ccf12 = correlate(dqtoa_array[:,q1], dqtoa_array[:,q2], mode='full')
         acf2 = correlate(dqtoa_array[:,q2], dqtoa_array[:,q2], mode='full')
         ccf12n = ccf12 / sqrt(max(acf1)*max(acf2)) 
         ccp, pp = stats.pearsonr(dqtoa_array[:,q1], dqtoa_array[:,q2])
         if q1 == 0 and q2 == 1:
              maxccf = max(ccf12n)
              minccf = min(ccf12n)
         if max(ccf12n) > maxccf: maxccf = max(ccf12n)
         if min(ccf12n) < minccf: minccf = min(ccf12n)
         subplot(3,3,q2+q1*3)
         plot(lags, ccf12n)
         axmin = minccf - 0.15*(maxccf-minccf)
         axmax = maxccf + 0.15*(maxccf-minccf)
         #axmin = mintoa - 0.15*(maxtoa-mintoa) 
         #axmax = maxtoa + 0.15*(maxtoa-mintoa)
         axis(ymin=axmin, ymax=axmax)
         #title(r'$\rho(%d,%d) = %5.2f$'%(q1, q2, ccp))
         #plot((axmin, axmax), (axmin, axmax), lw=1)
         subplots_adjust(wspace=0.35, hspace=0.35)
         title(r'CCF(%d, %d)'%(q1, q2), fontsize=10)
         if q2 == q1+1:
             xlabel(r'$\rm Time \, Lags$')
             ylabel(r'$\rm CCF$')
   annotate(r'CCFs of $\delta$TOAs (bins) Between Bandpass Quadrants', xy=(0.5, 0.95), xycoords='figure fraction', ha='center',fontsize=15)
   savefig(plotfile)
   return

def toa_analysis_tf(template, profiles, Nf1=0, Nf2=0, noff1=0, noff2=0, textlabel=''):
    """
    Calculates TOAs for profiles for each channel and subintegration

    Input:
        template
 	profiles for subintegrations, channels vs phase
        Nf1, Nf2 = channel numbers to use
        noff1, noff2 = pulse profile bins to use

    Usage example:
    toa2d, b2d = toa_analysis_tf(template,channel_profiles_fixed,20,400,100,400)

    """
    Nt,Nf,Nphi = shape(profiles)

    # default channels to use:
    if Nf1 == 0 and Nf2 ==0:
        Nf1 = 0
        Nf2 = Nf

    # define off-pulse region if not inputted:
    # need a more sensible, automatic choice of noff1, noff2
    # by finding peak and avoiding it; possibly two peaks
    if noff1 == 0 and noff2 == 0:
         noff1 = 0
         noff2 = 100
    # get TOAs and scale factors for channel profiles
    toa_array = zeros((Nt, Nf))
    toaerr_array = zeros((Nt, Nf))
    b_array = zeros((Nt, Nf))
    berr_array = zeros((Nt, Nf))
 
    for nt in range(Nt):
        print("processing subintegration ", nt)
        for nf in range(Nf1,Nf2):
            profile = profiles[nt, nf]
            rmsnoise = std(profile[noff1:noff2])
            #print "    nf, rmsoff = ", nf, rmsnoise
            t,b,dt,db = get_toa(template, profile, rmsnoise)
            toa_array[nt,nf] = t
            toaerr_array[nt,nf] = dt
            b_array[nt,nf] = b
            berr_array[nt,nf] = db
    return toa_array, b_array 

def acf_toa2d(toa2d, b2d):
    """
    Calculates and plots the weighted ACF of t-f TOAs along the time-lag axis.

    Input:
        toa2d = array of TOAs
        b2d   = array of amplitudes (scale factors from template fitting)
    """

    Nt,Nf = shape(toa2d)
    

    ntlags = Nt
    acft=zeros(ntlags)
    dtoa2d=zeros(shape(toa2d))

    # remove means at fixed frequency:
    for nf in range(Nf):
        dtoa2d[:,nf] = toa2d[:,nf] - mean(toa2d[:,nf])
    
    for nl in range(ntlags):
       sumt = 0.
       sumw = 0.
       for nt in range(Nt-nl):
           wvec1 = sign(b2d[nt])
           wvec2 = sign(b2d[nt+nl])
           # weighted means:
           m1 = mean(wvec1*dtoa2d[nt])
           m2 = mean(wvec2*dtoa2d[nt+nl])
           for nf in range(Nf):
               w1 = wvec1[nf]
               w2 = wvec2[nf]
               dt1 = dtoa2d[nt,nf]-m1
               dt2 = dtoa2d[nt+nl,nf]-m2
               sumt += w1*w2*dt1*dt2
               sumw += w1*w2
       if sumw > 0.: acft[nl] = sumt / sumw
    
    plot(acft/acft[0], drawstyle='steps-post')
    a,b,c,d=axis()
    axis(xmin=a-0.05*(b-a), xmax=b+0.05*(b-a), ymin=c-0.05*(d-c), ymax=d+0.05*(d-c))
    xlabel('Time Lag (bins)')
    ylabel('Normalized ACF')
    return acft
