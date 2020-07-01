'''
De-dispersion exercise
Michael Lam, July 17, 2014
'''
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import sys
sys.path.append("dataFiles/")
import SinglePulse as SP
import utilities as u

#rc('text',usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern'],'size':14})

MP_WINDOW = np.arange(500,1500)


def DMdelay(nu,DM):#nu in GHz, return delay in ms
    return 4.15*DM*(nu**-2) 


def initializer(dataSet):

    DIR = "dataFiles/"

    if dataSet=='Set1':
        x = np.load(DIR+"J1713+0747.GBT.npz")

    elif dataSet=='Set2':
        x = np.load(DIR+"J1455-3330.GBT.npz")
    elif dataSet=='Set3':
        x = np.load(DIR+"J1600-3053.GBT.npz")
    elif dataSet=='Set4':
        x = np.load(DIR+"J1643-1224.GBT.npz")
    elif dataSet=='Set5':
        x = np.load(DIR+"J2145-0750.GBT.npz")

    elif dataSet=='Set6': #impossible? need binary parameters?
        x = np.load(DIR+"J0613-0200.GBT.npz") #the hard one?

    data = x['data']
    period = x['period']
    freqs = x['freqs']

    dt = period/2048 * 1e3
    
    bins = np.arange(2048)*dt

    freqs /= 1000

    return data,bins,freqs




def plot_pulseAmplitudes(data,bins,freqs):
    bmin = np.min(bins)
    bmax = np.max(bins)
    fmin = np.min(freqs)
    fmax = np.max(freqs)
                
    plt.figure()
    gs = gridspec.GridSpec(5,1)
    plt.subplots_adjust(hspace=0.001)
    
    ax1 = plt.subplot(gs[:-1,0])
    u.imshow(data,ax=ax1,extent = [bmin,bmax,fmin,fmax])
    ax1.set_xticklabels([])
    ax1.set_ylabel("Frequency (GHz)")

    dt = freqs[1] - freqs[0]

    def freq2chan(freq):
        chan = (freq - freqs[0])/dt
        return chan

    def chan2freq(chan):
        freq = freqs[0] + chan*dt
        return freq
    
    ax1_secondary = ax1.secondary_yaxis('right', functions=(freq2chan, chan2freq))
    ax1_secondary.set_ylabel('Channel number')

    ax2 = plt.subplot(gs[-1,0])
    summedprof = np.mean(data,axis=0)
    ax2.plot(bins,summedprof,'k')
    ax2.set_xlim(bmin,bmax)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Phase (ms)")

    sp = SP.SinglePulse(u.center_max(summedprof),mpw=MP_WINDOW)

    fwhm = sp.getFWHM()
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    t = np.arange(2048)
    template = np.exp(-0.5*((t-1024)/sigma)**2)

    retval = sp.fitPulse(template)
    snr = retval[-2]
    
    sigma_TOA = retval[3]
    
    ax1.set_title(r"$\mathrm{S/N}=%0.1f, \sigma_{\mathrm{TOA}}=%0.3f \mathrm{\;ms}$"%(snr,sigma_TOA))

    plt.show()

def getDM(timeDiff,freqLow,freqHigh):#freqs in GHz, difference in ms
    return float(timeDiff)/(4.15*(freqHigh**-2 - freqLow**-2))


def getChannelTOA(data,bins,freqChannel):
    '''
    Assume gaussian template withh FWHM equal to FWHM of pulse.
    '''
    sp = SP.SinglePulse(data[freqChannel])
    fwhm = sp.getFWHM()
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    t = np.arange(2048)
    template = np.exp(-0.5*((t-1024)/sigma)**2)

    retval = sp.fitPulse(template)
    tauhat = retval[1]
    bhat = retval[2]

    dt = bins[1]-bins[0]
    
    #print("TOA: %f" % ((tauhat + 1024)*dt))
    return (tauhat + 1024)*dt


def fft_roll(a, shift):
    '''
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of 
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    '''
    try:
        shift = shift[...,np.newaxis]
    except (TypeError, IndexError): pass
    phase = -2j*np.pi*shift*np.fft.rfftfreq(a.shape[-1])
    return np.fft.irfft(np.fft.rfft(a)*np.exp(phase))


def dedisperse(data,bins,freqs,trialDM):
    retval = np.zeros(np.shape(data))
    
    dt = bins[1] - bins[0]
    period = bins[-1] #? + dt

    tshifts = (DMdelay(freqs,trialDM) % period)
    bshifts = -tshifts/dt
    shiftdata = fft_roll(data, bshifts)
    return shiftdata




def significance(dedispersedValues):
    '''
    Modified from Dusty's code
    '''
    collapsed = np.mean(dedispersedValues,axis=0)

    sp = SP.SinglePulse(u.center_max(collapsed),mpw=MP_WINDOW)
    fwhm = sp.getFWHM()
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    t = np.arange(2048)
    template = np.exp(-0.5*((t-1024)/sigma)**2)

    retval = sp.fitPulse(template)
    snr = retval[-2]
    return snr
    
    #Roughly the same speed but better is above.
    mean = np.mean(collapsed)
    X2 = 0
    for i in range(len(collapsed)):
        X2 += (collapsed[i]-mean)**2
    X2 /= float(len(collapsed))
    return X2



def DMSearch(data,bins,freqs,DMLB,DMUB,nDM):
    '''
    DM search following Dusty's periodSearch() function
    '''
    X2s = []
    trialDMs = np.linspace(DMLB,DMUB,nDM)
    for i in range(nDM):
        dedispersedValues = dedisperse(data,bins,freqs,trialDMs[i])
        X2s.append(significance(dedispersedValues))
        print(str(100*float(i+1)/nDM)+'% complete')
    return trialDMs,X2s

def plot_X2s(DMs,X2s):
    '''
    Modified from Dusty's code
    '''
    fig = plt.figure()
    ax = fig.add_axes([0.15,0.12,0.78,0.78])
    ax.plot(DMs,X2s)
    ax.set_xlim(min(DMs),max(DMs))
    ax.set_xlabel('DM [pc/cc]')
    ax.set_ylabel('S/N')
    plt.show()
    




