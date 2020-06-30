'''
Michael Lam
Last updated: 12/1/2013
'''
from __future__ import division, print_function

import numpy as np
import scipy.fftpack as fft
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.interpolate as interpolate
from scipy.special import erfc
from scipy.signal import fftconvolve,correlate
from matplotlib import mlab
import matplotlib.pyplot as plt
#import sympy.mpmath as mp
#from sympy.utilities import lambdify



'''
Power Spectrum
freq = 1/period
'''
def ps(x,mean_subtracted=True):
    return powerspectrum(x,mean_subtracted=mean_subtracted)
def powerspectrum(series,mean_subtracted=True):
    if mean_subtracted:
        series-=np.mean(series)

    #Why does this not work?
    
    sp = fft.fft(series)
    #sp=np.fft.fft(series)
    freq = fft.fftfreq(len(series)) # series.shape[-1] for numpy array
    freq = fft.fftshift(freq)
    sp = fft.fftshift(sp)
    return freq,np.abs(sp)**2
    
    '''
    length=len(series)
    ps,freq=mlab.psd(series,NFFT=2**10)#,Fs=1.0/length,NFFT=length)
    return freq,ps
    '''
    #http://users.aims.ac.za/~mike/python/env_modelling.html ???





'''
ACF
var=True: calculate variance, var=False, do not calculate. var=number: use as number
Include mean subtraction?
'''
def acf(array,var=True,norm_by_tau=False):
    array=np.array(array)
    N=len(array)
    if var==True:
        var=np.var(array)
    elif var==False:
        var=1
        
    if norm_by_tau:
        taus=np.concatenate((np.arange(1,N+1),np.arange(N-1,0,-1)))
        return np.correlate(array,array,"full")/(var*taus)
    return np.correlate(array,array,"full")/(var*N)
   
def acf_unequal(t,x,dtau=1):
    length = len(x)
    num_lags = np.ceil((np.max(t) - np.min(t))/dtau) + 1 #+1?
    taus = np.arange(num_lags) * dtau
    tau_edges = (taus[:-1] + taus[1:])/2.0
    tau_edges = np.hstack((tau_edges,[tau_edges[-1]+dtau]))
    N_taus = np.zeros(num_lags)
    retval = np.zeros(num_lags)

    # this could be sped up several ways
    for i in xrange(length):
        for j in xrange(length):
            dt = np.abs(t[i]-t[j])
            index = np.where(dt < tau_edges)[0][0]
            #try:
            #    index = np.where(dt < tau_edges)[0][0]
            #except: 
            #    print t-np.min(t),dt,tau_edges
            #    raise SystemExit
            N_taus[index] += 1
            retval[index] += x[i]*x[j]
    

    #divide by zero problem!
    retval = retval / N_taus
    #mirror each:
    taus = np.concatenate((-1*taus[::-1][:-1],taus))
    retval = np.concatenate((retval[::-1][:-1],retval))
    retval /= 2 #double counting, can speed this up!

    return taus,retval

#Do not provide bins but provide edges?
#error bars?
def lagfunction(func,t,x,dtau=1,tau_edges=None,mirror=False):
    length = len(x)
    if tau_edges == None:
        num_lags = np.ceil((np.max(t) - np.min(t))/dtau) + 1 #+1?
        taus = np.arange(num_lags) * dtau
        tau_edges = (taus[:-1] + taus[1:])/2.0
        tau_edges = np.hstack((tau_edges,[tau_edges[-1]+dtau]))
        N_taus = np.zeros(num_lags)
        retval = np.zeros(num_lags)
    else:
        N_taus = np.zeros(len(tau_edges)-1)
        retval = np.zeros(len(tau_edges)-1)

    # this could be sped up several ways
    for i in xrange(length):
        for j in xrange(length):
            dt = np.abs(t[i]-t[j])
            index = np.where(dt < tau_edges)[0] #<=?
            if len(index)==0:
                continue
        
            index = index[0] #get the lowest applicable lag value
            #print dt,tau_edges[index-1:index+1]
            #try:
            #    index = np.where(dt < tau_edges)[0][0]
            #except: 
            #    print t-np.min(t),dt,tau_edges
            #    raise SystemExit
            try: 
                N_taus[index-1] += 1
            except:
                print(dt,tau_edges,index,len(N_taus))
            retval[index-1] += func(x[i],x[j])
    

    #divide by zero problem!
    retval = retval / N_taus
    if mirror: #fix this
    #mirror each:
        #taus = np.concatenate((-1*taus[::-1][:-1],taus))
        retval = np.concatenate((retval[::-1][:-1],retval))
        retval /= 2 #double counting, can speed this up!
    return tau_edges,retval

def acf_unequal2(t,x,**kwargs):
    func = lambda x,y: x*y
    return lagfunction(func,t,x,mirror=False,**kwargs)


def sf_unequal(t,x,**kwargs):
    func = lambda x,y: (x-y)**2
    return lagfunction(func,t,x,mirror=False,**kwargs)


def acf2d(array,speed='fast',mode='full'):
    ones = np.ones(np.shape(array))
    norm = fftconvolve(ones,ones,mode=mode) #very close for either speed
    if speed=='fast':
        return fftconvolve(array,np.flipud(np.fliplr(array)),mode=mode)/norm
    else:
        return correlate(array,array,mode=mode)/norm

def ccf2d(array1,array2,speed='fast',mode='full'):
    #check shape of arrays, fix ones array
    ones = np.ones(np.shape(array1))
    norm = fftconvolve(ones,ones,mode=mode) #very close for either speed
    if speed=='fast':
        return fftconvolve(array1,np.flipud(np.fliplr(array2)),mode=mode)/norm
    else:
        return correlate(array1,array2,mode=mode)/norm
    


#Following Edelson and Krolik
def dcf(tx,x,ex,ty,y,ey,subtract_mean=True):
    t = np.array(t)
    x = np.array(x)
    ex = np.array(ex)
    ty = np.array(ty)
    y = np.array(y)
    ey = np.array(ey)
    
    if subtract_mean:
        meanx = np.mean(x)
        meany = np.mean(y)
        x -= meanx
        y -= meany

    UDCF = np.zeros((len(x),len(y)))
    for i in xrange(len(x)):
        for j in xrange(len(y)):
            #UDCF[i,j] = 
            pass
    


    '''
    array=np.array(array)
    length=len(array)
    retval=np.zeros(length)
    slide_x=np.zeros(2*length-1)
    slide_x[0:length]=array
    slide_y=np.zeros(2*length-1)
    for i in range(length):
        slide_y[i:length+i]=array
        retval[i]=slide_x*slide_y
    '''


#Taken from diagnostics.py, set default threshold=3
def zct(series,threshold=3,full=False,meansub=False):
    count=0
    N=len(series)
    current=np.sign(series[0])
    if meansub:
        series-=np.mean(series)
    for i in range(1,N):
        #print np.sign(series[i])
        if np.sign(series[i]) != current:
            count+=1 #Crossed zero, add to count
            current*=-1 #Flip sign
    average_zw=float(N-1)/2
    sigma_zw=np.sqrt(N-1)/2
    if (average_zw - threshold*sigma_zw) <= count <= (average_zw + threshold*sigma_zw):
        if full:
            return True,abs(count-average_zw)/sigma_zw,count
        return True
    else:
        if full:
            return False,abs(count-average_zw)/sigma_zw,count
        return False 




#Linear fit function here?
def MSE(func,fitfunc):
    return np.sqrt(RSS(func,fitfunc)/len(func))
def RSS(func,fitfunc):
    return np.sum((fitfunc-func)**2)
def TSS(func):
    return RSS(func,np.mean(func))

def R2(func,fitfunc):
    return 1 - RSS(func,fitfunc)/TSS(func)



'''
Radiometer Equation
sigma_T = T_sys / sqrt(dnu_rf*tau)
Where T_sys is the system temperature
dnu_rf is the Bandwidth around nu_rf
tau is the integration time

Example:
Arecibo: T_sys~30K
PUPPI: 800 MHz bandwidth
'''
def radiometer(T_sys,dnu_rf,tau,n_pol=1):
    return T_sys / np.sqrt(n_pol*dnu_rf * tau)








'''
Smoothing Function
Taken from Scipy Cookbook's Signal Smooth
'''

def smooth(x,window_len=3,window='flat'):
    #TODO: the window parameter could be the window itself if an array instead of a string
    #NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    if len(x) < window_len or window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

'''
Decimate the daat
Be careful with window_len!
'''
def decimate(x,window_len):
    if window_len==1:
        return x
    length=len(x)
    retval=np.zeros(length//window_len)
    for i in range(window_len):
        retval+=x[i:length:window_len]
    return retval/window_len


def imshow(x,ax=None,**kwargs):
    if ax!=None:
        im=ax.imshow(x,origin='lower',interpolation='nearest',aspect='auto',**kwargs)
    else:
        im=plt.imshow(x,origin='lower',interpolation='nearest',aspect='auto',**kwargs)
    return im
def loadtxt(x):
    return np.transpose(np.loadtxt(x))

'''
Histogram

Allow for intervals or number of bins
bins: Provide an array of bins
'''
def histogram(values,interval=1.0,bottom=None,full=False,bins=None,plot=False,show=True,horizontal=False,**kwargs):
    if bins==None:
        factor=1.0/interval
        if bottom==None:
            minval=(np.fix(factor*min(values))-1)/factor
        else:
            minval=bottom
        maxval=(np.ceil(factor*max(values))+1)/factor
        bins=np.arange(minval,maxval,interval)
    else:
        minval=bins[0]
        maxval=bins[-1]
    hist,bins=np.histogram(values,bins=bins)
    center=(bins[:-1]+bins[1:])/2.0

    if plot:
        plothistogram(center,hist,interval,show=show,horizontal=horizontal,**kwargs)
        return

    if full:
        return center,hist,bins,minval,maxval
    return center,hist

def plothistogram(center,hist,interval=1.0,bins=None,steps=False,show=True,horizontal=False,**kwargs):
    if steps or bins!=None:
        binsize = np.mean(np.diff(center))
        center = np.concatenate(([center[0]-binsize],center,[center[-1]+binsize]))
        hist = np.concatenate(([0],hist,[0]))
        plt.plot(center,hist,drawstyle='steps-mid',**kwargs)
    else:
        if horizontal:
            plt.barh(center,hist,height=interval,align='center',**kwargs)
        else:
            plt.bar(center,hist,width=interval,align='center',**kwargs)
    if show:
        plt.show()


'''
Reports moments: mean, variance, skewness, kurtosis
Flags:
sigma: report sigma instead of variance
ddof for skew and kurtosis?

should use scipy.stats.decribe? use ddof=1 to match
'''
def moments(series,sigma=False,ddof=0,fisher=True):
    a=np.mean(series)
    if sigma:
        b=np.std(series,ddof=ddof)
    else:
        b=np.var(series,ddof=ddof)
    c=stats.skew(series)
    d=stats.kurtosis(series,fisher=fisher)
    return a,b,c,d

'''
Returns variance of a discrete PDF (integrated)
Need a method to integrate (quad) from 0 to +inf, not by assumption!
'''
def variance(x,PDF,mu=None):
    return PDFmoment(x,PDF,2,mu)

#Generalization of above
#Standardized: mu_k/sigma^k
def PDFmoment(x,PDF,k,mu=None,standardized=False,excess=False):
    x=np.array(x)
    PDF=np.array(PDF)
    #Clip NaN values
    goodinds=np.where(np.logical_not(np.isnan(PDF)))[0]
    x=x[goodinds]
    PDF=PDF[goodinds]

    area=np.trapz(PDF,x=x)
    normedPDF=PDF/area
    if mu==None:
        mu=np.trapz(normedPDF*x,x=x)
    if k==1:
        return mu
    denom=1
    if standardized:
        sigmasq=np.trapz(normedPDF*(x-mu)**2,x=x)
        denom=np.sqrt(sigmasq)**k
    modifier=0
    if excess and k==4:
        modifier=-3
    #from matplotlib.pyplot import *
    #plot(x,normedPDF)
    #show()
    return np.trapz(normedPDF*(x-mu)**k,x=x)/denom + modifier

#Generalization of moments()
#Returns mean, variance, standardized skewness, standardized excess kurtosis
def PDFmoments(x,PDF):
    a=PDFmoment(x,PDF,1)
    #b=variance(x,PDF,mu=a)
    b=PDFmoment(x,PDF,2,mu=a)
    c=PDFmoment(x,PDF,3,mu=a,standardized=True)
    d=PDFmoment(x,PDF,4,mu=a,standardized=True,excess=True)
    return a,b,c,d


'''
Return RMS
'''
def RMS(series,subtract_mean=False):
    if subtract_mean:
        series -= np.mean(series)
    return np.sqrt(np.mean(np.power(series,2)))


def harmonic_mean(series):
    return len(series)/np.sum(1.0/series)

'''
Return weighted sample mean and std
http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance
'''
def weighted_moments(series,weights,unbiased=False,harmonic=False):
    series=np.array(series)
    weights=np.array(weights)
    weightsum=np.sum(weights)
    weightedmean = np.sum(weights*series)/weightsum
    weightedvariance = np.sum(weights*np.power(series-weightedmean,2))
    if harmonic:
        return weightedmean, harmonic_mean(1.0/weights)
    elif unbiased:
        weightsquaredsum=np.sum(np.power(weights,2))
        return weightedmean, np.sqrt(weightedvariance * weightsum / (weightsum**2 - weightsquaredsum))
    else:
        return weightedmean, np.sqrt(weightedvariance / weightsum)




'''
Return Chi-squared test statistic
Given: List of observed frequencies and list of expected frequencies
'''
def chisquared(observed,expected):
    o=np.array(observed)
    e=np.array(expected)
    return np.sum(np.power(o-e,2))/np.sum(o)
    #return np.sum(np.power(o-e,2)/e)



'''
Normalize an array to unit height
Below: normalize 
'''
def normalize(array,simple=False,minimum=None):
    if simple:
        return array/np.max(array)
    maximum=np.max(array)
    if minimum==None:
        minimum=np.min(array)
    return (array-minimum)/(maximum-minimum)
def normalize_area(array,x=None,full=False):
    if x==None:
        x=np.arange(len(array))
    area=np.trapz(array,x=x)
    if full:
        return array/area,area
    return array/area



'''
Center the maximum value of the array
Follows profiles.py
'''
def center_max(array):    
    maxind=np.argmax(array)
    length=len(array)
    centerind=length//2
    diff=centerind-maxind
    return np.roll(array,diff)



#Following stackoverflow.com/questions/8094374/python-matplotlib-find-intersection-of-lineplots
#Uses Piecewise polynomials
#Potentially slow!
def roots(func,x1,x2=None):
    if x2==None:
        xs=x1
    else:
        xs=np.r_[x1,x2]
        xs.sort()
    x_min=xs.min()
    x_max=xs.max()
    x_mid=xs[:-1]+np.diff(xs)/2 #get midpoints of xs array (combined x1/x2 arrays)
    rootlist=set()
    for val in x_mid:
        root,infodict,ier,mesg = optimize.fsolve(func,val,full_output=True)
        # ier==1 indicates a root has been round
        if ier==1 and x_min<root<x_max:
            print(root[0])
            rootlist.add(root[0])
    return list(rootlist)




#Follow profiles.py
#notcentered is very rudimentary
#have norm be simple
def FWHM(series,norm=True,simple=False,notcentered=False):
    if norm:
        series=normalize(series) #assumes these are floats, not integers!
    y=np.abs(series-0.5)
    
    N=len(series)
    half=N//2

    wL = 0
    wR = N-1

    
    #initial solution
    if notcentered:
        series = center_max(series)
#        half=np.argmax(series)
    iL=np.argmin(y[:half])
    iR=np.argmin(y[half:])+half
    if not simple:
        x=np.arange(len(series))
        f=interpolate.interp1d(x,series-0.5)

        negindsL = np.where(np.logical_and(series<0.5,x<half))[0]
        negindsR = np.where(np.logical_and(series<0.5,x>half))[0]
        iL=optimize.brentq(f,negindsL[-1],negindsL[-1]+1)#half)
        iR=optimize.brentq(f,negindsR[0]-1,negindsR[0])#half,wR)
    return iR-iL




#scrunch all given arrays into single points given an epochsize
#decimals follows numpy's around format
#Loosely follows residuals.py

#Rather than return binned epochs, return time averaged epochs?
#e.g. if 1.2 and 1.6 fall into a bin, return 1.4, not 1 or 2
#or just use finer decimals?

def epoch_scrunch(toas,data=None,errors=None,epochs=None,decimals=0,getdict=False,weighted=False,harmonic=False):
    if epochs==None:    
        epochsize=10**(-decimals)
        bins=np.arange(np.around(min(toas),decimals=decimals)-epochsize,np.around(max(toas),decimals=decimals)+epochsize,epochsize)
        freq,bins=np.histogram(toas,bins)
        validinds=np.where(freq!=0)[0]
    
        epochs=np.sort(bins[validinds])
        diffs=np.array(map(lambda x: np.around(x,decimals=decimals),np.diff(epochs)))
        epochs=np.append(epochs[np.where(diffs>epochsize)[0]],[epochs[-1]])
    else:
        epochs=np.array(epochs)
    reducedTOAs=np.array(map(lambda toa: epochs[np.argmin(np.abs(epochs-toa))],toas))

    if data==None:
        return epochs

    Nepochs=len(epochs)

    if weighted and errors!=None:
        averaging_func=lambda x,y: weighted_moments(x,1.0/y**2,unbiased=True,harmonic=harmonic)
    else:
        averaging_func=lambda x,y: (np.mean(x),np.std(y)) #is this correct?


    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval=np.zeros(Nepochs)
        retvalerrs=np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch=epochs[i]
        inds=np.where(reducedTOAs==epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors != None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors==None:
                retval[i]=np.mean(data[inds]) #this is incomplete
                retvalerrs[i]=np.std(data[inds]) #temporary
            else:
                retval[i],retvalerrs[i]=averaging_func(data[inds],errors[inds])
#            print(data[inds],errors[inds])
    if getdict and errors==None: #is this correct?
        return epochs,retval
    return epochs,retval,retvalerrs
    




