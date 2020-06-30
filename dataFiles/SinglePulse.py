'''
Michael Lam
Last updated: 12/31/2013

Define interpulse alignment as putting the peak value at len/4. Interpulse will be roughly at 3*len/4


Need to handle what to do if no opw for offpulse noise, etc.

Figure out way to add/average SPs.

'''
from __future__ import division, print_function
from matplotlib.pyplot import *

import numpy as np
import utilities as u
import scipy.optimize as optimize
import sys

#sys.path.append('/home/dizzy4/mlam/source/jimcode')

import waveforms
get_toa = waveforms.get_toa3
#import ffttoa
#get_toa = ffttoa.get_toa

#ACF=lambda p: np.correlate(p,p,"full") #no longer used

class SinglePulse:
    def __init__(self,data,mpw=None,ipw=None,opw=None,prepare=False,align=None,period=None):
        self.data=np.array(data)
        if mpw is not None:
            self.mpw = np.array(mpw)
        else:
            self.mpw = None
        if ipw is not None:
            self.ipw = np.array(ipw)
        else:
            self.ipw = None
        #Define off pulse
        self.nbins = len(data)
        bins=np.arange(self.nbins)

        if opw is None:
            if self.mpw is None and self.ipw is None:
                self.opw=None #do not define any windows
            elif self.ipw is None:
                self.opw=bins[np.logical_not(np.in1d(bins,mpw))]
            elif self.mpw is None:
                self.opw=bins[np.logical_not(np.in1d(bins,ipw))]
            else:
                self.opw=bins[np.logical_not(np.logical_or(np.in1d(bins,mpw),np.in1d(bins,ipw)))]
        else:
            self.opw=np.array(opw)

        if self.mpw is None and self.ipw is None and self.opw is None:
            self.mpw=np.arange(self.nbins)

        if align:
            if align!=0:
                self.data = np.roll(self.data,align)
                #prepare=True #? #keep this for 1937?
            #self.shiftit(align,save=True)



        if prepare: #change this for jitter (prepare set to False here)
            self.interpulse_align()
            #self.normalize() #do not do this

        self.period = period

        self.null = False
        if np.all(self.data==0) or np.all(np.isnan(self.data)):
            self.null = True


    def interpulse_align(self):
        self.data = np.roll(u.center_max(self.data),-len(self.data)//4)

    def center_align(self):
        self.data = u.center_max(self.data)
        
    def normalize(self):
        minimum=np.mean(self.getOffpulse())
        #print minimum
        self.data=u.normalize(self.data,minimum=minimum)
        

    def getFWHM(self,simple=False,timeunits=True):
        #remove baseline? what if no offpulse window?
        dbin = u.FWHM(self.data,notcentered=True)#,window=800)
        factor=1
        if timeunits and self.period is not None:
            factor = self.period/self.nbins
        return factor*dbin
        


    def getWeff(self,fourier=False,sumonly=False,timeunits=True):
        if not timeunits or self.period is None:
            return None
        P=self.period
        N=self.nbins
        U=u.normalize(self.data,simple=True) #remove baseline?
        
        tot=np.sum(np.power(U[1:]-U[:-1],2))
        if sumonly:
            return tot
        self.weff=P/np.sqrt(N*tot)
        return self.weff




    def remove_baseline(self,save=True):
        if self.opw is None:
            #print "No Offpulse" #do this?
            return
        opmean = np.mean(self.getOffpulse())
        if save:
            self.data = self.data - opmean
            return self.data
        return self.data - opmean

    

    def getMainpulse(self):
        if self.mpw is None:
            return None
        return self.data[self.mpw]
    def getInterpulse(self):
        if self.ipw is None:
            return None
        return self.data[self.ipw]
    def getOffpulse(self):
        if self.opw is None:
            return None
        return self.data[self.opw]
    def getAllpulse(self):
        return self.getMainpulse(),self.getInterpulse(),self.getOffpulse()

    def getMainpulseACF(self):
        mp=self.getMainpulse()
        return u.acf(mp,var=False,norm_by_tau=True)
    def getInterpulseACF(self):
        if self.ipw is None:
            return None
        ip=self.getInterpulse()
        return u.acf(ip,var=False,norm_by_tau=True)
    def getOffpulseACF(self):
        if self.opw is None:
            return None
        op=self.getOffpulse()
        return u.acf(op,var=False,norm_by_tau=True)
    def getAllACF(self):
        return self.getMainpulseACF(),self.getInterpulseACF(),self.getOffpulseACF()
    
    def getOffpulseNoise(self,full=False):
        if self.opw is None:
            return None
        op=self.getOffpulse()
        if full:
            return np.mean(op),np.std(op)
        return np.std(op)

    def getOffpulseZCT(self):
        return u.zct(self.getOffpulse(),full=True,meansub=True)



    def fitPulse(self,template,fixedphase=False,rms_baseline=None):
        """
        Returns taucff, tauhat, bhat, sigma_Tau,sigma_b, snr, rho
        """
        if self.null:
            return None
        if rms_baseline is None:
            self.remove_baseline()
        if fixedphase: #just return S/N
            p0 = [np.max(self.data)]
            p1,cov,infodict,mesg,ier = optimize.leastsq(lambda p,x,y: np.abs(p[0])*x - y,p0[:],args=(np.asarray(template,np.float64),np.asarray(self.data,np.float64)),full_output=True) #conversion to np.float64 fixes bug with Jacobian inversion
            noise = self.getOffpulseNoise()
            return np.abs(p1[0])/noise#,np.sqrt(cov[0][0])/noise
        if self.opw is None:
            if rms_baseline is not None:
                try:
                    return get_toa(template,self.data,rms_baseline)
                except:
                    print(self.data)
                    plot(self.data)
                    show()
                    raise SystemExit
            return get_toa(template,self.data,1)
        try: #problem?
            return get_toa(template,self.data,self.getOffpulseNoise())#,nlagsfit=1001)
        except:
            return None

        
    #define this so a positive shift is forward
    def shiftit(self,shift,save=False):
        x = waveforms.shiftit(self.data,-1*shift)
        if save:
            self.data = x
        return x



    def getPeriod(self):
        return self.period

    def getNBins(self):
        return len(self.data)
