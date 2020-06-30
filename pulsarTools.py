from __future__ import division, print_function

def initializer(dataSet):
    import numpy as np
    dt = 6.54761904761905e-05
    if dataSet == 'Set1':
        values = np.fromfile('dataFiles/ts_B0523+11_DM79.30.dat','float32')
    if dataSet == 'Set2':
        values = np.fromfile('dataFiles/ts_B1737+13_DM48.70.dat','float32')
    if dataSet == 'Set3':
        values = np.fromfile('dataFiles/ts_B1848+12_DM70.60.dat','float32')
    times = np.linspace(0,dt*len(values),len(values))
    return times,values

def fft(times,values):
    import numpy as np
    dt = times[1]-times[0]
    N = len(times)
    freqs = np.fft.fftfreq(N,d=dt)
    fourier = np.fft.fft(values)
    freqs = np.fft.fftshift(freqs)
    fourier = np.fft.fftshift(fourier)
    return freqs,fourier

def folder(times,values,trialPeriod):
    import numpy as np
    dt = float(times[1]-times[0])
    nTimes = int(np.ceil(trialPeriod/dt))
    excess = nTimes*dt-trialPeriod
    accumulatedExcess = 0
    nRemain = len(values)
    foldedValues = []
    j = 0
    while nRemain > nTimes:
        accumulatedExcess += excess
        if accumulatedExcess < dt:
            foldedValues.append(values[j:j+nTimes])
            nRemain -= nTimes
            j += nTimes
        else:
            accumulatedExcess = 0
            nRemain += 1
            j -= 1
            foldedValues.append(values[j:j+nTimes])
            nRemain -= nTimes
            j += nTimes
    return np.array(foldedValues)
        

def significance(foldedValues):
    import numpy as np
    collapsed = np.zeros(len(foldedValues[0]))
    for i in range(len(foldedValues)):
        collapsed += foldedValues[i]
    collapsed /= len(foldedValues)
    mean = np.mean(collapsed)
    X2 = 0
    for i in range(len(collapsed)):
        X2 += (collapsed[i]-mean)**2
    X2 /= float(len(collapsed))
    return X2

def periodSearch(times,values,periodLB,periodUB,nPeriod):
    import numpy as np
    X2s = []
    trialPeriods = np.linspace(periodLB,periodUB,nPeriod)
    for i in range(nPeriod):
        foldedValues = folder(times,values,trialPeriods[i])
        X2s.append(significance(foldedValues))
        print str(100*float(i+1)/nPeriod)+'% complete'
    return trialPeriods,X2s

def plot_powerSpectrum(freqs,fourier):
    import numpy as np
    import pylab as py
    powerSpectrum = np.real(fourier*np.conj(fourier))
    for i in range(len(freqs)):
        if np.abs(freqs[i]) < 0.3:
            powerSpectrum[i] = 0
    fig = py.figure()
    ax = fig.add_axes([0.15,0.12,0.78,0.78])
    ax.plot(freqs,powerSpectrum)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [Arbitrary Units]')
    ax.set_xlim(min(freqs),max(freqs))
    py.show()

def plot_folded(trialPeriod,foldedValues):
    import numpy as np
    import pylab as py
    nPeriod = len(foldedValues)
    nTimes = len(foldedValues[0])
    fig = py.figure()
    ax = fig.add_axes([0.15,0.12,0.78,0.78])
    ax.pcolormesh(foldedValues)
    ax.set_xlim(0,nTimes)
    ax.set_ylim(0,nPeriod)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Time [s]')
    xtick_locs = np.linspace(0,nTimes,5)
    xtick_lbls = np.linspace(0,1,5)
    for i in range(len(xtick_lbls)):
        xtick_lbls[i] = float(str(xtick_lbls[i]))
    ytick_locs = np.linspace(0,nPeriod,5)
    ytick_lbls = np.linspace(0,nPeriod*trialPeriod,5)
    for i in range(len(ytick_lbls)):
        ytick_lbls[i] = float(str(ytick_lbls[i])[0:min(6,len(str(ytick_lbls[i]))-1)])
    py.xticks(xtick_locs,xtick_lbls)
    py.yticks(ytick_locs,ytick_lbls)
    py.show()

def plot_collapsed(foldedValues):
    import numpy as np
    import pylab as py
    nPeriod = len(foldedValues)
    nTimes = len(foldedValues[0])
    collapsed = np.zeros(len(foldedValues[0]))
    for i in range(len(foldedValues)):
        collapsed += foldedValues[i]
    collapsed /= len(foldedValues)
    fig = py.figure()
    ax = fig.add_axes([0.15,0.12,0.78,0.78])
    ax.plot(collapsed)
    ax.set_xlim(0,nPeriod)
    ax.set_ylabel('Intensity [Arbitrary Units]')
    ax.set_xlabel('Phase')
    xtick_locs = np.linspace(0,nTimes,5)
    xtick_lbls = np.linspace(0,1,5)
    for i in range(len(xtick_lbls)):
        xtick_lbls[i] = float(str(xtick_lbls[i])[0:min(6,len(str(xtick_lbls[i]))-1)])
    py.xticks(xtick_locs,xtick_lbls)
    py.show()
    
def plot_X2s(periods,X2s):
    import pylab as py
    fig = py.figure()
    ax = fig.add_axes([0.15,0.12,0.78,0.78])
    ax.plot(periods,X2s)
    ax.set_xlim(min(periods),max(periods))
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('Significance')
    py.show()

