# Filename: signanalysis.py

import numpy.fft as fft
import numpy as np
from scipy import signal
from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d

def gaussian_smoothing(signal, idt_sbsmpl=10):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(signal, idt_sbsmpl)

def autocorrel(signal, tmax, dt):
    """
    argument : signal (np.array), tmax and dt (float)
    tmax, is the maximum length of the autocorrelation that we want to see
    returns : autocorrel (np.array), time_shift (np.array)
    take a signal of time sampling dt, and returns its autocorrelation
     function between [0,tstop] (normalized) !!
    """
    steps = int(tmax/dt) # number of steps to sum on
    Signal = (signal-signal.mean())/signal.std()
    cr = np.correlate(Signal[steps:],Signal)/steps
    time_shift = np.arange(len(cr))*dt
    return cr/cr.max(), time_shift

def crosscorrel(signal1, signal2, tmax, dt):
    """
    argument : signal1 (np.array()), signal2 (np.array())
    returns : np.array()
    take two signals, and returns their crosscorrelation function 

    CONVENTION:
    --------------------------------------------------------------
    when the peak is in the past (negative t_shift)
    it means that signal2 is delayed with respect to signal 1
    --------------------------------------------------------------
    """
    if len(signal1)!=len(signal2):
        print('Need two arrays of the same size !!')
        
    steps = int(tmax/dt) # number of steps to sum on
    time_shift = dt*np.concatenate([-np.arange(1, steps)[::-1], np.arange(steps)])
    CCF = np.zeros(len(time_shift))
    for i in np.arange(steps):
        ccf = np.corrcoef(signal1[:len(signal1)-i], signal2[i:])
        CCF[steps-1+i] = ccf[0,1]
    for i in np.arange(steps):
        ccf = np.corrcoef(signal2[:len(signal1)-i], signal1[i:])
        CCF[steps-1-i] = ccf[0,1]
    return CCF, time_shift

def crosscorrel_norm(signal1,signal2):
    """
    computes the cross-correlation, and takes into account the boundary
    effects ! so normalizes by the weight of the bins !!
    the two array have t have the same size @
    
    argument : signal1 (np.array()), signal2 (np.array())
    returns : np.array()
    take two signals, and returns their crosscorrelation function 
    """
    if signal1.size!=signal2.size:
        print("problem no equal size vectors !!")
    signal1 = (signal1-signal1.mean())
    signal2 = (signal2-signal2.mean())
    cr = signal.fftconvolve(signal1,signal2,"full")/signal1.std()/signal2.std()
    ww = np.linspace(signal1.size,0,-1)
    bin_weight = np.concatenate((ww[::-1],ww[1:]))
    return cr/bin_weight

def butter_lowpass(Fcutoff, Facq, order=5):
    nyq = 0.5 * Facq
    normal_cutoff = Fcutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, Fcutoff, Facq, order=5):
    b, a = butter_lowpass(Fcutoff, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(Fcutoff, Facq, order=5):
    nyq = 0.5 * Facq
    normal_cutoff = Fcutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, Fcutoff, Facq, order=5):
    b, a = butter_highpass(Fcutoff, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, Facq, order=5):
    nyq = 0.5 * Facq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, Facq, order=5):
    b, a = butter_bandpass(lowcut, highcut, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def low_pass_by_convolve_with_exp(data, T, dt):
    """
    function to be worked out, normalization not ok
    """
    tt = np.arange(int(5.*T/dt))*dt
    tmid = tt[int(len(tt)/2.)]
    exp = np.array([np.exp(-(t2-tmid)/T) if t2>=tmid else 0 for t2 in tt])
    conv_number = signal.convolve(np.ones(len(data)),
                                  .5*(1+np.sign(tt-tmid)),
                                  mode='same')
    output = signal.convolve(data, exp,
                             mode='same')/conv_number
    return output

if __name__=='__main__':

    import matplotlib.pyplot as plt
    
    t = np.linspace(0, np.pi*2, int(1e3))
    signal1 = np.sin(3*t)
    signal2 = np.cos(3*t-np.pi/4)
    cr, t_shift = crosscorrel(signal1, signal2, np.pi/2, t[1]-t[0])
    _, ax = plt.subplots(2)
    ax[0].plot(t, signal1, 'k')
    ax[0].plot(t, signal2)
    ax[1].plot(t_shift, cr, '-')

    plt.show()
