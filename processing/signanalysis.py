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
    signal = (signal-signal.mean())/signal.std()
    cr = np.correlate(signal[steps:],signal)/steps
    time_shift = np.arange(len(cr))*dt
    return cr/cr.max(), time_shift

def crosscorrel(signal1,signal2):
    """
    argument : signal1 (np.array()), signal2 (np.array())
    returns : np.array()
    take two signals, and returns their crosscorrelation function 
    """
    signal1 = (signal1-signal1.mean())
    signal2 = (signal2-signal2.mean())
    cr = np.correlate(signal1,signal2,"full")/signal1.std()/signal2.std()
    return cr

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

    import sys
    import matplotlib.pyplot as plt
    
    if len(sys.argv)>1:
        filtertype = sys.argv[-1]
    else:
        filtertype = 'low-pass'
        
    # Filter requirements.
    order = 10
    fs = 300.0       # sample rate, Hz
    cutoff = 30.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    if filtertype=='low-pass':
        b, a = butter_lowpass(cutoff, fs, order)
    elif filtertype=='high-pass':
        b, a = butter_highpass(cutoff, fs, order)
    elif filtertype=='band-pass':
        b, a = butter_bandpass(cutoff, cutoff/10., fs, order)

    # Plot the frequency response.
    w, h = signal.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title(filtertype+" Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0         # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)+5.

    # Filter the data, and plot both the original and filtered signals.
    if filtertype=='low-pass':
        y = butter_lowpass_filter(data, cutoff, fs, order)
    elif filtertype=='high-pass':
        y = butter_highpass_filter(data, cutoff, fs, order)
    elif filtertype=='band-pass':
        y = butter_bandpass(cutoff, cutoff/10., fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()
