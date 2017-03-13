import numpy.fft as fft
import numpy as np
from scipy import signal

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

if __name__=='__main__':

    import sys
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
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
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()
