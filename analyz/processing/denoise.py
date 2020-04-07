import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

def remove_50Hz(t, signal):
    """
    removes the 50 Hz on signal by fitting a sinewave and substracting it
    """
    if (t[-1]-t[0])>10:
        # no need of more than 10 seconds to accrately capture the 
        t2, signal2 = t[:int(10/(t[1]-t[0]))], signal[:int(10/(t[1]-t[0]))]
    else:
        t2, signal2 = t, signal
    f = 50.
    def sinewave(x, t=t2):
        return np.sin(2.*np.pi*f*t+x[0])*x[1]
    def min_sinewave(x):
        return np.power(signal2-signal2.mean()-sinewave(x),2).sum()
    res = minimize(min_sinewave, (0, signal.std()))
    print(res)
    # return signal-sinewave(res.x)
    return signal, sinewave(res.x, t=t)
    
def remove_mean_drift(t, data, T=1.):
    """
    evaluate the mean of a signal over a sliding window of size T
    and substract this mean to the signal !
    """
    # the convolution function is a Heaviside function to get the mean
    conv_func = np.ones(int(T/(t[1]-t[0]))) 
    # the number of convoluted points is variable (boundary effect)
    conv_number = convolve(np.ones(len(data)), conv_func,
                                      mode='same')
        # the sliding mean that depends on the frequency
    sliding_mean = convolve(data, conv_func,
                                   mode='same')/conv_number
    return data-sliding_mean
    
