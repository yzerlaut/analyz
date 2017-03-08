import numpy as np
from scipy.optimize import minimize


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
    
    
