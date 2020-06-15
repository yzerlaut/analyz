# Filename: fits.py

import numpy as np
from scipy.optimize import leastsq, minimize

def leastsq_fit(x, data, waveform, coeffs, verbose=False):
    """
    'waveform' has to be a fnction waveform(x, coeffs) :
    """
    def residuals(coeffs, y, x):
        return np.abs(y - waveform(x, coeffs)).sum()
    
    C, flag = leastsq(residuals, coeffs, args=(data, x))

    if verbose:
        print(flag)
    return C

def curve_fit(x, y, waveform, coeffs, bounds=None, verbose=False):
    """
    'waveform' has to be a fnction waveform(x, coeffs) :
    """
    def residual(coeff):
        return np.abs(y - waveform(x, coeffs)).sum()
    
    res = minimize(residual, coeffs, bounds=bounds)

    if verbose:
        print(res)
    return res


if __name__=='__main__':


    import numpy as np
    from datavyz import ge

    def waveform(x, coeffs=[.3, .2, 2.]):
        return coeffs[2]*(np.sign(x-coeffs[0])+1.)*np.exp(-(x-coeffs[0])/coeffs[1])

    import inspect
    print(inspect.signature(waveform).coeffs)
    # x = np.linspace(0, 1, 100)
    # y = np.random.randn(100)+waveform(x)

    # ge.plot(x, y)
    # ge.show()
