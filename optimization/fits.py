# Filename: fits.py

import numpy as np
from scipy.optimize import leastsq


def leastsq_fit(x, data, waveform, coeffs, verbose=False):
    """
    'waveform' has to be a fnction waveform(x, coeffs) :
    """
    def residuals(coeffs, y, x):
        return y - waveform(x, coeffs)    
    
    C, flag = leastsq(residuals, coeffs, args=(data, x))

    if verbose:
        print(flag)
    return C

