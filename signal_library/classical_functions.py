import numpy as np

def gaussian(x, mean, std):
    """
    """
    return np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std

