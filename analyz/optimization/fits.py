# Filename: fits.py

import numpy as np
from scipy.optimize import leastsq, minimize
from analyz.workflow.funcs import get_default_args

def leastsq_fit(x, y, waveform, coeffs=None, verbose=False):
    """
    'waveform' has to be a fnction waveform(x, coeffs) :
    """
    if coeffs is None:
        try:
            coeffs = get_default_args(waveform)['coeffs']
        except NameError:
            print('/!\ No default "coeffs" found in function, need to pass it explicitely !')

            
    def residual(c):
        return (y - waveform(x, coeffs=c))
    
    C, flag = leastsq(residual, [.3, .2, 2.])

    if verbose:
        print(flag)
    return C

def curve_fit(x, y, waveform, coeffs=None,
              bounds=None,
              method=None,
              options={'maxiter':10000},
              verbose=False):
    """
    'waveform' has to be a fnction waveform(x, coeffs) :
    """
    if coeffs is None:
        try:
            coeffs = get_default_args(waveform)['coeffs']
        except NameError:
            print('/!\ No default "coeffs" found in function, need to pass it explicitely !')
            
    def residual(c):
        return np.sum((y-waveform(x, c))**2)/len(y)

    if verbose:
        options['disp']=True

    if (method is None) and (bounds is not None):
        res = None
    else:
        res = minimize(residual, coeffs,
                       method='Nelder-Mead',
                       options=options)
    if verbose:
        print(res)
        
    return res

def curve_fit(x, y, waveform, coeffs=None,
              bounds=None,
              method=None,
              options={'maxiter':10000},
              verbose=False):
    # """
    # 'waveform' has to be a fnction waveform(x, coeffs) :
    # """
    if coeffs is None:
        try:
            coeffs = get_default_args(waveform)['coeffs']
        except NameError:
            print('/!\ No default "coeffs" found in function, need to pass it explicitely !')

    def to_minimize(c):
        return np.sum((y-waveform(x, c))**2)/len(y)

    if (method is None) and (bounds is None):
        method='Nelder-Mead' # default method

    if (bounds is not None):
        res = minimize(to_minimize, coeffs,
                       method=method, bounds=bounds, options=options)
    else:
        res = minimize(to_minimize, coeffs,
                       method=method, options=options)
        
    if verbose:
        print(res)

    return res

if __name__=='__main__':


    import numpy as np
    from datavyz import ge

    def waveform(x, coeffs=[.3, .2, 2.]):
        return coeffs[2]*(np.sign(x-coeffs[0])+1.)*np.exp(-(x-coeffs[0])/coeffs[1])

    x = np.linspace(0, 1, 100)
    y = np.random.randn(100)+waveform(x, [0.5, .2, 4.])

    res = curve_fit(x, y, waveform)

    ge.plot(x, Y=[y, waveform(x, res.x)])
    ge.show()
