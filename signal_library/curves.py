import numpy as np

def gaussian(x, mu, sigma):
    if sigma<=0:
        # print('the standard deviation has to be strictly postitive')
        return 0*x
    else:
        return 1./np.sqrt(2*np.pi)/sigma*np.exp(-((x-mu)/sigma)**2)
