import numpy as np
from scipy.special import erf

def heaviside(x, x0=0):
    return (np.sign(x-x0)+1.)/2.
    
def gaussian(x, mean=0., std=1.):
    """
    """
    return np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std

def logistic(x, mean=0., std=1.):
    """
    """
    return np.exp((x-mean)/std)/(1+np.exp((x-mean)/std))

def error(x, mean=0., std=1.):
    """
    """
    return erf((x-mean)/std)

def gaussian_cumproba(x, mean=0., std=1.):
    """
    """
    return 1/2 * (1 + erf((x-mean)/std/np.sqrt(2)))


def gaussian_2d(x, y,
                mu=(0,0),
                sigma=(1,1)):
    
    xcomp = 1./np.sqrt(2*np.pi)/sigma[0]*np.exp(-((x-mu[0])/sigma[0])**2)
    ycomp = 1./np.sqrt(2*np.pi)/sigma[1]*np.exp(-((y-mu[1])/sigma[1])**2)
    
    return xcomp*ycomp

if __name__=='__main__':

    from datavyz.main import graph_env
    ge = graph_env()
    
    x = np.linspace(-3,3)
    fig, ax = ge.figure(with_legend_space=True)
    for func, label in zip([gaussian, logistic, error, gaussian_cumproba],
                ['gaussian', 'logistic', 'error', 'gaussian\ncum.proba']):
        ge.plot(x, func(x), ax=ax, no_set=True, label=label)
    ge.set_plot(ax)
    ax.legend(loc=(1.,.2))
    ge.show()
