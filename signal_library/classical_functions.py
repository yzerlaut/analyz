import numpy as np
from  scipy.special import erf

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


if __name__=='__main__':
    import sys
    sys.path.append('../../')
    from graphs.my_graph import graphs
    mg = graphs()
    x = np.linspace(-3,3)
    fig, ax = mg.figure(with_legend_space=True)
    for func, label in zip([gaussian, logistic, error, gaussian_cumproba],
                ['gaussian', 'logistic', 'error', 'gaussian\ncum.proba']):
        mg.plot(x, func(x), ax=ax, no_set=True, label=label)
    mg.set_plot(ax)
    ax.legend(loc=(1.,.2))
    mg.show()
