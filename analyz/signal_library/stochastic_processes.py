import numpy as np

def Wiener_Process(mu, sigma, dt=0.1, tstop=100, seed=0):
    """ 
    """
    np.random.seed(seed)
    return np.random.randn(int(tstop/dt))*sigma+mu

def OrnsteinUhlenbeck_Process(mu, sigma, tau, dt=0.1, tstop=100, seed=1):
    """ 
    Ornstein-Uhlenbeck Process

    from joint work with bartosz telenczuk:
    https://github.com/yzerlaut/transfer_functions/blob/master/tf_filters.py
    """
    np.random.seed(seed)
    
    diffcoef = 2*sigma**2/tau
    y0 = mu
    n_steps = int(tstop/dt)
    A = np.sqrt(diffcoef*tau/2.*(1-np.exp(-2*dt/tau)))
    noise = np.random.randn(n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    for i in range(n_steps-1):
        y[i+1] = y0 + (y[i]-y0)*np.exp(-dt/tau)+A*noise[i]
    return y


if __name__ == '__main__':
    
    import sys
    sys.path.append('../../../')
    from graphs.my_graph import *
    from classical_functions import gaussian
    sys.path.append('../')
    from processing.signanalysis import autocorrel

    tstop=10000
    mean, std, tau = 1, 2, 30
    y = OrnsteinUhlenbeck_Process(mean, std, tau, tstop=tstop, dt=1)
    
    fig, AX = figure(axes_extents=[[[2,1]],[[1,1],[1,1]]], hspace=1.8, wspace=1.5)
    
    plot(y[:1000], ax=AX[0][0], xlabel='time', ylabel='signal')
    AX[0][0].grid()

    hist(y, ax=AX[1][0], normed=True, ylabel='occurence', xlabel='signal')
    AX[1][0].plot(np.linspace(y.min(), y.max()), gaussian(np.linspace(y.min(), y.max()), mean, std))

    acf, ts = autocorrel(y, 4*tau, 1)
    plot(ts, acf, ax=AX[1][1])
    AX[1][1].plot(ts, np.exp(-ts/tau), lw=2, alpha=.5, label='theory')
    AX[1][1].legend()
    set_plot(AX[1][1], ylabel='autocorrelation', xlabel='time shift')
    
    show()
