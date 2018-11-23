import numpy as np

def Wiener_Process(mu, sigma, dt=0.1, tstop=100):
    """ 
    """
    return np.random.randn(int(tstop/dt))*sigma+mu

def OrnsteinUhlenbeck_Process(mu, sigma, tau, dt=0.1, tstop=100):
    """ 
    Ornstein-Uhlenbeck Process

    from joint work with bartosz telenczuk:
    https://github.com/yzerlaut/transfer_functions/blob/master/tf_filters.py
    """ 
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
    from data_analysis.signal_library
    
    mean, std, tau = 1, 2, 10
    y = OrnsteinUhlenbeck_Process(mean, std, tau, tstop=1000)
    
    fig, AX = figure(axes_extents=[[[2,1]],[[1,1],[1,1]]])
    plot(y, ax=AX[0][0])
    AX[0][0].grid()

    hist(y, ax=AX[1][0], normed=True)

    show()
