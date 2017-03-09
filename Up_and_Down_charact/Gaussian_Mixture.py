import numpy as np
from sklearn import mixture

def hvsd(x): # heaviside step function
    return .5*(1.+np.sign(x))*x

def gaussian(x, mean, std):
    output = np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std
    return output

def fit_3gaussians(Vm, n=1000, ninit=3, bound1=-90, bound2=-35):
    clf = mixture.GaussianMixture(n_components=3, max_iter=n, n_init=ninit,
                                  means_init=((-80,), (-65,), (-50,)), covariance_type='spherical')
    clf.fit(np.array((Vm[(Vm>bound1) & (Vm<bound2)],)).T)
    return clf.weights_, clf.means_.flatten(), np.sqrt(clf.covariances_)

def fit_2gaussians(Vm, n=1000, ninit=3, bound1=-90, bound2=-35, means_init=((-80,), (-50,))):
    clf = mixture.GaussianMixture(n_components=2, max_iter=n, n_init=ninit,
                                  covariance_type='spherical')
    clf.fit(np.array((Vm[(Vm>bound1) & (Vm<bound2)],)).T)
    return clf.weights_, clf.means_.flatten(), np.sqrt(clf.covariances_)

def determine_thresholds(weights, means, stds, down_state_security=1.):
    """ Gives the thresholds given the Gaussian Mixture"""
    i0, i1 = np.argmin(means[0:2]), np.argmax(means[1:3])+1
    alpha = 1.-np.exp(-hvsd(means[i1]-2.*stds[i1]-means[i0]-2.*stds[i0]-down_state_security)/5.)
    return means[i0]+2.*alpha*stds[i0]+down_state_security, means[i1]-2.*alpha*stds[i1]

def loop_over_sliding_window(data, window_size=5., window_update=2.5):

    # Size of X windows
    WS = int(window_size/data['dt']) 
    # Number of those windows
    N_windows = int(data['t'][-1]/window_update)
    WS_small = int(window_update/data['dt'])
    threshold1, threshold2 = 0.*data['t'], 0.*data['t']
    for ii in range(N_windows-int(window_size/window_update)):
        icenter = WS/2.+ii*WS_small
        i0, i1 = int(icenter-WS/2.), int(icenter+WS/2.)
        try:
            t1, t2 = determine_thresholds(*fit_3gaussians(data['Vm'][i0:i1]))
        except ValueError: # means overfitting
            t1, t2 = determine_thresholds(*fit_2gaussians(data['Vm'][i0:i1]))
        threshold1[i0:], threshold2[i0:] = t1+0.*data['t'][i0:], t2+0.*data['t'][i0:]

    return threshold1, threshold2 # adding 1mV to Down state

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from data_analysis.IO.load_data import load_file, get_formated_data
    import state_classification
    data = get_formated_data('/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2016_12_6/16_48_19_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin')
    t1, t2 = loop_over_sliding_window(data)
    UD_transitions, DU_transitions = state_classification.get_transition_times(data['t'], data['Vm'], t1, t2)
    import matplotlib.pylab as plt
    T0, T1 = 50., 70.
    zoom = (data['t']>T0) & (data['t']<T1)
    plt.plot(data['t'][zoom], data['Vm'][zoom], 'k-')
    plt.plot(data['t'][zoom], t1[zoom], 'r-')
    plt.plot(data['t'][zoom], t2[zoom], 'b-')
    for tt in UD_transitions[(UD_transitions>T0) & (UD_transitions<T1)]:
        plt.plot([tt, tt], [data['Vm'].min(), data['Vm'].max()], 'r-', lw=2)
    for tt in DU_transitions[(DU_transitions>T0) & (DU_transitions<T1)]:
        plt.plot([tt, tt], [data['Vm'].min(), data['Vm'].max()], 'b-', lw=2)
    plt.show()
