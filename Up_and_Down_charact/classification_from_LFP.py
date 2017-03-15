"""
Some functions are identical (so imported from) the classification_from_Vm.py file
"""
import numpy as np
from scipy.optimize import minimize
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.Up_and_Down_charact.classification_from_Vm import get_state_intervals, gaussian
from data_analysis.freq_analysis.wavelet_transform import my_cwt
from data_analysis.processing.signanalysis import butter_lowpass_filter

def get_time_variability(Pow_vs_t, dt, T=10e-3):
    iT = int(T/dt)
    var = 0.*Pow_vs_t
    for i in range(len(Pow_vs_t)):
        var[i] = Pow_vs_t[max([0,i-iT]):min([i+iT,len(Pow_vs_t)])].std()
    return var

def processLFP(LFP, freqs, dt, lin_combination=None, smoothing=10e-3):

    if lin_combination is None:
        # by default all frequencies are equally weighted !
        lin_combination = np.ones(len(freqs))*1./len(freqs)
        
    coefs = my_cwt(LFP, freqs, dt)
    tfPow = (lin_combination*np.abs(coefs).T).T

    # output = butter_lowpass_filter(tfPow.mean(axis=0), 1./smoothing, 1./dt, order=5)
    output = get_time_variability(tfPow.mean(axis=0), dt, T=5e-3)
    output = butter_lowpass_filter(output, 1./smoothing, 1./dt, order=5)
    
    return output
     
def fit_2gaussians(pLFP, n=1000, nbins=200):
    """
    take the histogram of the Vm values
    and fits two gaussians using the least square algorithm

    'upper_bound' cuts spikes !

    returns the weights, means and standard deviation of the
    gaussian functions
    """
    vbins = np.linspace(pLFP.min(), pLFP.max(), nbins) # discretization of Vm for histogram
    hist, be = np.histogram(pLFP, vbins, normed=True) # normalized distribution
    vv = 0.5*(be[1:]+be[:-1]) # input vector is center of bin edges
    
    def to_minimize(args):
        w, m1, m2, s1, s2 = args
        double_gaussian = w*gaussian(vv, m1, s1)+(1.-w)*gaussian(vv, m2, s2)
        return np.power(hist-double_gaussian, 2).sum()

    # initial values
    mean0, std0 = pLFP.mean(), pLFP.std()
    w, m1, m2, s1, s2 = 0.5, mean0-std0, mean0+std0, std0/2., std0/2.
    
    res = minimize(to_minimize, [w, m1, m2, s1, s2],
                   method='L-BFGS-B',
                   bounds = [(.05, .95),
                             (vv.min(), vv.max()), (vv.min(), vv.max()),
                             (1e-2, vv.max()-vv.min()), (1e-2, vv.max()-vv.min())],
                   options={'maxiter':n})

    w, m1, m2, s1, s2 = res.x
    
    return (w, 1-w), (m1, m2), (s1, s2)

def determine_threshold(weigths, means, stds, with_amp=False):
    """ Gives the thresholds given the Gaussian Mixture"""
    
    i0, i1 = np.argmin(means), np.argmax(means) # find the upper and lower distrib

    if stds[i0]/stds[i1]<.5:
        vv = np.linspace(means[i0], means[i1], 1e2) # the point is in between the two means
        gaussian1 = weigths[i0]*gaussian(vv, means[i0], stds[i0])
        gaussian2 = weigths[i1]*gaussian(vv, means[i1], stds[i1])
        ii = np.argmin(np.power(gaussian1-gaussian2, 2))
        threshold, amp = vv[ii], gaussian1[ii]
    else:
        threshold, amp = means[i0], weigths[i0]*gaussian(0,0,stds[i0])
        
    if with_amp:
        return threshold, amp
    else:
        return threshold

    
def determine_threshold_on_sliding_window(t, pLFP, sliding_window=20):

    dt = t[1]-t[0]
    islw = int(sliding_window/dt)
    THRESHOLDS, INTERVALS = [], []
    i=0
    while i<len(t):
        INTERVALS.append([t[i], t[i+islw]])
        weights, means, stds = fit_2gaussians(pLFP[i:i+islw])
        THRESHOLDS.append(determine_threshold(weigths, means, stds))
        i+=islw
    INTERVALS[-1][1] = np.inf

    # now making a function out of this
    return threshold_func

if __name__=='__main__':
    
    import sys
    sys.path.append('../..')
    from data_analysis.IO.load_data import load_file, get_formated_data
    t, [Vm, _, _, _, _, _, LFP] = load_file('/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2016_12_6/16_48_19_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin',
                                            zoom=[0, 20])
    weights, means, stds = fit_2gaussians(LFP, n=300)
    
    # for w, m, s in zip(weights, means, stds):
    #     plt.plot(w*gaussian(vbins, m, s), vbins, ':', lw=1, color='k')
    # # find threshold as the interesction of the two Gaussians
    # threshold_up, threshold_down = determine_thresholds(weights, means, stds)    
    # Up_intervals, Down_intervals = get_state_intervals(Vm[t<tstop], threshold,
    #                                                    threshold_up, threshold_down,
    #                                                    t[-1]-t[0],
    #                                                    min_duration_criteria=100e-3)
    
    # t1, t2 = loop_over_sliding_window(data)
    # UD_transitions, DU_transitions = state_classification.get_transition_times(data['t'], data['Vm'], t1, t2)
    # import matplotlib.pylab as plt
    # T0, T1 = 50., 70.
    # zoom = (data['t']>T0) & (data['t']<T1)
    # plt.plot(data['t'][zoom], data['Vm'][zoom], 'k-')
    # plt.plot(data['t'][zoom], t1[zoom], 'r-')
    # plt.plot(data['t'][zoom], t2[zoom], 'b-')
    # for tt in UD_transitions[(UD_transitions>T0) & (UD_transitions<T1)]:
    #     plt.plot([tt, tt], [data['Vm'].min(), data['Vm'].max()], 'r-', lw=2)
    # for tt in DU_transitions[(DU_transitions>T0) & (DU_transitions<T1)]:
    #     plt.plot([tt, tt], [data['Vm'].min(), data['Vm'].max()], 'b-', lw=2)
    # plt.show()
    


