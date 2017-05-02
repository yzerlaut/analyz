"""
Some functions are identical (so imported from) the classification_from_Vm.py file
"""
import numpy as np
from scipy.optimize import minimize
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.Up_and_Down_charact.classification_from_Vm import gaussian, get_thresholded_intervals, apply_duration_criteria
from data_analysis.freq_analysis.wavelet_transform import my_cwt
from data_analysis.processing.filters import butter_bandpass_filter, butter_lowpass_filter
from data_analysis.processing.signanalysis import gaussian_smoothing

def get_gaussian_mixture_for_several_data(DATA, key='ExtraCort', std_window=5e-3):
    array = np.empty(0)
    for data in DATA:
        compute_smooth_time_varying_std(data, std_window=std_window)
        array = np.concatenate([array, data[key+'_var_smoothed']])
        # get the gaussian mixture
    W, M, S = fit_2gaussians(array, n=1000, nbins=200)
    for data in DATA:
        data[key+'_var_W'], data[key+'_var_M'], data[key+'_var_S'] = W, M, S
    
def get_threshold_given_gaussian_mixture(data, key='ExtraCort'):
    """
    this is the part that is different from Mukovski et al.
    """
    W, M, S = data[key+'_var_W'], data[key+'_var_M'], data[key+'_var_S']


    i0, i1 = np.argmin(data[key+'_var_M']), np.argmax(data[key+'_var_M']) # find the upper and lower distrib
    # the lower gaussian is the quiescent one, Up states are above that one
    data[key+'_var_threshold'] = M[i0]+S[i0]
    
    # now down states are below the intersection between the two distrib
    vv = np.linspace(data[key+'_var_M'][i0], data[key+'_var_M'][i1], 1e2) # the point is in between the two means
    gaussian1 = data[key+'_var_W'][i0]*gaussian(vv, data[key+'_var_M'][i0], data[key+'_var_S'][i0])
    gaussian2 = data[key+'_var_W'][i1]*gaussian(vv, data[key+'_var_M'][i1], data[key+'_var_S'][i1])
    ii = np.argmin(np.power(gaussian1-gaussian2, 2))
    data[key+'_var_threshold_low'] = vv[ii]
    

def compute_smooth_time_varying_std(data, key='ExtraCort',
                                    std_window=5e-3, smoothing=50e-3,
                                    g_band=[20, 100]):
    
    # band-pass filter
    data[key+'_filtered'] = butter_bandpass_filter(data[key],
                                                   g_band[0], g_band[1],
                                                   1./data['dt'], order=3)
    n = int(std_window/data['dt'])
    t, var = [], []
    n1, n2 = 0, n
    while n2<len(data['t']):
        t.append(data['t'][int(n1+n/2)])
        var.append(data[key+'_filtered'][n1:n2].std())
        n1, n2 = int(n1+n/2), int(n2+n/2)
    data['t_var'] = np.array(t)
    data['dt_var'] = data['t_var'][1]-data['t_var'][0]
    data[key+'_var'] = np.array(var)
    
    # smooth the time-varying fluct.
    nsmooth = int(smoothing/data['dt_var'])
    data[key+'_var_smoothed'] = gaussian_smoothing(data[key+'_var'], nsmooth)

    
    
def Mukovski_method(data, key='ExtraCort',
                    min_duration=100e-3, max_duration=np.inf,
                    std_window=5e-3, smoothing=50e-3,
                    with_down_intervals=False,
                    GAUSSIAN_MIXTURE=None):

    
    if GAUSSIAN_MIXTURE is not None:
        W, M, S = GAUSSIAN_MIXTURE
    else:
        # the time-varying std was not computed before 
        compute_smooth_time_varying_std(data, key=key,
                                        std_window=std_window,
                                        smoothing=smoothing)
        # get the gaussian mixture
        W, M, S = fit_2gaussians(data[key+'_var_smoothed'], n=1000, nbins=200)
        data[key+'_var_W'], data[key+'_var_M'], data[key+'_var_S'] = W, M, S
        
    get_threshold_given_gaussian_mixture(data, key=key)

    data['intervals'] = get_thresholded_intervals(data['t_var'],
                                                  data[key+'_var_smoothed'],
                                                  data[key+'_var_threshold'],
                                                  where='above')

    data['intervals'] = apply_duration_criteria(data['intervals'],
                                                min_duration=min_duration,
                                                max_duration=max_duration)

    if with_down_intervals:
        data['down_intervals'] = get_thresholded_intervals(data['t_var'],
                                                           data[key+'_var_smoothed'],
                                                           data[key+'_var_threshold_low'],
                                                           where='below')
    
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
    dL = (pLFP.max()-pLFP.min())
    # discretization of Vm for histogram
    vbins = np.linspace(pLFP.min()-pLFP.std(), pLFP.max()+pLFP.std()/2., nbins) 
    hist, be = np.histogram(pLFP, vbins, normed=True) # normalized distribution
    vv = 0.5*(be[1:]+be[:-1]) # input vector is center of bin edges
    
    def to_minimize(args):
        w, m1, m2, s1, s2 = args
        double_gaussian = w*gaussian(vv, m1, s1)+(1.-w)*gaussian(vv, m2, s2)
        return np.power(hist-double_gaussian, 2).sum()

    # initial values
    mean0, std0 = pLFP.mean(), pLFP.std()
    w, m1, m2, s1, s2 = 0.5, mean0-std0, mean0+std0, std0/4., std0
    
    res = minimize(to_minimize, [w, m1, m2, s1, s2],
                   method='L-BFGS-B',
                   bounds = [(.05, .95),
                             (vv.min(), vv.max()), (vv.min(), vv.max()),
                             (pLFP.std()/100., 10*pLFP.std()), (pLFP.std()/100., 10*pLFP.std())],
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
    


