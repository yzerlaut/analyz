import numpy as np
from scipy.optimize import minimize

def gaussian(x, mean, std):
    output = np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std
    return output

def fit_2gaussians(Vm, n=1000, upper_bound=-35, nbins=100):
    """
    take the histogram of the Vm values
    and fits two gaussians using the least square algorithm

    'upper_bound' cuts spikes !

    returns the weights, means and standard deviation of the
    gaussian functions
    """
    vbins = np.linspace(Vm.min(), upper_bound, nbins) # discretization of Vm for histogram
    hist, be = np.histogram(Vm, vbins, normed=True) # normalized distribution
    vv = 0.5*(be[1:]+be[:-1]) # input vector is center of bin edges
    
    def to_minimize(args):
        w, m1, m2, s1, s2 = args
        double_gaussian = w*gaussian(vv, m1, s1)+(1.-w)*gaussian(vv, m2, s2)
        return np.power(hist-double_gaussian, 2).sum()

    # initial values
    mean0, std0 = Vm.mean(), Vm.std()
    w, m1, m2, s1, s2 = 0.5, mean0-std0, mean0+std0, std0/2., std0/2.
    
    res = minimize(to_minimize, [w, m1, m2, s1, s2],
                   method='L-BFGS-B',
                   bounds = [(.05, .95),
                             (vv.min(), vv.max()), (vv.min(), vv.max()),
                             (1e-2, vv.max()-vv.min()), (1e-2, vv.max()-vv.min())],
                   options={'maxiter':n})

    w, m1, m2, s1, s2 = res.x
    
    return (w, 1-w), (m1, m2), (s1, s2)


def determine_threshold(weigths, means, stds, with_amp=False, distance_to_min_criteria=5, Vm_min=-80):
    """ Gives the thresholds given the Gaussian Mixture"""
    
    i0, i1 = np.argmin(means), np.argmax(means) # find the upper and lower distrib

    # if np.abs(means[i0]-Vm_min)<distance_to_min_criteria:
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

def determine_thresholds(weigths, means, stds):
    """ Gives the thresholds given the Gaussian Mixture"""
    
    i0, i1 = np.argmin(means), np.argmax(means) # find the upper and lower distrib

    # find the intersection between the two distrib
    vv = np.linspace(means[i0], means[i1], 1e2) # the point is in between the two means
    gaussian1 = weigths[i0]*gaussian(vv, means[i0], stds[i0])
    gaussian2 = weigths[i1]*gaussian(vv, means[i1], stds[i1])
    ii = np.argmin(np.power(gaussian1-gaussian2, 2))
    
    if stds[i0]/stds[i1]<.5:
        # low variance of down state, i.e. very visible down state
        threshold_down, amp = vv[ii], gaussian1[ii]
    else:
        threshold_down, amp = means[i0], weigths[i0]*gaussian(0,0,stds[i0])
    threshold_up = max([threshold_down, means[i1]-stds[i1]])
    return threshold_up, threshold_down


def get_state_intervals(t, Vm, threshold_up, threshold_down,
                        fluct_duration_criteria=30e-3,
                        min_duration_criteria=100e-3):
    """
    single threshold strategy for state transition characterization
    with a duration criteria

    """

    Up_intervals, Down_intervals = [], []
    dt = t[1]-t[0]

    for i in range(len(t)):
        

        
    iup = np.argwhere(Vm>threshold_up).flatten() # basic characterization: being above threshold
    idown = np.argwhere(Vm<threshold_down).flatten() # basic characterization: being below threshold

    INTERVALS = []
    for istate in [iup, idown]:
        
        statetransitions = t[np.argwhere(np.diff(istate)>1).flatten()]
        print(statetransitions)
        if istate[0]==0:
            # we start with the state state
            statetransitions = np.concatenate([[0], statetransitions])
        print(statetransitions)
        if len(statetransitions)%2>0:
            statetransitions = np.concatenate([statetransitions, [t[-1]]])
        INTERVALS.append(statetransitions.reshape((int(len(statetransitions)/2),2)))
    Up_intervals, Down_intervals = INTERVALS[0], INTERVALS[1]
    
    print(Up_intervals)
    print(Down_intervals)
        
    # for istate, intervals in zip([iup, idown],
    #                              [Up_intervals, Down_intervals]):
    #     print('state change')
    #     # same strategy for up and down states
    #     i0, i= 0, 0
    #     while i<len(istate)-1:
    #         # while ((istate[i]-istate[i-1])<2) & (i<len(istate)-1):
    #         while ((t[istate[i]]-t[istate[i-1]])<fluct_duration_criteria) & (i<len(istate)-1):
    #             # meaning we're in the same up state
    #             i+=1
    #             # print(((t[istate[i]]-t[istate[i-1]])<fluct_duration_criteria), (i<len(istate)-1))
    #         # print(t[istate[i0]], t[istate[i]], t[istate[i+1]]-t[istate[i]], fluct_duration_criteria)
    #         # then this interval is finished
    #         intervals.append([t[istate[i0]], t[istate[i]]])
    #         i+=1
    #         i0 = i
    #     # print(intervals)
    #     # then removing too short intervals
    #     # i = 0
    #     # while i<len(intervals):
    #     #     if (intervals[i][1]-intervals[i][1])<min_duration_criteria:
    #     #         intervals.remove(intervals[i])
    #     #     i+=1

    return Up_intervals, Down_intervals

                
if __name__=='__main__':
    
    import sys
    sys.path.append('../..')
    from data_analysis.IO.load_data import load_file, get_formated_data
    from graphs.my_graph import show
    tstop = 2
    t, [Vm, _, _, _, _, LFP] = load_file(\
            '/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2016_12_6/16_48_19_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin', zoom=[0, tstop])
    vbins = np.linspace(Vm.min(), -30)
    weights, means, stds = fit_2gaussians(Vm)
    threshold_up, threshold_down = determine_thresholds(weights, means, stds)    
    Up_intervals, Down_intervals = get_state_intervals(t, Vm,
                                                       threshold_up, threshold_down)
    i=0
    import matplotlib.pylab as plt
    plt.subplot2grid((2, 6), (i,0), colspan=5)
    plt.plot(t[t<tstop], Vm[t<tstop], lw=1)
    plt.ylim([vbins[0],vbins[-1]])
    plt.gca().set_xticklabels([])
    plt.ylabel('$V_m$ (mV)')
    plt.annotate('('+'i'*(i+1)+')', (-.2, 1), xycoords='axes fraction')    
    ax = plt.gca()
    # compute histogram and gaussian mixture (histogram over full data !)
    Vmhist, be = np.histogram(Vm, bins=vbins, normed=True)
    plt.subplot2grid((1+2, 6), (i,5))
    plt.barh(.5*(be[1:]+be[:-1]), Vmhist, height=be[1]-be[0])
    # establish gaussian mixture
    for w, m, s in zip(weights, means, stds):
        plt.plot(w*gaussian(vbins, m, s), vbins, ':', lw=1, color='k')
    # find threshold as the interesction of the two Gaussians
    ax.plot(t[t<tstop], threshold_up+0.*t[t<tstop], '--', lw=1.5, label='Up threshold')
    ax.plot(t[t<tstop], threshold_down+0.*t[t<tstop], '--', lw=1.5, label='Down threshold')
    # threshold, amp = determine_threshold(weights, means, stds,
    #                                      Vm_min=Vm.min(), with_amp=True)
    # plt.plot([amp], [threshold], 'ko')
    # ax.plot(t[t<tstop], threshold+0.*t[t<tstop], '--', lw=1.5, label='threshold')
    # determine up and down states
    # Up_intervals, Down_intervals = get_state_intervals(t[t<tstop], Vm[t<tstop], threshold,
    #                                                    duration_criteria=100e-3)
    for (t1, t2) in Up_intervals:
        ax.fill_between([t1,t2], ax.get_ylim()[0]*np.ones(2),
                        ax.get_ylim()[1]*np.ones(2), color='r', alpha=.1, lw=0)
    # for (t1, t2) in Down_intervals:
    #     ax.fill_between([t1,t2], ax.get_ylim()[0]*np.ones(2),
    #                     ax.get_ylim()[1]*np.ones(2), color='b', alpha=.1, lw=0)
    plt.ylim([vbins[0],vbins[-1]])
    plt.gca().set_yticklabels([])
    plt.xticks([])


    show(plt)
