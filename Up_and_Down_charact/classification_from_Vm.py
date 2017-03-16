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

def passing_the_up_and_down_states_criteria(weigths, means, stds,
                                            std_ratio=1./3., std_shift=3.):
    
    i0, i1 = np.argmin(means), np.argmax(means) # find the upper and lower distrib
    
    if (stds[i0]/stds[i1]<std_ratio) and (means[i1]>means[i0]+std_shift*stds[i0]):
        return True
    else:
        return False
    
def determine_threshold(weigths, means, stds, with_amp=False):
    """ Gives the single threshold given the Gaussian Mixture"""
    
    i0, i1 = np.argmin(means), np.argmax(means) # find the upper and lower distrib

    vv = np.linspace(means[i0], means[i1], 1e2) # the point is in between the two means
    gaussian1 = weigths[i0]*gaussian(vv, means[i0], stds[i0])
    gaussian2 = weigths[i1]*gaussian(vv, means[i1], stds[i1])
    ii = np.argmin(np.power(gaussian1-gaussian2, 2))
    threshold, amp = vv[ii], gaussian1[ii]
        
    if with_amp:
        return threshold, amp
    else:
        return threshold

def get_thresholded_intervals(t, Vm, threshold, where='above-and-below',\
                              adding_the_boundary_intervals=False):
    """
    returns the interval below/above/below-and-above a threshold
    """

    Up_intervals, Down_intervals = [], []

    iupward = np.argwhere( (Vm[:-1]<threshold) & (Vm[1:]>=threshold)).flatten()
    idownward = np.argwhere( (Vm[:-1]>threshold) & (Vm[1:]<=threshold)).flatten()

    if iupward[0]<idownward[0]:
        # we start by down state
        start_with_down = True
        if adding_the_boundary_intervals:
            Down_intervals.append([t[0], t[iupward[0]]])
        # then we fill in order
        for i in range(min([len(idownward), len(iupward)-1])):
            Down_intervals.append([t[idownward[i]], t[iupward[i+1]]])
        for i in range(min([len(idownward), len(iupward)])):
            Up_intervals.append([t[iupward[i]], t[idownward[i]]])
    else:
        start_with_down = False
        # we start by up state
        if adding_the_boundary_intervals:
            Up_intervals.append([t[0], t[idownward[0]]])
        # then we fill in order
        for i in range(min([len(iupward), len(idownward)-1])):
            Up_intervals.append([t[iupward[i]], t[idownward[i+1]]])
        for i in range(min([len(idownward), len(iupward)])):
            Down_intervals.append([t[idownward[i]], t[iupward[i]]])

    if adding_the_boundary_intervals:
        if iupward[-1]>idownward[-1]:
            # we finish by Up state
            Up_intervals.append([t[iupward[-1]], t[-1]])
        else:
            Down_intervals.append([t[idownward[-1]], t[-1]])

    if where=='above-and-below':
        return Up_intervals, Down_intervals
    elif where=='above':
        return Up_intervals
    elif where=='below':
        return Down_intervals
    else:
        print('need to specify a valid setting for the interval you want !')

def apply_duration_criteria(state_interval,
                            min_duration=150e-3, max_duration=1500e-3):
    i=0
    while i<len(state_interval):
        length = state_interval[i][1]-state_interval[i][0]
        if (length<min_duration) or (length>max_duration):
            state_interval.remove(state_interval[i])
            i-=1
        i+=1
    return state_interval


def matching_cond_for_duplicates(ref_interval, test_interval):
    tr1, tr2 = ref_interval
    tt1, tt2 = test_interval
    if (((tt1>=tr1) and (tt1<=tr2)) or \
        ((tt2>=tr1) and (tt2<=tr2)) or \
        ((tr1>=tt1) and (tr1<=tt2)) or \
        ((tr2>=tt1) and (tr2<=tt2))):
        # return tr1, tr2
        return [min([tr1, tt1]), max([tr2, tt2])] # merging intervals
    else:
        return None

def remove_duplicates(state_interval):
    i=0
    new_intervals = []
    while len(state_interval)>0:
        # the strategy is to put the intervals into the "duplicates"
        # and remove them from the main array
        to_remove = [0]
        for i in range(1, len(state_interval)):
            new = matching_cond_for_duplicates(state_interval[0], state_interval[i])
            if new is not None:
                to_remove.append(i) # we store the one to remove
                state_interval[0] = new # we replace the standard interval with updated values
        new_intervals.append(state_interval[0])
        for i in to_remove[::-1]: # need to start from the end
            state_interval.pop(i) # we remove
    return new_intervals

def procedure_over_sliding_window(t, Vm,
                                  sliding_window=5, sliding_shift=.5):
    """
    Goes over a sliding window, fits two gaussians, decide whether 
    the distrib is bimodal enough to be likely to have up and down
    states, if yes, classifiesthe state
    """
    dt = t[1]-t[0]
    isl, iss = int(sliding_window/dt), int(sliding_shift/dt)

    Up_intervals, Down_intervals = [], []
    for i in range(int(len(t)/iss)):
        i0, i1 = max([0, int(i*iss-isl/2)]), min([len(t), int(i*iss+isl/2)])
        weights, means, stds = fit_2gaussians(Vm[i0:i1])
        if passing_the_up_and_down_states_criteria(weights, means, stds):
            threshold = determine_threshold(weights, means, stds)    
            upI, downI = get_thresholded_intervals(t[i0:i1], Vm[i0:i1],
                                                   threshold)
            Up_intervals = Up_intervals + upI
            Down_intervals = Down_intervals + downI

    # check duration criteria
    Up_intervals = apply_duration_criteria(Up_intervals)
    Down_intervals = apply_duration_criteria(Down_intervals)
    # remove duplicates
    Up_intervals = remove_duplicates(Up_intervals)
    Down_intervals = remove_duplicates(Down_intervals)
    Up_intervals = remove_duplicates(Up_intervals)
    Down_intervals = remove_duplicates(Down_intervals)
    
    return Up_intervals, Down_intervals
    
    
if __name__=='__main__':
    
    import sys
    sys.path.append('../..')
    from data_analysis.IO.load_data import load_file, get_formated_data
    from graphs.my_graph import show
    t1, t2 = 17, 26
    t, DATA = load_file(\
          # '/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2016_12_6/16_48_19_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin',
          '/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2017_01_17/16_17_12_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin',
                                         zoom=[t1, t2])
    Vm = DATA[0]
    vbins = np.linspace(Vm.min(), -30)
    
    Up_intervals, Down_intervals = procedure_over_sliding_window(t, Vm)
    
    import matplotlib.pylab as plt
    plt.figure(figsize=(8,3))
    plt.style.use('ggplot')
    plt.subplot2grid((1,6), (0,0), colspan=5)
    plt.plot(t, Vm, lw=1)
    plt.ylim([vbins[0],vbins[-1]])
    plt.ylabel('$V_m$ (mV)')
    for (t1, t2) in Up_intervals:
        plt.fill_between([t1,t2], plt.gca().get_ylim()[0]*np.ones(2),
                        plt.gca().get_ylim()[1]*np.ones(2), color='r', alpha=.1, lw=0)
    for (t1, t2) in Down_intervals:
        plt.fill_between([t1,t2], plt.gca().get_ylim()[0]*np.ones(2),
                        plt.gca().get_ylim()[1]*np.ones(2), color='b', alpha=.1, lw=0)
    # compute histogram and gaussian mixture (histogram over full data !)
    Vmhist, be = np.histogram(Vm, bins=vbins, normed=True)
    plt.subplot2grid((1,6), (0,5))
    plt.barh(.5*(be[1:]+be[:-1]), Vmhist, height=be[1]-be[0])
    plt.ylim([vbins[0],vbins[-1]])
    plt.gca().set_yticklabels([])
    plt.xticks([])

    plt.show()
    # show(plt)
