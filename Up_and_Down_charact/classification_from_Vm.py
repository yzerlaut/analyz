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

    return means[i1]-stds[i1], means[i1]-stds[i1]


def get_state_intervals(Vm, threshold_up, threshold_down, dt,
                        fluct_duration_criteria=30e-3,
                        min_duration_criteria=100e-3):
    """
    single threshold strategy for state transition characterization
    with a duration criteria

    """

    Up_intervals, Down_intervals = [], []

    #### =========== #####
    ###   Up states   ###
    #### =========== #####

    iup = np.argwhere(Vm>threshold_up) # basic characterization: being above threshold
    i=0
    while i<len(iup):
        i0 = i
        while (iup[i+1]-iup[i])<int(fluct_duration_criteria/dt):
            # meaning we're in the same up state
            i+=1
        # then this interval is finished
        print(i)
        Up_intervals.append([iup[i0], iup[i]])

    # then removing too short intervals
    i = 0
    while i<len(Up_intervals):
        if (Up_intervals[i][1]-Up_intervals[i][1])<min_duration_criteria:
            Up_intervals.remove(Up_intervals[i])

    #### =========== #####
    ###   Down states   ###
    #### =========== #####

    idown = np.argwhere(Vm<threshold_down) # basic characterization: being below threshold
    i=0
    while i<len(idown):
        i0 = i
        while (idown[i+1]-idown[i])<int(fluct_duration_criteria/dt):
            # meaning we're in the same down state
            i+=1
        # then this interval is finished
        print(i)
        Down_intervals.append([idown[i0], idown[i]])

    # then removing too short intervals
    i = 0
    while i<len(Down_intervals):
        if (Down_intervals[i][1]-Down_intervals[i][1])<min_duration_criteria:
            Down_intervals.remove(Down_intervals[i])

    return Up_intervals, Down_intervals

def get_state_intervals2(t, Vm, threshold,
                        duration_criteria=100e-3):
    """
    single threshold strategy for state transition characterization
    with a duration criteria

    """

    Up_intervals, Down_intervals = [], []

    iupward = np.argwhere( (Vm[:-1]<threshold) & (Vm[1:]>=threshold)).flatten()
    idownward = np.argwhere( (Vm[:-1]>threshold) & (Vm[1:]<=threshold)).flatten()

    if iupward[0]<idownward[0]:
        # we start by down state
        start_with_down = True
        Down_intervals.append([t[0]-duration_criteria, t[iupward[0]]])
        # then we fill in order
        for i in range(min([len(idownward), len(iupward)-1])):
            Down_intervals.append([t[idownward[i]], t[iupward[i+1]]])
        for i in range(min([len(idownward), len(iupward)])):
            Up_intervals.append([t[iupward[i]], t[idownward[i]]])
    else:
        start_with_down = False
        # we start by up state
        Up_intervals.append([t[0]-duration_criteria, t[idownward[0]]])
        # then we fill in order
        for i in range(min([len(iupward), len(idownward)-1])):
            Up_intervals.append([t[iupward[i]], t[idownward[i+1]]])
        for i in range(min([len(idownward), len(iupward)])):
            Down_intervals.append([t[idownward[i]], t[iupward[i]]])

    if iupward[-1]>idownward[-1]:
        # we finish by Up state
        Up_intervals.append([t[iupward[-1]], t[-1]])
    else:
        Down_intervals.append([t[idownward[-1]], t[-1]])

    # now we check whether the duration criteria is matched
    # and we correct

    iup, idown = 0, 0
        
    while (iup<len(Up_intervals)) and (idown<len(Down_intervals)):
        if Down_intervals[idown][0]<Up_intervals[iup][0]:
            # need to look at down state duration
            if (Down_intervals[idown][1]-Down_intervals[idown][0])>duration_criteria:
                # we pass the criteria
                idown += 1
            else:
                # we don't pass
                # we remove this down state interval
                Down_intervals.remove(Down_intervals[idown])
                # and the next up state as it is prolongs with the previous one !
                Up_intervals[iup-1][1] = Up_intervals[iup][1]
                Up_intervals.remove(Up_intervals[iup])
        else:
            # need to look at up state duration
            if (Up_intervals[iup][1]-Up_intervals[iup][0])>duration_criteria:
                # we pass the criteria
                iup += 1
            else:
                # we don't pass
                # we remove this up state interval
                Up_intervals.remove(Up_intervals[iup])
                # and the next down state as it is merged with the previous one !
                Down_intervals[idown-1][1] = Down_intervals[idown][1]
                Down_intervals.remove(Down_intervals[idown])
                
    return Up_intervals, Down_intervals
                
if __name__=='__main__':
    
    import sys
    sys.path.append('../..')
    from data_analysis.IO.load_data import load_file, get_formated_data
    t, [Vm, _, _, _, _, LFP] = load_file('/Users/yzerlaut/DATA/Exps_Ste_and_Yann/2016_12_6/16_48_19_VM-FEEDBACK--OSTIM-AT-VARIOUS-DELAYS.bin', zoom=[0, 20])
    weights, means, stds = fit_2gaussians(Vm)
    threshold_up, threshold_down = determine_thresholds(weights, means, stds)    
    Up_intervals, Down_intervals = get_state_intervals(Vm,
                                                       threshold_up, threshold_down,
                                                       t[-1]-t[0],
                                                       min_duration_criteria=100e-3)
    print(Up_intervals, Down_intervals)
