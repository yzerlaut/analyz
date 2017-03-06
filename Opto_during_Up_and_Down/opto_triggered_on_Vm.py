import numpy as np
import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.IO.from_csv_to_cells import List_cells_and_choose_one
import data_analysis.IO.load_data as L
import data_analysis.Up_and_Down_charact.Gaussian_Mixture as UDGM
import data_analysis.Up_and_Down_charact.state_classification as state_classification
import matplotlib.pylab as plt

####################################################################################
########### Extract the data samples around the detected transitions
###################################################################################

def get_DU_transition_samples(data, up_transitions, window=[-100e-3, 800e-3], verbose=True):
    """
    """
    t_window = window[0]+np.arange(int((window[1]-window[0])/data['dt'])+1)*data['dt']

    if verbose:
        print('[...] Getting transition samples')
        
    TESTS = {'delay':[], 'amplitude':[], 'duration':[], 'times':[], 'traces':[]}
    CTRLS = {'times':[], 'traces':[]}

    
    for tup1, tup2 in zip(up_transitions[:-1], up_transitions[1:]):
        cond = (data['t']>=tup1+window[0]) & (data['t']<=tup1+window[1]) # conditions for analysis
        vv = data['Vm'][cond]
        cond = (data['t']>=tup1) & (data['t']<=tup1+window[1]) # insure that stim is after stim
        if len(vv)==len(t_window): # only if we are not in the borders to have same number of points
            # then we check whether there was a stimulation
            if np.diff(data['Laser'][cond]).max()>0.:
                # there was a laser input
                i0 = np.argmax(np.diff(data['Laser'][cond]))
                i1 = np.argmin(np.diff(data['Laser'][cond])) # no limit, in case switch again...
                TESTS['traces'].append(vv)
                TESTS['times'].append(tup1)
                TESTS['delay'].append(data['t'][cond][i0]-tup1)
                TESTS['duration'].append(data['t'][i1-i0])
                TESTS['amplitude'].append(np.diff(data['Laser'][cond]).max())
            else: # no laser -> blank
                CTRLS['times'].append(tup1)
                CTRLS['traces'].append(vv)
    
    return t_window, TESTS, CTRLS

def get_UD_transition_samples(data, down_transitions, window=[-100e-3, 800e-3], verbose=True):
    """
    """
    t_window = window[0]+np.arange(int((window[1]-window[0])/data['dt'])+1)*data['dt']

    if verbose:
        print('[...] Getting transition samples')
        
    TESTS = {'delay':[], 'amplitude':[], 'duration':[], 'times':[], 'traces':[]}
    CTRLS = {'times':[], 'traces':[]}

    for tdown1, tdown2 in zip(down_transitions[:-1], down_transitions[1:]):
        cond = (data['t']>=tdown1+window[0]) & (data['t']<=tdown1+window[1]) # conditions for analysis
        vv = data['Vm'][cond]
        cond = (data['t']>=tdown1) & (data['t']<=tdown1+window[1]) # insure that stim is after stim
        if len(vv)==len(t_window): # only if we are not in the borders to have same number of points
            # then we check whether there was a stimulation
            if np.diff(data['Laser'][cond]).max()>0.:
                # there was a laser input
                i0 = np.argmax(np.diff(data['Laser'][cond]))
                i1 = np.argmin(np.diff(data['Laser'][cond])) # no limit, in case switch again...
                TESTS['traces'].append(vv)
                TESTS['times'].append(tdown1)
                TESTS['delay'].append(data['t'][cond][i0]-tdown1)
                TESTS['duration'].append(data['t'][i1-i0])
                TESTS['amplitude'].append(np.diff(data['Laser'][cond]).max())
            else: # no laser -> blank
                CTRLS['times'].append(tdown1)
                CTRLS['traces'].append(vv)
    
    return t_window, TESTS, CTRLS

####################################################################################
########### Cross-validate the transitions
###################################################################################

def cross_validation_of_DU_transitions(data, TESTS, CTRLS,
                                       window=7, criteria=100e-3,
                                       verbose=True, hp=False, debug=False):

    TESTS['cross-validated'], TESTS['next_up'], TESTS['next_down'], TESTS['t1'], TESTS['t2'] = [], [], [], [], []
    CTRLS['cross-validated'], CTRLS['next_up'], CTRLS['next_down'], CTRLS['t1'], CTRLS['t2'] = [], [], [], [], []
    
    if verbose:
        print('[...] Cross-validating transitions')
    if hp:
        n, ninit = 500, 10
    else:
        n, ninit = 4, 2
        
    for COND, s in zip([TESTS, CTRLS], ['test conditions', 'control conditions']):
        for tt in COND['times']:
            cond = (data['t']>=tt-window/2) & (data['t']<=tt+window/2.)
            try:
                t1, t2 = UDGM.determine_thresholds(*UDGM.fit_3gaussians(data['Vm'][cond], n=n, ninit=ninit))
            except (ValueError, IndexError): # means overfitting
                t1, t2 = UDGM.determine_thresholds(*UDGM.fit_2gaussians(data['Vm'][cond], n=n, ninit=ninit))
            UDtr, DUtr = state_classification.get_transition_times(data['t'][cond], data['Vm'][cond],\
                                                                   t1+0.*data['t'][cond], t2+0.*data['t'][cond])
            i0 = np.argmin(np.abs(tt-UDtr))
            COND['cross-validated'].append((np.abs(UDtr[i0]-tt)<=criteria)) # cross-validated ? True or False
            COND['t1'].append(t1)
            COND['t2'].append(t2)
            try:
                COND['next_up'].append(DUtr[i0+1])
            except IndexError:
                COND['next_up'].append(np.inf)
            try:
                COND['next_down'].append(UDtr[UDtr>DUtr[i0]][0])
            except IndexError:
                COND['next_down'].append(np.inf)
            if debug:
                fig, ax = plt.subplots()
                ax.plot(data['t'][cond], data['Vm'][cond], 'k-')
                ax.plot(data['t'][cond], t1+0.*data['t'][cond], 'r-')
                ax.plot(data['t'][cond], t2+0.*data['t'][cond], 'b-')
                plt.show()
                

        COND['cross-validated'] = np.array(COND['cross-validated'], dtype=bool)
        if verbose:
            print('percentage of cross-validated transitions: ',\
                  round(100.*len(np.array(COND['times'])[COND['cross-validated']])/len(COND['times'])), '% in', s)


def cross_validation_of_UD_transitions(data, TESTS, CTRLS,
                                       window=7, criteria=100e-3,
                                       verbose=True, hp=False, debug=False):

    TESTS['cross-validated'], TESTS['next_up'], TESTS['next_down'], TESTS['t1'], TESTS['t2'] = [], [], [], [], []
    CTRLS['cross-validated'], CTRLS['next_up'], CTRLS['next_down'], CTRLS['t1'], CTRLS['t2'] = [], [], [], [], []
    
    if verbose:
        print('[...] Cross-validating transitions')
    if hp:
        n, ninit = 500, 10
    else:
        n, ninit = 4, 2
        
        
    for COND, s in zip([TESTS, CTRLS], ['test conditions', 'control conditions']):
        for tt in COND['times']:
            cond = (data['t']>=tt-window/2) & (data['t']<=tt+window/2.)
            try:
                t1, t2 = UDGM.determine_thresholds(*UDGM.fit_3gaussians(data['Vm'][cond], n=n, ninit=ninit))
            except ValueError: # means overfitting
                t1, t2 = UDGM.determine_thresholds(*UDGM.fit_2gaussians(data['Vm'][cond], n=n, ninit=ninit))
            UDtr, DUtr = state_classification.get_transition_times(data['t'][cond], data['Vm'][cond],\
                                                                   t1+0.*data['t'][cond], t2+0.*data['t'][cond])
            i0 = np.argmin(np.abs(tt-UDtr))
            COND['cross-validated'].append((np.abs(UDtr[i0]-tt)<=criteria)) # cross-validated ? True or False
            COND['t1'].append(t1)
            COND['t2'].append(t2)
            try:
                COND['next_up'].append(DUtr[DUtr>UDtr[i0]][0])
            except IndexError:
                COND['next_up'].append(np.inf)
            try:
                COND['next_down'].append(UDtr[i0+1])
            except IndexError:
                COND['next_down'].append(np.inf)
            if debug:
                fig, ax = plt.subplots()
                ax.plot(data['t'][cond], data['Vm'][cond], 'k-')
                ax.plot(data['t'][cond], t1+0.*data['t'][cond], 'r-')
                ax.plot(data['t'][cond], t2+0.*data['t'][cond], 'b-')
                plt.show()

        COND['cross-validated'] = np.array(COND['cross-validated'], dtype=bool)
        if verbose:
            print('percentage of cross-validated transitions: ',\
                  round(100.*len(np.array(COND['times'])[COND['cross-validated']])/len(COND['times'])), '% in', s)
        
            
####################################################################################
########### Insure that remains in a given state
###################################################################################

def insure_that_remains_in_up_state(t_window, TESTS, CTRLS, DELAY, std_factor=2.):
    """ """
    
    cond = (t_window>0) & (t_window<DELAY) # pre stimulus
    
    for COND, s in zip([TESTS, CTRLS], ['test conditions', 'control conditions']):
        COND['remains_in_up_state'] = [False for i in range(len(COND['traces']))]
        for i in range(len(COND['times'])):
            if (COND['traces'][i][cond].min()>COND['t1'][i]):
                COND['remains_in_up_state'][i] = True
            

def insure_that_remains_in_down_state(t_window, TESTS, CTRLS, DELAY, std_factor=2.):
    """ """
    
    cond = (t_window>0) & (t_window<DELAY) # pre stimulus
    TESTS['remains_in_down_state'] = [False for i in range(len(TESTS['traces']))]
    CTRLS['remains_in_down_state'] = [False for i in range(len(CTRLS['traces']))]
    
    mean = .5*(np.mean(np.array([v[cond].mean() for v in TESTS['traces']]))+\
               np.mean(np.array([v[cond].mean() for v in CTRLS['traces']])))
    std = .5*(np.std(np.array([v[cond].mean() for v in TESTS['traces']]))+\
              np.std(np.array([v[cond].mean() for v in CTRLS['traces']])))

    for i in range(len(TESTS['times'])):
        if (TESTS['traces'][i][cond].max()<mean+std_factor*std) and (TESTS['traces'][i][cond].min()>mean-std_factor*std):
            TESTS['remains_in_down_state'][i] = True
    for i in range(len(CTRLS['times'])):
        if (CTRLS['traces'][i][cond].max()<mean+std_factor*std) and (CTRLS['traces'][i][cond].min()>mean-std_factor*std):
            CTRLS['remains_in_down_state'][i] = True

            
####################################################################################
########### Get responses
###################################################################################

def get_stimulus_responses(filename, window=[-100e-3, 800e-3], criteria=200e-3,
                           with_crossvalidation=True, crossvalidation_window=10,
                           debug=False, hp=False):
    """
    get stimulus response (triggered on Real-Time analysis) with or without the cross-validation
    """

    # load raw data
    data, params = L.get_formated_data(filename, with_params_only=True)
    
    if data['flag_for_state_stimulation']=='1':
        # if stimulation in Up state !
        up_transitions = data['t'][np.argwhere(np.diff(data['UP_FLAG'])==1).flatten()]
        t_window, TESTS, CTRLS = get_DU_transition_samples(data, up_transitions, window=window)
        if with_crossvalidation:
            cross_validation_of_DU_transitions(data, TESTS, CTRLS,\
                                               window=crossvalidation_window, criteria=criteria,
                                               debug=debug, hp=hp)
    elif data['flag_for_state_stimulation']=='2':
        # if stimulation in Down state !
        down_transitions = data['t'][np.argwhere(np.diff(data['UP_FLAG'])==-1).flatten()]
        t_window, TESTS, CTRLS = get_UD_transition_samples(data, down_transitions, window=window)
        if with_crossvalidation:
            cross_validation_of_UD_transitions(data, TESTS, CTRLS,\
                                               window=crossvalidation_window, criteria=criteria,
                                               debug=debug, hp=hp)
        
    print('Done !')

    TESTS['delay'] = np.round(np.array(TESTS['delay'])*1e2,0)*1e1 # rounding and transformation to ms
    TESTS['duration'] = np.round(np.array(TESTS['duration'])*1e2,0)*1e1 # rounding and transformation to ms

    return 1e3*t_window, TESTS, CTRLS


####################################################################################
########### Merge data of two files
###################################################################################

def merge_two_cells(t_window, TESTS1, CTRLS1, TESTS2, CTRLS2):
    for COND1, COND2 in zip([TESTS1, CTRLS1], [TESTS2, CTRLS2]):
        for key in COND1.keys():
            try:
                COND1[key] = np.concatenate([COND1[key], COND2[key]])
            except ValueError: # 0-size array
                pass

