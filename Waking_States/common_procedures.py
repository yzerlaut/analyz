import sys, os
sys.path.append('../../')
from data_analysis.IO import rtxi # download data_analysis module at https://bitbucket.org/yzerlaut/data_analysis
import numpy as np
from graphs.my_graph import *  # download graphs module at https://bitbucket.org/yzerlaut/graphs
from data_analysis.manipulation.files import * # download data_analysis module at https://bitbucket.org/yzerlaut/
from data_analysis.processing.signanalysis import gaussian_smoothing

def find_keys(data,
              keys_to_find=['Vm', 'LFP', 'Isyn', 'nGe', 'Iout']):
    
    keys = list(data.keys())
    for key in keys:
        for key_to_find in keys_to_find:
            if len(key.split(key_to_find))>1:
                data[key_to_find] = key

    data['t'] = np.arange(len(data[data['Vm']]))*data['dt']

            
def compute_network_states_and_responses(data, args,
                                         keys=['seed_vector']):

    for key in keys:
        data[key.upper()] = []

    LFP_levels, Vm_Responses, Spike_Responses = [], [], []
    Firing_levels_pre, Firing_levels_post = [], []
    Depol_levels_pre, Depol_levels_post = [], []

    t_window = np.arange(int((data['stim_duration']+2e-3*args.pre_window)/data['dt']))*data['dt']-1e-3*args.pre_window
    
    i=0
    while (data['stop_vector'][i]<data['t'][-1]):

        for key in keys:
            data[key.upper()].append(data[key][i])

        cond = (data['t']>(data['start_vector'][i]-1e-3*args.pre_window)) &\
               (data['t']<=(data['stop_vector'][i]+1e-3*args.pre_window))
        # extracting Vm level
        vec = 0*t_window+data[data['Vm']][cond][-1]
        vec[:np.min([len(vec), len(data[data['Vm']][cond])])] = data[data['Vm']][cond][:np.min([len(vec), len(data[data['Vm']][cond])])]
        # exctracting spikes
        ispikes = np.argwhere((vec[1:]>1e-3*args.Vspike) & (vec[:-1]<=1e-3*args.Vspike)).flatten()
        Spike_Responses.append(t_window[ispikes])
        vec[vec>1e-3*args.Vspike] = 1e-3*args.Vspike
        Vm_Responses.append(vec)
        
        # extracting LFP level
        LFP_levels.append(np.mean(data[data['LFP']][cond]))
        
        # extracting spike rate and depol levels before and after
        # -- pre
        pre_cond = (data['t']>(data['start_vector'][i]-data['stim_duration'])) &\
                   (data['t']<=data['start_vector'][i])
        ispikes = np.argwhere((data[data['Vm']][pre_cond][1:]>1e-3*args.Vspike) & (data[data['Vm']][pre_cond][:-1]<=1e-3*args.Vspike)).flatten()
        Firing_levels_pre.append(len(ispikes)/data['stim_duration'])
        Depol_levels_pre.append(np.mean(data[data['Vm']][pre_cond]))
        # -- post
        post_cond = (data['t']>data['start_vector'][i]) &\
                    (data['t']<=(data['stop_vector'][i]+data['stim_duration']))
        ispikes = np.argwhere((data[data['Vm']][post_cond][1:]>1e-3*args.Vspike) & (data[data['Vm']][post_cond][:-1]<=1e-3*args.Vspike)).flatten()
        Firing_levels_post.append(len(ispikes)/data['stim_duration'])
        Depol_levels_post.append(np.mean(data[data['Vm']][post_cond]))
        i+=1

    data['t_window'] = t_window
    for key in keys:
        data[key.upper()] = np.array(data[key.upper()])

    data['Vm_Responses'] = np.array(Vm_Responses)
    data['Spike_Responses'] = Spike_Responses
    data['LFP_levels'] = np.array(LFP_levels)
    data['Firing_levels_pre'] = np.array(Firing_levels_pre)
    data['Firing_levels_post'] = np.array(Firing_levels_post)
    data['Depol_levels_pre'] = np.array(Depol_levels_pre)
    data['Depol_levels_post'] = np.array(Depol_levels_post)

    for i in range(args.N_state_discretization):
        lower = np.percentile(data['LFP_levels'], i*100./args.N_state_discretization)
        higher = np.percentile(data['LFP_levels'], (i+1)*100./args.N_state_discretization)
        data['cond_state_'+str(i+1)] =  (data['LFP_levels']>=lower) & (data['LFP_levels']<=higher)
  
