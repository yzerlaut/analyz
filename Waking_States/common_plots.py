import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.IO import rtxi # download data_analysis module at https://bitbucket.org/yzerlaut/data_analysis
from data_analysis.Waking_States import common_procedures as preprocessing # download data_analysis module at https://bitbucket.org/yzerlaut/data_analysis
from data_analysis.manipulation.files import * # download data_analysis module at https://bitbucket.org/yzerlaut/
from graphs.my_graph import *  # download graphs module at https://bitbucket.org/yzerlaut/graphs

def make_raw_data_figure(data,
                         # args,
                         figsize=(.8,.16),
                         keys = ['Isyn', 'Vm', 'LFP'],
                         colors = [Kaki, 'k', Grey],
                         extent_factors = [1, 3, 1],
                         tzoom = [0,10],
                         Vm_color=Blue,
                         Iinj_color=Orange,
                         nGe_color=Green,
                         LFP_color=Grey,
                         Vm_spike_highlight={'marker':'*', 'ms':3},
                         Vpeak = -10e-3,
                         lw = 1,
                         spike_ms=4.):
    """

    """

    fig, ax = figure(figsize=figsize, left=.1, bottom=.1)
    
    # time conditions
    cond = (data['t']>tzoom[0]) & (data['t']<tzoom[1])
    
    for k, key, color, extent_f in zip(range(len(colors)), keys, colors, extent_factors):

        trace = data[data[key]][cond]
        
        if k==0:
            Tmin0, Tmax0 = np.min(trace), np.max(trace)
            dT0 = Tmax0-Tmin0
            
        Tmin, Tmax = np.min(trace), np.max(trace)
        dT = Tmax-Tmin

        print(np.array(extent_factors)[:k].sum())
        ax.plot(data['t'][cond], trace*dT0/dT*extent_f+np.array(extent_factors)[:k].sum()*dT0+Tmin0,
                color=color,
                lw=lw)
        
    set_plot(ax, [], xlim=[data['t'][cond][0], data['t'][cond][-1]])
    
    # # from Iinj to Vm
    # ax.plot(data['t'][cond], data[data['Isyn']][cond], color=Iinj_color, lw=1)
    # Imin, Imax = np.min(data[data['Isyn']][cond]), np.max(data[data['Isyn']][cond])

    # # normalized conductance input color
    # nGemin, nGemax = np.min(data[data['nGe']][cond]), np.max(data[data['nGe']][cond])
    # ax.plot(data['t'][cond], (data[data['nGe']][cond]-nGemin)/(nGemax-nGemin)*(Imax-Imin)+(Imax-Imin)+Imin, color=nGe_color, lw=1)

    # # # Vm plot with spikes
    # Vmin, Vmax = np.min(data[data['Vm']][cond]), np.max(data[data['Vm']][cond])
    # ispikes = np.argwhere((data[data['Vm']][cond][1:]>Vpeak) & (data[data['Vm']][cond][:-1]<=Vpeak)).flatten()
    # for ii in ispikes:
    #     ax.plot([data['t'][cond][ii]], [(Vpeak-Vmin)*Vm_enhancement_factor/(Vmax-Vmin)*(Imax-Imin)+2*(Imax-Imin)+Imin], '*',  color=Vm_color, ms=spike_ms)
    # data[data['Vm']][data[data['Vm']]>Vpeak] = Vpeak
    # ax.plot(data['t'][cond], (data[data['Vm']][cond]-Vmin)*Vm_enhancement_factor/(Vmax-Vmin)*(Imax-Imin)+2*(Imax-Imin)+Imin, color=Vm_color, lw=1)

    # # # LFP plot
    # LFPmin, LFPmax = np.min(data[data['LFP']][cond]), np.max(data[data['LFP']][cond])
    # ax.plot(data['t'][cond], (data[data['LFP']][cond]-LFPmin)/(LFPmax-LFPmin)*(Imax-Imin)+(1.6+Vm_enhancement_factor)*(Imax-Imin)+Imin, color=LFP_color, lw=1)

    # condD = np.array(data['start_vector'])<args.tzoom[1]
    # for ts, te, ss in zip(data['start_vector'][condD], data['stop_vector'][condD], data['seed_vector'][condD]):
    #     ax.fill_between([ts, te], Imin*np.ones(2), np.ones(2)*2*(Imax-Imin)+Imin, color=Pink, alpha=0.5, lw=0)
    #     if args.debug:
    #         ax.annotate('Pattern '+str(int(ss+1)),  (te, Imax+Imin))
                
                
    # ax.annotate('$I_{inj}$', (args.tzoom[0], Imin), color=Iinj_color)
    # ax.annotate(r'$G_{e}$/$G_L$', (args.tzoom[0], Imax), color=nGe_color)
    # ax.annotate('$V_{m}$', (args.tzoom[0], 2*(Imax-Imin)+Imin), color=Vm_color)
    # ax.annotate('LFP', (args.tzoom[0], 5*(Imax-Imin)+Imin), color=LFP_color)
    # ax.plot([args.tzoom[0]+.1*np.diff(args.tzoom)[0],args.tzoom[0]+.1*np.diff(args.tzoom)[0]+args.Tbar], [Imin, Imin], 'k-', lw=2)
    # ax.plot([args.tzoom[0]+.1*np.diff(args.tzoom)[0],args.tzoom[0]+.1*np.diff(args.tzoom)[0]], [Imin, Imin+args.Ibar*1e-12], 'k-', lw=2)
    # if args.Tbar<1:
    #     ax.annotate(str(int(1e3*args.Tbar))+'ms', (args.tzoom[0]+.1*np.diff(args.tzoom)[0], Imin), color='k')
    # else:
    #     ax.annotate(str(np.round(args.Tbar,1))+'s', (args.tzoom[0]+.1*np.diff(args.tzoom)[0], Imin), color='k')
    # ax.annotate(str(int(args.Ibar))+'pA', (args.tzoom[0]+.1*np.diff(args.tzoom)[0], Imin+args.Ibar*1e-12), color=Iinj_color, rotation=90)
    # ax.annotate(str(np.round(1e3*args.Ibar*1e-12/Vm_enhancement_factor*(Vmax-Vmin)/(Imax-Imin),1))+'mV',\
    #             (args.tzoom[0], Imin+Imax+args.Ibar*1e-12), color=Vm_color, rotation=90)
    # ax.annotate(str(np.round(args.Ibar*1e-12*(nGemax-nGemin)/(Imax-Imin),1)),\
    #             (args.tzoom[0], Imin+2*Imax+args.Ibar*1e-12), color=nGe_color, rotation=90)
    # ax.annotate(str(np.round(1e6*args.Ibar*1e-12*(LFPmax-LFPmin)/(Imax-Imin),1))+'uV',\
    #             (args.tzoom[0], Imin+3*Imax+args.Ibar*1e-12), color=LFP_color, rotation=90)
    return fig, ax


def make_trial_average_figure(data, args):
    """
    """
    # run analysis
    compute_network_states_and_responses(data, args)

    data['seed_levels'] = np.unique(data['SEED_VECTOR'])
    
    fig, AX = figure(figsize=(.2*len(data['seed_levels']),.4),
                     axes=(2, len(data['seed_levels'])),
                     wspace=0.2,
                     left=0.25, top=.8)
    
    number_of_common_trials = 1000
    for a, f in enumerate(data['seed_levels']):
        # loop over frequency levels
        cond = (data['SEED_VECTOR']==f)
        for i in range(args.N_state_discretization):
            true_cond = data['cond_state_'+str(i+1)] & cond
            AX[1][a].plot(1e3*data['t_window'],
               1e3*data['Vm_Responses'][true_cond,:].mean(axis=0),
                          '-', color=COLORS[i], lw=2)
            AX[1][a].fill_between(1e3*data['t_window'],
                                  1e3*data['Vm_Responses'][true_cond,:].mean(axis=0)+1e3*data['Vm_Responses'][true_cond,:].std(axis=0),
                                  1e3*data['Vm_Responses'][true_cond,:].mean(axis=0)-1e3*data['Vm_Responses'][true_cond,:].std(axis=0),
                                  lw=0., color=COLORS[i], alpha=.3)
            # for the raster plot, we want a vcommon trial number
            number_of_common_trials = np.min([number_of_common_trials,\
                                              len(data['SEED_VECTOR'][true_cond])])
            print(len(data['SEED_VECTOR'][true_cond]))

    for a, f in enumerate(data['seed_levels']):
        # loop over seeduency levels
        cond = (data['SEED_VECTOR']==f)
        for i in range(args.N_state_discretization):
            true_cond = data['cond_state_'+str(i+1)] & cond
            for k, s in enumerate(np.arange(len(true_cond))[true_cond][:number_of_common_trials]):
                spk_train = data['Spike_Responses'][s]
                AX[0][a].plot(1e3*spk_train, 0*spk_train+k+i*(number_of_common_trials+2), 'o',
                              color=COLORS[i], ms=args.ms)

            AX[0][a].fill_between([1e3*data['t_window'][0],1e3*data['t_window'][-1]],
                                  i*(number_of_common_trials+2)*np.ones(2)-1, 
                                  i*(number_of_common_trials+2)*np.ones(2)+number_of_common_trials, 
                                  color=COLORS[i], alpha=.3, lw=0)
                
        AX[0][a].set_title('Pattern '+str(int(f+1)))
        AX[1][a].plot([0,0], args.Vm_lim, 'w.', ms=1e-8, alpha=0)
        if (a==0):
            AX[0][a].plot(1e3*data['t_window'][0]*np.ones(2),
                      args.N_state_discretization*(number_of_common_trials+2)-np.arange(2)*number_of_common_trials-2,
                      'k-', lw=1)
            AX[0][a].annotate(str(number_of_common_trials)+'trials', (1e3*data['t_window'][0],
                                                                      args.N_state_discretization*(number_of_common_trials+2)))
            set_plot(AX[1][a], xlabel='time from stim. (ms)', ylabel='Vm (mV)', ylim=args.Vm_lim)
            set_plot(AX[0][a], ['bottom'], ylabel='Spikes', ylim =[-3, AX[0][a].get_ylim()[1]+3])
        else:
            set_plot(AX[0][a], ['bottom'])
            set_plot(AX[1][a], xlabel='time from stim. (ms)', yticks_labels=[], ylim=args.Vm_lim)
            
    return fig, AX

if __name__ == '__main__':

    filename = sys.argv[-1]
    data = rtxi.load_continous_RTXI_recording(filename, with_metadata=True)
    preprocessing.find_keys(data, keys_to_find = ['Iout', 'Vm', 'LFP'])
    make_raw_data_figure(data,
                         keys = ['Iout', 'Vm', 'LFP'])
    show()
