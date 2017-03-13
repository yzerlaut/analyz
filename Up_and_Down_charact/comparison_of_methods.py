import numpy as np
import itertools

def making_grid_of_linear_combination(freqs, level_number=2):

    SETS_OF_FREQ_LEVELS = [] # we will store all combinations of levels for summation
    possible_levels = np.linspace(0, 1, level_number, endpoint=True)
    
    for freq_levels in itertools.product(*[possible_levels for f in freqs]):
        #
        if np.sum(freq_levels)!=0:
            SETS_OF_FREQ_LEVELS.append(np.array(freq_levels)/np.sum(freq_levels))

    return SETS_OF_FREQ_LEVELS


def CoincidenceIndex(Up_intervals1, Down_intervals1, Up_intervals2, Down_intervals2,
                     dt=5e-5, tstop=10):
    """
    Implementing the Coincidence index of Mukovski et al. Cerebral Cortex 2007
    see their Figure 5, for a visual representation !
    """
    t = np.arange(int(tstop/dt))*dt
    
    # calculating the means from the intervals
    mean_up = .5*(\
                  np.sum([t2-t1 for (t1, t2) in Up_intervals1])+\
                  np.sum([t2-t1 for (t1, t2) in Up_intervals2]))
    mean_down = .5*(\
                  np.sum([t2-t1 for (t1, t2) in Down_intervals1])+\
                  np.sum([t2-t1 for (t1, t2) in Down_intervals2]))
    
    # then we compute the intersection
    Up_state1, Up_state2, Down_state1, Down_state2 = 0.*t, 0.*t, 0.*t, 0.*t
    # dt for the first sequence
    for vec, interval in zip([Up_state1, Down_state1],
                             [Up_intervals1, Down_intervals1]):
        for (t1, t2) in interval:
            vec[(t>=t1) & (t<t2)] = dt
    # 1 for the second sequence to get a time
    for vec, interval in zip(\
                             [Up_state2, Down_state2],
                             [Up_intervals2, Down_intervals2]):
        for (t1, t2) in interval:
            vec[(t>=t1) & (t<t2)] = 1

    # cross product to get coincidence index
    # print(np.sum(Up_state1*Up_state2), mean_up1, mean_up2)
    CoInUp = np.sum(Up_state1*Up_state2)/mean_up
    # print(np.sum(Down_state1*Down_state2), mean_down1, mean_down2)
    CoInDown = np.sum(Down_state1*Down_state2)/mean_down
    
    return 100*.5*(CoInUp+CoInDown)
    
def make_comp_plot(t, Vm, LFP,
                   axVm, axLFP,
                   Up_statesVm=[], Up_statesLFP=[]):

    import matplotlib.pylab as plt
    dt = t[1]-t[0]

    # Vm plot
    axVm.plot(t, Vm, lw=1)
    axVm.set_ylabel('$V_m$ (mV)')
    axVm.set_xlim([t[0], t[-1]])
    axVm.set_xticklabels([])
    # LFP plot
    axLFP.plot(t, LFP, lw=1)
    axLFP.set_ylabel('LFP ($\mu$V)')
    axLFP.set_xlim([t[0], t[-1]])
    axLFP.set_xticklabels([])
    # 
    for ax, Up_states in zip([axVm, axLFP],
                             [Up_statesVm, Up_statesLFP]):
        y1, y2 = ax.get_ylim()
        for (t1, t2) in Up_states:
            if t1<t[-1] and t2>t[0]:
                if t2>t[-1]:
                    t2=t[-1]
                if t1<t[0]:
                    t1=t[0]
                ax.fill_between([t1,t2],
                                y1*np.ones(2)+.01*(y2-y1),
                                y2*np.ones(2)-.01*(y2-y1),
                                color='k', alpha=.1, lw=0)

if __name__ == '__main__':
    
    freqs = np.linspace(10, 100, 10)
    
    SETS_OF_FREQ_LEVELS = making_grid_of_linear_combination(freqs, level_number=2)
    
    import matplotlib.pylab as plt
    for ii in range(5):
        ii = np.random.randint(10000)%len(SETS_OF_FREQ_LEVELS)
        plt.bar(freqs, SETS_OF_FREQ_LEVELS[ii],
                width=freqs[1]-freqs[0])
    
    plt.show()
