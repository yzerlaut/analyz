import numpy as np

def get_transition_times(t, LFP, window=25e-3, factor=0.1):
    """
    2 thresholds strategy for state transition characterization
    """
    dt = t[1]-t[0]
    wdw = int(window/dt)

    # subsampling
    new_t = t[::wdw]
    varLFP = np.array([np.std(LFP[max([0,int(tt/dt-wdw)]):min([int(t[-1]/dt), int(tt/dt+wdw)])]) for tt in new_t])

    hist, be = np.histogram(varLFP, bins=np.linspace(varLFP.min(), varLFP.max(), 100))
    be_hist = .5*(be[1:]+be[:-1])

    meanThre = be_hist[np.argmax(hist)] # mean of the two threshold
    stdThre = factor*np.std(varLFP)

    DU_threshold = meanThre+stdThre
    UD_threshold = max([meanThre-stdThre, 1.2*varLFP.min()])

    down_flag = False
    if varLFP[0]>UD_threshold:
        down_state = True
    UD_transitions, DU_transitions = [], [],
    for i in range(len(varLFP)):
        if varLFP[i]>DU_threshold and down_flag:
            DU_transitions.append(new_t[i]-window)
            down_flag = False
        if varLFP[i]<UD_threshold and not down_flag:
            UD_transitions.append(new_t[i]-window)
            down_flag = True
            
    return np.array(UD_transitions), np.array(DU_transitions)


def get_down_state_array(t, LFP, window=50e-3, factor=0.1):
    """
    2 thresholds strategy for state transition characterization
    """
    dt = t[1]-t[0]
    wdw = int(window/dt)

    # subsampling
    new_t, new_i = t[::wdw], np.arange(len(t))[::wdw]
    varLFP = np.array([np.std(LFP[max([0,int(tt/dt-wdw)]):min([int(t[-1]/dt), int(tt/dt+wdw)])]) for tt in new_t])

    meanThre = np.mean(varLFP)
    stdThre = factor*np.std(varLFP)
    # print(meanThre-stdThre, meanThre, stdThre)
    
    down_state = 0.*t
    for i in new_i:
        if varLFP[int(i/wdw)]<meanThre-stdThre:
            down_state[max([0,i-int(wdw/2)]):min([len(t),i+int(wdw/2)])] = 1
    return down_state

if __name__ == '__main__':
    
    import matplotlib.pylab as plt
    plt.style.use('ggplot')
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    import data_analysis.IO.load_data as L
    from data_analysis.freq_analysis.wavelet_transform import my_cwt
    from data_analysis.processing.denoise import remove_50Hz
    from graphs.my_graph import show, set_plot
    from graphs.time_freq import time_freq_plot
    filename = '/Users/yzerlaut/DATA/Data_Ste_Zucca/2017_02_24/17_46_32_VCLAMP-WITH-THAL-AND-CORTEX-EXTRA.abf'
    t, [_, LFP, _] = L.load_file(filename)
    dt, tstop = t[1]-t[0], 10
    plt.plot(t[:int(tstop/dt)], LFP[:int(tstop/dt)])
    signal, sine = remove_50Hz(t[:int(tstop/dt)], LFP[:int(tstop/dt)])
    plt.plot(t[:int(tstop/dt)], signal[:int(tstop/dt)])
    plt.plot(t[:int(tstop/dt)], sine[:int(tstop/dt)])
    # freqs = np.linspace(0.5, 100)
    # coefs = my_cwt(LFP[:int(tstop/dt)], freqs, dt)
    # time_freq_plot(t[:int(tstop/dt)], freqs, LFP[:int(tstop/dt)], coefs)
    # show(plt)
    plt.show()
