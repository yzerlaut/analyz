
import numpy as np
from scipy import signal

### MORLET WAVELET, definition, properties and normalization
def Morlet_Wavelet(t, f, w0=6.):
    x = 2.*np.pi*f*t
    output = np.exp(1j * x)
    output *= np.exp(-0.5 * ((x/w0) ** 2)) # (Normalization comes later)
    return output

def Morlet_Wavelet_Decay(f, w0=6.):
    """
    Time value of the wavelet where the amplitude decays of 
    """
    return 2 ** .5 * (w0/(np.pi*f))

def from_fourier_to_morlet(freq):
    x = np.linspace(0.1/freq, 2.*freq, 1e3)
    return x[np.argmin((x-freq*(1-np.exp(-freq*x)))**2)]
    
def get_Morlet_of_right_size(f, dt, w0=6., with_t=False):
    Tmax = Morlet_Wavelet_Decay(f, w0=w0)
    t = np.arange(-int(Tmax/dt), int(Tmax/dt)+1)*dt
    if with_t:
        return t, Morlet_Wavelet(t, f, w0=w0)
    else:
        return Morlet_Wavelet(t, f, w0=w0)

def norm_constant_th(freq, dt, w0=6.):
    # from theoretical calculus:
    n = (w0/2./np.sqrt(2.*np.pi)/freq)*(1.+np.exp(-w0**2/2))
    return n/dt

def my_cwt(data, frequencies, dt, w0=6.):
    """
    wavelet transform with normalization to catch the amplitude of a sinusoid
    """
    output = np.zeros([len(frequencies), len(data)], dtype=np.complex)

    for ind, freq in enumerate(frequencies):
        wavelet_data = np.conj(get_Morlet_of_right_size(freq, dt, w0=w0))
        sliding_mean = signal.convolve(data,
                                       np.ones(len(wavelet_data))/len(wavelet_data),
                                       mode='same')
        # the final convolution
        wavelet_data_norm = norm_constant_th(freq, dt, w0=w0)
        output[ind, :] = signal.convolve(data-sliding_mean+0.*1j,
                                         wavelet_data,
                                         mode='same')/wavelet_data_norm
    return output



###########################################################
## Plotting function ######################################
###########################################################

def illustration_plot(t, freqs, data, coefs, dt, tstop, freq1, freq2, freq3):
    """
    a plot to illustrate the output of the wavelet analysis
    """
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12,6))
    plt.subplots_adjust(wspace=.8, hspace=.5, bottom=.2)
    # signal plot
    plt.subplot2grid((3, 8), (0,0), colspan=6)
    plt.plot(1e3*t, data, 'k-', lw=2)
    plt.ylabel('signal')
    for f, tt in zip([freq2, freq1, freq3], [200,500,800]):
        plt.annotate(str(int(f))+'Hz', (tt, data.max()))
    plt.xlim([1e3*t[0], 1e3*t[-1]])
    # time frequency power plot
    ax1 = plt.subplot2grid((3, 8), (1,0), rowspan=2, colspan=6)
    c = plt.contourf(1e3*t, freqs, coefs, cmap='PRGn', aspect='auto')
    plt.xlabel('time (ms)')
    plt.ylabel('frequency (Hz)')
    plt.yticks([10, 40, 70, 100]);
    # inset with legend
    acb = plt.axes([.4, .4, .02, .2])
    plt.colorbar(c, cax=acb, label='coeffs (a.u.)', ticks=[-1, 0, 1])
    # mean power plot over intervals
    plt.subplot2grid((3, 8), (1, 6), rowspan=2)
    for t1, t2 in zip([0,300e-3,700e-3], [300e-3,700e-3, 1000e-3]):
        cond = (t>t1) & (t<t2)
        plt.barh(freqs, np.power(coefs[:,cond],2).mean(axis=1)*dt,\
                 label='t$\in$['+str(int(1e3*t1))+','+str(int(1e3*t2))+']')
    plt.legend(prop={'size':'small'}, loc=(0.1,1.1))
    plt.yticks([10, 40, 70, 100]);
    plt.xticks([]);
    plt.xlabel(' mean \n power \n (a.u.)')
    # max of power over intervals
    plt.subplot2grid((3, 8), (1, 7), rowspan=2)
    for t1, t2 in zip([0,300e-3,600e-3], [300e-3,600e-3, 1000e-3]):
        cond = (t>t1) & (t<t2)
        plt.barh(freqs, np.power(coefs[:,cond],2).max(axis=1)*dt,\
                 label='t$\in$['+str(int(1e3*t1))+','+str(int(1e3*t2))+']')
    plt.yticks([10, 40, 70, 100]);
    plt.xticks([]);
    plt.xlabel(' max. \n power \n (a.u.)');
    return fig

def time_freq_plot(t, freqs, data, coefs):
    """
    a plot to illustrate the output of the wavelet analysis
    """
    dt = t[1]-t[0]
    import matplotlib.pylab as plt

    fig = plt.figure(figsize=(8,5))
    plt.subplots_adjust(wspace=.8, hspace=.5, bottom=.2)
    # signal plot
    plt.subplot2grid((3, 8), (0,0), colspan=6)
    plt.plot(1e3*t, data, 'k-', lw=2)
    plt.ylabel('signal')
    plt.xlim([1e3*t[0], 1e3*t[-1]])
    # time frequency power plot
    ax1 = plt.subplot2grid((3, 8), (1,0), rowspan=2, colspan=6)
    c = plt.contourf(1e3*t, freqs, coefs, cmap='PRGn', aspect='auto')
    plt.xlabel('time (ms)')
    plt.ylabel('frequency (Hz)')
    # inset with legend
    acb = plt.axes([.4, .4, .02, .2])
    plt.colorbar(c, cax=acb, label='coeffs (a.u.)', ticks=[-1, 0, 1])
    # mean power plot over intervals
    plt.subplot2grid((3, 8), (1, 6), rowspan=2)
    plt.barh(freqs, np.power(coefs,2).mean(axis=1)*dt)
    plt.xticks([]);
    plt.xlabel(' mean \n power \n (a.u.)')
    # max of power over intervals
    plt.subplot2grid((3, 8), (1, 7), rowspan=2)
    plt.barh(freqs, np.power(coefs,2).max(axis=1)*dt)
    plt.xticks([]);
    plt.xlabel(' max. \n power \n (a.u.)');
    return fig

if __name__ == '__main__':

    import argparse
    parser=argparse.ArgumentParser(description='Wavelet',
                                   formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-w", "--wavelet", default='ricker')
    parser.add_argument("--noise_level",help="noise level",\
                        type=float, default=0.)
    parser.add_argument("--nfreq",help="discretization of frequencies",\
                        type=int, default=20)
    parser.add_argument("-sw", "--show_wavelet", help="show_wavelet", action="store_true")
    args = parser.parse_args()
    
    import numpy as np
    import matplotlib.pylab as plt

    plt.style.use('ggplot')

    # temporal sampling
    dt, tstop = 1e-4, 1.
    t = np.arange(int(tstop/dt))*dt
    
    make_wavelet_of_right_size = get_Morlet_of_right_size
        
    if args.show_wavelet:
        for f in [10., 40., 70.]:
            plt.plot(*make_wavelet_of_right_size(f, dt, with_t=True))
    else:

        # ### artificially generated signal, transient oscillations
        freq1, width1, freq2, width2, freq3, width3 = 10., 100e-3, 40., 40e-3, 70., 20e-3
        data  = 3.2+np.cos(2*np.pi*freq1*t)*np.exp(-(t-.5)**2/2./width1**2)+\
                np.cos(2*np.pi*freq2*t)*np.exp(-(t-.2)**2/2./width2**2)+\
                np.cos(2*np.pi*freq3*t)*np.exp(-(t-.8)**2/2./width3**2)

        # ### adding colored noise to test robustness
        data += args.noise_level*np.convolve(np.exp(-np.arange(1000)*dt/400e-3),\
                            np.random.randn(len(t)), mode='same') # a slow one
        data += args.noise_level*np.convolve(np.exp(-np.arange(1000)*dt/5e-3),\
                            np.random.randn(len(t)), mode='same') # a faster one

        # Continuous Wavelet Transform analysis
        freqs = np.linspace(1, 90, args.nfreq)
        coefs = my_cwt(data, freqs, dt)

        # illustration_plot(t, freqs, data, coefs)
        from datavyz import ge
        ge.time_freq_plot(t, freqs, data, coefs)
    plt.show()
