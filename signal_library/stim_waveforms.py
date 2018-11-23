import numpy as np

def double_exponential(amp, t0, T_rise, T_decay, dt=0.1, width_extent_factor=6., with_t=False):
    """
    You need to be careful with the parameters

    width_extent_factor>5
    dt<min([T_rise, Tdecay])/10.
    T_decay>T_rise/2.
    [...]
    """
    tt = np.arange(int(width_extent_factor*T_decay/dt))*dt
    output = 0*tt
    output = (1.-np.exp(-tt/T_rise))*np.exp(-tt/T_decay)
    # get the normalization factor analytically:
    factor = (1.-T_rise/(T_rise+T_decay))*np.power(T_rise/(T_rise+T_decay), T_rise/T_decay)
    if with_t:
        return tt, output/factor*amp
    else:
        return output/factor*amp
