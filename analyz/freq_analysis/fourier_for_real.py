# Filename: fourier_for_real.py
"""
We shortly adapt the Fourier transform modulus of numpy to be used in
combination with analytical calculus under the 'mathematician' Fourier
transform convention:

Run the modulus as main to have an example !


DETAILS:

Indeed we start from the definitions:

$\hat{f}(\nu) = \int_{-\infty}^{\infty} e^{ - 2 i \pi \nu t} f(t) \, dt $
$f(t) = \int_{-\infty}^{\infty} e^{2 i \pi \nu t} \hat{f}(\nu) \, d \nu$

We discretize: $t_k = k\,dt$, $\nu_l = l\,d\nu$ with $d\nu=\frac{1}{N dt}$

$\hat{f}(\nu_l) = \sum_{k=0}^{N-1} dt e^{ - 2 i \pi l d\nu k dt } f(t_k) $
$\hat{f}(\nu_l) = dt \sum_{k=0}^{N-1} e^{ - 2 i \pi \frac{l k}{N}} f(t_k) $
where $\sum_{k=0}^{N-1} e^{ - 2 i \pi \frac{l k}{N}} f(t_k)$ is what is
computed by the rfft numpy.fft function

Similarly for the inverse Fourier transform:
$f(t_k) = d\nu \sum_{l=0}^{N-1} e^{ 2 i \pi \frac{l k}{N}} f(\nu_l) $
$f(t_k) = \frac{1}{N dt} \sum_{l=0}^{N-1} e^{ 2 i \pi \frac{l k}{N}} f(\nu_l) $
where $\frac{1}{N} \sum_{l=0}^{N-1} e^{ 2 i \pi \frac{l k}{N}} f(\nu_l)$
 is what is computed by the irfft numpy.fft function


the whole documentation and conventions for the discrete fourier transform
are available at :
http://docs.scipy.org/doc/numpy/reference/routines.fft.html
"""

import numpy as np

def time_to_freq(N, dt):
    return np.array(np.fft.rfftfreq(N,d=dt))

def FT(signal, N, dt):
    ft = np.fft.rfft(signal)*dt
    return ft

def inv_FT(signal, N, dt):
    ft = np.fft.irfft(signal, N)/dt
    return ft
    
if __name__ == '__main__':
    """
    illustrate the conventions used in this module
    Takes a time varying signal that allows analytic computation of the fourier
    transform and its inverse to show that this module can be used in complement
    of analytic derivations
    """

    import matplotlib.pylab as plt

    # #### 1) conventions used in this module !
    plt.figure(figsize=(4,3.5))
    ax = plt.axes(frameon=False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    plt.title('Fourier transform \n conventions !')
    plt.annotate('$\hat{f}(\\nu) = \int_{-\infty}^{\infty} e^{ - 2 i \pi \\nu t} f(t) \, dt $',(.1,.6))
    plt.annotate('$f(t) = \int_{-\infty}^{\infty} e^{2 i \pi \\nu t} \hat{f}(\\nu) \, d \\nu$',(.1,.2))
    plt.tight_layout()
    plt.show()

    #### 2) Illustrating Example
    plt.figure(figsize=(4,3.5))
    ax = plt.axes(frameon=False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    plt.title('Illustrating Example')
    plt.annotate('$f(t) = Q \, e^{\\frac{t-t_0}{\\tau}} H(t-t_0)$',(.1,.7))
    plt.annotate('$\hat{f}(\omega) = e^{2 i \pi \\nu t_0} \, \\frac{Q}{2 i \pi \\nu +\\frac{1}{\\tau}}$',(.1,.4))
    # plt.annotate(' with $t_0 = 5$, $\\tau=10$, $Q=0.1$',(.1,.1))
    plt.tight_layout()
    plt.show()


    Q, ts, Tau = .1, 10., 10.
    dt, N, t0 = 0.063, 1230, 0.
    t = t0+np.arange(N)*dt # time vector
    f = time_to_freq(len(t), dt)

    func_th = [Q*np.exp(-(tt-ts)/Tau) if tt>=ts else 0 for tt in t ]
    TFfunc_th = Q*np.exp(-1j*2*np.pi*ts*f)/(1j*2*np.pi*f+1./Tau)

    TFfunc_from_tf = FT(func_th, N, dt)
    func_from_tf = inv_FT(TFfunc_th, N, dt)

    print(TFfunc_from_tf[:3], TFfunc_th[:3])
    print(f[:3])
    
    #### 3) Plots in the temporal domain
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    plt.title('temporal domain')
    plt.plot(func_th, 'k', lw=2, label='$f(t)$')
    plt.plot(np.real(func_from_tf),'r--',\
             lw=2,label='R[TF$^{-1}[\hat{f}]]$')
    plt.legend(frameon=False, prop={'size':'small'})
    plt.xlabel('t (s)')
    plt.tight_layout()
    plt.show()


    #### 4) Plots in the frequency domain
    plt.figure(figsize=(12,4))
    plt.suptitle('frequency domain')
    ax = plt.subplot(121)
    plt.semilogx(np.real(TFfunc_th), 'k', lw=2, label='R[$\hat{f}(\\nu)$]')
    plt.semilogx(np.real(TFfunc_from_tf),\
                 'r--', lw=2, label='R[TF[f]]')
    plt.legend(frameon=False, prop={'size':'small'})
    plt.xlabel('$\\nu$ (Hz)')
    ax = plt.subplot(122)
    plt.semilogx(np.imag(TFfunc_th),\
                 'k', lw=2, label='Im[$\hat{f}(\\nu)]$')
    plt.semilogx(np.imag(TFfunc_from_tf),\
                 'r--',lw=2, label='Im[TF[f]]')
    plt.legend(frameon=False, loc='lower right', prop={'size':'small'})
    plt.xlabel('$\\nu$ (Hz)')
    plt.tight_layout()
    plt.show()
