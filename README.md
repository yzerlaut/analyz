<div><img src="https://github.com/yzerlaut/datavyz/raw/master/docs/logo.png" alt="datavyz logo" width="45%" align="right" style="margin-left: 10px"></div>

# analyz

*A wrap up of tools for data analysis in neuroscientific context. Covering data import, data exploration, data mining, time-series analysis, statistics, ...*

# Components

The package is organized around the follwoing 

## IO (import-export interface)

Interface to load datafiles common data formats into numpy arrays, it currently supports:

- *Axon* (*Molecular Instruments*) datafiles
- *Elphy* datafiles
- *HDF5* datafiles
- binary datafiles
- npz (numpy) datafiles
- neuronexus (numpy) datafiles

# optimization

Imlements: 

- minimization procedures
- curve fits
- ...

# freq_analysis

Implements: 

- Fourier transform
- Wavelet transform

# signal_library

Imlements: 

- generation of stochastic processes (Wiener process, Ornstein-Uhlenbeck process, ...)
- set of classical functions (Gaussian, Heaviside, ...)

- set of classical waveforms (Step, etc...)

# processing

Imlements: 

- filtering 
- denoising
- smoothing
- ...

# statistics

Statistical library including classical and specific tests (based on *scipy.stats*):

- permutation test
- ...

# workflow_tools

Set of procedures for:

- parameter search
- array manipulation
- file organization
- saving strategies
- shell interaction
- ...
