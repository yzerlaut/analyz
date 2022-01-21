<div><img src="https://github.com/yzerlaut/datavyz/raw/master/docs/logo.png" alt="datavyz logo" width="45%" align="right" style="margin-left: 10px"></div>

# analyz

*A wrap up of tools for data analysis in neuroscientific context. Covering data import, data exploration, data mining using machine learning, time-series analysis, signal processing, statistics, ...*

Part of the software suite for data science: [analyz](https://github.com/yzerlaut/analyz), [datavyz](https://github.com/yzerlaut/datavyz), [finalyz](https://github.com/yzerlaut/finalyz)

# Components

The package is organized around the following components:

## IO - the import-export interface 

Interface to load datafiles common data formats into numpy arrays, it currently supports:

- *Axon* (*Molecular Instruments*) datafiles
- *Elphy* datafiles
- *HDF5* datafiles
- binary datafiles
- npz (numpy) datafiles
- neuronexus (numpy) datafiles

## ML - the machine learning toolbox

Based on scipy & tensorflow

- classification
- dim_reduce
- regression
- convolutional_nn
- ensemble
- recurrent_nn
- tf_cookbook (tensorflow cookbook)

## optimization - the minimization/fitting toolbox

Imlements: 

- minimization procedures
- curve fits
- ...

## freq_analysis - the spectral analysis toolbox

Implements: 

- Fourier transform
- Wavelet transform

## signal_library - the library of standard signal

Implements: 

- generation of stochastic processes (Wiener process, Ornstein-Uhlenbeck process, ...)
- set of classical functions (Gaussian, Heaviside, ...)
- set of classical waveforms (Step, etc...)

## processing - signal processing toolbox

Classical signal processing tools. It currently implements:

- filtering 
- denoising
- smoothing
- ...

## statistics - the statistics toolbox

Statistical library including classical and specific tests (based on *scipy.stats*):

- permutation test
- ...

## workflow_tools - various tools used in analysis pipeline

Set of procedures for:

- parameter search
- array manipulation
- file organization
- saving strategies
- shell interaction
- ...
