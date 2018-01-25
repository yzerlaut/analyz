import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.IO.move_axon_to_binary import *

def write_param_file(args):
    S = """
experiment_name =  '%(name)s'
prb_file = '/Users/yzerlaut/work/data_analysis/spike_sorting/probe_files/single_electrode.prb'

traces = dict(
    raw_data_files=['%(name)s.dat'],
    voltage_gain=1.,
    sample_rate=%(facq)f,
    n_channels=1,
    dtype='float32',
)

spikedetekt = dict(
    filter_low=500.,  # Low pass frequency (Hz)
    filter_high_factor=0.95 * .5,
    filter_butter_order=3,  # Order of Butterworth filter.

    filter_lfp_low=0,  # LFP filter low-pass frequency
    filter_lfp_high=300,  # LFP filter high-pass frequency

    chunk_size_seconds=1,
    chunk_overlap_seconds=.015,

    n_excerpts=50,
    excerpt_size_seconds=1,
    threshold_strong_std_factor=4.,
    threshold_weak_std_factor=2.,
    detect_spikes='negative',

    connected_component_join_size=1,

    extract_s_before=16,
    extract_s_after=16,

    n_features_per_channel=4,  # Number of features per channel.
    pca_n_waveforms_max=1000,
)

klustakwik2 = dict(
    num_starting_clusters=10000,
)

    """ % args
    f = open(args['filename'].replace('abf', 'prm'), 'w')
    f.write(S)
    f.close()

if __name__ == '__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Generating random sample of a given distributions and
     comparing it with its theoretical value
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("filename")

    parser.add_argument("--sample_freq",help="starting time of export", type=float, default=32000)

    parser.add_argument("--tstart",help="starting time of export", type=float, default=0.)
    parser.add_argument("--tend",help="starting time of export", type=float, default=np.inf)
    
    args = parser.parse_args()
    
    KEYS = list_keys(args)
    ikey = int(input('For a 1-D npy export, choose a key among those: '))
    print('You chose', KEYS[ikey], ' [...] export to .dat file')
    infos = translate(args, ikey, KEYS)
    args = vars(args)
    args['name'] = args['filename'].replace('.abf', '')
    for k in infos:
        args[k] = infos[k]
    write_param_file(args)

    
    print("""
    Now just run:

    klusta %(name)s.prm

    """ % args)
