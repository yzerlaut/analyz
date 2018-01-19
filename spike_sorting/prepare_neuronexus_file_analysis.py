import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from data_analysis.IO.move_neuronexus_to_binary import main as convert_to_binary


def write_param_file(args):
    S = """
experiment_name =  '%(exp_name)s'
prb_file = '/Users/yzerlaut/work/data_analysis/spike_sorting/probe_files/16_channels_neuronexus.prb'

traces = dict(
    raw_data_files=['%(exp_name)s.dat'],
    voltage_gain=1.,
    sample_rate=32000,
    n_channels=16,
    dtype='int16',
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
    f = open(args['exp_name']+'.prm', 'w')
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

    parser.add_argument('--input_names', nargs='*',
                        default= ['CSC4.ncs', 'CSC1.ncs', 'CSC8.ncs', 'CSC5.ncs',
                                  'CSC2.ncs', 'CSC3.ncs', 'CSC16.ncs', 'CSC13.ncs',
                                  'CSC6.ncs', 'CSC7.ncs', 'CSC12.ncs', 'CSC9.ncs',
                                  'CSC10.ncs', 'CSC11.ncs', 'CSC14.ncs', 'CSC15.ncs'])


    parser.add_argument('-e', "--exp_name", help="name of the protocol", default='exp')
    parser.add_argument("--sample_freq",help="starting time of export", type=float, default=32000)

    parser.add_argument("--tstart",help="starting time of export", type=float, default=0.)
    parser.add_argument("--tmax",help="starting time of export", type=float, default=np.inf)
    
    args = parser.parse_args()
    
    write_param_file(vars(args))
    convert_to_binary(args.input_names, args.exp_name+'.dat', max_chunk_number=6000)
    
    print("""
    Now just run:

    klusta %(name)s

    """ % {'name':args.exp_name+'.prm'})
