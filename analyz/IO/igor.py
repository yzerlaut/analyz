import string, sys, os, platform
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.pardir))

import numpy as np

from IO.hdf5 import load_dict_from_hdf5


def reshape_data_from_Igor(data,
                           dt_subsampling=0,
                           verbose=False):
    """
    dt_subampling in [ms]
    """
    # creating a new dictionary
    new_data = {'recordings':{},
                'stimulations':{},
                'Metadata':{'original_dt':data['SampleInterval']}}
    
    isubsampling = max([1, int(dt_subsampling/data['SampleInterval'])])
    
    new_data['Metadata']['dt'] = data['SampleInterval']*isubsampling # in ms
    
    nsample = int(data['SamplesPerWave'])
    new_data['t'] = data['SampleInterval']*np.arange(nsample)[::isubsampling]

    if verbose:
        print('- temporal sampling, original time step: %.3fms' % data['SampleInterval'])
        if isubsampling:
            print('    --> subsampled at %.3fms' % new_data['Metadata']['dt'])
        # print('- n=%i episodes' % new_data['Metadata']['nepisodes'])
    
    ##############################################
    ## Find the key of the protocol --> MORE TO FIND IN THAT protocol_key quantity
    for key in data:
        if (type(data[key]) is dict) and ('BoardConfigs' in data[key]):
            protocol_data = data[key]
            protocol_key = key
            
    ##############################################
    ## Read the "BoardConfig" of the protocol and build the 'recording' and 'stimulation' outputs
    #
    # read recording info
    VRC = np.isfinite(protocol_data['BoardConfigs']['ADCboard']) # valid recording channels
    rec_channels = protocol_data['BoardConfigs']['ADCchan'][VRC]
    rec_names = protocol_data['BoardConfigs']['ADCname'][VRC]
    rec_units = protocol_data['BoardConfigs']['ADCunits'][VRC]
    if verbose:
        print('- Recordings channels:')
    for i, name in enumerate(rec_names):
        if verbose:
            print('  * %i) %s in %s' % (i, name, rec_units[i]))
        new_data['recordings'][name], j = [], 0
        while 'Record%s%i' % (string.ascii_uppercase[i], j) in data:
            new_data['recordings'][name].append(data['Record%s%i'%(string.ascii_uppercase[i],j)][::isubsampling])
            j+=1
        new_data['recordings'][name] = np.array(new_data['recordings'][name])
        new_data['Metadata']['%s_unit' % name] = rec_units[i]
    #
    # read stimulation info
    VSC = np.isfinite(protocol_data['BoardConfigs']['DACboard']) # valid stimulation channels
    stim_channels = protocol_data['BoardConfigs']['DACchan'][VSC]
    stim_names = protocol_data['BoardConfigs']['DACname'][VSC]
    stim_gain = protocol_data['BoardConfigs']['DACscale'][VSC]
    stim_units = protocol_data['BoardConfigs']['DACunits'][VSC]
    if verbose:
        print('- Stimulation channels:')
    for i, name in enumerate(stim_names):
        if verbose:
            print('  * %i) %s in %s' % (i, name, stim_units[i]))
        if 'DAC_%i_1' % i in protocol_data:
            new_data['stimulations'][name], j = [], 0
            while 'DAC_%i_%i' % (i,j) in protocol_data: # we loop over episodes
                new_data['stimulations'][name].append(stim_gain[i]*protocol_data['DAC_%i_%i' % (i,j)][::isubsampling])
                j+=1
        else:
            new_data['stimulations'][name] = stim_gain[i]*protocol_data['DAC_%i_0' % i][::isubsampling] # just one array
        new_data['Metadata']['%s_unit' % name] = stim_units[i]
    #
    ##############################################
    # IGOR Metadata
    KEYS = [key for key in data.keys() if (len(key.split('Record'))==1) and (key!=protocol_key)]
    # data keys
    for key in KEYS:
        new_data['Metadata'][key] = data[key]
    # protocol keys
    for key in data[protocol_key]:
        if sys.getsizeof(data[protocol_key][key])<1000: # if not a full-data array
            new_data['Metadata'][key] = data[protocol_key][key]
    
    # some specific processing here
    if 'CT_RecordMode' in data:
        if data['CT_RecordMode']==1:
            new_data['Metadata']['recording_type']='voltage-clamp'
        elif data['CT_RecordMode']==1:
            new_data['Metadata']['recording_type']='current-clamp'
            
    return new_data


def load_hdf5_exported_from_Igor(filename,
                                 dt_subsampling=0.,
                                 verbose=False,
                                 with_reshaping=True):
    data = load_dict_from_hdf5(filename)
    if with_reshaping:
        new_data = reshape_data_from_Igor(data,
                                          dt_subsampling=dt_subsampling,
                                          verbose=verbose)
        new_data['Metadata']['filename'] = filename
    else:
        return data

if __name__ == '__main__':

    filename = sys.argv[-1]
    data = load_hdf5_exported_from_Igor(filename)
    print(data)
    # print(filename)
    # dd = load_dict_from_hdf5(filename)
    # print(dd.keys())
    # for key, val in dd.items():
    #     print(key)
    # print(f['RecordB9'].dtype)

