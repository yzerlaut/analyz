import numpy as np
import sys
sys.path.append('../')
import IO.binary_to_python as binary_to_python

def get_formated_data(filename, zoom=[0,np.inf], with_params_only=False):
    data = get_metadata(filename)
    if filename.endswith('.bin'):
        t, DD = binary_to_python.load_file(filename, zoom=zoom)
    # if filename.endswith('.abf'):
    #     t, DD = axon_to_python.load_file(filename, zoom=zoom)
    else:
        t, DD = 0, [[0]]
    data['t'] = t
    data['Vm'], data['Iinj'], data['Laser'], data['UP_FLAG'], data['Vm_LP'], data['CMD'] = DD[0], DD[1], DD[2], DD[3], DD[4], DD[5]
    data['dt'] = (t[1]-t[0]) # data['dt'] in ms (transition times are stored in ms)
    if with_params_only:
        return data, get_metadata(filename)
    else:
        return data

def get_metadata(filename):
    if filename.endswith('.bin'):
        return binary_to_python.get_metadata(filename)
    # if filename.endswith('.abf'):
    #     return axon_to_python.get_metadata(filename)
    if filename.endswith('.npz'):
        return {'main_protocol':'modeling_work'}
    else:
        return {}


    
