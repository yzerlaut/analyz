import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import data_analysis.IO.hdf5 as hdf5

def classify_RTXI_recordings_according_to_protocols(data):
    """
    
    """
    print(data['params'])

    data['protocol_type'] = ''


def load_continous_RTXI_recording(filename, with_metadata=True):
    """
    ....
    """
    data = hdf5.load_dict_from_hdf5(filename)['Trial1']
    formatted_data = {}
    formatted_data['Downsampling Rate'] = data['Downsampling Rate']
    formatted_data['Date'] = data['Date']
    # time step
    formatted_data['dt'] = 1e-9*float(data['Period (ns)'])
    # parameters
    formatted_data['params'] = {}
    for key, val in data['Parameters'].items():
        formatted_data['params'][key] = float(val[0][1])
    for i in range(data['Synchronous Data']['Channel Data'].shape[1]):
        formatted_data[list(data['Synchronous Data'].keys())[i]] = data['Synchronous Data']['Channel Data'][:,i]

    classify_RTXI_recordings_according_to_protocols(formatted_data)    
    
    return formatted_data

