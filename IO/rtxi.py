import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import data_analysis.IO.hdf5 as hdf5

def classify_RTXI_recordings_according_to_protocols(data):
    """
    to be implemented
    """
    # print(data['params'])

    data['protocol_type'] = ''


def find_metadata_file_and_add_parameters(data):
    """

    """
    file_directory = os.path.dirname(data['filename'])
    time_stamp_filename = from_filename_to_time_stamp(data['filename'], extension='.RTXI.h5')
    file_list= os.listdir(file_directory)
    json_list, time_stamps = [], []
    for ff in file_list:
        if len(ff.split('.JSON'))>1:
            json_list.append(ff)
            time_stamps.append(from_filename_to_time_stamp(ff, extension='.JSON'))

    if len(json_list)>0:
        # the right file is the one right before the recording
        ii = np.argmin(np.abs(np.array(time_stamps)-time_stamp_filename))
        param_filename = json_list[ii]
        print('-----------------------------------------')
        print('the file:', data['filename'])
        print('was associated to the metadata file: **', param_filename, '**')
        print('-----------------------------------------')
        with open(file_directory+os.path.sep+param_filename, 'r') as fn:
            exec('dd='+fn.read()) # the
        # we loop through the dd dictionary and set its key and values in the data dictionary
        for key in locals()['dd']:
            data[key] = locals()['dd'][key]
    else:
        print('-----------------------------------------')
        print('no metadata file found in this folder')
        print('-----------------------------------------')
            
        
def load_continous_RTXI_recording(filename,
                                  with_metadata=False):
    """
    ....
    """
    data = hdf5.load_dict_from_hdf5(filename)['Trial1']
    formatted_data = {'filename':filename}
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

    if with_metadata:
        find_metadata_file_and_add_parameters(formatted_data)
    
    return formatted_data

def from_filename_to_time_stamp(filename, extension='.RTXI.h5'):
    new_filename = filename.split(os.path.sep)[-1] # to be sure to have the last extension
    time_stamp = 0
    for val, factor in zip(new_filename.split(extension)[0].split(':'), [3600., 60., 1.]):
        time_stamp += factor*float(val)
    return time_stamp
