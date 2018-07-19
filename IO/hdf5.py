"""
taken from:
http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py?newreg=f582be64155a4c0f989a2aa05ee67efe

Updated July 2018 
the string encoding has changed
so string keys are translated to bytes
the for the decoding, bytes strings are decoded to utf-8 strings
"""

import numpy as np
import h5py
import os


def make_writable_elements(value):
    # all cases to be covered should be:
    # np.ndarray, np.int64, np.float64, bytes, dict, tuple, list, str
    if isinstance(value, (float, int)):
        return np.ones(1)*value
    elif isinstance(value, str):
        return np.string_(value)
    elif isinstance(value, tuple):
        return np.array(value)
    else:
        return value

def make_writable_list(List):
    list_to_return = []
    try:
        # if list of lists
        if isinstance(List[0], (list, np.ndarray)):
            list_to_return = [make_writable_list(List[i]) for i in range(len(List))]
        else:
            list_to_return = [make_writable_elements(List[i]) for i in range(len(List))]
    except IndexError:
        list_to_return = [make_writable_elements(List[i]) for i in range(len(List))]
    return list_to_return


def make_writable_dict(dic):
    dic2 = dic.copy()
    for key, value in dic.items():
        if isinstance(value, (list, np.ndarray)):
            dic2[key] = make_writable_list(value)
        elif isinstance(value, dict):
            dic2[key] = make_writable_dict(value)
        else:
            dic2[key] = make_writable_elements(value)
    return dic2

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    dic2 = make_writable_dict(dic)
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic2)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        new_key = np.string_(path+key)
        if isinstance(item, np.ndarray):
            h5file[new_key] = item
        elif isinstance(item, (np.int64, np.float64, bytes)):
            h5file[new_key] = item
        elif isinstance(item, str):
            print('/!\ Problem ! there should be no strings anymore at that stage !!')
            print('key ', str(new_key), 'of value:', item , 'is still a string')
            # h5file[new_key] = np.string_(item)
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, tuple):
            h5file[new_key] = np.array(item)
        elif isinstance(item, list):
            # h5file[new_key] = np.array(item)
            try:
                if type(item[0])==str:
                    item = np.array([np.string_(ii) for ii in item])
            except (IndexError, AttributeError):
                pass
            h5file[new_key] = np.array(item)
        elif isinstance(item, (float, int)):
            h5file[new_key] = np.array(item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def make_readable_elements(value):
    # all cases to be covered should be:
    # np.ndarray, np.int64, np.float64, bytes, dict, tuple, list, str
    if isinstance(value, bytes):
        return str(value, 'utf-8')
    else:
        return value

def make_readable_list(List):
    list_to_return = []
    try:
        # if list of lists
        if isinstance(List[0], (list, np.ndarray)):
            list_to_return = [make_readable_list(List[i]) for i in range(len(List))]
        else:
            list_to_return = [make_readable_elements(List[i]) for i in range(len(List))]
    except IndexError:
        list_to_return = [make_readable_elements(List[i]) for i in range(len(List))]
    return list_to_return


def make_readable_dict(dic):
    dic2 = dic.copy()
    for key, value in dic.items():
        if isinstance(value, (list, np.ndarray)):
            dic2[key] = make_readable_list(value)
        elif isinstance(value, dict):
            dic2[key] = make_readable_dict(value)
        else:
            dic2[key] = make_readable_elements(value)
    return dic2

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        dic = recursively_load_dict_contents_from_group(h5file, '/')
    return make_readable_dict(dic)

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            print(path, key, item.value, type(item.value))
            if isinstance(item.value, bytes):
                to_be_put = str(item.value,'utf-8')
            else:
                to_be_put = item.value
            try:
                if len(to_be_put)>1:
                    if isinstance(to_be_put[0], bytes):
                        to_be_put = np.array([str(ii,'utf-8') for ii in to_be_put])
            except TypeError:
                pass
            ans[str(key)] = to_be_put
        elif isinstance(item, h5py._hl.group.Group):
            ans[str(key)] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans



if __name__ == '__main__':

    data = {'x': 'astring',
            'y': np.arange(10),
            '0':['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd'],
            '1':np.array([['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd'],['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd']]),
            '2':[['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd'],['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd']],
            'd': {'z': np.ones((2,3)),
                  'sdkfjh':'',
                  'dict_of_dict':{'234':'kjsdfhsdjfh','z': np.ones((1,3))},
                  'b': b'bytestring'}}
    print(data)
    filename = 'test.h5'
    save_dict_to_hdf5(data, filename)
    dd = load_dict_from_hdf5(filename)
    # should test for bad type
    print(dd)
