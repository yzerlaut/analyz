import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import data_analysis.IO.axon_to_python as axon
import data_analysis.IO.binary_to_python as binary

def load_file(filename, zoom=[0,np.inf]):
    if filename.endswith('.bin'):
        return binary.load_file(filename, zoom=zoom)
    elif filename.endswith('.abf'):
        print(filename)
        return axon.load_file(filename, zoom=zoom)
    else:
        return None

def get_metadata(filename, infos={}):
    print('filename is', filename)
    if filename.endswith('.bin'):
        return binary.get_metadata(filename, infos=infos)
    elif filename.endswith('.abf'):
        return axon.get_metadata(filename, infos=infos)
    elif filename.endswith('.npz'):
        return {'main_protocol':'modeling_work'}
    else:
        return None

def get_formated_data(filename):

    t, VEC = load_file(filename)
    meta = get_metadata(filename)
    data = {'t':t, 'Vm':VEC[0],
            'infos':meta, 'dt':t[1]-t[0]}

    return data
    
    
if __name__ == '__main__':
    import sys
    import matplotlib.pylab as plt
    filename = sys.argv[-1]
    print(get_metadata(filename))
    t, data = load_file(filename, zoom=[-5.,np.inf])
    plt.plot(t[10000:], data[0][10000:])
    plt.show()
