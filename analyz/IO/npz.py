import numpy as np

def save_dict(filename, data):

    if '.npz' not in filename:
        print('/!\ The filename need to have the "npz" extension')
        print('        ------- renamed to:', filename+'.npz')
        np.savez(filename+'.npz', **data)
    else:
        np.savez(filename, **data)

def load_dict(filename):

    data = np.load(filename, allow_pickle=True)
    output = {}
    for key in data.files:
        output[key] = data[key]
    return output
        
if __name__=='__main__':

    data = {'x':np.arange(10),
            'y':{'blabla':'1',
                 'blabla2':[34, 35]},
    }
    save_dict('data.npz', data)
    data = load_dict('data.npz')
    print(data['x'], data['y'])
