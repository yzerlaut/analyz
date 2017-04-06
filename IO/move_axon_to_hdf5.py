from neo.io import AxonIO
from hdf5 import save_dict_to_hdf5
import numpy as np
import os
import matplotlib.pylab as plt

def translate(args):

    keys = list_keys(args.filename)
    
    data = {} # dictionary for hdf5 export
    
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)
    RT = Block.rec_datetime #
    
    data['day'] = ("%02d" % RT.day) + '_' + ("%02d" % RT.month) + '_' + ("%04d" % RT.year)
    data['time'] = ("%02d" % RT.hour) + '_' + ("%02d" % RT.minute) + '_' + ("%02d" % RT.second)
    data['dt'] =  np.array([Block.segments[0].analogsignals[0].sampling_period])

    if not os.path.exists(data['day']):
        os.makedirs(data['day'])

    protocol = args.protocol
    args.filename = data['day']+os.path.sep+data['time']+'_'+protocol+'.h5'
    print(args.filename)

    if args.new_keys is not None:
        if len(args.new_keys)==len(keys):
            keys = new_keys
        else:
            print('need to give a new to all former keys (in the order)')

    if len(Block.segments)==1:
        # if only one episode, we swith to continuous
        args.force_continuous = True
        
    for i in range(len(keys)):
        data[keys[i]] = []
        for s in range(len(Block.segments)):
             data[keys[i]].append(Block.segments[s].analogsignals[i])
        if args.force_continuous:
            data[keys[i]] = np.array(data[keys[i]]).flatten()
        data[keys[i]+'_unit'] = Block.segments[s].analogsignals[i][0].dimensionality.string
        
    save_dict_to_hdf5(data, args.filename)

    
def list_keys(args):
    """ list the keys of the """
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)
    KEYS = []
    for i in range(len(Block.segments[0].analogsignals)):
        print('key', i+1, '-->', Block.segments[0].analogsignals[i].name)
        KEYS.append(Block.segments[0].analogsignals[i].name)
    return KEYS

def show(args):
    
    keys = list_keys(args.filename)
    
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)

    fig, AX = plt.subplots(len(keys), figsize=(5,3*len(keys)))
    for i in range(len(keys)):
        AX[i].set_title(keys[i])
        AX[i].plot(Block.segments[0].analogsignals[i][:10000])
    plt.show()
    
if __name__ == '__main__':
    
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Generating random sample of a given distributions and
     comparing it with its theoretical value
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("args.filename")
    parser.add_argument('-nk', "--new_keys",help="new keys to be assigned", nargs='*')
    parser.add_argument("--std",help="std of the random values",\
                        type=float, default=10.)
    parser.add_argument('-p', "--protocol", help="name of the protocol", default='Sample-Recording')
    parser.add_argument("-fc", "--force_continuous", help="force continuous", action="store_true")
    parser.add_argument("-s", "--show", help="show data with initial keys",
                        action="store_true")
    parser.add_argument("-lk", "--list_keys", help="list the available keys in data file",\
                        action="store_true")
    
    parser.add_argument("--args.filename", '-f', help="args.filename",type=str, default='data.npz')
    args = parser.parse_args()

    if args.show:
        show(args)
    elif args.list_keys:
        list_keys(args)
    else:
        translate(args)
        
