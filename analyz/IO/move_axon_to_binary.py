from neo.io import AxonIO
import numpy as np
import os, quantities
import matplotlib.pylab as plt


def translate(args, ikey, keys):

    data = {} # dictionary for hdf5 export
    
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)
    RT = Block.rec_datetime #

    params = {} # parameters stored here
    params['day'] =  ("%04d" % RT.year) + '_' + ("%02d" % RT.month) + '_' + ("%02d" % RT.day) 
    params['time'] = ("%02d" % RT.hour) + '_' + ("%02d" % RT.minute) + '_' + ("%02d" % RT.second)
    
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)
    params['dt'] =  np.array([Block.segments[0].analogsignals[0].sampling_period])

    new_filename = args.filename.replace('.abf', '.dat')

    if len(Block.segments)==1:
        # if only one episode, we swith to continuous
        args.force_continuous = True

    t = np.arange(len(Block.segments[0].analogsignals[0]))*params['dt']
    cond = (t>=args.tstart) & (t<=args.tend)
    
    # ARRAY = []
    # for s in range(len(Block.segments)):
    #      ARRAY.append(Block.segments[s].analogsignals[ikey][cond])

    X = Block.segments[0].analogsignals[ikey][cond]/quantities.mV
    # 'short int' (int16) has values from −32,768 to 32,767
    # SWITCH TO int16 !!!
    # X -= X.mean()
    # scale = 30000/np.abs(X).max()
    # X2 = np.reshape(np.array(scale*X), (len(X), 1)).astype('int16')
    # X2.tofile(new_filename.replace('abf', 'dat'))

    # float32 by waiting
    scale = 1.
    X2 = np.reshape(np.array(X), (len(X), 1)).astype('float32')

    
    X2.tofile(new_filename.replace('abf', 'dat'))
    print('file exported as: ', new_filename)

    return {'facq':1./params['dt'], 'gain':scale}


def list_keys(args):
    """ list the keys of the """
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)
    KEYS = []
    for i in range(len(Block.segments[0].analogsignals)):
        print('key', i, '-->', Block.segments[0].analogsignals[i].name)
        KEYS.append(Block.segments[0].analogsignals[i].name)
    return KEYS


def show(args):
    
    keys = list_keys(args)
    
    Block = AxonIO(args.filename).read_block(lazy=False, cascade=True)

    fig, AX = plt.subplots(len(keys), figsize=(5,3*len(keys)))
    for i in range(len(keys)):
        if len(keys)>1:
            ax = AX[i]
        else:
            ax = AX
        ax.set_title(keys[i])
        ax.plot(Block.segments[0].analogsignals[i][:10000])
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

    parser.add_argument("filename")
    parser.add_argument('-nk', "--new_keys",help="new keys to be assigned", nargs='*')
    parser.add_argument("--std",help="std of the random values",\
                        type=float, default=10.)
    parser.add_argument("--tstart",help="starting time of export", type=float, default=0.)
    parser.add_argument("--tend",help="starting time of export", type=float, default=np.inf)
    parser.add_argument('-p', "--protocol", help="name of the protocol", default='Sample-Recording')
    parser.add_argument("-fc", "--force_continuous", help="force continuous", action="store_true")
    parser.add_argument("-s", "--show", help="show data with initial keys",
                        action="store_true")
    parser.add_argument("-lk", "--list_keys", help="list the available keys in data file",\
                        action="store_true")
    
    parser.add_argument("--args.filename", '-f', help="args.filename",type=str, default='data.npz')
    args = parser.parse_args()

    KEYS = list_keys(args)
    ikey = int(input('For a 1-D npy export, choose a key among those: '))
    print('You chose', KEYS[ikey], ' [...] export to .dat file')
    infos = translate(args, ikey, KEYS)
    
    # X2 = np.memmap(args.filename, dtype='float32', shape=(26339840, 1))
    # plt.plot(X2)
    # plt.show()
