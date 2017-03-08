import numpy as np
import json, os

def load_file(filename, zoom=[0,np.inf]):

    P = get_metadata(filename)
    nChannels = int(P['ChannelCount'])
    nEpisode = int(P['EpCount'])
    dt = 1e-3/float(P['f_acq'])

    if zoom[0]<0:
        tend = dt*int(os.path.getsize(filename)/nChannels/nEpisode/np.dtype(np.float32).itemsize)
        zoom[0] = tend+zoom[0]
        
    # loading the data file
    try:
        data = np.fromfile(filename, dtype=np.float32)
        if nEpisode==1:
            npoints = int(len(data)/nChannels)
            data = data.reshape(nChannels,npoints)
            t = np.arange(npoints)*dt
            return t[(t>=zoom[0]) & (t<=zoom[1])], np.array([data[i][(t>=zoom[0]) & (t<=zoom[1])] for i in range(nChannels)])
        else:
            npoints = int(len(data)/nEpisode/nChannels)
            data = data.reshape(nEpisode, nChannels, npoints)
            t = np.arange(npoints)*dt
            return t[(t>=zoom[0]) & (t<=zoom[1])], np.array([[data[j][i][(t>=zoom[0]) & (t<=zoom[1])] for j in range(nEpisode)] for i in range(nChannels)])
    except FileNotFoundError:
        print('File not Found !')
        return [[], []]

def get_metadata(filename, infos={}):
    with open(filename.replace('.bin', '.json'), 'r') as json_data:
        data= json_data.read().replace('\n', '').replace('\\', '\\\\')
        bd = json.loads(data)
    return {**bd,**infos}
    
if __name__ == '__main__':
    import sys
    import matplotlib.pylab as plt
    filename = sys.argv[-1]
    print(get_metadata(filename))
    t, data = load_file(filename, zoom=[-5.,np.inf])
    # for i in range(10):
    #     plt.plot(t, data[0][i])
    plt.plot(t, data[0])
    plt.show()
