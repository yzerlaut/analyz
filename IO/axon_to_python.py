from neo.io import AxonIO
import numpy as np
import os

def load_file(filename, zoom=[0,np.inf]):

    # loading the data file
    try:
        data = AxonIO(filename).read_block(lazy=False, cascade=True)
        dt =  float(data.segments[0].analogsignals[0].sampling_period)
        if zoom[0]<data.segments[0].analogsignals[0].t_start:
            zoom[0]=data.segments[0].analogsignals[0].t_start
        if zoom[1]>data.segments[-1].analogsignals[0].t_stop:
            zoom[1]=data.segments[-1].analogsignals[0].t_stop
        ### 
        ii = 0
        while (ii<len(data.segments)) and (float(data.segments[min(ii,len(data.segments)-1)].analogsignals[0].t_start)<=zoom[0]):
            ii+=1
        tt = np.array(data.segments[ii-1].analogsignals[0].times)
        cond = (tt>=zoom[0]) & (tt<=zoom[1])
        VEC = [tt[cond]]
        for j in range(1, len(data.segments[ii-1].analogsignals)+1):
            VEC.append(np.array(data.segments[ii-1].analogsignals[j-1])[cond])
        ### 
        while (ii<len(data.segments)) and ((float(data.segments[min(ii,len(data.segments)-1)].analogsignals[0].t_start)<=zoom[1])):
            tt = np.array(data.segments[ii].analogsignals[0].times)
            cond = (tt>=zoom[0]) & (tt<=zoom[1])
            VEC[0] = np.concatenate([VEC[0],\
                np.array(data.segments[ii].analogsignals[0].times)[cond]])
            for j in range(1, len(data.segments[ii].analogsignals)+1):
                VEC[j] = np.concatenate([VEC[j],\
                    np.array(data.segments[ii].analogsignals[j-1])[cond]])
            ii+=1
        return VEC[0], VEC[1:]
    except FileNotFoundError:
        print('File not Found !')
        return [[], []]

def get_protocol_name(filename):
    fn = filename.split(os.path.sep)[-1] # only the filename without path
    protocol = '' # empty by default
    if len(fn.split('_'))>0:
        fn2 = fn.split('_')
        for ss in fn2[3:-1]:
            protocol+=ss+'_'
        protocol += fn2[-1].split('.')[0] # removing extension
    return protocol

def get_metadata(filename, infos={}):
    protocol = get_protocol_name(filename)
    if protocol!='':
        bd = {'main_protocol':'classic_electrophy',
                'protocol':protocol,
                'clamp_index':2}
    else:
        bd = {'main_protocol':'spont-act-sampling', 'clamp_index':1}
    return {**bd,**infos}



if __name__ == '__main__':
    import sys
    import matplotlib.pylab as plt
    filename = sys.argv[-1]
    # AxonIO(filename).read_block(lazy=False, cascade=True)
    print(get_metadata(filename))
    t, data = load_file(filename, zoom=[-5.,np.inf])
    # for i in range(10):
    #     plt.plot(t, data[0][i])
    plt.plot(t, data[0])
    plt.show()
