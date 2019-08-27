from neo.io import ElphyIO
import numpy as np
import sys
import json

def get_analogsignals_continuous(filename):
    """
    simplified version, requires ElphyIO in neo==0.4, only on python 2 !
    """
    
    # loading the data file
    try:
        File = str(filename)
        reader = ElphyIO(filename=File)
        data = ElphyIO(filename).read_block(lazy=False, cascade=True)
        dt =  float(data.segments[0].analogsignals[0].sampling_period)

        t = np.array(data.segments[0].analogsignals[0].times, dtype='float32')
        VEC = [t]
        for j in range(len(data.segments[0].analogsignals)):
            VEC.append(np.array(data.segments[0].analogsignals[j], dtype='float32'))

        return t, VEC
    except Exception as e:
        print(e)
        return None

def get_analogsignals(filename, zoom=[0,np.inf]):
    
    # loading the data file
    try:
        File = str(filename)
        reader = ElphyIO(filename=File)
        data = ElphyIO(filename).read_block(lazy=False, cascade=True)
        dt =  float(data.segments[0].analogsignals[0].sampling_period)
        if zoom[0]<data.segments[0].analogsignals[0].t_start:
            zoom[0]=data.segments[0].analogsignals[0].t_start
        if zoom[1]>data.segments[-1].analogsignals[0].t_stop:
            zoom[1]=data.segments[-1].analogsignals[0].t_stop

        ### 
        ii = 0
        while float(data.segments[ii].analogsignals[0].t_start)<=zoom[0]:
            ii+=1
            
        tt = np.array(data.segments[ii-1].analogsignals[0].times, dtype='float32')
        cond = (tt>=zoom[0]) & (tt<=zoom[1])
        VEC = [tt[cond]]
        for j in range(1, len(data.segments[ii-1].analogsignals)+1):
            VEC.append(np.array(data.segments[ii-1].analogsignals[j-1], dtype='float32')[cond])
        ### 
        while ((float(data.segments[ii].analogsignals[0].t_start)<=zoom[1]) and (ii<len(data.segments))):
            print(ii)
            tt = np.array(data.segments[ii].analogsignals[0].times, dtype='float32')
            cond = (tt>=zoom[0]) & (tt<=zoom[1])
            print(cond)
            VEC[0] = np.concatenate([VEC[0],\
                np.array(data.segments[ii].analogsignals[0].times, dtype='float32')[cond]])
            for j in range(1, len(data.segments[ii].analogsignals)+1):
                VEC[j] = np.concatenate([VEC[j],\
                    np.array(data.segments[ii].analogsignals[j-1], dtype='float32')[cond]])
            print(len(VEC[0]), len(VEC[1]))
            ii+=1
        return VEC
    except Exception as e:
        print(e)
        return None
    

def get_metadata(filename):
    with file(filename) as fid:
        params = json.load(fid)
    return params

def save_as_npz(t, data, filename, filename2):

    npz = {'dt': t[1]-t[0]}
    for i in range(len(data)):
        npz['Channel_%i' % i] = data[i]
    np.savez(filename2, **npz)
    
if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    t, data = get_analogsignals_continuous(filename)
    save_as_npz(t, data, filename, sys.argv[2])

