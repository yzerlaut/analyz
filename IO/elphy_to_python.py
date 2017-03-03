from neo.io import ElphyIO
import numpy as np
import sys
import json

def get_analogsignals(filename):
    
    # loading the data file
    try:
        File = str(filename)
        reader = ElphyIO(filename=File)
        seg = reader.read_block(lazy=False, cascade=True)
        
    #     data = ElphyIO(filename).read_block(lazy=False, cascade=True)
    #     dt =  float(data.segments[0].analogsignals[0].sampling_period)
    #     if zoom[0]<data.segments[0].analogsignals[0].t_start:
    #         zoom[0]=data.segments[0].analogsignals[0].t_start
    #     if zoom[1]>data.segments[-1].analogsignals[0].t_stop:
    #         zoom[1]=data.segments[-1].analogsignals[0].t_stop

    #     ### 
    #     ii = 0
    #     while float(data.segments[ii].analogsignals[0].t_start)<=zoom[0]:
    #         ii+=1
    #     print(ii)
    #     tt = np.array(data.segments[ii-1].analogsignals[0].times, dtype='float32')
    #     cond = (tt>=zoom[0]) & (tt<=zoom[1])
    #     VEC = [tt[cond]]
    #     for j in range(1, len(data.segments[ii-1].analogsignals)+1):
    #         VEC.append(np.array(data.segments[ii-1].analogsignals[j-1], dtype='float32')[cond])
    #     ### 
    #     print(len(VEC[0]), len(VEC[1]))
    #     while ((float(data.segments[ii].analogsignals[0].t_start)<=zoom[1]) and (ii<len(data.segments))):
    #         print(ii)
    #         tt = np.array(data.segments[ii].analogsignals[0].times, dtype='float32')
    #         cond = (tt>=zoom[0]) & (tt<=zoom[1])
    #         print(cond)
    #         VEC[0] = np.concatenate([VEC[0],\
    #             np.array(data.segments[ii].analogsignals[0].times, dtype='float32')[cond]])
    #         for j in range(1, len(data.segments[ii].analogsignals)+1):
    #             VEC[j] = np.concatenate([VEC[j],\
    #                 np.array(data.segments[ii].analogsignals[j-1], dtype='float32')[cond]])
    #         print(len(VEC[0]), len(VEC[1]))
    #         ii+=1
    #     return VEC
    except FileNotFoundError:
        print('File not Found !')
        return [[], []]
    

def get_metadata(filename):
    with file(filename) as fid:
        params = json.load(fid)
    return params

if __name__ == '__main__':
    import sys
    filename = sys.argv[-1]
    data = get_analogsignals(filename)
    print(data[0])
