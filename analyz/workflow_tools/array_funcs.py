# Filename: array_funcs.py
import numpy as np

def find_coincident_duplicates_in_two_arrays(array1, array2, with_ids=False):

    X = np.array([array1, array2])
    isort = np.lexsort(X, axis=0)
    Id = X.T[isort]
    
    iduplic = [np.diagonal(Id[jj]==[Id[jj-1]])[0] for jj in range(1, Id.shape[0])]
    output =[]
    for ii in range(len(iduplic)):
        if iduplic[ii]:
            output.append(Id[ii])

    if with_ids:
        return output, np.arange(Id.shape[0])[iduplic]
    else:
        return output

if __name__=='__main__':

    array1 = np.array([1,2,2,4,5,5])
    array2 = np.array([1,1,1,2,2,2])
    print(find_coincident_duplicates_in_two_arrays(array1, array2, with_ids=True))
