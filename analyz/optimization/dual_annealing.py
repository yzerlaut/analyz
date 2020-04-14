from scipy.optimize import dual_annealing
import numpy as np

def run_dual_annealing(func_to_minimize, x0=None,
                       bounds=None,
                       seed=1,
                       maxiter=100, maxfun=200,
                       no_local_search=False,
                       verbose=False):

    if bounds is None:
        bounds = [None for x in x0]

    res = dual_annealing(func_to_minimize, x0=x0,
                         bounds=bounds, seed=seed,
                         maxiter=maxiter, maxfun=maxfun,
                         no_local_search=no_local_search)
    if verbose:
        print(res)
        
    return res

if __name__=='__main__':
    
    func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    for seed in np.arange(10):
        print('\n Seed %i' % seed)
        res = run_dual_annealing(func, np.zeros(10), bounds=[[-5.,5.] for i in range(10)],
                                 seed=seed, verbose=True)

