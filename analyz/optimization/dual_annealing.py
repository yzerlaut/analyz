from scipy.optimize import dual_annealing
import numpy as np

def run_dual_annealing(func_to_minimize, x0,
                       bounds=None, seed=1, maxiter=1000, verbose=False):

    if bounds is None:
        bounds = [None for x in x0]

    res = dual_annealing(func_to_minimize,
                         bounds=bounds, seed=seed, maxiter=maxiter)
    if verbose:
        print(res)
    return res.x

if __name__=='__main__':
    
    func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    for seed in np.arange(10):
        print('\n Seed %i' % seed)
        res = run_dual_annealing(func, np.zeros(10), bounds=[[-5.,5.] for i in range(10)],
                                 seed=seed, verbose=True)

