import numpy as np
from scipy.stats import pearsonr

def linear_correlation(x, y,
                       N=100,
                       seed=1,
                       return_distrib=False):

    np.random.seed(seed)
    Correl_Coeff_Distribution = np.zeros(N)
    
    for i in range(N):
        Correl_Coeff_Distribution[i] = pearsonr(np.random.permutation(x), y)[0]

    c = pearsonr(x, y)[0]
    if c>0:
        pvalue = np.sum(Correl_Coeff_Distribution>c)/N
    else:
        pvalue = np.sum(Correl_Coeff_Distribution<c)/N
        
    if return_distrib:
        return c, pvalue, Correl_Coeff_Distribution
    else:
        return c, pvalue
        

if __name__ == '__main__':

    import sys
    sys.path.append('../../')
    from graphs.my_graph import *
    
    x = np.linspace(0, 1, 20)
    y = .5*np.random.randn(len(x))+x

    fig, ax = plt.subplots(figsize=(3,3))
    plt.subplots_adjust(left=.3, bottom=.3)
    ax.set_title('True Data, c='+str(np.round(pearsonr(x,y)[0],3)), fontsize=14)
    ax.plot(x, y, 'ko', ms=2)
    pol = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(pol, x), 'r--')
    set_plot(ax, xlabel='x-data', ylabel='y-data', num_xticks=3)

    fig2, AX = plt.subplots(1, 5, figsize=(10,3))
    plt.subplots_adjust(left=.1, bottom=.3, top=.8)
    plt.suptitle('(x-label) Shuffled Data', fontsize=14)
    for i, ax in enumerate(AX):
        x2 = np.random.permutation(x)
        ax.plot(x2, y, 'ko', ms=2)
        ax.set_title('Sample #'+str(i+1)+', c='+str(np.round(pearsonr(x2,y)[0],3)), fontsize=10)
        pol = np.polyfit(x2, y, 1)
        ax.plot(x, np.polyval(pol, x), 'r--')
        set_plot(ax, xlabel='x-data', yticks_labels=[], num_xticks=3)
    set_plot(AX[0], xlabel='x-data', ylabel='y-data', num_xticks=3)

    c, pvalue, Correl_Coeff_Distribution = linear_correlation(x, y, return_distrib=True, N=int(1e5), seed=2)
    print(pvalue)
    fig3, ax = plt.subplots(figsize=(3,3))
    ax.hist(Correl_Coeff_Distribution, bins=100)
    ax.plot([c,c], ax.get_ylim(), 'r-')
    set_plot(ax, xlabel='correlation-coef', ylabel='occurence')
    
    show()
    
