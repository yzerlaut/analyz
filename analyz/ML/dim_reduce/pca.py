from sklearn.decomposition import PCA as sklPCA
import numpy as np


def PCA(data,
        n_components=2, desired_var_explained=None):

    if desired_var_explained is not None:
        pca = sklPCA(n_components=desired_var_explained)
    else:
        pca = sklPCA(n_components=n_components)
        
    pca.fit_transform([data[key] for key in data])

    # for i in range(len(pca.explained_variance_ratio_)):
        
    
    return pca

if __name__=='__main__':

    from sklearn.datasets import load_breast_cancer
    import sys
    sys.path.append('../..')
    from graphs.my_graph import graphs
    from graphs.features_plot import *
    from graphs.cross_correl_plot import *
    
    mg = graphs()

    raw, data = load_breast_cancer(), {}
    for i, key in enumerate(raw['feature_names']):
        data[key] = np.log10(raw['data'][i])

    features_plot(mg, data, features=list(data.keys())[:6])
    cross_correl_plot(mg, data, features=list(data.keys())[:6])
    
    pca = PCA(data, n_components = len(data)) # desired_var_explained=0.9)
    mg.plot(100.*pca.explained_variance_ratio_, xlabel='component #', ylabel='% var. expl.')

    mg.show()

    

    
        
