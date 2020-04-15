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

    from datavyz import ges

    raw, data = load_breast_cancer(), {}
    for i, key in enumerate(raw['feature_names']):
        data[key] = np.log10(raw['data'][i])

    ges.features_plot(data, features=list(data.keys())[:6])
    ges.cross_correl_plot(data, features=list(data.keys())[:6])
    
    pca = PCA(data, n_components = len(data)) # desired_var_explained=0.9)
    
    ges.plot(100.*pca.explained_variance_ratio_,
              m='o', ms=4, xlabel='component #', ylabel='% var. expl.')

    ges.parallel_plot(pca.components_,
                      fig_args=dict(figsize=(3,1)))

    ges.show()

    

    
        
