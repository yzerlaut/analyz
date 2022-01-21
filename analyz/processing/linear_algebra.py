import numpy as np

def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca    


def find_ellipse_props_of_binary_image_from_PCA(x, y, img):

    mu = np.mean(x[img==1]), np.mean(y[img==1])
    xdist = x[img==1].flatten()-mu[0]
    ydist = y[img==1].flatten()-mu[1]

    xydist = np.concatenate([xdist[:,np.newaxis],
                             ydist[:,np.newaxis]],
                            axis=1)

    
    C = np.dot(xydist.T, xydist) / (xydist.shape[0]-1)
    
    EVALs, EVECS = np.linalg.eig(C)
    # eigen_vecs = []
    eigen_vals, eigen_vec_angles, eigen_vec_stds = [np.zeros(xydist.shape[1]) for i in range(3)]
    for e, i in enumerate(np.argsort(EVALs)[::-1]):
        eigen_vals[e] = EVALs[i]
        # eigen_vecs.append(EVECS[i])
        eigen_vec_angles[e] = (-np.arctan(EVECS[i][1]/(1e-5+EVECS[i][0]))+np.pi)%(np.pi)
        eigen_vec_stds[e] = 4*np.sqrt(EVALs[i])

        
    return mu, eigen_vec_stds, eigen_vec_angles
    

if __name__=='__main__':

    
    import matplotlib.pyplot as plt

    def inside_ellipse_cond(X, Y, xc, yc, sx, sy, alpha):
        return ( ( (X-xc)*np.cos(alpha) + (Y-yc)*np.sin(alpha) )**2 / (sx/2.)**2 +\
                 ( (X-xc)*np.sin(alpha) - (Y-yc)*np.cos(alpha) )**2 / (sy/2.)**2 ) < 1


    def ellipse_coords(xc, yc, sx, sy, alpha, n=100):
        t = np.linspace(0, 2*np.pi, n)
        return xc+sx/2.*np.cos(t)*np.cos(alpha)-sy/2.*np.sin(t)*np.sin(alpha),\
            yc+sx/2.*np.cos(t)*np.sin(alpha)+sy/2.*np.sin(t)*np.cos(alpha)
    

    # BUILDING A TEST IMAGE
    xc, yc, std1, std2, angle = 8, 12, 8, 15, -np.pi/10.
    print('before: std1=%.1f, std2=%.1f, angle=%.0f' % (std1, std2, 180./np.pi*angle))
    x, y = np.meshgrid(np.linspace(0, 20, 100), np.linspace(0, 20, 100))
    cond = inside_ellipse_cond(x, y, xc, yc, std1, std2, angle)
    img = np.zeros(x.shape)
    img[cond]= 1


    import time
    tic = time.time()
    mu, stds, angles = find_ellipse_props_of_binary_image_from_PCA(x, y, img)
    print('%.1f ms' % (1e3*(time.time()-tic)))
    plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
    print(*mu, *stds, 180/np.pi*angles[0])
    plt.plot(*ellipse_coords(*mu, *stds, angles[0]), 'r')
    
    plt.show()
