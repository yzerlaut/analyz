import numpy as np

def gaussian(x, mu, sigma):
    if sigma<=0:
        # print('the standard deviation has to be strictly positive')
        return 0*x
    else:
        return 1./np.sqrt(2*np.pi)/sigma*np.exp(-((x-mu)/sigma)**2)


def gaussian_2d(x, y,
                mu=(0,0),
                sigma=(1,1)):
    
    xcomp = 1./np.sqrt(2*np.pi)/sigma[0]*np.exp(-((x-mu[0])/sigma[0])**2)
    ycomp = 1./np.sqrt(2*np.pi)/sigma[1]*np.exp(-((y-mu[1])/sigma[1])**2)
    
    return xcomp*ycomp
    

if __name__=='__main__':


    from datavyz.main import graph_env
    ge = graph_env()

    x, y = np.meshgrid(np.linspace(-2,2),
                       np.linspace(-2,2))
    
    z = gaussian_2d(x, y,
                    mu=(0,0),
                    sigma=(1,1))
    
    ge.twoD_plot(x.flatten(), y.flatten(), z.flatten())
    
    ge.show()
