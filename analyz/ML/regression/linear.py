import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def choose_regression_algorithm(alpha, regularization):
    """
    find the right method 
    """
    if alpha==0.:
        lin_reg = LinearRegression()
    else:
        if regularization=='Ridge':
            lin_reg = Ridge(alpha)
        elif regularization=='Lasso':
            lin_reg = Lasso(alpha)
        else:
            print("regularization type '%s' not understood" % regularization)
            lin_reg = None

    return lin_reg
    

def OLS_1d(x, y,
           alpha=0.,
           regularization='Ridge'):
    """
    single variable linear regression using ordinary least square minimization
    f(x) = a*x+b

    arguments:
    x -> 1d array
    y -> 1d array

    returns: dictionary with the prediction function and coefficients
    {'predict_func':predict_func, 'a':lin_reg.coef_[0], 'b':lin_reg.intercept_[0]}
    """

    lin_reg = choose_regression_algorithm(alpha, regularization)
    lin_reg.fit(x.reshape(len(x),1), y.reshape(len(x),1))

    def predict_func(x):
        return lin_reg.predict(x.reshape(len(x),1))
    
    return {'predict_func':predict_func, 'a':lin_reg.coef_[0], 'b':lin_reg.intercept_[0]}


def OLS_Md(X, Y,
           alpha=0.,
           regularization='Ridge'):
    """
    multidimensional linear regression using ordinary least square minimization
    f(X) = A*X+B

    arguments:
    X -> Md array of shape (Nsamples, Nfeatures)
    y -> 1d array of shape (Nsamples,)

    returns: dictionary with the prediction function and coefficients
    {'predict_func':predict_func, 'a':lin_reg.coef_[0], 'b':lin_reg.intercept_[0]}
    """
    if len(Y)!=X.shape[0]:
        print(""" Problem with the dimensions of the input, they should be:
        X -> Md array of shape (Nsamples, Nfeatures)
        y -> 1d array of shape (Nsamples,)
        """)

    lin_reg = choose_regression_algorithm(alpha, regularization)
    lin_reg.fit(X, Y)

    return {'predict_func':lin_reg.predict, 'A':lin_reg.coef_, 'B':lin_reg.intercept_}

##############################################
#### Visualization functions #################
##############################################

def ols_1D_viz(graph, x, y, ml_output):
    fig, ax = mg.figure(top=3.)
    ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
    mg.plot(x, ml_output['predict_func'](x), ax=ax, label='f(x)')
    ax.set_title('f(x)=a*x+b, a=%.2f, b=%.2f' % (ml_output['a'], ml_output['b']))
    return fig


def ols_MD_viz(graph, x, y, ml_output):
    fig, AX = mg.figure(axes=(1,x.shape[1]), top=8.)
    for ax, x, xl, a in zip(AX, X.T, ['x1', 'x2', 'x3'], ml_output['A']):
        ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
        ax.plot(x, ml_output['predict_func'](X), 'o', color=mg.red, label='prediction')
        ax.plot(x, ml_output['B']+a*x, c='k', ls='--')
        mg.set_plot(ax, xlabel=xl)
        ax.set_title('$a_{%s}$=%.2f' % (xl, a))
    AX[0].legend()
    return fig

if __name__=='__main__':

    import sys

    # visualization module
    sys.path.append('../..')
    from graphs.my_graph import graphs
    mg = graphs('screen')
    
    if sys.argv[-1]=='1D':
        # generating random data
        x = np.linspace(0,1, 100)
        y = np.random.randn(len(x))+1.23*x-0.24
        # --- plot
        fig = ols_1D_viz(mg, x, y, OLS_1d(x, y))
        mg.annotate(fig, '1.23*x-0.24', (.6,.2), color=mg.blue)
        mg.show()
    elif sys.argv[-1]=='1D-RGL':
        # generating random data
        x = np.linspace(0,2, 100)
        y = np.random.randn(len(x))+2.*np.sin(2*np.pi*x)
        # --- plot
        fig, AX = mg.figure(axes=(1,2), top=5.)
        for ax, method in zip(AX, ['Ridge', 'Lasso']):
            ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
            for d, alpha in enumerate([0., 0.2, 20.]):
                mg.plot(x, OLS_1d(x, y, alpha=alpha, regularization=method)['predict_func'](x), ax=ax,
                        color=mg.viridis(d/2.), lw=2, label='$\\alpha$=%.1f'%alpha)
                ax.set_title(method)
            ax.legend(prop={'size':'xx-small'})
        mg.show()
    elif sys.argv[-1]=='MD':
        # generating random data
        nsamples = 100 # and we add three features:
        # uncorrelated variables:
        X = np.random.randn(nsamples,3)
        y = np.random.randn(nsamples)-0.4*X.T[0]-0.5*X.T[1]+3.*X.T[2]-12.
        fig = ols_MD_viz(mg, X, y, OLS_Md(X, y))
        mg.annotate(fig, '-0.4*x1-0.5*x2+3.x3', (.55,.18), color=mg.blue, ha='center')
        # correlated variables:
        X = np.array([np.linspace(0,1,nsamples), np.linspace(-4,3,nsamples), np.linspace(-4,3,nsamples)]).T
        y = np.random.randn(nsamples)-0.4*X.T[0]-0.5*X.T[1]+3.*X.T[2]-12.
        fig = ols_MD_viz(mg, X, y, OLS_Md(X, y))
        # --- plot
        mg.annotate(fig, '-0.4*x1-0.5*x2+3.x3', (.55,.18), color=mg.blue, ha='center')
        mg.show()
    else:
        print("""
        ------------------------------------------
        Please choose one of the available method:
        '1D' : one dimensional regression
        '1D-REG' : one dimensional regression with regularization
        'MD' : multidimensional linear regression
        ------------------------------------------
        and pass it as an argument to the script
        """)
        
