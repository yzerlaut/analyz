import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from linear import choose_regression_algorithm

def build_pipeline_for_polynomial_regression(degree,
                                             alpha=0.,
                                             regularization='Ridge'):
    """
    using the pipeline feature of sklearn
    """
    
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    std_scaler = StandardScaler()
    lin_reg = choose_regression_algorithm(alpha, regularization)
    
    polynomial_regression = Pipeline([
        ("poly_features", poly_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg)])
    
    return poly_features, std_scaler, lin_reg, polynomial_regression

    
def OLS_1d(x, y, degree,
                             alpha=0.,
                             regularization='Ridge'):
    """
    single variable linear regression using ordinary least square minimization
    f(x) = Pol(x, degree)+b
    where Pol is a polynomial expression of x (with no bias term)

    arguments:
    x -> 1d array
    y -> 1d array

    returns: dictionary with the prediction function and coefficients
    {'predict_func':predict_func, 'a':lin_reg.coef_[0], 'b':lin_reg.intercept_[0]}
    """

    
    
    poly_features, std_scaler,\
        lin_reg, polynomial_regression = build_pipeline_for_polynomial_regression(degree,
                                                                                  alpha=alpha,
                                                                                  regularization=regularization)
    polynomial_regression.fit(x.reshape(len(x),1), y)
    
    def predict_func(x):
        return polynomial_regression.predict(x.reshape(len(x),1))

    output = {}
    for deg in range(degree):
        output['a_x%i' % (deg+1)] = polynomial_regression['lin_reg'].coef_[deg]
        
    output['predict_func'] = predict_func
    return output


def OLS_Md(X, y, degree,
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
    if len(y)!=X.shape[0]:
        print(""" Problem with the dimensions of the input, they should be:
        X -> Md array of shape (Nsamples, Nfeatures)
        y -> 1d array of shape (Nsamples,)
        """)

    poly_features, std_scaler,\
        lin_reg, polynomial_regression = build_pipeline_for_polynomial_regression(degree,
                                                                                  alpha=alpha,
                                                                                  regularization=regularization)
    
    polynomial_regression.fit(X, y)
    
    def predict_func(X):
        return polynomial_regression.predict(X)

    output = {}
    
    # for deg in range(degree):
    #     output['a_x%i' % (deg+1)] = polynomial_regression['lin_reg'].coef_[deg]
        
    output['predict_func'] = predict_func
    return output



if __name__=='__main__':

    import sys

    # visualization module
    sys.path.append('../..')
    from graphs.my_graph import graphs
    mg = graphs('screen')
    
    if sys.argv[-1]=='1D':
        # generating random data
        x = np.linspace(0,1, 100)
        y = np.random.randn(len(x))+np.sin(2*np.pi*x)
        # --- plot
        fig, ax = mg.figure(top=3.)
        ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
        for d in range(1,5):
            mg.plot(x, OLS_1d(x, y, d)['predict_func'](x), ax=ax,
                    color=mg.viridis((d-1)/3.), lw=2, label='deg=%i'%d)
        ax.legend(prop={'size':'xx-small'})
        mg.show()
    elif sys.argv[-1]=='1D-RGL':
        # generating random data
        x = np.linspace(0,2, 100)
        y = np.random.randn(len(x))+2.*np.sin(2*np.pi*x)
        # --- plot
        fig, AX = mg.figure(axes=(1,2), top=3.)
        for ax, method in zip(AX, ['Ridge', 'Lasso']):
            ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
            for d, alpha in enumerate([0.,0.5,.9]):
                mg.plot(x, OLS_1d(x, y, 3, alpha=alpha, regularization=method)['predict_func'](x), ax=ax,
                        color=mg.viridis(d/2.), lw=2, label='$\\alpha$=%.1f'%alpha)
            ax.legend(prop={'size':'xx-small'})
        mg.show()
    elif sys.argv[-1]=='MD':
        # generating random data
        nsamples = 200 # and we add three features:
        # uncorrelated variables:
        X = np.array([np.linspace(0,1,nsamples) for i in range(3)]).T
        # X = np.array([1+np.random.randn(nsamples) for i in range(3)]).T
        y = np.random.randn(nsamples)\
            -np.sin(2*np.pi*X.T[0])\
            -np.sin(2*np.pi*X.T[1])\
            +np.sin(2*np.pi*X.T[2])-12.

        fig, AX = mg.figure(axes=(1,X.shape[1]), top=8.)
        for ax, x, xl in zip(AX, X.T, ['x1', 'x2', 'x3']):
            ax.plot(x, y, 'o', ms=3, label='data', color=mg.blue)
        for d in range(1,5):
            ml_output = OLS_Md(X, y, d)
            for ax, x, xl in zip(AX, X.T, ['x1', 'x2', 'x3']):
                mg.plot(x, ml_output['predict_func'](X), ax=ax,
                        color=mg.viridis((d-1)/3.), lw=2, label='deg=%i'%d)
                mg.set_plot(ax, xlabel=xl)
        AX[0].legend(prop={'size':'xx-small'})
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
        
