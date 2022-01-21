import numpy as np
from sklearn.linear_model import LogisticRegression


def Logit_1D_example(data):
    
    X = data["data"][:, 3:]
    y = (data["target"] == 2).astype(np.int)

    fig, AX = mg.figure(axes=(1,2), top=6.)
    for C, ax in zip([1., 10**10], AX):
        log_reg = LogisticRegression(solver="liblinear", C=C)
        log_reg.fit(X, y)

        # plot proba
        X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
        y_proba = log_reg.predict_proba(X_new)
        ax.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="IV")
        ax.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not IV")
        ax.plot(X[y==0][::2], y[y==0][::2], "bs")
        ax.plot(X[y==1][::2], y[y==1][::2], "g^")
        ax.set_title('Regularization C=%.1e' % C)
        ax.legend()
        mg.set_plot(ax, xlabel='petal width', ylabel='proba')
    mg.show()

def Logit_2D_example(data):
    
    X = data["data"][:, (2, 3)]  # petal length, petal width
    y = data["target"]
    
    fig, AX = mg.figure(axes=(1,2), top=6.)
    for C, ax in zip([1., 100], AX):
        softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=C, random_state=42)
        softmax_reg.fit(X, y)


        x0, x1 = np.meshgrid(
                np.linspace(0, 8, 500).reshape(-1, 1),
                np.linspace(0, 3.5, 200).reshape(-1, 1),
            )
        X_new = np.c_[x0.ravel(), x1.ravel()]


        y_proba = softmax_reg.predict_proba(X_new)
        y_predict = softmax_reg.predict(X_new)

        zz1 = y_proba[:, 1].reshape(x0.shape)
        zz = y_predict.reshape(x0.shape)

        ax.plot(X[y==2, 0], X[y==2, 1], "g^", label="IVi")
        ax.plot(X[y==1, 0], X[y==1, 1], "bs", label="IVe")
        ax.plot(X[y==0, 0], X[y==0, 1], "yo", label="IS")

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

        ax.contourf(x0, x1, zz, cmap=custom_cmap)
        contour = ax.contour(x0, x1, zz1, cmap=mg.viridis)
        ax.clabel(contour, inline=1, fontsize=12)
        ax.legend(loc="center left", fontsize=14, prop={'size':'xx-small'})
        ax.axis([0, 7, 0, 3.5])
        ax.set_title('Regularization C=%.1e' % C)

        mg.set_plot(ax, ylabel='petal width', xlabel='petal length')
    mg.show()

if __name__=='__main__':

    import sys

    # visualization module
    sys.path.append('../..')
    from graphs.my_graph import graphs
    mg = graphs('screen')

    # load dataset
    from sklearn import datasets
    data = datasets.load_iris()

    if sys.argv[-1]=='1D':
        Logit_1D_example(data)
    elif sys.argv[-1]=='2D':
        Logit_2D_example(data)
    else:
        print("""
        ------------------------------------------
        Please choose one of the available method:
        '1D' : one dimensional regression
        '2D' : two dimensional regression
        ------------------------------------------
        and pass it as an argument to the script
        """)
        


    
