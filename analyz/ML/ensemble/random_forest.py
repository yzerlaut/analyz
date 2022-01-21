import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__=='__main__':

    # import sys, os

    # # visualization module
    # sys.path.append('../..')
    # from graphs.my_graph import graphs
    # mg = graphs('screen')

    # X, y = make_moons(n_samples=400, noise=0.30, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    
    # # Single Decision Tree Classifier
    # tree = DecisionTreeClassifier(max_depth=16)
    # tree.fit(X_train, y_train)

    # # Random Forest Classifier
    # forest_clf = RandomForestClassifier(max_leaf_nodes=16, 
    #                                     n_estimators=500, n_jobs=-1)
    # forest_clf.fit(X_train, y_train)
    

    # fig, AX = mg.figure(axes=(1,2))
    # for ax in AX:
    #     mg.scatter(X=[X_train[:,0][y_train==1],X_train[:,0][y_train==0]],
    #            Y=[X_train[:,1][y_train==1],X_train[:,1][y_train==0]],
    #            xlabel='x1', ylabel='x2', COLORS=[mg.b, mg.o],
    #            LABELS=['y=1', 'y=0'], ax=ax)


    # x1, x2 = np.meshgrid(np.linspace(X_train[:,0].min(), X_train[:,0].max(), 200),
    #                      np.linspace(X_train[:,1].min(), X_train[:,1].max(), 200))
    # y_pred_full = tree.predict(np.array([x1.flatten(), x2.flatten()]).T)
    # mg.twoD_plot(x1.flatten(), x2.flatten(), y_pred_full, alpha=0.3, ax=AX[0])

    # y_pred_full = forest_clf.predict(np.array([x1.flatten(), x2.flatten()]).T)
    # mg.twoD_plot(x1.flatten(), x2.flatten(), y_pred_full, alpha=0.3, ax=AX[1])

    # print(tree.)

