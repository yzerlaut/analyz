import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
dnn_clf.train(input_fn=input_fn)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)


y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)

# print showing results
(_, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
import sys
sys.path.append('../..')
from graphs.my_graph import graphs
mg = graphs()
fig, AX = mg.figure(axes=(1,10))
for i in range(10):
    mg.image(X_test[i], ax=AX[i])
    mg.annotate(AX[i], 'predicted: %i' % int(y_pred[i]['class_ids']), (.1, -.3))
mg.show()
fig.savefig('mnist_pred.png')    

# printing performance summary
print(eval_results)


