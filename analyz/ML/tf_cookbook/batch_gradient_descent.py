import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Loading dataset from sklearn !
from sklearn.datasets import fetch_california_housing


scaler = StandardScaler()
housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

tf.reset_default_graph()


X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

def fetch_batch(epoch, batch_index, batch_size):
    indices = np.random.randint(m, size=batch_size) 
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

# setting up the tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "logs/run-{}/".format(now)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # fetching the new batch of data
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            # 
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str, step)
            # running the training on the batch
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

