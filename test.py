import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.add(x, x)

with tf.Session() as sess:
  print sess.graph_def
  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
