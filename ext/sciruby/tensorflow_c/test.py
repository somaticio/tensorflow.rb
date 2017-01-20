import tensorflow as tf
import numpy as np

x = tf.Variable(0)
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
assign_op = x.assign(1)
sess.run(assign_op)  # or `assign_op.op.run()`
print(x.eval())
print sess.graph_def
