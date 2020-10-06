import tensorflow as tf
import numpy as np
#
# M = np.array([[[2], [1], [2], [-1]], [[0], [-1], [3], [0]],
# [[2], [1], [-1], [4]], [[-2], [0], [-3], [4]]], dtype="float32").reshape([1, 4, 4, 1])
# # 一行有四列， 一列有一行， 一行又有四列
# filter_weight = tf.get_variable("weights", [2, 2, 1, 1], initializer=tf.constant_initializer([[-1, 4], [2, 1]]))
# biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(1))
# x = tf.placeholder('float32', [1, None, None, 1])
# conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
# add_bias = tf.nn.bias_add(conv, biases)
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init_op.run()
#     M_conv = sess.run(add_bias, feed_dict={x: M})
# print("After convolution:", M_conv)
# 加入池化层
M = np.array([[[-2], [2], [0], [3]], [[1], [2], [-1], [2]], [[0], [-1], [1], [0]]],dtype="float32").reshape(1, 3, 4, 1)
filter_weight = tf.get_variable("weights", [2, 2, 1, 1], initializer=tf.constant_initializer([[2, 0], [-1, 1]]))
biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(1))
x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
add_bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.max_pool(add_bias, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    init_op.run()
    M_conv = sess.run(add_bias, feed_dict={x: M})
    M_pool = sess.run(pool, feed_dict={x: M})
print("After convolution:", M_pool)