#coding=utf-8
import tensorflow as tf
import numpy as np
# day1 27.7
# （default graph)
# print(np.__version__)
# print("Tensorflow-version:{}".format(tf.__version__))
# a = tf.constant([1.0, 2.0], name='a')
# b = tf.constant([3.0, 4.0], name='b')
# result = a + b
# print(a.graph is tf.get_default_graph())
# print(b.graph is tf.get_default_graph())

# graph的初体验
# g1 = tf.Graph()
# with g1.as_default():
#     a = tf.get_variable("a", [2], initializer=tf.ones_initializer())
#     b = tf.get_variable("b", [2], initializer=tf.zeros_initializer())
# g2 = tf.Graph()
# with g2.as_default():
#     a = tf.get_variable("a", [2], initializer=tf.zeros_initializer())
#     b = tf.get_variable("b", [2], initializer=tf.ones_initializer())
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     # 初始化计算图中的所有变量
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("a")))
#         print(sess.run(tf.get_variable("b")))
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     # 初始化计算图中的所有变量
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("a")))
#         print(sess.run(tf.get_variable("b")))

# placeholder(动态输入):解决数据大时需要定义多个note的问题

# use placeholder to define a place
# a = tf.placeholder(tf.float32, shape={2}, name="input")
# b = tf.placeholder(tf.float32, shape={2}, name="input")
# result = a + b
# with tf.Session() as sess:
#     sess.run(result, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]})
#     print(result)
# a = tf.placeholder(tf.float32, shape=(2), name="input")
# b = tf.placeholder(tf.float32, shape=(4, 2), name="input")
# result = a + b
# with tf.Session() as sess:
#     print(sess.run(result, feed_dict={a: [1.0, 2.0], b: [[2.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]}))
#     print(result)

# tensorflow 变量
# random_normal 创建一个3*4的矩阵， 标准差为1，mean=0[0为默认值]（mean表示均值）,其属于正太分布
# mean 即μ stddev即α
# weights = tf.Variable(tf.random_normal([3, 4], stddev=1))
# weight = tf.zeros([2, 3], tf.int32)
# biases = tf.Variable(tf.zeros([3]))
# # 通过已赋值的变量赋值新的变量
# b1 = tf.Variable(biases.initial_value*3)
# # 矩阵的简单运算
# x = tf.constant([[1.0, 2.0]])
# # seed 可以控制随机数每次出现都一样
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(y))
#     print(w1)

# 关于变量空间的理解

# with tf.variable_scope("one"):
# #     a = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
# # with tf.variable_scope("one", reuse=True):
# #     a2 = tf.get_variable("a", [1])
# #     print(a.name, a2.name)
# # init_op = tf.global_variables_initializer()
# # with tf.Session() as sess:
# #     sess.run(init_op)
# #     print(sess.run(a))

# 变量空间的嵌套
# 变量空间外部创建变量
# a = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
# # 变量空间one中：
# with tf.variable_scope("one"):
#     a1 = tf.get_variable('a', [1], initializer=tf.constant_initializer(1.0))
#     print(a1.name)
# with tf.variable_scope("one"):
#     with tf.variable_scope("two"):
#         a2 = tf.get_variable('a', [1])
#         print(a2.name)
#     b = tf.get_variable('b', [1])
# with tf.variable_scope("", reuse=True):
#     a3 = tf.get_variable("one/two/a", [1])
#     print(a3)
#     print(a3 == a2)

# day2 28.7
# 前馈神经网络的简单模拟
# x为输入数据
# x = tf.constant([0.9, 0.85], shape=[1, 2])
# w1w2为权值，b1b2为偏值
# w1 = tf.constant([[0.2, 0.1, 0.3], [0.2, 0.4, 0.3]], shape=[2, 3], name="w1")
# w2 = tf.constant([[0.2], [0.5], [0.25]], shape=[3, 1], name="w2")
# b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name='b1')
# b2 = tf.constant([-0.3], shape=[1], name='b2')
# init_op = tf.global_variables_initializer()
# a = tf.matmul(x, w1) + b1
# y = tf.matmul(a, w2) + b2
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(y))
# 权值和偏值一般情况下难以预测，此情况下的初始化
# x = tf.constant([0.9, 0.85], shape=[1, 2])
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")
# b1 = tf.Variable(tf.zeros([1, 3]))
# b2 = tf.Variable(tf.ones([1]))
# init_op = tf.global_variables_initializer()
# a = tf.matmul(x, w1) + b1
# y = tf.matmul(a, w2) + b2
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(y))
# 8.8
# weights = tf.constant([[1.0, 2.0], [-3.0, -4.0]])
# regularizer_l2 = tf.contrib.layers.l2_regularizer(.5)
# regularizer_l1 = tf.contrib.layers.l1_regularizer(.5)
# with tf.Session() as sess:
#     print(sess.run(regularizer_l1(weights)))
#     print(sess.run(regularizer_l2(weights)))

# # 深度学习的一次模拟
# training_step = 30000
# data = []
# label = []
# for i in range(200):
#     x1 = np.random.uniform(-1, 1)
#     x2 = np.random.uniform(0, 2)
#     # 圈内添加0，圈外添加1
#     if x1**2 + x2**2 <= 1:
#         data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
#         label.append(0)
#     else:
#         data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
#         label.append(1)
# # hstack函数将输入的data排成一行，返回一个数组。 reshape一个个参数为-1时可以反转数组，2表示2个为一组
# # 这个-1可以琢磨琢磨
# data = np.hstack(data).reshape(-1, 2)
# label = np.hstack(label).reshape(-1, 1)
#
#
# # 定义向前传播的隐层
# def hidden_layer(input_tensor, weight1, bias1, weight2, bias2, weight3, bias3):
#     layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1)+bias1)
#     layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
#     return tf.matmul(layer2, weight3) + bias3
#
#
# x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')
# # 定义权值参数和偏值参数
# # truncated_normal 表示截断的正太分布，即差值大于两个stddev，则重新生成
# weight1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
# bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
# weight2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
# bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
# weight3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
# bias3 = tf.Variable(tf.constant(0.1, shape=[1]))
#
# # len()计算data数组的长度, 其长度为200
# sample_size = len(data)
# # 得到隐藏层向前传播结果
# y = hidden_layer(x, weight1, bias1, weight2, bias2, weight3, bias3)
# # 自定义损失函数
# error_loss = tf.reduce_sum(tf.pow(y_-y, 2)) / sample_size
# tf.add_to_collection("losses", error_loss)
#
# # 在权重参数上实现L2正则
# regularizer = tf.contrib.layers.l2_regularizer(0.01)
# regularization = regularizer(weight1) + regularizer(weight2) + regularizer(weight3)
# tf.add_to_collection("losses", regularization)
# # 加和运算add_n get_collection得到指定集合的所有元素
# loss = tf.add_n(tf.get_collection("losses"))
# # 定义一个优化器，学习率固定为0.01
# train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for i in range(training_step):
#         sess.run(train_op, feed_dict={x: data, y_: label})
#         # 训练30000轮，但每隔2000轮就输出一次loss值
#         if i % 2000 == 0:
#             loss_value = sess.run(loss, feed_dict={x: data, y_: label})
#             print("After %d steps, mse_loss: %f" % (i, loss_value))
# 8.9 8.16
# mnist 实现
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("F:/pycharm/tensorflow/datasets/mnist/MNIST_data/", one_hot=True)
batch_size = 100
learning_rate = 0.8
learning_rate_decay = 0.999  # 衰减率
max_steps = 30000
training_step = tf.Variable(0, trainable=False)


def hidden_layer(input_tensor, weights1, biases1, weights2, biases2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# [NONE, 784]MEANS 784列，行未知
x = tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-output")

weights1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[500]))
weights2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[10]))
# 前馈神经网络得到输出
y = hidden_layer(x, weights1, biases1, weights2, biases2, 'y')
# 初始化一个滑动平均类
averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)
averages_op = averages_class.apply(tf.trainable_variables())
average_y = hidden_layer(x, averages_class.average(weights1),
                         averages_class.average(biases1),
                         averages_class.average(weights2),
                         averages_class.average(biases2), "average-y")
# argmax函数
# argmax(input, axis) axis1代表取input每一行的最大值，返回其索引；0代表列
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
regularizer = tf.contrib.layers.l2_regularizer(0.0001)  # L2正则损失函数
regularization = regularizer(weights1) + regularizer(weights2)
loss = regularization + tf.reduce_mean(cross_entropy)
# 指数衰减法更改学习率
learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples/batch_size,
                                           learning_rate_decay)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)
# 跟新滑动平均值和参数？？？？？
with tf.control_dependencies([train_step, averages_op]):
    train_op = tf.no_op(name='train')
crorent_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
# print(crorent_prediction)
# cast 为转换函数，转换数值类型
accuracy = tf.reduce_mean(tf.cast(crorent_prediction, tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # prepare for the validation_data
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # prepare for the test_data
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):
        if i % 1000 == 0:
            # 计算滑动平均模型在验证数据集上的结果
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training step(s),validation accuracy""using average model is %g%%" %
                  (i, validate_accuracy*100))
        xs, ys = mnist.train.next_batch(batch_size=100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average""model is %g%%" % (i, test_accuracy*100))



