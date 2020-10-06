import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data as cd

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "F:/pycharm/tensorflow/datasets/cifar-10-binary/cifar-10-batches-bin"


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection(name="losses", value=weights_loss)
    return var


# 训练数据进行数据增强， 测试数据不数据增强
image_train, labels_train = cd.input(data_dir=data_dir, batch_size=batch_size, distorted=True)
image_test, labels_test = cd.input(data_dir=data_dir, batch_size=batch_size, distorted=None)
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 第一个卷积核
# 5*5卷积核， 3个深度通道，共64个卷积核（输出深度为64)【一个卷积核对输入进行一次完整的卷积】
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第二个卷积核
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 拉直数据
reshape = tf.reshape(pool2, [batch_size, -1])
# reshape[0]为batch_size;[1]为拉直的pool2的数据长度， [，-1]表示将数据拉直成一维结构
dim = reshape.get_shape()[1].value

# 第一个全连接层
# w1避免参数过多导致的过拟合现象
weight1 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_relu1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)


# matmul 为矩阵相乘
# multilpy 为矩阵对应元素相乘



# 第二个全连接
weight2 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_relu2 = tf.nn.relu(tf.matmul(fc_relu1, weight2) + fc_bias2)

# 第三个全连接
weight3 = variable_with_weight_loss([192, 10], stddev=1 / 192.0, w1=0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
result = tf.add(tf.matmul(fc_relu2, weight3), fc_bias3)

# 计算损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss  # 正则损失居然在这里加上？
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 输出分类准确率最高的时的数值
top_k_op = tf.nn.in_top_k(result, y_, 1)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 开启多线程
    tf.train.start_queue_runners()

    # 一次训练一个batch的数据量
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([image_train, labels_train])
        # 训练开始， 并return loss
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time

        # 每100个batch输出一次时间相关的信息
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d, loss = %.2f(%.1f examples/sec; %.3fsec/batch)" %
                  (step, loss_value, examples_per_sec, sec_per_batch))

    # 测试





