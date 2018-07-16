# -*-coding:utf-8-*-
import random

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 1-生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

sess = tf.Session()
features = tf.random_normal(shape=(num_examples, num_inputs)).eval(session=sess)
F = tf.placeholder(tf.float32, [None, num_inputs])
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels = labels.reshape(1000, 1)
L = tf.placeholder(tf.float32, [None, 1])
labels += tf.random_normal(shape=labels.shape).eval(session=sess) * 0.01

# 散点图数据可视
plt.scatter(features[:, 1], labels, 1)
plt.show()

# 读取数据
batch_size = 10


# 每次随机读取十个数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的。
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        # take 函数根据索引返回对应元素。
        yield np.hstack(
            ((features[:, 0].take(j)).reshape(-1, 1), (features[:, 1].take(j)).reshape(-1, 1))), labels.take(j)


# 初始化模型参数
W = tf.Variable(tf.random_normal([num_inputs, 1]))
b = tf.Variable(tf.zeros([1]))


# 定义模型
def linreg(X, w, b):
    return tf.add(tf.matmul(X, W), b)


# 定义损失函数
def squared_loss(y_hat, y):
    return tf.reduce_mean(
        tf.square(y_hat - y))


# 定义优化算法
def optimizer(lr, loss):
    return tf.train.GradientDescentOptimizer(lr).minimize(loss)


# 初始化
init = tf.global_variables_initializer()

# 启动图计算
sess = tf.Session()

# Run the initializer
sess.run(init)

# 训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

l = loss(net(F, W, b), L)
trainer = optimizer(lr, l)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        tmp = y.reshape(batch_size, 1)
        sess.run(trainer, feed_dict={F: X,
                                     L: tmp})
        c = sess.run(l, feed_dict={F: X,
                                   L: tmp})
    print "epoch %d" % (epoch + 1), "\n", \
        "loss %f" % c.mean(), "\n", \
        "w=", sess.run(W), "true_w=", true_w, "\n", \
        "b=", sess.run(b), "true_b=", true_b
