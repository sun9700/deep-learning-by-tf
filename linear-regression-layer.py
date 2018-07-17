# -*-coding:utf-8-*-
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

sess = tf.Session()
features = tf.random_normal(shape=(num_examples, num_inputs)).eval(session=sess)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels = labels.reshape(1000, 1)
labels += tf.random_normal(shape=labels.shape).eval(session=sess) * 0.01

# 散点图数据可视化
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


F = tf.placeholder(tf.float32, [None, num_inputs])
L = tf.placeholder(tf.float32, [None, 1])


def addlayer(inputs, in_size, out_size, activation_function=None):  # 构造线性回归框架
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    baise = tf.Variable(tf.zeros([1, out_size]))
    WX_PLUS_B = tf.matmul(inputs, Weight) + baise
    if activation_function is None:
        output = WX_PLUS_B
    else:
        output = activation_function(WX_PLUS_B)
    return output, Weight, baise


lr = 0.03
num_epochs = 10
l1, w_p, b_p = addlayer(F, 2, 1, activation_function=tf.nn.relu)
loss = tf.losses.mean_squared_error(L, l1)  # 计算误差
trainer = tf.train.AdamOptimizer(0.02).minimize(loss)

sess = tf.Session()  # 初始化
sess.run(tf.global_variables_initializer())

# 每次随即取十个数据训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        tmp = y.reshape(batch_size, 1)
        loss_, trainer_, pred_ = sess.run([loss, trainer, l1], feed_dict={F: X,
                                                                          L: tmp})
    print "epoch %d" % (epoch + 1), "\n", \
        "loss %f" % loss_.mean(), "\n", \
        "W:\n", sess.run(w_p), "\n", \
        "b:\n", sess.run(b_p), "\n"

# # 每次取全部数据进行训练
# for epoch in range(1000):
#     loss_, trainer_, pred_ = sess.run([loss, trainer, l1], feed_dict={F: features,
#                                                                       L: labels})
#     print "epoch %d" % (epoch + 1), "\n", \
#         "loss %f" % loss_.mean(), "\n", \
#         "W:\n", sess.run(w_p), "\n", \
#         "b:\n", sess.run(b_p), "\n"
