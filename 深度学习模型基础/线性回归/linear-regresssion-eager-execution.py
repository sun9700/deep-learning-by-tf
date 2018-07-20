# -*-coding:utf-8-*-
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# 开启eager API
tfe.enable_eager_execution()

# 生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

tmp_list1 = [np.random.normal() for i in range(num_examples)]
tmp_list2 = [np.random.normal() for i in range(num_examples)]
features = np.vstack((tmp_list1, tmp_list2)).transpose()

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels = labels + np.random.normal(scale=0.01)
labels = labels.reshape(1000, 1)

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


# 权重和偏置
W = tfe.Variable(np.array([np.random.randn() for i in range(2)]).reshape(2, 1))
b = tfe.Variable(np.array(np.random.randn()))


# 线性回归公式函数(Wx + b)
def linear_regression(inputs):
    return tf.add(tf.matmul(inputs
                            , W), b)


# 均方误差函数，计算损失
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * num_examples)


# 参数
learning_rate = 0.01
display_step = 100
num_epochs_random_data = 3
num_epochs_entire_data = 3000
# 随机梯度下降法作为优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 计算梯度
grad = tfe.implicit_gradients(mean_square_fn)

# 优化之前，初始化损失函数
print("Initial cost= {:.5f}".format(
    mean_square_fn(linear_regression, features, labels)),
      "W=", W.numpy(), "b=", b.numpy())

# # 每次选取十个数据训练,尚未完成，有问题
# for step in range(num_epochs_random_data):
#     for X, y in data_iter(batch_size, features, labels):
#         optimizer.apply_gradients(grad(linear_regression, X, y))
#     print "Epoch:", '%d' % (step + 1), \
#         "\n", "cost=", "{:.5f}".format(mean_square_fn(linear_regression, features, labels)), \
#         "\n", "W=", W.numpy(), "\n", \
#         "b=", b.numpy()

# 每次选取全部数据训练
for step in range(num_epochs_entire_data):
    optimizer.apply_gradients(grad(linear_regression, features, labels))
    if (step + 1) % display_step == 0 or step == 0:
        print "Epoch:", '%d' % (step + 1), \
            "\n", "cost=", "{:.5f}".format(mean_square_fn(linear_regression, features, labels)), \
            "\n", "W=", W.numpy(), "\n", \
            "b=", b.numpy()
