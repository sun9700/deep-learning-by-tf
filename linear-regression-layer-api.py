# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

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

# 散点图数据可视化
plt.scatter(features[:, 1], labels, 1)
plt.show()

# 定义模型
model = Sequential()
model.add(Dense(1, kernel_initializer='random_normal', bias_initializer='zeros'))

# 模型参数
learning_rate = 0.03
num_epochs = 3

# 训练过程
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate),
              loss='mse', metrics=['acc'])

# 2-训练模型
# batch_size 和epochs可调
batch_size = 10  # 随即(shuffle=True)读取batch_size个数据用于训练
for epoch in range(num_epochs):
    history = model.fit(features, labels, batch_size=batch_size, shuffle=True)
    print 'epoch ', epoch + 1, 'loss ', history.history['loss']
