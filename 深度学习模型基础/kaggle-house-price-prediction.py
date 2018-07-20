# -*-coding:utf-8-*-

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses

# 使用pandas读取训练数据和测试数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
all_features = pd.concat \
    ((train_data.loc[:, 'MSSubClass':'SaleCondition'],
      test_data.loc[:, 'MSSubClass':'SaleCondition']))

# print train_data.shape, test_data.shape

# 预处理训练数据，标准化特征
numeric_features = all_features.dtypes[all_features.dtypes
                                       != 'object'].index
all_features[numeric_features] = \
    all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.fillna(all_features.mean())

num_examples = train_data.shape[0]
train_features = all_features[:num_examples].values
train_labels = train_data.SalePrice.values
num_features_dim = train_features.shape[1]

# 定义模型
model = Sequential()
# 单层神经网络，线性模型
# glorot_normal-keras中的Xavier正态分布初始化
model.add(Dense(1, kernel_initializer=keras.initializers.glorot_normal(seed=None)))

# 训练过程
batch_size = 64
num_epochs = 800
lr = 5
# 1-cost function:msle;optimizer='Adam'
model.compile(optimizer=tf.train.AdamOptimizer(lr),
              loss=losses.mean_squared_logarithmic_error,
              metrics=['acc'])
# 2-训练模型
for epoch in range(num_epochs):
    history = model.fit(train_features, train_labels, batch_size=batch_size, shuffle=True)
    print 'epoch ', epoch + 1, 'loss ', history.history['loss']
# 预测过程
# 测试数据
test_features = all_features[num_examples:].values
# 预测
y = model.predict(test_features)

# 使用pandas生成csv
# ...
