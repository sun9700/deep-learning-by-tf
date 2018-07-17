# -*-coding:utf-8-*-

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

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
model.add(Dense(input_dim=num_features_dim, units=1))

# 训练过程
# 1-cost function:mse;optimizer='Adam'
model.compile(optimizer=tf.train.AdamOptimizer(0.2),
              loss='msle',
              metrics=['crossentropy'])
# 2-训练模型
# batch_size 和epochs可调
model.fit(train_features, train_labels, batch_size=16, epochs=400)

# 预测过程
# 测试数据
test_features = all_features[num_examples:].values
# 预测
y = model.predict(test_features)

# 使用pandas生成csv
# ...
