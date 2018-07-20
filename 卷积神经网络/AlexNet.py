# -*- coding: UTF-8 -*-

# 参数和mxnet完全相同，五次迭代精度90以上
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras import layers

# 获得数据并进行处理使之符合网络输入格式
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# 训练数据 60000张手写图片，28*28*1
# 测试数据 10000张手写图片，28*28*1
trainData = trainData.reshape(60000, 784)
testData = testData.reshape(10000, 784)

trainLabels = keras.utils.to_categorical(trainLabels.reshape(-1, 1), 10)
testLabels = keras.utils.to_categorical(testLabels.reshape(-1, 1), 10)

# tensorflow后端
trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
testData = testData.reshape(testData.shape[0], 28, 28, 1)

# AlexNet
# 定义模型
net = keras.Sequential()
net.add(layers.Convolution2D(filters=96, kernel_size=11, strides=4, activation='relu'))
net.add(layers.MaxPool2D(pool_size=2, strides=2))

net.add(layers.Convolution2D(filters=256, kernel_size=5, padding='same', activation='relu'))
net.add(layers.MaxPool2D(pool_size=2, strides=2))

net.add(layers.Convolution2D(filters=384, kernel_size=3, padding='same', activation='relu'))
net.add(layers.Convolution2D(filters=384, kernel_size=3, padding='same', activation='relu'))
net.add(layers.Convolution2D(filters=256, kernel_size=3, padding='same', activation='relu'))
net.add(layers.MaxPool2D(pool_size=1, strides=2))

net.add(layers.Flatten())

net.add(layers.Dense(4096, activation="relu"))
net.add(layers.Dropout(0.5))
net.add(layers.Dense(4096, activation="relu"))
net.add(layers.Dropout(0.5))

net.add(layers.Dense(10, activation="softmax"))

lr = 0.01
num_epochs = 5
batch_size = 128
net.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])

for epoch in range(num_epochs):
    start = time.time()
    history = net.fit(trainData, trainLabels, batch_size=batch_size, shuffle=True, verbose=0)
    score = net.evaluate(testData, testLabels, verbose=0)
    print 'epoch ', epoch + 1, ', loss %.3f' % history.history['loss'][0], \
        ', train acc %.9f' % history.history['acc'][0], ', test acc %.9f' % score[1], \
        ', time %.3f' % (time.time() - start), ' sec'

# # 输出模型图片
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=False)
