# -*- coding: UTF-8 -*-

# 参数和mxnet完全相同，有一些不稳定，原因待查
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras import layers, activations

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

# NiN
net = keras.Sequential()


# NiN Block
def nin_block(num_channels, kernel_size, strides, padding):
    net.add(layers.Convolution2D(filters=num_channels, padding=padding, kernel_size=kernel_size,
                                 strides=strides, activation='relu'))
    net.add(layers.Convolution2D(filters=num_channels, kernel_size=1, activation='relu'))
    net.add(layers.Convolution2D(filters=num_channels, kernel_size=1, activation='relu'))


nin_block(96, kernel_size=11, strides=4, padding='valid')
net.add(layers.MaxPool2D(pool_size=2, strides=2))
nin_block(256, kernel_size=5, strides=1, padding='same')
net.add(layers.MaxPool2D(pool_size=2, strides=2))
nin_block(384, kernel_size=3, strides=1, padding='same')
net.add(layers.MaxPool2D(pool_size=1, strides=2))
net.add(layers.Dropout(0.5))

nin_block(10, kernel_size=3, strides=1, padding='same')

net.add(layers.GlobalAvgPool2D())
net.add(layers.Activation('softmax'))

net.add(layers.Flatten())

lr = 0.1
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
# plot_model(net, to_file='model.png', show_shapes=True, show_layer_names=False)
# # 保存model
# json_string = model.to_json()
# open('my_model_architecture.json', 'w').write(json_string)
# model.save_weights('my_model_weights.h5')
