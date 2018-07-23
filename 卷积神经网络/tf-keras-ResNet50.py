# -*- coding: UTF-8 -*-

# 参数和mxnet完全相同，五次迭代精度90以上
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

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

# tf.Keras 自带的ResNet50，需要科学上网 否则下载速度太慢
model = ResNet50()
model.summary()

lr = 0.05
num_epochs = 5
batch_size = 256
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])

for epoch in range(num_epochs):
    start = time.time()
    history = model.fit(trainData, trainLabels, batch_size=batch_size, shuffle=True, verbose=0)
    score = model.evaluate(testData, testLabels, verbose=0)
    print 'epoch ', epoch + 1, ', loss %.3f' % history.history['loss'][0], \
        ', train acc %.9f' % history.history['acc'][0], ', test acc %.9f' % score[1], \
        ', time %.3f' % (time.time() - start), ' sec'
