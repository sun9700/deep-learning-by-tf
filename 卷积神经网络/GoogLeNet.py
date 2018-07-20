# -*- coding: UTF-8 -*-

# 使用的是keras版本的GoogLeNet，因为tensorflow.keras没有办法将模型
# 的参数通过tf.keras.models.Model整合从而构造出模型，也无法通过
# model.add进行简单网络的叠加

import time
from keras.datasets import mnist
from keras.optimizers import SGD
from tensorflow import keras
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, BatchNormalization
from keras.models import Model

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


# GooLeNet
# 定义模型

def inception_module(x, params, concat_axis, padding='same', activation='relu'):
    (branch1, branch2, branch3, branch4) = params

    # 1x1
    pathway1 = Convolution2D(filters=branch1[0], kernel_size=(1, 1), activation=activation)(x)

    # 1x1->3x3
    pathway2 = Convolution2D(filters=branch2[0], kernel_size=(1, 1), activation=activation)(x)
    pathway2 = Convolution2D(filters=branch2[1], kernel_size=(3, 3),
                             padding=padding, activation=activation)(pathway2)

    # 1x1->5x5
    pathway3 = Convolution2D(filters=branch3[0], kernel_size=(1, 1), activation=activation)(x)
    pathway3 = Convolution2D(filters=branch3[1], kernel_size=(5, 5),
                             padding=padding, activation=activation)(pathway3)

    # 3x3->1x1
    pathway4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding=padding)(x)
    pathway4 = Convolution2D(filters=branch4[0], kernel_size=(1, 1), activation=activation)(pathway4)

    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=3)


def create_model():
    INP_SHAPE = (28, 28, 1)
    img_input = Input(shape=INP_SHAPE)
    CONCAT_AXIS = 1
    NB_CLASS = 10

    # module 1
    x = Convolution2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # 为防止梯度弥散，在网络的第一模块最后加入批量规范化层可以避免随机性可能造成的梯度弥散，使精度停留在0.1左右
    x = BatchNormalization()(x)

    # module 2
    x = Convolution2D(filters=64, kernel_size=(1, 1))(x)
    x = Convolution2D(filters=192, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # module 3
    x = inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    # module 4
    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # module 5
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)
    # x = Dropout(.5)(x)
    # x = Dense(output_dim=NB_CLASS, activation='linear')(x)
    x = Dense(units=NB_CLASS, activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE


# Create the Model
x, img_input, CONCAT_AXIS, INP_SHAPE = create_model()
# Create a Keras Model
net = Model(inputs=img_input, outputs=[x])
# 显示模型详细信息
# net.summary()

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
# plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=False)
