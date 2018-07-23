# -*-coding:utf-8-*-

# 仿照mxnet构建了一个与之完全相同的ResNet，但是远远比keras自带的ResNet50要简单很多
import time

from keras.datasets import mnist
from keras.models import Model
import keras
from keras.layers import Convolution2D, BatchNormalization, add, \
    MaxPooling2D, Dense, Input, ZeroPadding2D, GlobalAvgPool2D
from keras.optimizers import SGD

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


# ResNet

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Convolution2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def resnet_(width, height, channel, classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)

    # conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    # conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    # conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = GlobalAvgPool2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


IM_WIDTH = 28
IM_HEIGHT = 28
NB_CLASSES = 10
model = resnet_(IM_WIDTH, IM_HEIGHT, 1, NB_CLASSES)

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
