#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#from keras import applications
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import *


def FCN32(nClasses, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=( input_height, input_width,3))

    model = VGG16(
        include_top=False,
        weights='imagenet',input_tensor=img_input,
        pooling=None,
        classes=1000)

    # def setup_to_transfer_learn(fcn32, model):
    #     for layer in model.layers:
    #         layer.trainable = False
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def add_new_last_layer(model, nClasses):
        x = Conv2D(filters=4096, kernel_size=(7, 7), padding="same", activation="relu", name="fc6")(model.output)
        x = Dropout(rate=0.5)(x)
        x = Conv2D(filters=4096, kernel_size=(1, 1), padding="same", activation="relu", name="fc7")(x)
        x = Dropout(rate=0.5)(x)

        x = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu",
                   kernel_initializer="he_normal",
                   name="score_fr")(x)

        x = Conv2DTranspose(filters=nClasses, kernel_size=(32, 32), strides=(32, 32), padding="valid", activation=None,
                            name="score2")(x)

        x = Reshape((-1, nClasses))(x)
        x = Activation("softmax")(x)

        fcn32 = Model(inputs=img_input, outputs=x)
        return fcn32

    fcn32 = add_new_last_layer(model, nClasses)
    setup_to_transfer_learn(fcn32, model)
    return fcn32


if __name__ == '__main__':
    model = FCN32(11,320, 320)
    model.summary()

