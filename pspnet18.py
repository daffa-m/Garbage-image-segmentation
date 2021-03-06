from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_loss(pred, actual):
    num = 2 * tf.reduce_sum((pred * actual), axis=-1)
    den = tf.reduce_sum((pred + actual), axis=-1)
    return 1 - (num + 1) / (den + 1)

def conv_block(input_tensor, filters, strides, d_rates):
    x = Conv2D(filters[0], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
#     x = BatchNormalization()(x)

    shortcut = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, filters, d_rates):
    x = Conv2D(filters[0], kernel_size=3, padding='same', dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
#     x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    h = input_tensor.shape[1]
    w = input_tensor.shape[2]

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
        x = Conv2D(512, kernel_size=1)(x)
        x = Lambda(lambda x: tf.image.resize(x, (h, w)))(x)

        concat_list.append(x)

    return concatenate(concat_list)


def pspnet50(num_classes, input_shape, lr_init):
    img_input = Input(input_shape)

    x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, filters=[64, 64], strides=(1, 1), d_rates=[1, 1])
    x = identity_block(x, filters=[64, 64], d_rates=[1, 1])
#     x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[128, 128], strides=(2, 2), d_rates=[1, 1])
    x = identity_block(x, filters=[128, 128], d_rates=[1, 1])
#     x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
#     x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[256, 256], strides=(1, 1), d_rates=[2, 2])
    x = identity_block(x, filters=[256, 256], d_rates=[2, 2])
#     x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
#     x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
#     x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
#     x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = conv_block(x, filters=[512, 512], strides=(1, 1), d_rates=[4, 4])
    x = identity_block(x, filters=[512, 512], d_rates=[4, 4])
#     x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])

    x = pyramid_pooling_block(x, [1, 2, 3, 6])

    x = Conv2D(512, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_classes, kernel_size=1)(x)
    x = Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init),
                  loss=dice_loss,
                  metrics=[dice_coef])

    return model

