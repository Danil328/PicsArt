import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose, Reshape, multiply
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda, Add
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout

import numpy as np

smooth = 1.
dropout_rate = 0.5
act = "relu"

def attention_gating_block(shortcut, gating_signal, inter_channels):
    theta = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (shortcut)
    phi = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (gating_signal)

    concat_theta_phi = Add()([theta, phi])
    psi = Activation('relu') (concat_theta_phi)
    compatibility_score = Conv2D(1, (1, 1), use_bias=True, padding='same') (psi)
    alpha = Activation('sigmoid') (compatibility_score)

    return multiply([alpha, shortcut])

def cSE_block(input_tensor, ratio=2):
    feature_maps = input_tensor._keras_shape[3]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, feature_maps))(se)
    se = Dense(feature_maps // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(feature_maps, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return multiply([input_tensor, se])

def sSE_block(input_tensor):
    se = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid', padding='same', use_bias=False) (input_tensor)

    return multiply([input_tensor, se])

def scSE_block(input_tensor):
    channel_se = cSE_block(input_tensor)
    spatial_se = sSE_block(input_tensor)

    return Add()([channel_se, spatial_se])

########################################
# 2D Standard
########################################

# def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
#
#     input = Conv2D(nb_filter, (kernel_size, kernel_size), activation=None, name='conv' + stage + '_0',
#                kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#
#     x = Activation('relu')(input)
#     x = BatchNormalization()(x)
#
#     x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1',
#                kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
#     x = BatchNormalization()(x)
#     # x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
#     x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=None, name='conv' + stage + '_2',
#                kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
#     # x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
#     x = Add()([x, input])
#     x = BatchNormalization()(x)
#     x = Activation(act)(x)
#     return x

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    input = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_0', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    input = BatchNormalization()(input)
    
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, input])
    return x

# def standard_unit(input_tensor, stage, nb_filter, kernel_size=3): 
#     x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#     x = BatchNormalization()(x)
#     x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
#     x = scSE_block(x)
#     x = BatchNormalization()(x)
#     return x

"""
wU-Net for comparison
Total params: 9,282,246
"""

def wU_Net(img_rows, img_cols, color_type=1, num_class=1):
    # nb_filter = [32,64,128,256,512]
    nb_filter = [35, 70, 140, 280, 560]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
        img_input = BatchNormalization()(img_input)
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal',
                         padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(input=img_input, output=unet_output)

    return model


"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""


def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    # nb_filter = [32, 64, 128, 256, 512]
    nb_filter = [16, 32, 64, 128, 256]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)
    pool2 = Dropout(dropout_rate)(pool2)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)
    pool3 = Dropout(dropout_rate)(pool3)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)
    pool4 = Dropout(dropout_rate)(pool4)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(inputs=img_input, outputs=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(inputs=img_input, outputs=[nestnet_output_4])

    return model


if __name__ == '__main__':

    model = wU_Net(96, 96, 1)
    model.summary()

    model = Nest_Net(96, 96, 1)
    model.summary()
