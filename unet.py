from keras.models import Input, Model, Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation
from keras.layers.merge import concatenate
from keras.initializers import he_normal
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation, BatchNormalization, Lambda
from keras.layers import multiply, add, concatenate

def attention_gating_block(shortcut, gating_signal, inter_channels):
    theta = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (shortcut)
    phi = Conv2D(inter_channels, (1, 1), use_bias=True, padding='same') (gating_signal)

    concat_theta_phi = add([theta, phi])
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

    return add([channel_se, spatial_se])

def deep_supervision(input_tensor, in_channels, base_name, activation, initializer):

    conv_a = Conv2D(
        in_channels, (1, 1),
        padding='same',
        activation=activation,
        name='conv_{0}_a'.format(base_name),
        kernel_initializer=initializer) (input_tensor)
    bn_a = BatchNormalization(name='bn_{0}_a'.format(base_name))(conv_a)
    conv_b = Conv2D(
        in_channels // 2, (1, 1),
        padding='same',
        activation=activation,
        name='conv_{0}_b'.format(base_name),
        kernel_initializer=initializer) (bn_a)
    bn_b = BatchNormalization(name='bn_{0}_b'.format(base_name))(conv_b)
    conv_c = Conv2D(
        1, (1, 1),
        padding='same',
        name='conv_{0}_c'.format(base_name),
        kernel_initializer=initializer) (bn_b)

    conv_score = Activation('sigmoid', name='conv_{0}_score'.format(base_name)) (conv_c)

    return conv_score


def hypercolumn(last_layer, *args):
    layers = list()
    layers.append(last_layer)

    for layer in args:
        layers.append(Lambda(resize_bilinear, arguments={'target_tensor': last_layer}) (layer))

    return add(layers)

def resize_bilinear(input_tensor, target_tensor):
    target_height = target_tensor.get_shape()[1]
    target_width = target_tensor.get_shape()[2]

    return tf.image.resize_bilinear(input_tensor, [target_height.value, target_width.value])

def vanilla_unet(input_shape, random_state=17):
    initializer = he_normal(random_state)
    activation = 'relu'

    inputs            = Input(input_shape, name='input')

    # 0
    bn_1              = BatchNormalization(name='bn_1') (inputs)
    conv_d0a_b        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_d0a-b', kernel_initializer=initializer) (bn_1)
    bn_2              = BatchNormalization(name='bn_2') (conv_d0a_b)
    conv_d0b_c        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_d0b-c', kernel_initializer=initializer) (bn_2)
    #conv_d0b_c        = scSE_block(conv_d0b_c)
    bn_3              = BatchNormalization(name='bn_3') (conv_d0b_c)
    pool_d0c_1a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d0c-1a') (bn_3)

    # 1
    conv_d1a_b        = Conv2D(128, (3, 3), dilation_rate=2, padding='valid', activation=activation, name='conv_d1a-b', kernel_initializer=initializer) (pool_d0c_1a)
    bn_4              = BatchNormalization(name='bn_4') (conv_d1a_b)
    conv_d1b_c        = Conv2D(128, (3, 3), padding='same', activation=activation, name='conv_d1b-c', kernel_initializer=initializer) (bn_4)
    #conv_d1b_c        = scSE_block(conv_d1b_c)
    bn_5              = BatchNormalization(name='bn_5') (conv_d1b_c)
    pool_d1c_2a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d1c-2a') (bn_5)

    # 2
    conv_d2a_b        = Conv2D(256, (3, 3), dilation_rate=2, padding='valid', activation=activation, name='conv_d2a-b', kernel_initializer=initializer) (pool_d1c_2a)
    bn_6              = BatchNormalization(name='bn_6') (conv_d2a_b)
    conv_d2b_c        = Conv2D(256, (3, 3), padding='same', activation=activation, name='conv_d2b-c', kernel_initializer=initializer) (bn_6)
    #conv_d2b_c        = scSE_block(conv_d2b_c)
    bn_7              = BatchNormalization(name='bn_7') (conv_d2b_c)
    pool_d2c_3a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d2c-3a') (bn_7)

    # 3
    conv_d3a_b        = Conv2D(512, (3, 3), dilation_rate=2, padding='valid', activation=activation, name='conv_d3a-b', kernel_initializer=initializer) (pool_d2c_3a)
    bn_8              = BatchNormalization(name='bn_8') (conv_d3a_b)
    conv_d3b_c        = Conv2D(512, (3, 3), padding='same', activation=activation, name='conv_d3b-c', kernel_initializer=initializer) (bn_8)
    #conv_d3b_c        = scSE_block(conv_d3b_c)
    bn_9              = BatchNormalization(name='bn_9') (conv_d3b_c)
    pool_d3c_4a       = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool_d3c-4a') (bn_9)

    # bottleneck
    conv_d4a_b        = Conv2D(1024, (3, 3), padding='valid', activation=activation, name='conv_d4a-b', kernel_initializer=initializer) (pool_d3c_4a)
    bn_10             = BatchNormalization(name='bn_10') (conv_d4a_b)
    conv_d4b_c        = Conv2D(1024, (3, 3), padding='valid', activation=activation, name='conv_d4a-c', kernel_initializer=initializer) (bn_10)
    bn_11             = BatchNormalization(name='bn_11') (conv_d4b_c)
    upconv_d4c_u3a    = Conv2DTranspose(512, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_d4c_u3a', kernel_initializer=initializer) (bn_11)
    bn_12             = BatchNormalization(name='bn_12') (upconv_d4c_u3a)

    # 3
    crop_d3c_d3cc     = Cropping2D(4, name='crop_d3c-d3cc') (conv_d3b_c)
    #crop_d3c_d3cc     = attention_gating_block(shortcut=crop_d3c_d3cc, gating_signal=bn_12, inter_channels=256)
    concat_d3cc_u3a_b = concatenate([crop_d3c_d3cc, bn_12], axis=3, name='concat_d3cc_u3a-b')
    #concat_d3cc_u3a_b = scSE_block(concat_d3cc_u3a_b)
    conv_u3b_c        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_u3b-c', kernel_initializer=initializer) (concat_d3cc_u3a_b)
    bn_13             = BatchNormalization(name='bn_13') (conv_u3b_c)
    conv_u3c_d        = Conv2D(512, (3, 3), padding='valid', activation=activation, name='conv_u3c-d', kernel_initializer=initializer) (bn_13)
    bn_14             = BatchNormalization(name='bn_14') (conv_u3c_d)
    upconv_u3d_u2a    = Conv2DTranspose(256, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u3d_u2a', kernel_initializer=initializer) (bn_14)
    bn_15             = BatchNormalization(name='bn_15') (upconv_u3d_u2a)

    #conv_aux3_score   = deep_supervision(bn_15, 128, 'aux3', activation, initializer)

    # 2
    crop_d2c_d2cc     = Cropping2D(16, name='crop_d2c-d2cc') (conv_d2b_c)
    #crop_d2c_d2cc     = attention_gating_block(shortcut=crop_d2c_d2cc, gating_signal=bn_15, inter_channels=128)
    concat_d2cc_u2a_b = concatenate([crop_d2c_d2cc, bn_15], axis=3, name='concat_d2cc_u2a-b')
    #concat_d2cc_u2a_b = scSE_block(concat_d2cc_u2a_b)
    conv_u2b_c        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_u2b-c', kernel_initializer=initializer) (concat_d2cc_u2a_b)
    bn_16             = BatchNormalization(name='bn_16') (conv_u2b_c)
    conv_u2c_d        = Conv2D(256, (3, 3), padding='valid', activation=activation, name='conv_u2c-d', kernel_initializer=initializer) (bn_16)
    bn_17             = BatchNormalization(name='bn_17') (conv_u2c_d)
    upconv_u2d_u1a    = Conv2DTranspose(128, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u2d_u1a', kernel_initializer=initializer) (bn_17)
    bn_18             = BatchNormalization(name='bn_18') (upconv_u2d_u1a)

    #conv_aux2_score   = deep_supervision(bn_18, 64, 'aux2', activation, initializer)

    # 1
    crop_d1c_d1cc     = Cropping2D(40, name='crop_d1c-d1cc') (conv_d1b_c)
    #crop_d1c_d1cc     = attention_gating_block(shortcut=crop_d1c_d1cc, gating_signal=bn_18, inter_channels=64)
    concat_d1cc_u1a_b = concatenate([crop_d1c_d1cc, bn_18], axis=3, name='concat_d1cc_u1a-b')
    #concat_d1cc_u1a_b = scSE_block(concat_d1cc_u1a_b)
    conv_u1b_c        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_u1b-c', kernel_initializer=initializer) (concat_d1cc_u1a_b)
    bn_19             = BatchNormalization(name='bn_19') (conv_u1b_c)
    conv_u1c_d        = Conv2D(128, (3, 3), padding='valid', activation=activation, name='conv_u1c-d', kernel_initializer=initializer) (bn_19)
    bn_20             = BatchNormalization(name='bn_20') (conv_u1c_d)
    upconv_u1d_u0a    = Conv2DTranspose(64, (2, 2), padding='valid', strides=2, activation=activation, name='upconv_u1d_u0a', kernel_initializer=initializer) (bn_20)
    bn_21             = BatchNormalization(name='bn_21') (upconv_u1d_u0a)

    #conv_aux1_score   = deep_supervision(bn_21, 32, 'aux1', activation, initializer)

    # 0
    crop_d0c_d0cc     = Cropping2D(88, name='crop_d0c-d0cc') (conv_d0b_c)
    #crop_d0c_d0cc     = attention_gating_block(shortcut=crop_d0c_d0cc, gating_signal=bn_21, inter_channels=32)
    concat_d0cc_u0a_b = concatenate([crop_d0c_d0cc, bn_21], axis=3, name='concat_d0cc_u0a-b')
   # concat_d0cc_u0a_b = scSE_block(concat_d0cc_u0a_b)
    conv_u0b_c        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_u0b-c', kernel_initializer=initializer) (concat_d0cc_u0a_b)
    bn_22             = BatchNormalization(name='bn_22') (conv_u0b_c)
    conv_u0c_d        = Conv2D(64, (3, 3), padding='valid', activation=activation, name='conv_u0c-d', kernel_initializer=initializer) (bn_22)
    bn_24             = BatchNormalization(name='bn_24') (conv_u0c_d)
    conv_u0d_score    = Conv2D(1, (1, 1), padding='valid', activation='sigmoid', name='conv_u0d-score', kernel_initializer=initializer) (bn_24)

    model = Model(
        inputs=inputs,
        outputs=[conv_u0d_score],
        name='vanilla_unet'
    )

    return model

def encoder_block(input_tensor, n_filter):
    x = Conv2D(n_filter, (3, 3), padding='same')(input_tensor)
    x = Conv2D(n_filter, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    return x

def bottleneck(input_tensor, n_filter):
    x = Conv2D(n_filter, (3, 3), padding='same')(input_tensor)
    x = Conv2D(n_filter, (3, 3), padding='same')(x)
    return x

def decoder_block(input_tensor, n_filter, n_crop):
    #x = Cropping2D(cropping=((2, 2), (4, 4)))(input_tensor)
    x = Conv2D(n_filter, (3, 3), padding='same')(input_tensor)
    x = Conv2D(n_filter, (3, 3), padding='same')(x)
    x = Conv2DTranspose(n_filter, (2, 2), padding='same', strides=2)(x)
    return x


if __name__ == '__main__':
    pass
    #vanilla_unet((240,320,3))


    input = Input(shape=(240, 320, 3))
    x = encoder_block(input, n_filter=64)
    x = encoder_block(x, n_filter=128)
    x = encoder_block(x, n_filter=256)
    x = encoder_block(x, n_filter=512)
    x = bottleneck(x, 1024)
    x = decoder_block(x, 512, 4)
    x = decoder_block(x, 256, 16)
    x = decoder_block(x, 128, 40)
    x = decoder_block(x, 64, 88)

    model = Model( inputs = input, outputs = x)

    model.summary()