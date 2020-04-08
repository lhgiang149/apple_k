from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import keras
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation   

backend = keras.backend
layers = keras.layers
models = keras.models

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

def identity_block(input_tensor, kernel_size, filters, stage, block,activate = 'relu'):
    
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation(activate)(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation(activate)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation(activate)(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               activate = 'relu'):
    
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation(activate)(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation(activate)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation(activate)(x)
    return x


def ResNet50(include_top=True,
             weights='',
             input_tensor=None,
             input_shape=(224,224,3),
             classes=4,
             activate = 'relu',
             dropout_rate = None,
             **kwargs):
   
    if weights == '':
        print('Train from beginning!!')
    elif weights == 'imageNet':
        weights = r'C:\Users\GE3F.P1\Desktop\toby_new\new\model\imageNetNoTop.h5'
        include_top = False
    elif not os.path.isfile(weights):
        raise Exception('Wrong path, this file doesn\'t exist')
    elif not weights.endswith('h5'):
        raise Exception('Wrong file, weight file must end with h5')

    img_input = layers.Input(shape=input_shape)
    
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation(activate)(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),activate=activate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', activate= activate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', activate= activate)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', activate= activate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', activate= activate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', activate= activate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', activate= activate)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', activate= activate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', activate= activate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', activate= activate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', activate= activate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', activate= activate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', activate= activate)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', activate= activate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', activate= activate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', activate= activate)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate:
        x = layers.Dropout(rate = dropout_rate, name= 'drop')(x)

    if include_top:    
        x = layers.Dense(classes, activation='softmax', kernel_initializer='he_normal', name='fc_layer', \
                            kernel_regularizer=keras.regularizers.l2(l=0.2))(x)
        

    # Create model.
    model = models.Model(img_input, x, name='resnet50')
    
    # Load weights.
    model.load_weights(weights)

    if not include_top:
        out = layers.Dense(classes, activation='softmax', kernel_initializer='he_normal', name='fc_layer', \
                                kernel_regularizer=keras.regularizers.l2(l=0.2))(model.output)
        model = models.Model(inputs = [model.input], outputs=[out])

    return model