import numpy as np
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from efficientnet.model import *

import keras
import efficientnet.tfkeras
# from tensorflow.keras.models import load_model
from keras.models import load_model


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from data_processing import *

from multiprocessing import Pool, cpu_count

from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})


def _main():
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/images/'
    labels_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'
    # image_path = 'C:/Users/ADMINS/Desktop/apple_k/data/images/'
    # labels_path = 'C:/Users/ADMINS/Desktop/apple_k/data/train.csv'

    freeze = 1

    # step = 607
    num_train = 1500
    num_val = 321
    

    weights_path = 'model/model.h5'
    # model = load_model(weights_path)
    
    model = EfficientNetB7(weights=weights_path,
                        backend = keras.backend,
                        layers = keras.layers,
                        models = keras.models,
                        utils = keras.utils)

    
    # temporary use Adam and
    adam = Adam(lr = 0.0001)

    # Freeze FC layers
    for i in range((len(model.layers)-freeze)):
        model.layers[i].trainable = False
    model.compile(optimizer = adam, 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
    model.fit_generator(data_generator(image_path, labels_path, 30),
        epochs = 20, 
        steps_per_epoch = num_train//30 , 
        validation_data = data_generator(image_path, labels_path, 30, True),
        validation_steps = num_val//30,
        initial_epoch = 0)
    model.save_weights('model.h5')
    # model.fit_generator(aug.flow(X,y,batch_size = 20), epochs = 20, step_per_epoch = num_train//batch_size  , metrics = ['accuracy'])

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


if __name__ == "__main__":
    _main()