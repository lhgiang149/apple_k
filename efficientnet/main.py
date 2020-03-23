import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


import efficientnet.tfkeras
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.models import load_model

from multiprocessing import Pool, cpu_count
from keras import backend as K
import tensorflow as tf
from data_processing import *


def _main():
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/images/'
    labels_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'
    data = process_data(image_path,label_path)
    weights_path = 'model/model.h5'
    model = load_model(weights_path)
    adam = Adam(lr = 0.0001)
    model.compile(optimizers = adam, loss = [focal_loss])
    model.fit_generator(data, epochs = 20, ,batch_size, metrics = ['accuracy'])

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed