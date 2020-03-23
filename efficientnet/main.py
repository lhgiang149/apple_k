import numpy as np
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


import efficientnet.tfkeras
from tensorflow.keras.models import load_model

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from data_processing import *


def _main():
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/images/'
    labels_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'

    freeze = 1

    # step = 607
    # data, num_train = process_data(image_path,label_path)
    temp_data = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/'
    X = np.load(temp_data+'train_data.npy')
    y = np.load(temp_data+'train_label.npy')

    num_train = 1821
    aug = ImageDataGenerator(rescale=1.)
    batch_size = 20

    weights_path = 'model/model.h5'
    model = load_model(weights_path)

    # temporary use Adam and
    adam = Adam(lr = 0.0001)

    # Freeze FC layers
    for i in range((len(model.layers)-freeze)):
        model.layers[i].trainable = False
    model.compile(optimizers = adam, loss = [focal_loss])
    # model.fit_generator(data, epochs = 20, , step_per_epoch = num_train//20  , metrics = ['accuracy'])
    model.fit_generator(aug.flow(X,y,batch_size = 20), epochs = 20, step_per_epoch = num_train//batch_size  , metrics = ['accuracy'])

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


if __name__ == "__main__":
    _main()