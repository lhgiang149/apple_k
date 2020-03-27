import numpy as np

from efficientnet.model import *
import efficientnet.tfkeras

import tensorflow as tf

from data_processing import *
import keras
import keras.backend as K

from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

def _main():
    image_path = '/home/Projects/UTL/toby/apple_k/data/train/'
    labels_path = '/home/Projects/UTL/toby/apple_k/train.csv'
    # image_path = 'C:/Users/ADMINS/Desktop/apple_k/data/images/'
    # labels_path = 'C:/Users/ADMINS/Desktop/apple_k/data/train.csv'
    csv = pd.read_csv(labels_path)
    image_named = np.array(csv['image_id'])
    y = csv.loc[:, 'healthy':].values
    image_named , y = unison_shuffled_copies(image_named, y)
    image_named = list(image_named)
    
    num_train = 1500
    train = image_named[:num_train]
    val = image_named[num_train:]
    y_train = np.copy(y[:num_train])
    y_val = np.copy(y[num_train:])
    

    freeze = 1

    # step = 607
    num_train = 1500
    num_val = 321
    
    log_dir = 'model/log/'
    weights_path = 'model/model.h5'
    
    base_model = EfficientNetB7(input_shape = (600,600,3),
                        weights = weights_path,
                        backend = keras.backend,
                        layers = keras.layers,
                        models = keras.models,
                        utils = keras.utils,
                        include_top=True)
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    output = keras.layers.Dense(4, activation='softmax', 
                        kernel_initializer=DENSE_KERNEL_INITIALIZER,
                        name='probs_1')(base_model.output)
    model = keras.models.Model(inputs = [base_model.input], outputs=[output])
    
    # # temporary use Adam and
    adam = Adam(lr = 0.0001)
    # logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, mode = 'min')
    # mcp_save = keras.callbacks.callbacks.ModelCheckpoint('.mdl_wts.hdf5', 
    #                                                 save_best_only=True, 
    #                                                 monitor='val_loss', 
    #                                                 mode='min')



    # Freeze FC layers
    for i in range((len(model.layers)-freeze)):
        model.layers[i].trainable = False
    model.compile(optimizer = adam, 
                # loss = 'categorical_crossentropy',
                loss = [categorical_focal_loss(alpha=.25, gamma=2)],
                metrics = ['accuracy'])

    model.fit_generator(train_generator(image_path, train, 3, y, num_train),
        epochs = 20, 
        steps_per_epoch = num_train//3, 
        validation_data = val_generator(image_path, val, 3, y, num_val),
        validation_steps = num_val//3,
        initial_epoch = 0,
        verbose = 1,
        callbacks=[checkpoint])
    model.save_weights(log_dir+'model.h5')
    # model.fit_generator(aug.flow(X,y,batch_size = 20), epochs = 20, step_per_epoch = num_train//batch_size  , metrics = ['accuracy'])

# def focal_loss(gamma=2., alpha=.25):
# 	def focal_loss_fixed(y_true, y_pred):
# 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
# 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
# 		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
# 	return focal_loss_fixed
def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

if __name__ == "__main__":
    _main()