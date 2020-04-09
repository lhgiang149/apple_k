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
from keras import metrics

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

get_custom_objects().update({'swish': Activation(swish)})

def _main():
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train_image/'
    labels_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'
    # image_path = 'C:/Users/ADMINS/Desktop/apple_k/data/images/'
    # labels_path = 'C:/Users/ADMINS/Desktop/apple_k/data/train.csv'
    csv = pd.read_csv(labels_path)
    image_named = np.array(csv['image_id'])
    y = csv.loc[:, 'healthy':].values
    # image_named , y = unison_shuffled_copies(image_named, y)
    # image_named = list(image_named)
    
    # num_train = 1500
    # train = image_named[:num_train]
    # val = image_named[num_train:]
    # y_train = np.copy(y[:num_train])
    # y_val = np.copy(y[num_train:])
    train ,val, y_train,y_val = train_test_split(image_named, y, test_size = 0.22, random_state = 42, shuffle = True)

    freeze = 1

    # step = 607
    num_train = 1420
    num_val = 401
    
    log_dir = 'log/imageNet/normal/'
    check_path(log_dir)
    weights_path = 'model/model.h5'
    
    base_model = EfficientNetB7(input_shape = (600,600,3),
                        weights = weights_path,
                        backend = keras.backend,
                        layers = keras.layers,
                        models = keras.models,
                        utils = keras.utils,
                        include_top=True)
    
    output = keras.layers.Dense(4, activation='softmax', 
                        kernel_initializer=DENSE_KERNEL_INITIALIZER,
                        name='probs')(base_model.layers[-2].output)
    model = keras.models.Model(inputs = [base_model.input], outputs=[output])
    # # temporary use Adam and
    adam = Adam(lr = 0.0001)
    # logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, mode = 'min')
   


    # Freeze FC layers
    # for i in range((len(model.layers)-freeze)):
    #     model.layers[i].trainable = False

    
    model.compile(optimizer = adam, 
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

    batch_size  = 6
    
    model.fit_generator(train_generator(image_path, train, batch_size, y_train, num_train),
        epochs = 50, 
        steps_per_epoch = num_train//batch_size, 
        validation_data = val_generator(image_path, val, batch_size, y_val, num_val),
        validation_steps = num_val//batch_size,
        initial_epoch = 0,
        verbose = 1,
        callbacks=[checkpoint])
    model.save_weights(log_dir+'model.h5')
   

def check_path(path):
    os.makedirs(path) if not os.path.exists(path) else None

if __name__ == "__main__":
    _main()