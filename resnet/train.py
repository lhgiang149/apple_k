import numpy as np

import tensorflow as tf

from data_processing import *
import keras
import keras.backend as K

from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from model import ResNet50

def _main():
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train_image/'
    labels_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'
    csv = pd.read_csv(labels_path)
    image_named = np.array(csv['image_id'])
    y = csv.loc[:, 'healthy':].values
    
    train ,val, y_train,y_val = train_test_split(image_named, y, test_size = 0.22, random_state = 42, shuffle = True)

    freeze = 1

    # step = 607
    num_train = 1420
    num_val = 401
    
    log_dir = 'log/imageNet/non_letter_image_Swish/'
    check_path(log_dir)
    weights_path = 'imageNet'

    model = ResNet50(weights = 'imageNet',activate = 'swish')
    
    
    # # temporary use Adam and
    adam = Adam(lr = 0.001)
    # logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, mode = 'min')
   
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr = 1e-6)


    # Freeze FC layers
    # for i in range((len(model.layers)-freeze)):
    #     model.layers[i].trainable = False

    # model.compile(optimizer = adam, 
    #             loss = 'categorical_crossentropy',
    #             # loss = [categorical_focal_loss(alpha=.25, gamma=2)],
    #             metrics = ['accuracy'])
    model.compile(optimizer = adam, 
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

    batch_size  = 16
    
    history = model.fit_generator(train_generator(image_path, train, batch_size, y_train, num_train),
        epochs = 20, 
        steps_per_epoch = num_train//batch_size, 
        validation_data = val_generator(image_path, val, batch_size, y_val, num_val),
        validation_steps = num_val//batch_size,
        initial_epoch = 0,
        verbose = 1,
        callbacks=[checkpoint])
    model.save_weights(log_dir+'model.h5')

    import pickle
    with open(log_dir + 'history', 'wb') as f:
        pickle.dump(history.history, f)
   

def check_path(path):
    os.makedirs(path) if not os.path.exists(path) else None

if __name__ == "__main__":
    _main()