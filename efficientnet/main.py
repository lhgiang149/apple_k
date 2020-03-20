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


def _main():
    input_shape = (600,600)
    num_classes = 4
    weights_path = 'model/model.h5'
    model = load_model(weights_path)
    adam = Adam(lr = 0.0001)
    model.compile(optimizers = adam, loss = 'categorical_crossentropy')
    model.fit(X,y, epochs = 40, ,batch_size, metrics = ['accuracy'])

def _process_data(image_path, label_path):
    import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
workers = cpu_count()
executor = Pool(processes=workers)   
from PIL import Image

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def readAndProcess(image_path):
    path = image_path + image + '.jpg'
    im = Image.open(path)
    im = np.array(letterbox_image(im, (600,600)))
    im = np.reshape(im,[1]+list(im.shape))
    return im 
    if 'X' not in locals():
        X = im
    else:
        X = np.concatenate((X,im))

def multithread_preprocess_data(image_path, csv_path):
    workers = cpu_count()
    executor = Pool(processes=workers)
    csv = pd.read_csv(csv_path)
    labels = True if csv.shape[1] == 5 else False
    path = []
    for image in csv['image_id']:
        path.append[image_path + image + '.jpg']
    X = np.vstack(executor.map(readAndProcess,path))
    X = X/255
    if labels:
        y = csv.loc[:, 'healthy':].values
        return X,y
    else:
        return X
    
    