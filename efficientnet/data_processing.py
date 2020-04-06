import imgaug as ia
import imgaug.augmenters as iaa
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator




import imgaug as ia
import imgaug.augmenters as iaa

def augument(images, labels, number_aug = 2):
    seq = iaa.Sequential([
    iaa.Affine(rotate=(-50, 50)),
    iaa.Crop(percent=(0, 0.2))], random_order=True)

    temp_image = np.copy(images)
    temp_labels = np.copy(labels)
    for _ in range(number_aug):
        image_aug = seq(images=images)
        temp_image = np.concatenate((temp_image,image_aug))
        temp_labels = np.concatenate((temp_labels,labels))
        
    return temp_image, temp_labels

def letterbox_image(image, size):
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
    path = image_path
    im = Image.open(path)
    # im = im.resize((600,600), Image.BICUBIC)
    # im = np.array(im)
    im = np.array(letterbox_image(im, (600,600)))
    # im = np.reshape(im,[1]+list(im.shape))
    return im 
    

def unison_shuffled_copies(x, y):
    assert len(x) == len(y)
    np.random.seed(2000)
    p = np.random.permutation(len(x))
    return x[p], y[p]
    
def train_generator(image_dir, train, batch_size, y, num_train):
    i = 0
    batch_size = batch_size/3

    while True :
        image_data = []
        base = i
        # only use for train
        for b in range(int(batch_size)):
            num = (base+b)%num_train
            image = readAndProcess(image_dir + str(train[num]) + '.jpg')
            image_data.append(image)
            i = (i+1)%num_train
        # image_data = np.array(image_data)
        y_true = np.copy(y[base:i])
        if len(y_true) != len(image_data):
            y_true = np.copy(y[base:len(y)])
            y_true = np.concatenate((y_true,np.copy(y[:i]))) 
        image_data, y_true = augument(image_data, y_true)
        image_data = image_data/255
        yield [image_data, y_true]
        

def val_generator(image_dir, val, batch_size, y, num_val):
    i = 0
    while True:
        image_data = []
        base = i
        for b in range(int(batch_size)):
            num = (base+b)%num_val
            image = readAndProcess(image_dir + str(val[num]) + '.jpg')
            image_data.append(image)
            i = (i+1)%num_val
        image_data = np.array(image_data)
        image_data = image_data / 255
        y_true = np.copy(y[base:i])
        if len(y_true) != len(image_data):
            y_true = np.copy(y[base:len(y)])
            y_true = np.concatenate((y_true,np.copy(y[:i])))
        yield [image_data, y_true]


