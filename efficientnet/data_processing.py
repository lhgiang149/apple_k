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
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [

            iaa.Fliplr(0.5), 
            iaa.Flipud(0.2), 

            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
                rotate=(-45, 45), 
                shear=(-16, 16), 
                order=[0, 1], 
                cval=(0, 255), 
                mode=ia.ALL 
            )),

            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), 
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), 
                        iaa.AverageBlur(k=(2, 7)), 
                        iaa.MedianBlur(k=(3, 11)), 
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), 
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), 
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), 
                    iaa.Add((-10, 10), per_channel=0.5), 
                    iaa.AddToHueAndSaturation((-20, 20)), 
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), 
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), 
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), 
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    temp_image = np.copy(images)
    temp_labels = np.copy(labels)
    for _ in range(number_aug):
        image_aug = seq(images=images)
        # images_aug = np.vstack(seq(images=images))
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
    im = np.array(letterbox_image(im, (600,600)))
    # im = np.reshape(im,[1]+list(im.shape))
    return im 
    

def multithread_preprocess_data(image_path, csv_path, executor):
    csv = pd.read_csv(csv_path)
    labels = True if csv.shape[1] == 5 else False
    path = []
    for image in csv['image_id']:
        path.append(image_path + image + '.jpg')
    X = np.vstack(executor.map(readAndProcess,path))
    if labels:
        y = csv.loc[:, 'healthy':].values
        return X,y
    else:
        return X

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
            image = readAndProcess(image_dir + str(train[base+b]) + '.jpg')
            image_data.append(image)
            i = (i+1)%num_train

        image_data = np.array(image_data)
        y_true = np.copy(y[base:i])
        if len(y_true) != len(image_data):
            y_true = np.copy(y[base:len(y)])
            # y_true = np.concatenate((y_true,np.copy(y[:i]))) 
        image_data, y_true = augument(image_data, y_true)
        # print(image_data.shape, y_true.shape)
        yield [image_data, y_true]
        

def val_generator(image_dir, train, batch_size, y, num_val):
    i = 0
    while True:
        image_data = []
        base = i
        for b in range(int(batch_size)):
            image = readAndProcess(image_dir + str(train[base+b]) + '.jpg')
            image_data.append(image)
            i = (i+1)%num_val
        image_data = np.array(image_data)
        y_true = np.copy(y[base:i])
        if len(y_true) != len(image_data):
            y_true = np.copy(y[base:len(y)])
            # y_true = np.concatenate((y_true,np.copy(y[:i])))
        # print(image_data.shape, y_true.shape,i)
        yield [image_data, y_true]