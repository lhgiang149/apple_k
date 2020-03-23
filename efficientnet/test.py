import sys
sys.path.append('..')  
import efficientnet.tfkeras
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# model = load_model('model.h5')
# image_path = 'C:/Users/emage/OneDrive/Desktop/tom_jerry.png'
# image = cv2.imread(image_path)
# image_size = model.input_shape[1]
# x = center_crop_and_resize(image, image_size=image_size)
# # print(image.shape)
# x = preprocess_input(x)
# x = np.expand_dims(x, 0)

# # make prediction and decode
# y = model.predict(x)
# print(y.shape)
# print(np.max(y))
# # model = efn.EfficientNetB7(weights='imagenet')
# # model.save_weights('abc.h5')
# # model.save('model.h5')
# # model = 

import efficientnet.keras as efn 

model = efn.EfficientNetB7(weights='noisy-student') 
model.save_weights('model.h5')