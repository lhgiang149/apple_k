import pandas as pd
import cv2
import numpy as np
from model import *
from PIL import Image

if __name__ == "__main__":
    # weight_path = r'C:\Users\ADMINS\Desktop\apple_k\resnet\model\collect_model\drop\swish\96.8_per.h5'
    weight_path = r'C:\Users\ADMINS\Desktop\apple_k\resnet\model\collect_model\non_drop\relu\95.5_per.h5'
    resnet = ResNet50(weights = weight_path, activate = 'relu')
                        # , dropout_rate = 0.5)

    def readAndProcess(image_path):
        path = image_path
        im = Image.open(path)
        im = im.resize((224,224), Image.BICUBIC)
        im = np.array(im)
        im = im/255
        # im = np.array(letterbox_image(im, (224,224)))
        # im = np.reshape(im,[1]+list(im.shape))
        return im 

    image_path = 'C:/Users/ADMINS/Desktop/apple_k/data/test_image/'
    test_path = 'C:/Users/ADMINS/Desktop/apple_k/data/test.csv'

    csv = pd.read_csv(test_path)
    name = csv.image_id
    csv['healthy'] = csv['multiple_diseases'] = csv['rust'] = csv['scab'] = 0
    csv.reset_index()


    for each in name:
        num = int(each.split('_')[-1])
        print('Image num: ', num)
        path = image_path + each +'.jpg'
        image = readAndProcess(path)
        image = np.expand_dims(image, axis=0)
        result = list(np.round(resnet.predict(image),2))[0]
        csv.loc[num,1:]= result
        

    # csv.loc[0,1:]= np.array([1,2,3,4])
    # print(csv.head())
    csv.to_csv (r'C:\Users\ADMINS\Desktop\result_relu.csv', index = False, header=True)
