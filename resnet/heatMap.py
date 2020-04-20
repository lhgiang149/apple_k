import pandas as pd
import cv2
import numpy as np
from model import *
from PIL import Image
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk

def readAndProcess(image_path):
    path = image_path
    im = Image.open(path)
    im = im.resize((224,224), Image.BICUBIC)
    im = np.array(im)
    im = im/255
    return im 

def plot(weight_path, image_path, labels_path, save_fig_path):
    resnet = ResNet50(weights = weight_path, activate = 'swish', dropout_rate = 0.3)
    csv = pd.read_csv(labels_path)
    name = csv['image_id']
    y = csv.loc[:, 'healthy':].values
    final = []
    for each in name:
        num = each
        path = image_path + str(each) +'.jpg'
        image = readAndProcess(path)
        image = np.expand_dims(image, axis=0)
        result = list(np.round(resnet.predict(image),2))[0]
        final.append(result)
    final = np.round(np.array(final)).astype(np.int)
    confusion = np.round(sk.confusion_matrix(np.argmax(y,axis=1),np.argmax(final, axis=1)))
    df_cm = pd.DataFrame(confusion, index = [i for i in ["Healthy", "Both", "Rust", "scab"]],
                    columns = [i for i in ["Healthy", "Both", "Rust", "scab"]])
    plt.figure(figsize = (10,8))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt="d")
    plt.savefig(save_fig_path)

def _main():
    # weight_path = r'C:\Users\emage\OneDrive\Desktop\95_per.h5'
    weight_path = r'C:\Users\emage\OneDrive\Desktop\collect_model\drop\swish\96.8_per.h5'
    image_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train_image/'
    test_path = 'C:/Users/emage/OneDrive/Desktop/apple_k/data/train.csv'
    save_path = 'C:/Users/emage/OneDrive/Desktop/image_1.png'
    plot(weight_path, image_path,test_path,save_path)

if __name__ == "__main__":
    _main()