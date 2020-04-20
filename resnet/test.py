from tensorflow.keras.utils import plot_model as plt
from model import *
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

weights = 'log/imageNet/non_letter_image_Swish/ep001-loss2.044-val_loss1.241.h5'
model = ResNet50(weights = weights,activate = 'swish', include_top= True)
# model = y.yolo_model
# model.summary()
plt(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)