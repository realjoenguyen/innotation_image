#This file will read all image from the filepath and extract to the vec folder
# coding: utf-8
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from keras.preprocessing import image

model = VGG16(include_top=False, weights='imagenet')
model.summary()

def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    reshape_feature = features.reshape(1,-1)
    return reshape_feature

filepath = './data_small/test/image/'
filenames = glob.glob(filepath + '*.jpg')

for filename in filenames:
    vec_filename = filename.replace('/image/','/vec/').replace('.jpg','.vec')
    feature = extract_feature(filename)
    feature.tofile(vec_filename)
