#This file will read all image from the filepath and extract to the vec folder
# Example run: python ./image/ ./vec/
# coding: utf-8
from __future__ import division
import sys
source_dir = sys.argv[1]
vec_dir = sys.argv[2]

import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import glob

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
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    reshape_feature = features.reshape(1,-1)
    return reshape_feature


def main(source_dir,vec_dir):
    #print(source_dir)
    filepath = source_dir
    filenames = glob.glob(filepath + '/*.jpg')
    #print(filepath)
    #print(filenames)
    #os.chdir(vec_dir)
    for filename in filenames:
        vec_filename = filename.replace(filepath,'').replace('.jpg','.vec')
        feature = extract_feature(filename)
        feature.tofile(vec_dir + '/' + vec_filename)

    
if __name__ == "__main__":
    main(source_dir,vec_dir)
