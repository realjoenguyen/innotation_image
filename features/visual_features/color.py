
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import cPickle as pkl
import sys
source_dir = sys.argv[1]
res_dir = sys.argv[2]

from os import listdir
from os.path import isfile, join
def compute_color_vector(img_path):
    img = cv2.imread(img_path)
    color = ('b','g','r')
    vec = []
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])

        vec.extend(histr.flatten())
        # print histr

    vec = map(int, vec)
    return vec

def cal_all_in_list():
    print 'Converting ...'
    file_list= [ f for f in listdir(source_dir) if isfile(join(source_dir,f)) ]
    cnt = 0
    for filename in file_list:
        if cnt % 1000 == 0:
             print cnt
        cnt += 1
        file_path = os.path.join(source_dir, filename)
        cal_vec = compute_color_vector(file_path)
        cal_vec = np.array(cal_vec)
        cal_vec.tofile(os.path.join(res_dir, filename.replace('jpg', 'vec')))
    print 'Done!'

if __name__ == '__main__':
    cal_all_in_list()
