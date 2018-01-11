
from __future__ import division

import cv2
import sys
import numpy as np

def sift(img):

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    (kps, descs) = sift.detectAndCompute(img, None)
    return descs

import sys
source_dir = sys.argv[1]
result_path = sys.argv[2]

import os
import cPickle as pkl

from os import listdir
from os.path import isfile, join

train_file_list = [ f for f in listdir(source_dir) if isfile(join(source_dir,f)) ]

# train_file_list = train_file_list[:100]
print len(train_file_list)
from collections import defaultdict
res_sift = {}
cnt = 1
size_subsift = []
all_dcs = []

def cal_sift():
    global size_subsift
    global all_dcs
    global cnt
    print 'Cal sift ... '

    cnt_sift = 0
    for filename in train_file_list:
        if cnt % 1000 == 0:
             print cnt

        file_path = os.path.join(source_dir, filename)
        img = cv2.imread(file_path, 0)
        cv2.imshow('test', img)
        cal = np.array(sift(img))
        if cal.size > 1:
            # print cal.shape
            res_sift[filename] = cal
            size_subsift.append(cal.shape[0])
            all_dcs.extend(cal)

            cnt_sift += cal.shape[0]

        cnt += 1

    # print size_subsift[:5]
    print 'cnt_sift =', cnt_sift
    print len(all_dcs)
    sift_path = os.path.join(result_path, 'sift_vectors.pkl')
    # print 'Saving into', sift_path
    # pkl.dump(res_sift,open(sift_path ,'wb'))
    # pkl.dump(size_subsift,open(sift_path ,'wb'))
    # pkl.dump(all_dcs,open(sift_path ,'wb'))
    print 'Done!'


def load_sift(file_path):
    global res_sift
    global size_subsift
    global all_dcs
    print 'Loading sift features ...'
    res_sift = pkl.load(open(file_path, 'rb'))
    size_subsift = pkl.load(open(file_path, 'rb'))
    all_dcs = pkl.load(open(file_path, 'rb'))
    print len(all_dcs)

from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict
from sklearn.cluster import KMeans
import numpy as np
def kmean():
    global all_dcs
    print 'Performing codebook ...'

    all_dcs = np.array(all_dcs)
    print 'all_dcs.shape =', all_dcs.shape
    kmeans = KMeans(n_clusters=1000).fit(all_dcs)
    all_labels = kmeans.labels_
    each_labels = np.split(all_labels,np.cumsum(size_subsift))[:-1]
    print 'Done KMean!'

    print 'Converting vector..'

    v = DictVectorizer(sparse=False)
    print 'len size_subsift=', len(size_subsift)
    print 'len each_labels=', len(each_labels)
    X = v.fit_transform(Counter(f) for f in each_labels)
    # pkl.dump(X, open(os.path.join(result_path, 'features.pkl'), 'wb'))
    # print 'Saving features into', os.path.join(result_path, 'features.pkl')

    # print X[0]
    # sys.exit(0)
    for id, filename in enumerate(train_file_list):
        # print X[id]
        # X[id] = np.array(X[id])
        X[id].tofile(os.path.join(result_path, filename.replace('jpg', 'vec')))

    print 'Done'
    return X

if __name__ == '__main__':
    cal_sift()
    sift_path = os.path.join(result_path, 'sift_vectors.pkl')
    # load_sift(sift_path)
    X = kmean()
    # print X[1]
    # print type(X)
