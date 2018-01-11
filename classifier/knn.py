from __future__ import division

import sys
train_dir = sys.argv[1]
test_dir = sys.argv[2]
# print train_dir, test_dir

import os
import cPickle as pkl

from os import listdir
from os.path import isfile, join
import glob
import numpy as np

# def KNN(X, Y):
    # k = 20
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X['train'], Y['train'])
    # Y_predicted = knn.predict(X['test'])
    # return Y_predicted

# def load_X_Y(file,
k = 20
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import math

vocab = None
sigma = 1
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, find
from sklearn.svm import LinearSVC

return_num_tags = 3

X = {}
Y = {}
labels = {}
vocab = []

def load_X_Y(file_path):
    global X
    global Y
    global labels

    with open(file_path, 'rb') as f:
        X = pkl.load(f)
        Y = pkl.load(f)
        labels = pkl.load(f)

    print len(X['train']), Y['train'].shape
    print len(X['train']), Y['test'].shape
    print Y['train'][0]
    print labels['train'][0]
import operator
predicted = []

def fit(X, Y):
    global vocab
    global predicted
    global labels
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X['train'])

    for id, img_vec in enumerate(X['test']):
        dists, ids = nn.kneighbors(img_vec)
        dists = dists[0]
        ids = ids[0]
        # print labels['test'][id]

        # for each_nn in range(k):
            # classes_nn_img = labels['train'][ids[each_nn]]
            # print classes_nn_img

        #cal posterior probability
        p = {}
        # print p.shape
        for word in vocab:
            p[word] = 0
            for each_nn in range(k):
                # classes_nn_img = find(Y['train'][ids[each_nn]])[1]
                # print 'image id =', ids[each_nn]
                classes_nn_img = labels['train'][ids[each_nn]]
                # print classes_nn_img
                if word in classes_nn_img:
                    # print class_id, classes_nn_img
                    p[word] += math.exp(-dists[each_nn]**2/sigma)

        res = sorted(p.items(), key=lambda k: k[1], reverse=True)
        predicted.append([e[0] for e in res[:return_num_tags]])
    labels['predict'] = predicted

Count = {
        'ground' : {},
        'predict' : {},
        'correct' : {}
}

def test(labels):
    # print 'per-class precision =', precision_score(labels['test'], labels['predict'], average='macro')
    # print 'per-class recall = ', recall_score(labels['test'], labels['predict'], average='macro')

    # print 'global precision =', precision_score(labels['test'], labels['predict'], average='micro')
    # print 'global recall = ', recall_score(labels['test'], labels['predict'], average='micro')
    global Count
    print len(labels['test']), len(labels['predict'])
    acc_per_recall = 0
    acc_per_precision = 0
    print len(vocab)

    for word in vocab:
        Count['predict'][word] = 0
        Count['correct'][word] = 0
        Count['ground'][word] = 0

        for i in range(len(labels['test'])):
            each_pred = labels['predict'][i]
            each_test = labels['test'][i]
            if word in each_pred:
                Count['predict'][word] += 1

            if word in each_test:
                Count['ground'][word] += 1
                if word in each_pred:
                    Count['correct'][word] += 1

        # print word, Count['correct'][word], Count['ground'][word], Count['predict'][word]
        if Count['ground'][word] == 0: continue

        acc_per_recall += Count['correct'][word] / Count['ground'][word]
        if Count['predict'][word] > 0:
            acc_per_precision += Count['correct'][word] / Count['predict'][word]

    print acc_per_recall
    print acc_per_precision
    print 'per-class recall=', acc_per_recall
    print 'per-class precision=', acc_per_precision

if __name__ == '__main__':
    X_Y_file_path = '/home/ta/Projects/computer_vision/final/code/classifier/all_features_full.pkl'
    load_X_Y(X_Y_file_path)
    with open('vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)

    print 'len vocab = ', len(vocab)
    # print vocab[:5]
    fit(X, Y)
    test(labels)
