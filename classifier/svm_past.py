from __future__ import division
import sys
train_dir = sys.argv[1]
test_dir = sys.argv[2]
vec_dir = sys.argv[3]
# print train_dir, test_dir

import os
import cPickle as pkl

from os import listdir
from os.path import isfile, join
import glob


def get_img_label_path(source_dir):
    img_path = join(source_dir, 'image')
    img_lst = [ f for f in listdir(img_path) if isfile(join(img_path,f)) ]
    # label_lst =  [ f for f in listdir(labels_path) if isfile(join(labels_path,f)) ]
    # img_lst = glob.glob(img_path + '/*')
    return img_lst

import numpy as np
np.set_printoptions(threshold=5)

X = {}
labels = {}
Y = {}
take_first_labels = 3
filenames = {}

def normalize_text(x):
    x = x.lower()
    x = x.strip()
    return x

def get_labels(fname):
    with open(fname) as f:
        content = f.readlines()
        content = [normalize_text(x) for x in content]
    return content[:take_first_labels]

def get_X_labels(kind, source_dir, num=-1):
    global X
    global labels
    global Y
    print 'Get X & labels', kind, 'from', source_dir

    labels_path = join(source_dir, 'label')

    img_lst = get_img_label_path(source_dir)
    print 'len img = ', len(img_lst)
    X[kind] = []
    labels[kind] = []
    filenames[kind] = []

    for img_path in img_lst:
        filenames[kind].append(os.path.basename(img_path))
        X[kind].append(np.fromfile(join(vec_dir, img_path.replace('.jpg', '' + '.vec'))))
        labels[kind].append(get_labels(join(labels_path, img_path + '.desc')))

    if num != -1:
        X[kind] = X[kind][:num]
        labels[kind] = labels[kind][:num]

    X[kind] = np.array(X[kind])
    # labels[kind] = np.array(labels[kind])
    # return X, Y

from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict

vocab = []
map_vocab = {}

def create_Y(labels):
    print 'Building vocab'
    global vocab
    for kind in labels:
        # print 'len = ', kind, len(labels[kind])
        for e in labels[kind]:
            vocab.extend(e)

    # for e in labels['train']:
        # vocab.extend(e)

    vocab = list(set(vocab))
    pkl.dump(vocab, open('vocab.pkl', 'wb'))
    print 'Vectorize'

    print 'len vocab =', len(vocab)
    print vocab[:10]
    v = DictVectorizer(sparse=True)
    # Y_temp = v.fit_transform(Counter(f) for f in labels['train'])
    # Y['train'] = Y_temp

    # for id in range(len(labels['test'])):
        # labels['test'][id] = [e if e in vocab else 'NAK' for e in labels['test'][id]]

    print labels['train'][0]
    print labels['test'][0]
    # Y_temp = v.fit_transform(Counter(f) for f in np.concatenate(labels['train'],labels['test']))

    total_labels = labels['train'] + labels['test']
    Y_temp = v.fit_transform(Counter(f) for f in total_labels)

    global map_vocab
    with open('vocab.txt', 'wb') as f:
        print >> f,  v.vocabulary_
    map_vocab = dict((y,x) for x,y in v.vocabulary_.iteritems())

    Y['train'] = Y_temp[:len(labels['train'])]
    Y['test'] = Y_temp[len(labels['test']):]
    return Y

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, find
from sklearn.svm import LinearSVC

Y_predict = None
def classify(X, Y):
    print 'SVM...'
    global Y_predict
    import time
    start_time = time.time()
    svm= OneVsRestClassifier(LinearSVC(C =2), n_jobs=-1)
    svm.fit(X['train'], Y['train'])
    probs = svm.decision_function(X['test'])

    Y_predict = []
    for each_prob in probs:
        Y_predict.append(sorted(range(len(each_prob)), key=lambda i: each_prob[i], reverse=True)[:take_first_labels])

    # labels_predict = find(Y_predict)

    labels['predict'] = []
    for each_Y in Y_predict:
        labels['predict'].append([map_vocab[e] for e in each_Y])

    print 'Saving model into svm.pkl'
    pkl.dump(svm, open('svm.pkl', 'wb'))
    print 'Done!'

def load_X_Y(file_path):
    global X
    global Y
    with open(file_path, 'rb') as f:
        X = pkl.load(f)
        Y = pkl.load(f)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
Count = {
    'predict' : {},
    'correct' : {},
    'ground' : {}
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

if __name__ == "__main__":
    get_X_labels('train', train_dir, -1)
    get_X_labels('test', test_dir, -1)

    print 'Creating Y...'
    Y = create_Y(labels)


    print 'Train = ', len(X['train']), len(labels['train'])
    print 'Test = ', len(X['test']), len(labels['test'])

    # feature_name = raw_input('Name of these features: ')
    feature_name = 'all'
    print 'Saving X & Y into ', 'all_features_' + feature_name + '.pkl'
    print 'Y = '
    print Y['train'].shape
    print Y['test'].shape
    with open('all_features_' + feature_name + '.pkl', 'wb') as fw:
        pkl.dump(X, fw)
        pkl.dump(Y, fw)
        pkl.dump(labels, fw)
    print 'Done!'

    # load_X_Y('all_features.pkl')
    classify(X, Y)
    print 'Saving Y_predict to Y_predict.pkl...'
    with open('Y_predict.pkl', 'wb') as fw:
        pkl.dump(Y_predict, fw)
    print 'Done'

    print X['train'][0]
    print Y['train'][0]
    print X['test'][0]
    print Y['test'][0]

    print filenames['test'][0]
    print labels['test'][0]
    print labels['predict'][0]

    test(labels)
#here
