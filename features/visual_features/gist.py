
from PIL import Image
import leargist
import sys
source_dir = sys.argv[1]
result_path = sys.argv[2]
import os
from os import listdir
from os.path import isfile, join

train_file_list = [ f for f in listdir(source_dir) if isfile(join(source_dir,f)) ]
import glob
# train_file_list = glob.glob(source_dir + '/*')
print train_file_list[:5]
print len(train_file_list)

from collections import defaultdict
gists = defaultdict(list)
import numpy as np
# cnt = 0
# def make_gist_each_file(file_path):
    # cnt += 1
    # # print file_path
    # filename = os.path.basename(file_path)
    # print filename
    # im = Image.open(file_path)
    # dsr = leargist.color_gist(im)
    # dsr = np.array(dsr)
    # gists[filename] = dsr
    # if cnt % 500 == 0:
        # print cnt

cnt = 0
print 'Converting ...'

for filename in train_file_list:
    file_path = os.path.join(source_dir, filename)
    im = Image.open(file_path)
    try:
        dsr = leargist.color_gist(im)
        dsr = np.array(dsr)
        gists[filename] = dsr
        # print len(dsr)
        if cnt  % 100 == 0:
            print cnt
        cnt += 1
        dsr.tofile(os.path.join(result_path, filename.replace('jpg', 'vec')))

    except:
        print 'Error at', filename
        continue


print 'Done!'
import concurrent.futures
import glob

# if __name__ == '__main__':
    # with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map(make_gist_each_file, train_file_list)

# print 'Done Pool!'
# import cPickle as pkl
# pkl.dump(gists,open(result_path ,'wb'))
# print 'Done!'
