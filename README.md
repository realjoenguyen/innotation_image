#Problem#  
Given a image predict k tags describing that image 

#Method#

##Feature /features ##
###CNN pretrain ###
python CNN_feature.py \<source_dir_path\> \<vec_dir_path\>

###Visual features ###
* GIST gist.py 
* SIFT sift.py 
* HOG hog.py 
* Color histogram color.py 
All such features can be run by: 
python \<file\> <\source_dir\> \<vec_dir\>

##Classifier /classifers ##
###SVM ###
1. Edit bash file svm.sh 
* 1st parameter: train directory path 
* 2nd parameter: test directory path 
* 3rd parameter: vec directory path 

2. Run svm.sh 
###KNN ###
1. Edit bash file svm.sh 
* 1st parameter: train directory path 
* 2nd parameter: test directory path 
* 3rd parameter: vec directory path 

2. Run svm.sh 

