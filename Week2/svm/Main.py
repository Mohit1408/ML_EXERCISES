# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:39:00 2021

@author: Mohit Bisht
"""

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
X,y= datasets.fetch_openml('mnist_784', version=1, return_X_y=True)                                                                        #importing datset and labels straight from sklearn
Images = np.array(X, 'int16')                                                                                                              #converting list of Images into a numpy array.
labels = np.array(y, 'int')
list_hog_fd = []
for Image in Images:
    fd = hog(Image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False,block_norm='L2')    # using hog algorithm to extract features from each image
    list_hog_fd.append(fd)                                                                                                                  #adding the hog features extracted from the image into a list
hog_features = np.array(list_hog_fd, 'float64')                                                                                             #converting the HOG feature list into a numpy array

'''
Two line code to train entire dataset
load the LinearSVC() model into clf
and then fit hog_features and list into clf
'''
clf =


print('Accuracy is:',clf.score(hog_features,labels)*100,'%')
joblib.dump(clf, "digits_cls.pkl", compress=3)                                                                                             #storing the model into a file called 'digits_cls.pkl'