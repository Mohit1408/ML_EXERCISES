# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:15:05 2021

@author: Mohit Bisht
"""

import numpy as np
from sigmoid import sigmoid
def predict(theta, X, threshold=0.5):
  '''returns predicted value for X, given theta and threshold'''
  p = sigmoid(X.dot(theta.T)) >= threshold
  return(p.astype('int'))