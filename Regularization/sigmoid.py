# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:42:24 2021

@author: Mohit Bisht
"""

import numpy as np
def sigmoid(z):
  '''Returns sigmoid of z
  np.exp(A)==> returns element each element X in the form e^X'''
  sgm=1/(1+np.exp(-z))
  return sgm