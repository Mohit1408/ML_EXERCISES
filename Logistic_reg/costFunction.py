# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:42:24 2021

@author: Mohit Bisht
"""
import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h =sigmoid(X.dot(theta))

    J =(-1/m)*(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))

    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradient(theta, X, y):
    m = y.size
    theta=theta.reshape(-1,1)
    h=sigmoid(X.dot(theta))
    grad =(0.01/m)*X.T.dot(h-y) 
    return(grad.flatten())			# returns copy of array in one dimension
