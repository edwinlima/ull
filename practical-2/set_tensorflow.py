# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:01:35 2018

@author: Eigenaar
"""

from keras import backend as K
from os import environ

# user defined function to change keras backend
def set_keras_backend(backend):
    if K.backend() != backend:
       environ['KERAS_BACKEND'] = backend
       reload(K)
       assert K.backend() == backend

# call the function with "theano"
set_keras_backend("tensorflow")