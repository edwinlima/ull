# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:55:33 2017

@author: Eigenaar
"""

'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf



from nltk import sent_tokenize
from collections import defaultdict
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import SGD
import numpy as np
window_sz = 5 #five words left, five words right

sfile_path = ''
def read_input(fn):
    with open(fn, 'r') as content_file:
        content = content_file.read()
    #print(content)
    sentences = sent_tokenize(content)
    #print(sentences)
    punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\n']
    sentences_tokens = []
    corpus = []
    for sentence in sentences:
        s= [w for w in sentence.split() if w not in punctuation]
        sentences_tokens.append(s) 
        corpus = corpus + s
    corpus = set(corpus)
    word2idx, idx2word=encode_corpus(corpus)
    return word2idx, idx2word, sentences_tokens
#print(corpus)

def encode_corpus(corpus):
    word2idx = defaultdict(list)
    idx2word = defaultdict(list)
    for idx, word in enumerate(corpus):
        word2idx[word] = idx
        idx2word[idx] = word
    return word2idx, idx2word
    
#print('word 0=', idx2word[0], 'word to index=', word2idx[idx2word[0]]) 

def onehotencoding(idx,word2idx):
#c: corpus    
    hot_enc = list()
    hot_enc = np.zeros(len(word2idx))
    #idx = word2idx[word]
    hot_enc[idx] = 1.
    #print(idx, hot_enc[idx], hot_enc)
    return hot_enc
	#onehot_encoded.append(letter)
def get_features(sentences, word2idx, window_size, emb_sz):
    #sentences: set of sentences
    #word2idx dict with the word index of the whole corpus
    #window size: size of the context
    #Return: X, Y word pairs
    X=[]
    Y=[]
    
    for sentence in sentences:
        for idx, w_x in enumerate(sentence):
            pairs = []
            temp = [] #np.zeros((window_size*2, emb_sz*2))
            R = np.random.rand(len(word2idx),emb_sz)
            for i, w_y in enumerate(sentence[max(idx - window_size, 0) :\
                                    min(idx + window_size, len(sentence))]) :
                if w_y != w_x:
                    print('i=',i, 'w=', w_x, 'c=', w_y)
                    
                    #print('R_w=', R[word2idx[w_x]], 'R_c=', R[word2idx[w_y]])
                    temp.append(np.hstack((R[word2idx[w_x]], R[word2idx[w_y]])))

            X.append(temp)
            input() # comment this, we have this here only to check the output.
    print('shape X=', len(X), len(X[0]), len(X[1]), 's 0=',sentences[0], 's 1=', sentences[1])
    return X

batch_size = 100
latent_dim = 10
intermediate_dim = 50
epochs = 50
epsilon_std = 1.0
window_size=5
emb_sz = 50


tr_word2idx, tr_idx2word, sent_train = read_input('./data/dev.en') 
tst_word2idx, tst_idx2word,  sent_test = read_input('./data/test.en')
#print(tr_word2idx)
corpus_dim = len(tr_word2idx)
original_dim = corpus_dim
flatten_sz = (window_size*2+1)*original_dim
context_sz=window_size*2+1

x_train  = get_features(sent_train, tr_word2idx, window_size, emb_sz)
print('shape training set=',np.array(x_train).shape)

x_test = get_features(sent_test, tst_word2idx, window_size)