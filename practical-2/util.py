# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:55:33 2018

@author: Edwin Lima, Efi Athieniti
"""

'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

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

import numpy as np
import csv
from nltk.corpus import stopwords
from time import strftime, gmtime
from collections import Counter

window_sz = 5 #five words left, five words right
stopwords = set(stopwords.words('english'))
sfile_path = ''

def read_input(fn, most_common=None):
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), " Reading input sentences..")

    with open(fn, 'r') as content_file:
        content = content_file.read()
    sentences = sent_tokenize(content)
    punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\n']
    
    sentences_tokens = []
    corpus = []
    reserved = ['<null>' ,'<unk>']
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "File read")

    for sentence in sentences:
        #s= [w for w in sentence.split() if w not in punctuation]
        s=[]
        for w in sentence.split():
            if w not in punctuation:
                if w in stopwords:
                    w=reserved[1]
                s.append(w)

        sentences_tokens.append(s)
        corpus = corpus + s
    counts = Counter(corpus)
    print('len corpus=', len(set(corpus)))
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "created corpus")

    if most_common:
        corpus = set(map(lambda x: x[0], counts.most_common(most_common)))
    else:
        corpus = set(corpus)
    print("updated corpus len", len(corpus))

    corpus = corpus.union(set(reserved))
    print("updated corpus len", len(corpus))

    word2idx, idx2word=encode_corpus(corpus)

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Finished reading input sentences")

    return word2idx, idx2word, sentences_tokens

def encode_corpus(corpus):
    word2idx = defaultdict(list)
    idx2word = defaultdict(list)
    #stpwd_idx = 2

    
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
    #Return: X: concatenated context and central word deterministic embeddings,
    #           shape(central_words x window_size*2 x emb_sz*2)
    #        X_hot: context hot vectors shape (central_words*window_size*2 x vocab_size)

    X_hot = []
    X=[]

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Creating features..")
    R = np.random.rand(len(word2idx),emb_sz)


    for sentence in sentences:
        for idx, w_x in enumerate(sentence):
            if w_x in word2idx:
                temp = []

                for i, w_y in enumerate(sentence[max(idx - window_size, 0) :\
                                        min(idx + window_size, len(sentence))]) :

                    if i != idx:
                        word_y = w_y
                        if w_y not in word2idx:  # check if word in dict
                            #print(w_y,"word mapped to unk")
                            word_y = '<unk>'
                        temp.append([word2idx[w_x],word2idx[word_y]])
                temp = np.array(temp)

                #print('temp=',temp.shape)
                # pad if the contexts is smaller than window size
                if temp.shape[0] < window_size*2 and temp.shape[0]>0:

                   padding =  window_size*2 - temp.shape[0]

                   u = [np.hstack((word2idx[w_x],word2idx['<null>'] ))]
                   u_all = np.repeat(u, padding, axis=0)
                   #print(temp.shape, u_all.shape)
                   #print("temp",temp, "u_all",u_all)
                   if len(temp)>0:
                       temp = np.vstack((temp, u_all))

            if len(temp)>0:
                X.append(temp)
            
    X=np.vstack(X)
    print(X.shape)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Finished creating features")

    return X



def get_features(sentences, word2idx, window_size, emb_sz):
    #sentences: set of sentences
    #word2idx dict with the word index of the whole corpus
    #window size: size of the context
    #Return: X: concatenated context and central word deterministic embeddings,
    #           shape(central_words x window_size*2 x emb_sz*2)
    #        X_hot: context hot vectors shape (central_words*window_size*2 x vocab_size)

    X_hot = []
    X=[]

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Creating features..")
    R = np.random.rand(len(word2idx),emb_sz)


    for sentence in sentences:
        for idx, w_x in enumerate(sentence):
            if w_x in word2idx:
                temp = []

                for i, w_y in enumerate(sentence[max(idx - window_size, 0) :\
                                        min(idx + window_size, len(sentence))]) :

                    if i != idx:
                        word_y = w_y
                        if w_y not in word2idx:  # check if word in dict
                            #print(w_y,"word mapped to unk")
                            word_y = '<unk>'
                        temp.append( word2idx[word_y] )

                temp=np.array(temp)


                if len(temp) < window_size*2 and len(temp)>0:
                   padding =  window_size*2 - len(temp)

                   u = [ word2idx['<null>'] ]
                   u_all = np.repeat(u, padding, axis=0)
                   if len(temp)>0:
                       temp = np.hstack((temp, u_all))


            if len(temp)>0:
                word_contexts = [list(temp),word2idx[w_x]]
                for i in range(window_size*2)
                X.append(word_contexts)
                print("word_contexts",word_contexts)


    #X=np.vstack(X)
    print(X.shape)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Finished creating features")

    return X

def save_embeddings(embeddings_file, embeddings, idx2word):
    with open(embeddings_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        print(embeddings.shape)
        writer.writerow([embeddings.shape[1], embeddings.shape[0]])

        for i in range(embeddings.shape[1]):
            word = idx2word[i]
            embedding = embeddings[:,i]
            embedding = list(embedding)
            line = [word] + embedding
            writer.writerow(line)

