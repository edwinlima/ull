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
    """ Read a corpus file and create a list of sentences and vocabulary
    
    @param fn         : filename
    @param most_common: number of most frequent words to keep
    
    @return: 
        word2idx
        idx2word
        sentences_tokens: list of tokenized sentences
        
    """
    
    print(strftime("%H:%M:%S", gmtime()), " Reading input sentences..")

    with open(fn, 'r') as content_file:
        content = content_file.read()
    sentences = sent_tokenize(content)
    punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\n']
    
    sentences_tokens = []
    corpus = []
    reserved = ['<null>' ,'<unk>']
    print(strftime("%H:%M:%S", gmtime()), "File read")

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
    print(strftime(" %H:%M:%S", gmtime()), "created corpus")

    if most_common:
        corpus = set(map(lambda x: x[0], counts.most_common(most_common)))
    else:
        corpus = set(corpus)
    corpus = corpus.union(set(reserved))
    print("updated corpus len", len(corpus))
    word2idx, idx2word=encode_corpus(corpus)

    print(strftime(" %H:%M:%S", gmtime()),"Finished reading input sentences")

    return word2idx, idx2word, sentences_tokens, corpus

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
    hot_enc = list()
    hot_enc = np.zeros(len(word2idx))
    hot_enc[idx] = 1.
    return hot_enc
    
 

def get_features(sentences, word2idx, window_size, emb_sz):
    """
    Creates pairs of central words and contexts
    Note that contexts is 2d to keep the contexts for a central word together
    Adds null words ('<unk>') if contexts < window_size*2
    
    @param word2idx     : x dict with the word index of the whole corpus
    @param window size  : size of the context/2
    @param sentences    : list of tokenized sentences
    
    @return contexts    : 2d array of context words for each central word,
                          shape(N, window_size*2)
    @return targets     : 1d array of central words 
                          shape(N,1)   
    """
    X=[]
    Y=[]
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

                #temp=np.array(temp)


                if len(temp) < window_size*2 and len(temp)>0:
                   padding =  window_size*2 - len(temp)

                   u = [ word2idx['<null>'] ]
                   u_all = np.repeat(u, padding, axis=0)
                   if len(temp)>0:
                       temp = np.hstack((np.array(temp), u_all))


                if len(temp)>0:
                    word_contexts = np.hstack((temp, word2idx[w_x] ))
                    word_contexts = np.array([word_contexts])
                    #rep_word_contexts = np.repeat(word_contexts, window_size*2, axis=0)

                    X.append(word_contexts)


    X=np.vstack(X)
    contexts, targets = X[:,:10], X[:,10]
    print("contexts=",contexts.shape)
    print("targets=",targets.shape)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"Finished creating features")

    return contexts, targets


def save_embeddings(embeddings_file, embeddings, idx2word):
    """
    Write embeddings to file
    @param embeddings     : numpy array of the embeddings
                            shape(vocab_size, emb_sz)
    @param embeddings_file: filename to write embeddings to
    @param idx2word       : dictionary with word index as key
    """
    
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

