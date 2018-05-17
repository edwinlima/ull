# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:43:51 2018

@author: Eigenaar
"""
from nltk import sent_tokenize
from collections import defaultdict
import keras
import util
from keras.models import Sequential
import keras.backend as K
from keras.layers import Input, Dense, Lambda, Layer, Reshape, merge, Embedding
from keras.models import Model
from keras import backend as K
#from keras.optimizers import SGD
import numpy as np
from random import shuffle


    
#print('word 0=', idx2word[0], 'word to index=', word2idx[idx2word[0]]) 
window_sz = 5 #five words left, five words right
dataset = 'test'
dataset='hansards'


if dataset == 'hansards':
    filename = './data/hansards/training_25kL.txt'
else:
    filename='./data/test.en'

most_common=5000

def onehotencoding(idx,corpus):
#c: corpus    
    hot_enc = list()
    hot_enc = [0 for _ in range(len(corpus))]
    hot_enc[idx] = 1
    return hot_enc


def get_features(sentences, word2idx, window_size, corpus):
    #sentences: set of sentences
    #word2idx dict with the word index of the whole corpus
    #window size: size of the context
    #Return: X, Y word pairs
    X=[]
    Y=[]
    
    for sentence in sentences:
        for idx, w_x in enumerate(sentence):
            for w_y in sentence[max(idx - window_size, 0) :\
                                    min(idx + window_size, len(sentence)) + 1] : 
                if w_y != w_x and w_x in word2idx and w_y in word2idx:
                    X.append(word2idx[w_x])
                    Y.append(word2idx[w_y])
    return X, Y

def main():
    train=1
    window_sz = 5  #n words to the left, x words to the right
    embeddings_sz = 100 #
    epochs = 100
    if train:
        word2idx, idx2word,  sentences_tokens, corpus = util.read_input(filename, most_common=most_common)
        X,Y =get_features(sentences_tokens, word2idx, window_sz, corpus)
        #neg_X, neg_Y =X, Y
        #shuffle(neg_X)
        #shuffle(neg_Y)
        labels =  [1] * len(X)
        #neg_labels =  [0] * len(X)
        #X = X+neg_X
        #Y = Y+neg_Y
        #labels = labels + neg_labels
        X =np.array(X, dtype=float)
        Y =np.array(Y, dtype=float)

        labels = np.array(labels, dtype=float)

        print('X=',X.shape, 'Y=', Y.shape, 'corpus=',len(corpus))
        vocab_size = len(corpus)

        input_target = Input((1,))
        input_context = Input((1,))

        embedding = Embedding(vocab_size, embeddings_sz, input_length=1, name='embedding')
        target = embedding(input_target)
        target = Reshape((embeddings_sz, 1))(target)
        context = embedding(input_context)
        context = Reshape((embeddings_sz, 1))(context)

        similarity = merge([target, context], mode='cos', dot_axes=0)
        # now perform the dot product operation to get a similarity measure
        dot_product = merge([target, context], mode='dot', dot_axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)


        model = Model(input=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')


        for cnt in range(epochs):
            loss = model.train_on_batch([X, Y], labels)
            if cnt % 2 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))

    
    embeddings_file = "./output/embeddings_vocab_%s_%s_skipgram.txt"%(len(corpus), most_common)

    embeddings = np.transpose(model.get_layer(name='embedding').get_weights()[0])

    util.save_embeddings(embeddings_file, embeddings, idx2word)


    
if __name__ == "__main__":           
    main()

            
    