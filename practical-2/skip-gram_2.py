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
from keras.layers import Dense, Flatten
#from keras.optimizers import SGD
import numpy as np


    
#print('word 0=', idx2word[0], 'word to index=', word2idx[idx2word[0]]) 
window_sz = 5 #five words left, five words right
dataset = 'test'
dataset='hansards'

if dataset == 'hansards':
    filename = './data/hansards/training_25kL.txt'
else:
    filename='./data/test.en'

most_common=4000

def onehotencoding(idx,corpus):
#c: corpus    
    hot_enc = list()
    hot_enc = [0 for _ in range(len(corpus))]
    hot_enc[idx] = 1
    return hot_enc
	#onehot_encoded.append(letter)
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
                if w_y != w_x and w_y in word2idx and w_x in word2idx:
                    X.append(onehotencoding(word2idx[w_x], corpus))
                    Y.append(onehotencoding(word2idx[w_y], corpus))
    return X, Y

def main():
    train=1
    window_sz = 5  #n words to the left, x words to the right
    embeddings_sz = 10 #
    
    if train:

        word2idx, idx2word,  sentences_tokens, corpus = util.read_input(filename, most_common=most_common)
        X,Y =get_features(sentences_tokens, word2idx, window_sz, corpus)

        X =np.array(X, dtype=float)
        Y =np.array(Y, dtype=float)
        print('X=',X.shape, 'Y=', Y.shape, 'corpus=',len(corpus))
        print(len(X[0]))
        model = Sequential()
        model.add(Dense(embeddings_sz, activation='linear', input_dim=len(corpus)))    
        #model.add(Flatten())
        model.add(Dense(len(corpus), activation='softmax'))
        model.compile(loss='mse', optimizer='rmsprop')    
        model.fit(X,Y, epochs=2, batch_size=10)
    print('shape=', len(model.layers[0].get_weights()), 'weights=',model.layers[0].get_weights()[0])
    #score = model.evaluate()
    
    
    embeddings_file = "./output/embeddings_vocab_%s_%s_skipgram.txt"%(len(corpus), most_common)
    
    embeddings = np.transpose(model.layers[0].get_weights()[0])

    util.save_embeddings(embeddings_file, embeddings, idx2word)


    
if __name__ == "__main__":           
    main()

            
    