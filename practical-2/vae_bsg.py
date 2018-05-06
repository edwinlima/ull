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

    
    R = np.random.rand(emb_sz, len(word2idx))
    for sentence in sentences:
        for idx, w_x in enumerate(sentence):
            temp = np.zeros((window_size*2, len(word2idx)))
            for i, w_y in enumerate(sentence[max(idx - window_size, 0) :\
                                    min(idx + window_size, len(sentence)) + 1]) : 
                if w_y != w_x:
                    #print('x=', w_x, 'y=', w_y)
                    temp[i] = np.hstack(R[word2idx[w_y]], R[word2idx[w_x]]) #like this
                    temp[i] = np.hstack(R[word2idx[w_y]*word2idx[w_x]], R[word2idx[w_x]*word2idx[w_y]]) # or  like this
                    #X.append(onehotencoding(word2idx[w_x], word2idx))
                    #Y.append(onehotencoding(word2idx[w_y], word2idx))
            #print('i=',i)
            #print('shape temp=', temp.shape, 'shape r=', r.shape)
            X.append(temp)
    return X

batch_size = 100
latent_dim = 10
intermediate_dim = 50
epochs = 50
epsilon_std = 1.0
window_size=5


tr_word2idx, tr_idx2word, sent_train = read_input('./data/dev.en') 
tst_word2idx, tst_idx2word,  sent_test = read_input('./data/test.en')
#print(tr_word2idx)
corpus_dim = len(tr_word2idx)
original_dim = corpus_dim
flatten_sz = (window_size*2+1)*original_dim
context_sz=window_size*2+1

x_train  = get_features(sent_train, tr_word2idx, window_size)
print('shape training set=',np.array(x_train).shape)

x_test = get_features(sent_test, tst_word2idx, window_size)

x = Input(shape=(None,flatten_sz))
l = Dense(flatten_sz)(x)
relu=Dense(flatten_sz, activation='relu')(l)
relu=K.reshape(relu,(-1,window_size*2+1,original_dim))
h=K.sum(relu, axis=1)
print('shape h=', h.shape, 'type h=', type(h))
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim, activation='softplus')(h)




def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    
    #print("shape z_mean=", K.eval(K.shape(z_mean)[0]))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim)
decoder_mean = Dense(original_dim, activation='softmax')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
#log_p_x_give_z = tf.reduce_sum(tf.log(tf.multiply(x,x_decoded_mean) + tf.multiply((1-x),(1-x_decoded_mean))), axis=1)
#s=tf.Session()
#print("shape log_p_x_give_z=", log_p_x_give_z)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        print('shape x=', x.shape, 'x_decoded_mean shape=', x_decoded_mean.shape)
        #K.reshape(relu,(-1,window_size*2+1,original_dim))
        #x=K.sum(x, axis=1)
        s = Lambda(lambda f: K.sum(f, axis=1))(x)
        print('shape s=', s.shape, 'x_decoded_mean shape=', x_decoded_mean.shape)
        xent_loss = original_dim * metrics.binary_crossentropy(s, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([K.reshape(x,(-1,context_sz,original_dim)), x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)


# train the VAE on MNIST digits
#x_train, x_test = load_mnist_images(binarize=True)
#print("xtrain shape=", x_train.shape)        
#x_train=sentences        

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)#,
        #validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
