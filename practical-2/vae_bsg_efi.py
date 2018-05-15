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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf
import util
import csv
import time


from nltk import sent_tokenize
from collections import defaultdict
import keras
#from keras.optimizers import SGD
import numpy as np
window_sz = 5 #five words left, five words right

sfile_path = ''

# hansards
batch_size = 40
epochs = 100
epsilon_std = 1.0
window_size=5
emb_sz=100
context_sz=window_size*2
hidden=100
most_common = 6000

# test
batch_size = 4
epochs = 30
epsilon_std = 1.0
window_size=5
emb_sz=100
context_sz=window_size*2
hidden=100
most_common = 1500



#tr_word2idx, tr_idx2word, sent_train = util.read_input('./data/hansards/training.en')

tr_word2idx, tr_idx2word, sent_train = util.read_input('./data/test.en', most_common=most_common)
tst_word2idx, tst_idx2word,  sent_test = util.read_input('./data/test.en')
corpus_dim = len(tr_word2idx)
original_dim = corpus_dim
x_train, X_hot  = util.get_features(sent_train, tr_word2idx, window_size, emb_sz)
corpus_sz = len(tr_word2idx)
flatten_sz = x_train.shape[0]* x_train.shape[1]
emb_sz_2 = emb_sz*2

#x_train_hat = np.reshape(x_train, (flatten_sz,emb_sz_2))
#print('shape x_train_hat=', x_train_hat.shape)
print('shape X_hot=', X_hot.shape)

# ENCODER
x = Input(shape=(context_sz,emb_sz_2,))
x_hot = Input(shape=(context_sz,original_dim,))

print('shape x=', x.shape)
print('shape x_hot=', x_hot.shape)

M = Dense(hidden)(x)
print('shape M =', M.shape)
r=Dense(hidden, activation='relu')(M)
print('shape r=', r.shape)
#r=Lambda(lambda u: K.reshape(u,(x_train.shape[0], context_sz,emb_sz*2)))(r)
print('shape r reshape=', r.shape)
h = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(r)
print('shape h sum=', h.shape)
#h=K.transpose(h)
#h = Lambda(lambda x: K.transpose(x))(h)
#print('shape h transpose=', h.shape)
z_mean = Dense(emb_sz)(h) # L
print('shape z_mean=', z_mean.shape)
z_log_var = Dense(emb_sz, activation='softplus')(h) # S
print('shape z_log_var=', z_log_var.shape)




def sampling(args):
    z_mean, z_log_var = args
    print('shape z_mean sampling=', z_log_var.shape, 'shape z_log_var=', z_log_var.shape)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], emb_sz), mean=0.,
                              stddev=epsilon_std)
        
    print("shape epsilon=", epsilon.shape )
    
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(emb_sz,))([z_mean, z_log_var])

# Decoder
# we instantiate these layers separately so as to reuse them later
# Generator: We generate new data given the latent variable z
# These are the 'embeddings'
print('z dim=',z.shape)
decoder_h = Dense(original_dim, name="decoder")
# vector fw 
decoder_mean = Dense(original_dim, activation='softmax')
h_decoded = decoder_h(z)
print('h_decoded  dim=',h_decoded.shape)
x_decoded_mean = decoder_mean(h_decoded)
# need to recover the corpus size here
print('x_decoded_mean shape=', x_decoded_mean.shape)

x_decoded_mean = Lambda(lambda y: K.repeat_elements(y, context_sz, axis=0))(x_decoded_mean )
print('x_decoded_mean shape REPEAT=', x_decoded_mean.shape)


vae = Model(inputs=[x, x_hot],outputs=x_decoded_mean)

# VAE loss = mse_loss or xent_loss + kl_loss
# reshape here to flatten the contexts of each central word
x_hot_flat=K.reshape(x_hot, (-1,original_dim ))
print("x_decoded_mean=",x_decoded_mean.shape)
reconstruction_loss = original_dim * metrics.binary_crossentropy(x_hot_flat, x_decoded_mean)
print("rec_loss=", reconstruction_loss.shape)
reconstruction_loss *= original_dim
print("rec_loss=", reconstruction_loss.shape)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
print("z_log_var=",z_log_var.shape)
print("K.square(z_mean)=",K.square(z_mean).shape)

kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
kl_loss =  K.repeat_elements(kl_loss, context_sz, axis=0)
print("kl_loss=", kl_loss.shape)

vae_loss = K.mean(reconstruction_loss + kl_loss)
print("vae_loss=", vae_loss.shape)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

vae.fit([x_train, X_hot],
        shuffle=True,
        epochs=epochs,
        batch_size=None)



embeddings_file = "./output/embeddings_vocab_%s_ep_%s_emb_%s_hid_%s_%s_%s_test_bsg.txt"%(corpus_dim,epochs,emb_sz,hidden,batch_size, most_common)
embeddings = vae.get_layer("decoder").get_weights()[0]

util.save_embeddings(embeddings_file, embeddings, tr_idx2word)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(emb_sz,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
