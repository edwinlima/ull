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
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Input, Dense, Lambda, Layer, Reshape, Embedding, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import util
import csv
import time
import keras.callbacks as cbks


from nltk import sent_tokenize
from collections import defaultdict
import keras
#from keras.optimizers import SGD
import numpy as np
window_sz = 5 #five words left, five words right


sfile_path = ''


dataset = "hansards"
dataset = "test"

dataset = "hansards"
dataset = "test"


if dataset == "hansards":# hansards
    batch_size = 8
    epochs = 2
    emb_sz=100
    hidden=100
    most_common = 7000
   # filename = './data/hansards/training_25kL.txt'
    filename = './data/hansards/training_5kL.txt'
else:
    batch_size = 4
    epochs = 8
    emb_sz=100
    hidden=100
    most_common = 1600
    filename = './data/test.en'

epsilon_std = 1.0
window_size=5

context_sz=window_size*2

tr_word2idx, tr_idx2word, sent_train = util.read_input(filename, most_common=most_common)

tst_word2idx, tst_idx2word,  sent_test = util.read_input('./data/test.en')
corpus_dim = len(tr_word2idx)
original_dim = corpus_dim
x_train  = util.get_features(sent_train, tr_word2idx, window_size, emb_sz)
corpus_sz = len(tr_word2idx)
flatten_sz = x_train.shape[0]* x_train.shape[1]
emb_sz_2 = emb_sz*2

#x_train_hat = np.reshape(x_train, (flatten_sz,emb_sz_2))
#print('shape x_train_hat=', x_train_hat.shape)

# ENCODER
x = Input(shape=(2,))

R = Embedding(input_dim=original_dim,output_dim=emb_sz)(x)
R = Reshape((-1,emb_sz_2))(R)
print('shape R=', R.shape)

M = Dense(hidden)(R)
print('shape M =', M.shape)
r=Dense(hidden, activation='relu')(M)
print('shape r=', r.shape)
r=Lambda(lambda u: K.reshape(u,(-1, context_sz,hidden)))(r)
print('shape r reshape=', r.shape)
h = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(r)
print('shape h sum=', h.shape)
#h=K.transpose(h)
#h = Lambda(lambda x: K.transpose(x))(h)
#print('shape h transpose=', h.shape)
z_mean = Dense(emb_sz)(h) # LOC
print('shape z_mean=', z_mean.shape)
z_log_var = Dense(emb_sz, activation='softplus')(h) # SCALE
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

#x_hot_flat=K.reshape(x_hot, (-1,original_dim ))
#
#x_hot = tf.Print(data=[x_hot],input_=x_hot, message="x_hot")

x_hot_flat=K.reshape(x_hot, (-1,))
#x_hot_flat=K.flatten(x_hot)
print("shape x_hot=", x_hot_flat.shape)
x_hot_flat_2 = K.one_hot(x_hot_flat, original_dim)

#x_decoded_mean = tf.Print(data=[x_decoded_mean],input_=x_decoded_mean, message="x_dec")
print("shape x_hot_flat_2=", x_hot_flat_2.shape)
print("x_decoded_mean=",x_decoded_mean.shape)
#reconstruction_loss = original_dim * metrics.categorical_crossentropy(x_decoded_mean,x_hot_flat_2)
reconstruction_loss = original_dim * metrics.binary_crossentropy(x_decoded_mean,x_hot_flat_2)
#reconstruction_loss =  tf.Print(data=[reconstruction_loss],input_=reconstruction_loss, message="recon_loss")

print("rec_loss=", reconstruction_loss.shape)
print("rec_loss=", reconstruction_loss.shape)

kl_loss_posterior = K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
kl_loss_prior = K.sum(1 + prior_scale - K.square(prior_location) - K.exp(prior_scale), axis=-1)
total_kl_loss = kl_loss_posterior + kl_loss_prior
print("z_log_var=",z_log_var.shape)

print("K.square(z_mean)=",K.square(z_mean).shape)
kl_loss *= -0.5
kl_loss =  K.repeat_elements(kl_loss, context_sz, axis=0)
#kl_loss = tf.Print(data=[kl_loss],input_=kl_loss, message="kl_loss")
print("kl_loss=", kl_loss.shape)
vae_loss = K.mean(reconstruction_loss + kl_loss)
print("vae_loss=", vae_loss.shape)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)



embeddings_file = "./output/embeddings_vocab_%s_ep_%s_emb_%s_hid_%s_%s_%s_test_%s_cat_bsg.txt"%(corpus_dim,epochs,emb_sz,hidden,batch_size, dataset,most_common)
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
