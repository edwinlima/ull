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
    batch_size = 128
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
contexts, targets, output = util.get_features(sent_train, tr_word2idx, window_size, emb_sz)
corpus_sz = len(tr_word2idx)
emb_sz_2 = emb_sz*2

#x_train_hat = np.reshape(x_train, (flatten_sz,emb_sz_2))
#print('shape x_train_hat=', x_train_hat.shape)

def concat(input):
    return(K.concatenate([input[0], input[1]]))

### INFERENCE NETWORK
# ENCODER
x_con = Input(shape=(context_sz,))
x_tar = Input(shape=(1,))
y_true = Input(shape=(1,))
print('shape x_con=', x_con.shape)
print('shape x_tar=', x_tar.shape)
R_emb = Embedding(input_dim=original_dim,output_dim=emb_sz)
x_contexts = R_emb(x_con)
x_targets = R_emb(x_tar)

print('shape x_contexts=', x_contexts.shape)
print('shape x_targets=', x_targets.shape)
x_targets = Lambda(lambda y: K.repeat_elements(y, context_sz, axis=1))(x_targets)
targets_contexts = Lambda(concat)([x_targets, x_contexts])

print('shape contexts_targets=', targets_contexts.shape)
#x_targets = Reshape((-1,emb_sz_2))(R)

M = Dense(hidden)(targets_contexts)
print('shape M =', M.shape)
r=Dense(hidden, activation='relu')(M)
print('shape r=', r.shape)
h = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(r)
print('shape h sum=', h.shape)
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

### GENERATIVE MODEL
# Decoder
# we instantiate these layers separately so as to reuse them later
# Generator: We generate new data given the latent variable z
# These are the 'embeddings'
print('z dim=',z.shape)
decoder_h = Dense(original_dim, name="decoder")
decoder_mean = Dense(original_dim, activation='softmax')
h_decoded = decoder_h(z)
probs = decoder_mean(h_decoded)
# need to recover the corpus size here
print('probs=', probs.shape)

#x_decoded_mean = Lambda(lambda y: K.repeat_elements(y, context_sz, axis=0))(x_decoded_mean )
vae = Model(inputs=[x_con, x_tar, y_true],outputs=probs)

x_tar_2=K.reshape(x_tar, (-1,))

# VAE loss = mse_loss or xent_loss + kl_loss
negloglikelihood = metrics.sparse_categorical_crossentropy(y_true, probs)
print("neg_log=", negloglikelihood.shape)#]
#negloglikelihood=K.reshape(negloglikelihood, (-1,context_sz))
print("neg_log=", negloglikelihood.shape)
negloglikelihood = K.mean(negloglikelihood,axis=0)

print("neg_log=", negloglikelihood.shape)

# KL Divergence between 2 gaussians
L = Embedding(input_dim=original_dim,output_dim=emb_sz)
S = Embedding(input_dim=original_dim,output_dim=emb_sz)

prior_loc = L(x_tar_2)
prior_scale = Dense(emb_sz, activation='softplus')(S(x_tar_2))

def kl_divergence(mu_x, sigma_x, prior_mu, prior_scale):
    kl = K.log(prior_scale/sigma_x) +\
           ((K.square(sigma_x) +
             K.square(mu_x - prior_mu))/(2*K.square(prior_scale)))
    return kl

print("z_log_var=",z_log_var.shape)
print("z_mean=",z_mean.shape)
print("prior_loc=",prior_loc.shape)
print("prior scale=",prior_scale.shape)
kl = kl_divergence(z_mean, z_log_var, prior_loc, prior_scale=prior_scale)
kl = kl-0.5
print("kl=",kl.shape)

kl = K.mean(K.sum(kl, axis=1), axis=0)
print("kl=",kl.shape)

print("neglog=",negloglikelihood.shape)


print("K.square(z_mean)=",K.square(z_mean).shape)
#kl = tf.Print(data=[kl],input_=kl, message="kl_loss")
#negloglikelihood  = tf.Print(data=[negloglikelihood ],input_=negloglikelihood , message="neglog")

#elbo = negloglikelihood - kl
#vae_loss = -elbo
vae_loss = kl-negloglikelihood
print("vae_loss=", vae_loss.shape)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

vae.fit([contexts, targets, output],
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
