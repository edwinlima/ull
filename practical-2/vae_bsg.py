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
import util



from nltk import sent_tokenize
from collections import defaultdict
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import SGD
import numpy as np
window_sz = 5 #five words left, five words right

sfile_path = ''


batch_size = 100
latent_dim = 10

#intermediate_dim = 50
epochs = 50
epsilon_std = 1.0
window_size=5
emb_sz=50
context_sz=window_size*2


tr_word2idx, tr_idx2word, sent_train = util.read_input('./data/dev.en') 
tst_word2idx, tst_idx2word,  sent_test = util.read_input('./data/test.en')
#print(tr_word2idx)
corpus_dim = len(tr_word2idx)
original_dim = corpus_dim
flatten_sz = (window_size*2+1)*original_dim
hidden1=100

x_train  = util.get_features(sent_train, tr_word2idx, window_size, emb_sz)
print('shape training set=',np.array(x_train).shape, 'context_sz=', context_sz, 'emb sz=', emb_sz*2)

#x_test = get_features(sent_test, tst_word2idx, window_size, emb_sz)
x_train_hat = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))


print('shape=', context_sz*emb_sz*2)
hidden=x_train.shape[1]*x_train.shape[2]
latent_dim=hidden
sz=context_sz*emb_sz*2
x = Input(shape=(sz,))
M = Dense(hidden)(x)
r=Dense(hidden, activation='relu')(M)
r=K.reshape(r,(-1,context_sz,emb_sz*2))
h=K.sum(r, axis=1)
print('shape h=', h.shape, 'type h=', type(h))
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim, activation='softplus')(h)




def sampling(args):
    z_mean, z_log_var = args
    print('shape z_mean=', K.shape(z_mean)[0])
    epsilon = K.random_normal_variable(shape=(, latent_dim), mean=0.,scale=1.0)
    
    #print("shape z_mean=", K.eval(K.shape(z_mean)[0]))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# we instantiate these layers separately so as to reuse them later
# Generator: We generate new data given the latent variable z

decoder_h = Dense(emb_sz)
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

vae.fit(x_train_hat,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)#,
        #validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
