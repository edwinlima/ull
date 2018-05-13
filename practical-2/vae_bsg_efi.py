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

from keras.layers import Input, Dense, Lambda, Layer, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import tensorflow as tf
import util



from nltk import sent_tokenize
from collections import defaultdict
import keras
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
hidden=100
x_train, X_hot  = util.get_features(sent_train, tr_word2idx, window_size, emb_sz)
corpus_sz = len(tr_word2idx)
flatten_sz = x_train.shape[0]* x_train.shape[1]
emb_sz_2 = emb_sz*2

#x_test = get_features(sent_test, tst_word2idx, window_size, emb_sz)
x_train_hat = np.reshape(x_train, (flatten_sz,emb_sz_2))
print('shape x_train_hat=', x_train_hat.shape)
print('shape X_hot=', X_hot.shape)

x = Input(shape=(emb_sz_2,))
x_hot = Input(shape=(original_dim,))
print('shape x=', x.shape)
M = Dense(hidden)(x)
print('shape M =', M.shape)
r=Dense(hidden, activation='relu')(M)
print('shape r=', r.shape)
r=Lambda(lambda u: K.reshape(u,(x_train.shape[0], context_sz,emb_sz*2)))(r)
print('shape r reshape=', r.shape)
h = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(r)
print('shape h sum=', h.shape)
#h=K.transpose(h)
#h = Lambda(lambda x: K.transpose(x))(h)
#print('shape h transpose=', h.shape)
z_mean = Dense(emb_sz)(h) #L
print('shape z_mean=', z_mean.shape)
z_log_var = Dense(emb_sz, activation='softplus')(h) #S
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


# we instantiate these layers separately so as to reuse them later
# Generator: We generate new data given the latent variable z
# These are the 'embeddings'
print('z dim=',z.shape)
decoder_h = Dense(emb_sz)
# vector fw 
decoder_mean = Dense(corpus_sz, activation='softmax')
h_decoded = decoder_h(z)
print('h_decoded  dim=',h_decoded.shape)
x_decoded_mean = decoder_mean(h_decoded)
# need to recover the corpus size here
print('x_decoded_mean shape=', x_decoded_mean.shape)
#x_decoded_mean = Lambad(K.repeat_elements(y, context_sz, axis=0))

x_decoded_mean = Lambda(lambda y: K.repeat_elements(y, context_sz, axis=0))(x_decoded_mean )
print('x_decoded_mean shape REPEAT=', x_decoded_mean.shape)


#s=tf.Session()

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_hot, x_decoded_mean):
        #K.reshape(relu,(-1,window_size*2+1,original_dim))
        #x=K.sum(x, axis=1)
        #s = Lambda(lambda f: K.sum(f, axis=1))(x)
        #print('shape s=', s.shape, 'x_decoded_mean shape=', x_decoded_mean.shape)
        #x=K.flatten(x)
        #x_decoded_mean = K.flatten(x_decoded_mean)
        #print('shape xxxxxs=', x_hot.shape, 'x_decoded_mean shape=', x_decoded_mean.shape)
        xent_loss = original_dim * metrics.binary_crossentropy(x_hot, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = K.repeat_elements(kl_loss, context_sz, axis=0)
        #print("shape kl_loss=",kl_loss.shape)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x_hot = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x_hot, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

def vae_loss(x_hot, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    xent_loss = original_dim * metrics.binary_crossentropy(x_hot, x_decoded_mean)
    print('xent_loss shape=', xent_loss.shape)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #kl_loss = K.repeat_elements(kl_loss, context_sz, axis=0)
    kl_loss = Lambda(lambda y: K.repeat_elements(y, context_sz, axis=0))(kl_loss)

    print('kl_loss shape=', kl_loss.shape)

    return xent_loss + kl_loss

#y = CustomVariationalLayer()([x_hot, x_decoded_mean])
#print('shape x=', x.shape, 'x_decoded_mean shape=', x_decoded_mean.shape, 'type x=', type(x), 'type x_d_mean=', type(x_decoded_mean))
#_hat=Reshape([context_sz,emb_sz*2])(x)
#x = Lambda(lambda v: K.batch_flatten(v))(x)
#x_decoded_mean  = Lambda(lambda v: K.batch_flatten(v))(x_decoded_mean  )
#x_decoded_mean = K.flatten(x_decoded_mean)

vae = Model(inputs=[x, x_hot],outputs=x_decoded_mean)
#from keras.utils import vis_utils as vizu
#vizu.plot_model(vae, "ff.png", show_layer_names=False, show_shapes=True)
vae.compile(optimizer='rmsprop', loss=vae_loss)


# train the VAE on MNIST digits
#x_train, x_test = load_mnist_images(binarize=True)
#print("xtrain shape=", x_train.shape)        
#x_train=sentences        
#print('X_hot=', X_hot[0], x_train_hat[0])
#vae.fit([x_train_hat, X_hot],shuffle=True,epochs=epochs,batch_size=batch_size,validation_data=([x_train_hat,X_hot], None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(hidden,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
