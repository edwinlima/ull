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
from keras.layers import Dense, Flatten
#from keras.optimizers import SGD
import numpy as np
window_sz = 5 #five words left, five words right

file_path = ''
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
def get_features(sentences, word2idx, window_size):
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
                if w_y != w_x:
                    #print('x=', w_x, 'y=', w_y)
                    X.append(onehotencoding(word2idx[w_x], word2idx))
                    Y.append(onehotencoding(word2idx[w_y], word2idx))
    return X,Y

corpus_dim = 100
batch_size = 100
original_dim = corpus_dim
latent_dim = 10
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
window_size=5


tr_word2idx, tr_idx2word, sent_train = read_input('./data/dev.en') 
tst_word2idx, tst_idx2word,  sent_test = read_input('./data/test.en')
#print(tr_word2idx)

x_train, y_train = get_features(sent_train, tr_word2idx, window_size)
#print(x_train)

x_test, y_test = get_features(sent_test, tst_word2idx, window_size)

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim)(x)
z_mean = Dense(latent_dim, activation='relu')(h)
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
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
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
fig=plt.figure(figsize=(6, 6))
p=plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c='r')
#fig.colorbar(p, shrink=0.5, aspect=5)
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()