from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMTree
from keras.layers import LSTM, Dense, TimeDistributed


lr=0.001
batches_per_epoch=1000
max_epochs=60
batch_size=50

n_features_lstm=100

n=8
embed_size=10
tree_size=20
controller_size=20
weights_path='./ham.weights'
test=False

inputs = tf.placeholder(tf.float32, shape=[batch_size, n,embed_size], name='Input')
control = tf.placeholder(tf.float32, shape=[batch_size,controller_size], name='Control')
target = tf.placeholder(tf.float32, shape=[batch_size,n,embed_size], name='Target')

ham_ops=HAMOperations(embed_size,tree_size,controller_size)
tree=HAMTree(ham_ops=ham_ops)
tree.construct(n)

values = [tf.squeeze(x, [1]) for x in tf.split(1, n, inputs)]


model= Sequential()
model.add(LSTM(n_features_lstm,input_shape=(n,1),return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss="mean_squared_error", optimizer='adam')

