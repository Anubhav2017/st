from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMTree, HamLSTMCell

#import keras
from tensorflow.keras.layers import LSTMCell,RNN,Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
from datetime import datetime

#print(tf.executing_eagerly())

#print(tf.eagerly())
num_examples=10000
max_epochs=50
batch_size=200
n=8
embed_size=10
tree_size=20
controller_size=20
weights_path='./ham.weights'
test=False

initial_learning_rate = 0.1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=100000,decay_rate=0.96,staircase=True)

X=[]
Y=[]
for _ in range(num_examples):
	# arr=np.random.randint(0,high=pow(2, embed_size)-1, size=n)
	# arr_sort=np.argsort(arr)

	# arr_bin=[]

	# for i in range(n):
	# 	arr_bin.append(np.array(list(np.binary_repr(arr[i]).zfill(embed_size))).astype(np.float32))

	# arr_bin=np.array(arr_bin)
	arr_bin, arr_sort=create_example(bit_length=embed_size,n=n)

	X.append(arr_bin)
	Y.append(arr_sort)

X=np.array(X)
Y=np.array(Y)

X=np.tile(X,n)
X=X.reshape(num_examples,n,n,embed_size)
Y=Y.reshape(num_examples,n)

layer=RNN(HamLSTMCell(tree_size=tree_size,controller_size=controller_size,units=20),return_sequences=True)

model=tf.keras.Sequential()
model.add(layer)
#model.add(TimeDistributed(Dense(n,activation="relu")))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

os.system('rm -rf ./logs/')

logdir="logs/fit"+datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)

print("Yshape",Y.shape)

model.fit(X,Y,epochs=max_epochs,batch_size=batch_size, callbacks=[tensorboard_callback])

model.summary()
os.system('tensorboard --logdir logs')


#y=layer(X)

#print(y)
