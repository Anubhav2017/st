import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMTree, HamLSTMCell

from tensorflow.keras.layers import LSTMCell,RNN,Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

lr=0.001
batches_per_epoch=100
max_epochs=60
batch_size=10000
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
for _ in range(batch_size):
	arr=np.random.randint(0,high=pow(2, embed_size)-1, size=n)
	arr_sort=np.argsort(arr)

	arr_bin=[]

	for i in range(n):
		arr_bin.append(np.array(list(np.binary_repr(arr[i]).zfill(embed_size))).astype(np.float32))

	arr_bin=np.array(arr_bin)

	#arr_bin_sort=np.array(tf.one_hot(arr_sort,n))

	X.append(arr_bin)
	Y.append(arr_sort)

X=np.array(X)
Y=np.array(Y)

X=np.tile(X,n)
X=X.reshape(batch_size,n,n,embed_size)
Y=Y.reshape(batch_size,n)

layer=RNN(HamLSTMCell(tree_size=20,units=20),return_sequences=True)

model=tf.keras.Sequential()
model.add(layer)
model.add(TimeDistributed(Dense(n,activation="softmax")))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(X,Y,epochs=20)

#y=layer(X)

#print(y)
