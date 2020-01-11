from tensorflow import keras
import tensorflow as tf 
import numpy as np 

from data import create_example
from ham import HAMOperations, HAMTree, HamLSTMCell

from tensorflow.keras.layers import LSTMCell,RNN,Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

from datetime import datetime
#from packaging import version


################ input data and parameters initialize ##################################################

num_examples=10000
max_epochs=100
batch_size=100
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
	arr=np.random.randint(0,high=pow(2, embed_size)-1, size=n)
	arr_sort=np.argsort(arr)

	arr_bin=[]

	for i in range(n):
		arr_bin.append(np.array(list(np.binary_repr(arr[i]).zfill(embed_size))).astype(np.float32))

	arr_bin=np.array(arr_bin)

	X.append(arr_bin)
	Y.append(arr_sort)

X=np.array(X)
Y=np.array(Y)

X=np.tile(X,n)
X=X.reshape(num_examples,n,n,embed_size)
Y=Y.reshape(num_examples,n)


########################### building model ###################################################


inputs=keras.Input(shape=(n,n,embed_size))

layer=RNN(HamLSTMCell(tree_size=20,units=20),return_sequences=True)
