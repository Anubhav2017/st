from __future__ import absolute_import, division, print_function

import tensorflow as tf 

import datetime
import matplotlib.pyplot as plt 

import numpy as np
rng = np.random

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

xtrain=np.arange(100)
ytrain=np.arange(100)*3 + 1


lr=0.001

w = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

predict = lambda: w*xtrain +b

lossfun=lambda : tf.reduce_mean(tf.square(predict() - ytrain))





def train_step(inputs, predicts,opt=tf.optimizers.Adam(learning_rate=lr)):

	# with tf.GradientTape() as g:
	current_loss=lossfun()

	# grads=g.gradient(current_loss, [w, b])

	# opt.apply_gradients(zip(grads, [w, b]))
	opt.minimize(lossfun,[w,b])

	return current_loss


initial_lr=0.1
num_epochs=5000

for epoch in range(500):

	curr_loss=train_step(xtrain,ytrain)
	print("loss",curr_loss)
	print("w",w.numpy())
	print("b",b.numpy())

	with train_summary_writer.as_default():
		tf.summary.scalar('loss', curr_loss, step=epoch)


yfinal=predict(xtrain,w,b)

yfinal=tf.reshape(yfinal,[-1])

plt.plot(yfinal)
plt.plot(ytrain)
plt.show()