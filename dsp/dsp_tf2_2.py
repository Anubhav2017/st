import tensorflow as tf 
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential
import datetime
import matplotlib.pyplot as plt 
import numpy as np

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

frequency=20
sampling_rate=480
data=np.asarray([np.sin(2* np.pi * frequency * x/sampling_rate) for x in range(0,int(sampling_rate))])
data_clip=[]
clip=0.7
lr=0.001

for i in range(data.shape[-1]):
	if(data[i]>clip):
		data_clip.append(clip)

	elif(data[i]<-clip):
		data_clip.append(-clip)

	else:
		data_clip.append(data[i])

data_clip=np.array(data_clip,dtype=np.float32)


class Model(object):

	def __init__(self,x):
		self.w=tf.Variable(tf.reshape(tf.random.normal([5]),[-1,1,1]))
		self.x=x

	def __call__(self):
		return tf.nn.convolution(tf.cast(tf.reshape(self.x,[1,-1,1]),tf.float32), self.w,padding="SAME")

model=Model(data_clip)

#w=tf.Variable(tf.reshape(tf.random.normal([5]),[-1,1,1]))

# print(model.w.numpy())

predict=model

#predict=lambda: tf.nn.convolution(tf.reshape(data_clip,[1,-1,1]),w,padding="SAME")

def loss():
	return tf.math.reduce_sum(tf.math.square(tf.reshape(predict(),[-1])-data))

def train_step(lr=0.001, opt=tf.keras.optimizers.SGD(lr)):

	current_loss=loss()
	opt.minimize(loss, [model.w])
	return current_loss


initial_lr=0.05
num_epochs=5000

for epoch in range(num_epochs):

	curr_loss=train_step()

	with train_summary_writer.as_default():
		tf.summary.scalar('loss', curr_loss, step=epoch)
		

yfinal=tf.reshape(model(),[-1])

plt.plot(yfinal)
plt.plot(data_clip)
plt.plot(data)
plt.show()