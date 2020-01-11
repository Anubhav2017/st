import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

frequency=2000

sampling_rate=48000

data=np.asarray([np.sin(2* np.pi * frequency * x/sampling_rate) for x in range(int(sampling_rate/8))])

data_clip=[]

clip=0.7

for i in range(data.shape[-1]):

	if(data[i]>clip):
		data_clip.append(clip)

	elif(data[i]<-clip):
		data_clip.append(-clip)

	else:
		data_clip.append(data[i])


data_clip=np.array(data_clip)

data_fft=np.fft.fft(data_clip)

freq=np.fft.fftfreq(data_clip.shape[-1])

@tf.function
def compute_loss(x,y):
	loss1=tf.math.square(tf.cast(y,tf.float64)-x)
	loss=tf.math.reduce_sum(loss1)

def apply_filter(x, w):
	return (w[0]/x**2 + (w[1]/x) + w[0] + (w[2]*x)+(w[3]*x**2))

def train(inputs, output, filter_params):

	y=np.asarray([apply_filter(inputs[i],filter_params) for i in range(inputs.shape[-1])])
	opt=tf.keras.optimizers.Adam(lr=0.01)

	with tf.GradientTape() as t:
		loss=compute_loss(inputs,y)

	grads = t.gradient(loss,[filter_params])
	opt.apply_gradients(zip(grads,[filter_params]))
	print(current_loss)


#sess=tf.Session()

filter_params=tf.Variable(tf.random.normal((5,)))

for _ in range(10000):
	train(data_clip,data,filter_params)
