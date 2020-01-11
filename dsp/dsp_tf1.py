import numpy as np
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


frequency=20
sampling_rate=480
data=np.asarray([np.sin(2* np.pi * frequency * x/sampling_rate) for x in range(1,int(sampling_rate/8))])
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

data_clip=np.array(data_clip)
data_fft=np.fft.fft(data_clip)
freq=np.fft.fftfreq(data_clip.shape[-1])

def apply_filter(x, w):

	return tf.nn.convolution(tf.reshape(x,[1,-1,1]), tf.reshape(w,[-1,1,1]),padding="SAME")

X=tf.placeholder(dtype=tf.float32,shape=[int(sampling_rate/8)-1])
print(tf.shape(X))

Y=tf.placeholder(dtype=tf.float32,shape=[int(sampling_rate/8)-1])

optimizer=tf.train.AdamOptimizer(lr)

filter_params=tf.Variable(tf.random.truncated_normal([3]))

outputs=apply_filter(X,filter_params)

loss=tf.math.reduce_sum(tf.math.square(outputs-Y)/sampling_rate)

step=optimizer.minimize(loss)


sess=tf.Session()

sess.run(tf.global_variables_initializer())

print(sess.run(filter_params))

feed = {X: data_clip, Y: data}

_, loss_val, predicted = sess.run([step, loss, outputs], feed_dict=feed)
print("loss=", loss_val)
#print("outputs=", sess.run(predicted))

for i in range(1000):
	_, loss_val, predicted = sess.run([step, loss, outputs], feed_dict=feed)
	print("loss=", loss_val)
	print(sess.run(filter_params))
	#tf.Print(filter_params,[filter_params])