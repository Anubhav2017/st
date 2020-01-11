from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf 

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

frequency=20
sampling_rate=480
data=np.asarray([int(np.sin(2* np.pi * frequency * x/sampling_rate)*50)+50 for x in range(0,int(sampling_rate))], dtype=np.int32)

data_clip=[]


for i in range(data.shape[-1]):
	data_clip.append(int((data[i]**2)/100))


data_clip=np.array(data_clip,dtype=np.int32)

	
class Model(object):

	def __init__(self,x,y):

		self.w=tf.Variable(tf.random.uniform([101],dtype=tf.float32))
		
		u, indices = np.unique(x , return_index=True)

		self.xtrain=np.zeros(101)
		self.ytrain=np.zeros(101)

		self.xtrain[x[indices]]=x[indices]
		self.ytrain[x[indices]]=y[indices]

	@tf.function		
	def compute_loss(self):

		return tf.math.reduce_sum(tf.math.square(self.w-self.ytrain))

	def train_step(self, opt=tf.keras.optimizers.Adam(0.1)):
		current_loss=self.compute_loss()

		opt.minimize(self.compute_loss, [self.w])
		return current_loss

	def __call__(self,num_steps):

		for i in range(num_steps):
			curr_loss=self.train_step()
			print(curr_loss)
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', curr_loss, step=i)	


model=Model(data_clip,data)

num_epochs=5000


model(num_epochs)


wfinal=model.w.numpy()
yfinal=wfinal[data]

print(wfinal)
print(yfinal)
plt.plot(data_clip)

plt.plot(wfinal)


# plt.plot(data_clip)
# plt.plot(data)
plt.show()