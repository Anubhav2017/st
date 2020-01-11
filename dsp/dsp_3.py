from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf 


##used to generate tensorboard data
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

frequency=20
sampling_rate=480

#discretized data to be fit using LUT ranging from 
data_original=np.array([int(np.sin(2* np.pi * frequency * x/sampling_rate)*50+np.random.randn(1)*3) for x in range(0,int(sampling_rate))], dtype=np.int32)

#Ensuring that the data is >=0. The 
data_original=data_original-np.min(data_original)

#data to be fit using LUT
data_modified=np.array((data_original**2 / 100),dtype=np.int32)




class Model(object):

	def __init__(self,x,y):

		#The size of lookup table to be generated
		self.size=np.max(x)-np.min(x)+1

		#The parameter w represents the lookup table in our model. The size of lookup table is equal to the maximum number of discrete levels in the original signal   
		self.w=tf.Variable(tf.random.uniform([self.size],dtype=tf.float32))

		# print("min=",np.min(x))
		# print("max=", np.max(x))
		
		#This computes the indices of unique elements(in ascending order) in the original signal
		u, indices = np.unique(x , return_index=True)

		#We will store the distinct elements of the original signal in xtrain and their corresponding modified signal values in ytrain array
		self.xtrain=np.zeros(self.size)
		self.ytrain=np.zeros(self.size)

		self.xtrain[x[indices]]=x[indices]
		self.ytrain[x[indices]]=y[indices]

	#This function computes the loss	
	def compute_loss(self):

		return tf.math.reduce_sum(tf.math.square(self.w-self.ytrain))

	#We have used Adam optimizer to minimize the loss function. Simply change Adam to any other optimization function (such as SGD) if required	
	def train_step(self, opt=tf.keras.optimizers.Adam(0.1)):
		
		current_loss=self.compute_loss()
		opt.minimize(self.compute_loss, [self.w])
		return current_loss

	#This function is called first when the model is executed i.e starts training	
	def __call__(self,num_steps):

		for i in range(num_steps):

			curr_loss=self.train_step()

			#Print loss value 
			print(curr_loss)

			#Logging the loss value to be visualized in tensorboard (not mandatory to do the training)
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', curr_loss, step=i)	



#The original and modified data values are passed while declaring an instance of the model 
model1=Model(data_original,data_modified)



num_epochs=5000

#This step starts training the model for the number of epochs entered
model1(num_epochs)

#Accessing the LUT after training is over
wfinal1=model1.w.numpy()





#Model to compute inverse function
model2=Model(data_modified,data_original)

model2(num_epochs)

#Inverse LUT
wfinal2=model2.w.numpy()


plt.figure(1)
plt.scatter(np.arange(model1.size),wfinal1)
plt.plot(model1.ytrain - wfinal1)
plt.scatter(np.arange(model2.size),wfinal2)

plt.figure(2)
plt.plot(wfinal1[data_original])
plt.plot(data_modified)

plt.figure(3)
plt.plot(wfinal2[data_modified])
plt.plot(data_original)

plt.show()
