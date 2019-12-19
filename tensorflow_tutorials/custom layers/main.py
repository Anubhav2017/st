import numpy as np 

from keras.layers import Layer
from keras.models import Sequential

import tensorflow as tf 

class layer2(tf.keras.layers.Layer):
	def __init__(self, output_dim=1,b=1):
		super(layer2,self).__init__()
		self.b=b
		self.output_dim=output_dim

	def build(self,input_shape):
		print("in sahpe",input_shape)
		super(layer2, self).build(input_shape)
		self.W=self.add_variable(shape=[input_shape[1],self.output_dim],initializer='uniform',trainable='True')

	def call(self,x):

		return tf.matmul(x,self.W)+self.b


x=np.random.normal(size=(10,2))
x=x.reshape((10,2))

y=x*2
y=np.asarray([y[i][0] for i in range(y.shape[0])])
y=y.reshape((10,1))

print(y.shape)

model=tf.keras.Sequential()
layer=layer2(output_dim=1,b=1)
model.add(layer)
model.compile(loss="mean_squared_error", optimizer='adam',metrics=["accuracy"])
print(y.shape)
print(x.shape)
model.fit(x,y,epochs=5)
print(model.summary())


print(model.predict(x))
#model.add(layer2(1,1))

#model.compile(optimizer="Adam", loss="mean_squared_error", metrics=['accuracy'])

#model.fit(x,y, epochs=10)

#model.summary()



