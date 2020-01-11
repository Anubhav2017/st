import tensorflow as tf 
import numpy as np 

from tensorflow.keras.layers import Dense,LSTMCell

class custom_cell(tf.keras.layers.Layer):

	def __init__(self, units, *args, **kwargs):
		self.units=units
		self.cell=LSTMCell(units=units,state_size=(units,units))
		self.mid=Dense(10)
		super(custom_cell,self).__init__()

	def call(self, inputs, states):

		intermid=mid(inputs)

		opcell, ns=self.cell.call(intermid,states)

		return opcell,ns



		
