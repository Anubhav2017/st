import tensorflow as tf 
import numpy as np 
from custom_cell import custom_cell
from tensorflow.keras.layers import RNN

x=np.random.randn(100,10,10)
y=np.random.randn(100,10,2)

model=RNN(custom_cell(2,return_sequences=True))

model.compile(loss="mean_squared_error", optimizer="Adam")

print(model.summary())
model.fit(x,y,epochs=100)