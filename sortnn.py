import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D,Reshape,Flatten

ntrain=1000

x=np.ndarray((ntrain,100))
y=np.ndarray((ntrain,100))


for i in range(ntrain):
    x[i,:] = np.random.permutation(1024)[0:100]
    y[i,:]=np.sort(x[i,:])

print("x=",x)
print("y=",y)

x=x.reshape(ntrain,100,1)
y=y.reshape(ntrain,100,)



model= Sequential()
model.add(Conv1D(filters=32, kernel_size=10,activation="relu", input_shape=(100,1)))
model.add(Conv1D(filters=64, kernel_size=5,activation="relu"))
model.add(MaxPooling1D(pool_size=(5)))

model.add(Conv1D(32, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(5)))

model.add(Flatten())
model.add(Dense(100))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

epochs = 100
batch_size = 128
# Fit the model weights.
history = model.fit(x,y,batch_size=batch_size,
          epochs=epochs)