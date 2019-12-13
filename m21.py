import numpy as np 

from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import Sequential

l=5

x=np.arange(l)/l

x=x.reshape(1,l,1)
y=2*np.arange(l)/l 
y=y.reshape(1,l,1)

model=Sequential()

model.add(LSTM(10,input_shape=(l,1),return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss="mean_squared_error", optimizer='adam')
print(model.summary())

model.fit(x,y,epochs=1000, batch_size=1,verbose=2)

result=model.predict(x)
print(result)
