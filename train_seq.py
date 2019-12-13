import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM, Dense

n_steps = 8

x_file = 'data/x_{}.csv'.format(n_steps)
y_file = 'data/y_{}.csv'.format(n_steps)

split_at = 18000
batch_size = 100

hidden_size = 128
weights_file = 'model_weights/model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)
weights_file_1 = 'model_weights_1/model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)

x = np.loadtxt(x_file, delimiter=',', dtype=int)
y = np.loadtxt(y_file, delimiter=',', dtype=int)

x = np.expand_dims(x, axis=2)

YY = []
for y_ in y:
    YY.append(to_categorical(y_))
YY = np.asarray(YY)

x_train = x[:split_at]


x_test = x[split_at:]

y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]

assert (n_steps == x.shape[1])

model= Sequential()

model.add(LSTM(150, input_shape=(n_steps,1),return_sequences=True))
model.add(TimeDistributed(Dense))