from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
from sort_data import DataGenerator 
import numpy as np

def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

print("preparing dataset...")

t = DataGenerator()

X, dec_input, Y = t.next_batch(batch_size=100, N=10)

x_test,dec_input_test, y_test = t.next_batch(batch_size=1,N=10)

print("X shape ", X.shape)
print("Y shape ", Y.shape)

YY = []
for y in Y:
    YY.append(to_categorical(y))
YY = np.asarray(YY)

hidden_size = 128
seq_len = 10
nb_epochs = 10
learning_rate = 0.1

print("building model...")
main_input = Input(shape=(seq_len, 1), name='main_input')
print("main input shape", str(main_input.get_shape()))

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
print("encoder shape ", str(encoder.get_shape()))

decoder = PointerLSTM(hidden_size, output_dim=seq_len, name="decoder")(encoder)
print("decoder shape", str(decoder.get_shape()))

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
print(model.predict(x_test))
print("------")
print(to_categorical(y_test))
model.save_weights('model_weight_100.hdf5')
