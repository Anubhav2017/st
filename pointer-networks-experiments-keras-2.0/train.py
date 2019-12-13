#!/usr/bin/env python

"order integer sequences of length given by n_steps"

import numpy as np
from keras.layers import LSTM, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

from PointerLSTM import PointerLSTM

#

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

#

print("building model...")
main_input = Input(shape=(n_steps, 1), name='main_input')

encoder = LSTM(units=hidden_size, return_sequences=True, name="encoder")(main_input)
print(encoder)
decoder = PointerLSTM(hidden_size, units=hidden_size, name="decoder")(encoder)

model = Model(inputs=main_input, outputs=decoder)
model.summary()
print("loading weights from {}...".format(weights_file))
try:
    model.load_weights(weights_file)
except IOError:
    print("no weights file, starting anew.")

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('training and saving model weights each epoch...')

validation_data = (x_test, YY_test)


epoch_counter = 0

history = model.fit(x_train, YY_train, epochs=1, batch_size=batch_size,
                    validation_data=validation_data)

p = model.predict(x_test)

for y_, p_ in list(zip(y_test, p))[:5]:
    print("epoch_counter: ", epoch_counter)
    epoch_counter+=1
    print("y_test:", y_)
    print("p:     ", p_.argmax(axis=1))
    print()

x_test2=[]
i=0

for  p_ in list(p)[:1]:


    p_=p_.argmax(axis=1)
    print("p_ ", p_)
    x2=[]
    x2.append(p_[0])
    u,indices=np.unique(p_, return_index=True)
    print("u",u)
    print("indices",indices)
    indices=np.sort(indices)
    print("xtesti",x_test[i])
    x1=x_test[i]
    for el in u:
        x1=x1[x1 != el ]
    print("x1 ", x1)
    
    i1=0
    i2=0

    for i1 in range(1,len(p_)):
        if(p_[i1-1]==p_[i1]):
            x2.append(x1[i2])
            i2+=1
        else:
            x2.append(p_[i1])

    x_test2.append(x2)    
    print("x2", x2)


#x_test2=np.asarray(x_test2)

# x_test2 = np.expand_dims(x_test2, axis=2)

# p2=model.predict(x_test2)
# epoch_counter=0
# i=0
# for y_, p_ in list(zip(y_test, p2))[:5]:
#     print("epoch_counter: ", epoch_counter)
#     epoch_counter+=1
#     print("x_test2: ", x_test2[i])
#     print("y_test:", y_)
#     print("p:     ", p_.argmax(axis=1))
#     print()






#model.save(weights_file_1)
