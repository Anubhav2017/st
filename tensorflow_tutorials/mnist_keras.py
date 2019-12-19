import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Dense, Conv2D, Flatten
from keras import backend
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import subprocess 
import tempfile

#fashion_mnist = keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data() 

#print(train_images.shape)

#np.save("train_img.npy",train_images)

#np.save("train_lbl.npy", train_labels)

#np.save("test_img.npy", test_images)
#np.save("test_lbl.npy", test_labels)

train_images=np.load("train_img.npy")
train_labels=np.load("train_lbl.npy")
test_images=np.load("test_img.npy")
test_labels=np.load("test_lbl.npy")

train_images=train_images/255.0
test_images=test_images/255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model=keras.Sequential()

model.add(keras.layers.Conv2D(input_shape=(28,28,1),filters=8, kernel_size=3, strides=2, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

model.summary()

testing= False
epochs=5

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.fit(train_images,train_labels,epochs=epochs)

#test_loss, test_acc= model.evaluate(test_images,test_labels)

import tempfile

MODEL_DIR= tempfile.gettempdir()
version=1
export_path=os.path.join(MODEL_DIR, str(version))

if os.path.isdir(export_path):
  print('\nAlready saved a model, cleaning up\n')
  os.rmdir(export_path)

# tf.saved_model.simple_save(
# 	keras.backend.get_session(),export_path,inputs={'input_image': model.input},
# 	outputs={t.name:t for t in model.outputs})

checkpoint_path="training_1/cp.ckpt"

checkpoint_dir=os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, verbose=1)

#model.fit(train_images,train_labels,epochs=epochs,validation_data=(test_images,test_labels),callbacks=[cp_callback])

#model.save('model1.h5')

model1=tf.keras.models.load_model("model1.h5")

model1.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_loss,test_acc=model1.evaluate(test_images,test_labels)

print("test_acc", test_acc)