# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:57:18 2020

@author: jlking2
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_train[10000:20000,:,:]
x_train = x_train[0:10000,:,:]

y_test = y_train[10000:20000]
y_train = y_train[0:10000]

y_length_train = len(y_train)
y_length_test = len(y_test)

x_train_0D = np.zeros((y_length_train,784))
for k in range(0,y_length_train):
    xk_train_slice = x_train[k,:,:]
    x_train_0D[k,:] = xk_train_slice.flatten('C')

x_test_0D = np.zeros((y_length_test,784))
for k in range(0,y_length_test):
    xk_test_slice = x_test[k,:,:]
    x_test_0D[k,:] = xk_test_slice.flatten('C')

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train_0D = keras.utils.to_categorical(y_train, num_classes)
y_test_0D = keras.utils.to_categorical(y_test, num_classes)
#print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])

from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/

image_size = 784 # 28*28
num_classes = 10 # ten unique digits

#
epoch_default = 20
nodes_default = 30

epoch_list = np.linspace(10,60,6)
nodes_list = np.linspace(10,60,6)

## MSE Loss

MSE_epoch_test = np.zeros((len(epoch_list),1))
MSE_epoch_train = np.zeros((len(epoch_list),1))
MSE_nodes_test = np.zeros((len(nodes_list),1))
MSE_nodes_train = np.zeros((len(nodes_list),1))



# Epoch loop relu
for kk in range(0,len(epoch_list)):
    print('blah')

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=40, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    test_accuracy = accuracy
    train_accuracy = history.history['accuracy'][-1]
    
    print(train_accuracy)
    print(test_accuracy)


## Categorical Cross Entropy Loss
model = Sequential()

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
#model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=40, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)

test_accuracy = accuracy
train_accuracy = history.history['accuracy'][-1]

print(train_accuracy)
print(test_accuracy)







#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['training', 'validation'], loc='best')
#plt.show()
#
#print(f'Test loss: {loss:.3}')
#print(f'Test accuracy: {accuracy:.3}')


