# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:57:18 2020

@author: jlking2
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import time
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#x_test = x_train[10000:20000,:,:]
#x_train = x_train[0:1000,:,:]
#
#y_test = y_train[10000:20000]
#y_train = y_train[0:1000]

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
epoch_default = 30
nodes_default = 30

epoch_list = np.linspace(10,60,6)
nodes_list = np.linspace(10,60,6)

## MSE Loss

relu_MSE_epoch_test2L = np.zeros((len(epoch_list),1))
relu_MSE_epoch_train2L = np.zeros((len(epoch_list),1))
sigmoid_MSE_epoch_test2L = np.zeros((len(epoch_list),1))
sigmoid_MSE_epoch_train2L = np.zeros((len(epoch_list),1))

relu_MSE_nodes_test2L = np.zeros((len(nodes_list),1))
relu_MSE_nodes_train2L = np.zeros((len(nodes_list),1))
sigmoid_MSE_nodes_test2L = np.zeros((len(nodes_list),1))
sigmoid_MSE_nodes_train2L = np.zeros((len(nodes_list),1))

relu_MSE_epoch_test3L = np.zeros((len(epoch_list),1))
relu_MSE_epoch_train3L = np.zeros((len(epoch_list),1))
sigmoid_MSE_epoch_test3L = np.zeros((len(epoch_list),1))
sigmoid_MSE_epoch_train3L = np.zeros((len(epoch_list),1))

relu_MSE_nodes_test3L = np.zeros((len(nodes_list),1))
relu_MSE_nodes_train3L = np.zeros((len(nodes_list),1))
sigmoid_MSE_nodes_test3L = np.zeros((len(nodes_list),1))
sigmoid_MSE_nodes_train3L = np.zeros((len(nodes_list),1))

## Epoch loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_MSE_epoch_test2L[kk] = accuracy
    sigmoid_MSE_epoch_train2L[kk] = history.history['accuracy'][-1]

# Node loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_MSE_nodes_test2L[kk] = accuracy
    sigmoid_MSE_nodes_train2L[kk] = history.history['accuracy'][-1]

## Epoch loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=nodes_default, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_MSE_epoch_test3L[kk] = accuracy
    sigmoid_MSE_epoch_train3L[kk] = history.history['accuracy'][-1]

# Node loop sigmoid
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=int(nodes_list[kk]), activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='sigmoid', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    sigmoid_MSE_nodes_test3L[kk] = accuracy
    sigmoid_MSE_nodes_train3L[kk] = history.history['accuracy'][-1]





x_train_0D = x_train_0D.astype('float32') / 255
x_test_0D = x_test_0D.astype('float32') / 255





# Epoch loop relu
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_MSE_epoch_test2L[kk] = accuracy
    relu_MSE_epoch_train2L[kk] = history.history['accuracy'][-1]
    
# Node loop relu
for kk in range(0,len(nodes_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_MSE_nodes_test2L[kk] = accuracy
    relu_MSE_nodes_train2L[kk] = history.history['accuracy'][-1]

# Epoch loop relu
for kk in range(0,len(epoch_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=nodes_default, activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=int(epoch_list[kk]), verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_MSE_epoch_test3L[kk] = accuracy
    relu_MSE_epoch_train3L[kk] = history.history['accuracy'][-1]
   
# Node loop relu
for kk in range(0,len(nodes_list)):

    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=int(nodes_list[kk]), activation='relu', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='relu', input_shape=(image_size,)))
    #model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()
    
    model.compile(optimizer = 'sgd', loss='MSE', metrics=['accuracy'])
    history = model.fit(x_train_0D, y_train_0D, batch_size=128, epochs=epoch_default, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_0D, y_test_0D, verbose=False)
    
    relu_MSE_nodes_test3L[kk] = accuracy
    relu_MSE_nodes_train3L[kk] = history.history['accuracy'][-1]


plt.figure(3)
plt.plot(epoch_list, 1-relu_MSE_epoch_test2L,marker='o',markersize=12,color='black')
plt.plot(epoch_list, 1-relu_MSE_epoch_test3L,marker='v',markersize=12,color='black')
plt.plot(epoch_list, 1-sigmoid_MSE_epoch_test2L,marker='o',markersize=12,color='red')
plt.plot(epoch_list, 1-sigmoid_MSE_epoch_test3L,marker='v',markersize=12,color='red')
#plt.title('Predicted Labels, changing alpha')
plt.ylabel('Error Rate')
plt.xlabel('Epochs')
plt.legend(['relu, 1 layer', 'relu, 2 layers', 'sigmoid, 1 layer', 'sigmoid, 2 layers'], loc='upper right')
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(nodes_list, 1-relu_MSE_nodes_test2L,marker='o',markersize=12,color='black')
plt.plot(nodes_list, 1-relu_MSE_nodes_test3L,marker='v',markersize=12,color='black')
plt.plot(nodes_list, 1-sigmoid_MSE_nodes_test2L,marker='o',markersize=12,color='red')
plt.plot(nodes_list, 1-sigmoid_MSE_nodes_test3L,marker='v',markersize=12,color='red')
#plt.title('Predicted Labels, changing alpha')
plt.ylabel('Error Rate')
plt.xlabel('Width of hidden layers')
plt.legend(['relu, 1 layer', 'relu, 2 layers', 'sigmoid, 1 layer', 'sigmoid, 2 layers'], loc='upper right')
plt.grid(True)
plt.show()
#
#
#
#np.savetxt('relu_nodes_test2L_MSE.csv', relu_MSE_nodes_test2L, delimiter=',')
#np.savetxt('relu_nodes_test3L_MSE.csv', relu_MSE_nodes_test3L, delimiter=',')
#np.savetxt('sigmoid_nodes_test2L_MSE.csv', sigmoid_MSE_nodes_test2L, delimiter=',')
#np.savetxt('sigmoid_nodes_test3L_MSE.csv', sigmoid_MSE_nodes_test3L, delimiter=',')
#
#np.savetxt('relu_epoch_test2L_MSE.csv', relu_MSE_epoch_test2L, delimiter=',')
#np.savetxt('relu_epoch_test3L_MSE.csv', relu_MSE_epoch_test3L, delimiter=',')
#np.savetxt('sigmoid_epoch_test2L_MSE.csv', sigmoid_MSE_epoch_test2L, delimiter=',')
#np.savetxt('sigmoid_epoch_test3L_MSE.csv', sigmoid_MSE_epoch_test3L, delimiter=',')
#
#
#np.savetxt('Epochs_MSE.csv', epoch_list, delimiter=',')
#np.savetxt('noed_widths_MSE.csv', nodes_list, delimiter=',')


