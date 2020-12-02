# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:04:00 2020

@author: jlking2
"""

def logsig(_x):
    return 1/(1+np.exp(-_x))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.svm import LinearSVC

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_train[10000:20000,:,:]
x_train = x_train[0:1000,:,:]

y_test = y_train[10000:20000]
y_train = y_train[0:1000]

y_length_train = len(y_train)
y_length_test = len(y_test)

# Generate 0-degree polynomial X train and test matrices
x_train_0D = np.zeros((y_length_train,784))
for k in range(0,y_length_train):
    xk_train_slice = x_train[k,:,:]
    x_train_0D[k,:] = xk_train_slice.flatten('C')

x_test_0D = np.zeros((y_length_test,784))
for k in range(0,y_length_test):
    xk_test_slice = x_test[k,:,:]
    x_test_0D[k,:] = xk_test_slice.flatten('C')

def NN_2layer(alpha, L, M, Xdata_train, ydata_train): 
    p = np.shape(Xdata_train)[1] #features
    n = np.shape(Xdata_train)[0] #examples
    
    ## generate training data
    X = Xdata_train
    Y = ydata_train.reshape(-1, 1)
    
    ## Train NN
    Xb = np.hstack((np.ones((n,1)), X))
    q = np.shape(Y)[1] #number of classification problems
    
    ## initial weights
    V = np.random.randn(M+1, q); 
    W = np.random.randn(p+1, M);
    
    for epoch in range(L):
        ind = np.random.permutation(n)
        for i in ind:
            # Forward-propagate
            H = logsig(np.hstack((np.ones((1,1)), Xb[[i],:]@W)))
            #print(np.shape(H))
            Yhat = logsig(H@V)
            # Backpropagate
            delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
            Vnew = V-alpha*H.T@delta
            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
            Wnew = W - alpha*Xb[[i],:].T@gamma
            V = Vnew
            W = Wnew
        print(epoch)
    
    ## Final predicted labels (on training data)
    H = logsig(np.hstack((np.ones((n,1)), Xb@W)))
    Yhat = logsig(H@V)
    
    err_c1 = np.sum(abs(np.round(Yhat[:,0])-Y[:,0]))
    print('Errors, first classifier:', err_c1)
    
    return [V, W, err_c1, Yhat]

#testo_big = NN_2layer(.1, 10, 12, x_train_0D, y_train)

y_length = len(y_train)
y_eval = np.zeros(y_length)

for rr in range(0,10):
        number_eval = rr
        for k in range(0,y_length):
            if y_train[k] == number_eval:
                y_eval[k] = 1
            elif y_train[k] != number_eval:
                y_eval[k] = 0

testo_big = NN_2layer(.1, 20, 180, x_train_0D, y_eval)
#(alpha/step size , L/ epochs, M/# of hidden nodes, Xdata, Ydata)



#
#p = np.shape(x_train_0D)[1] #features
#n = np.shape(x_train_0D)[0] #examples
#
### generate training data
#X = x_train_0D
#Y = y_train.reshape(-1, 1)
#
### Train NN
#Xb = np.hstack((np.ones((n,1)), X))
#q = np.shape(Y)[1] #number of classification problems
#M = 12 #number of hidden nodes
#
### initial weights
#V = np.random.randn(M+1, q); 
#W = np.random.randn(p+1, M);
#
#alpha = 0.1 #step size
#L = 10 #number of epochs
#
#for epoch in range(L):
#    ind = np.random.permutation(n)
#    for i in ind:
#        # Forward-propagate
#        H = logsig(np.hstack((np.ones((1,1)), Xb[[i],:]@W)))
#        #print(np.shape(H))
#        Yhat = logsig(H@V)
#        # Backpropagate
#        delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
#        Vnew = V-alpha*H.T@delta
#        gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
#        Wnew = W - alpha*Xb[[i],:].T@gamma
#        V = Vnew
#        W = Wnew
#    print(epoch)
#
### Final predicted labels (on training data)
#H = logsig(np.hstack((np.ones((n,1)), Xb@W)))
#Yhat = logsig(H@V)
#
##plt.scatter(X[:,0], X[:,1], c=Yhat[:,0])
##plt.title('Predicted Labels, first classifier')
##plt.show()
#
##plt.scatter(X[:,0], X[:,1], c=Yhat[:,1])
##plt.title('Predicted Labels, second classifier')
##plt.show()
#
#err_c1 = np.sum(abs(np.round(Yhat[:,0])-Y[:,0]))
#print('Errors, first classifier:', err_c1)
#
##err_c2 = np.sum(abs(np.round(Yhat[:,1])-Y[:,1]))
##print('Errors, second classifier:', err_c2)









