# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:38:26 2020

@author: jlking2
"""

def logsig(_x):
    return 1/(1+np.exp(-_x))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def NN_3layer_log(alpha, L, M, X, ydata_train): 
    Y = y_eval.reshape(-1, 1)
    
    n = np.shape(x_train_1D)[0] #examples    
    m = np.shape(x_train_1D)[1] #features
    p = p_default #nodes
    q = 1 #number of classification problems - stick to 1 for now
    
    W = np.random.randn(p, m);
    V = np.random.randn(p, p); 
    U = np.random.randn(q, p);
    
    for epoch in range(L_default):
        ind = np.random.permutation(n)
        for i in ind:
            # Forward-propagate
            H = logsig(W@(x_train_1D[[i],:]).T)
            G = logsig(V@H)
            Yhat = logsig(U@G)
            
            # Back-propagate
            beta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
            Unew = U-alpha_default*beta@G.T    
            
            delta = np.multiply(U.T,np.multiply(G,(1-G)))@beta
            Vnew = V - alpha_default*delta@H.T
            
            gamma = np.multiply(V.T,np.multiply(H,(1-H)))@delta
            Wnew = W - alpha_default*gamma@x_train_1D[[i],:]
            
            U = Unew
            V = Vnew
            W = Wnew
            
        print(epoch)
            #time.sleep(5)
    
    H_big = logsig(W@x_train_1D.T)
    G_big = logsig(V@H_big)
    Yhat_big = (logsig(U@G_big)).T
    
    err_c1 = np.sum(abs(np.round(Yhat_big[:,0])-Y[:,0]))
    print('Errors, first classifier:', err_c1)
    
    err_rate = err_c1/y_length_train
    print('Errors, first classifier:', err_rate)
    
    train_output = [U, V, W, err_rate, Yhat]
    return(train_output)
        

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_train[10000:20000,:,:]
x_train = x_train[0:5000,:,:]

y_test = y_train[10000:20000]
y_train = y_train[0:5000]

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

# Generate 1-degree polynomial X matrices
x_train_1D = np.hstack((x_train_0D,np.ones((y_length_train,1))))
x_test_1D = np.hstack((x_test_0D,np.ones((y_length_test,1))))

## FIT FOR 1 INTEGER

alpha_default = .2
L_default = 500 #60 is good 
p_default = 500 #80 is good

#train_output = NN_3layer_log(alpha_default, L_default, p_default, x_train_1D, y_eval)
#[U, V, W, err_rate, Yhat] = train_output

y_length = y_length_train
y_eval = np.zeros(y_length)
w_list = [""]
v_list = [""]
u_list = [""]
y_list = np.zeros((y_length,10))
y_list_hat = np.zeros((y_length,10))

for rr in range(0,10):
    for k in range(0,y_length):
        if y_train[k] == rr:
            y_list[k,rr] = 1
    Y = y_eval.reshape(-1, 1)
    
n = np.shape(x_train_1D)[0] #examples    
m = np.shape(x_train_1D)[1] #features
p = p_default #nodes
q = 10 #number of classification problems - stick to 1 for now

W = np.random.randn(p, m);
V = np.random.randn(p, p); 
U = np.random.randn(q, p);

for epoch in range(L_default):
    ind = np.random.permutation(n)
    for i in ind:
        # Forward-propagate
        H = logsig(W@(x_train_1D[[i],:]).T)
        G = logsig(V@H)
        Yhat = np.squeeze(logsig(U@G))
        
        # Back-propagate
        beta = np.array([(Yhat-y_list[i,:])*Yhat*(1-Yhat)]).T
        Unew = U-alpha_default*beta@G.T    
        
        delta = np.multiply(U.T,np.multiply(G,(1-G)))@beta
        Vnew = V - alpha_default*delta@H.T
        
        gamma = np.multiply(V.T,np.multiply(H,(1-H)))@delta
        Wnew = W - alpha_default*gamma@x_train_1D[[i],:]
        
        U = Unew
        V = Vnew
        W = Wnew
        
    print(epoch)

H_big = logsig(W@x_train_1D.T)
G_big = logsig(V@H_big)
Yhat_big = (logsig(U@G_big)).T

counter = 0
y_result = np.zeros((y_length,1))
for bb in range (0,y_length):
    y_result[bb] = np.argmax(Yhat_big[bb,:])
    if y_result[bb] != y_train[bb]:
        counter = counter + 1

err_rate = counter / y_length
print('Errors, classifier training:', err_rate)            

##Testing
Yhat_test = np.zeros((y_length_test,10))
for i in range(0,y_length_test):
    H = logsig(W@(x_test_1D[[i],:]).T)
    G = logsig(V@H)
    Yhat_test[i,:] = np.squeeze(logsig(U@G))

counter2 = 0
y_result2 = np.zeros((y_length_test,1))
for bb in range (0,y_length_test):
    y_result2[bb] = np.argmax(Yhat_test[bb,:])
    if y_result2[bb] != y_test[bb]:
        counter2 = counter2 + 1

err_rate2 = counter2 / y_length_test
print('Errors, classifier testing:', err_rate2)


#time.sleep(5)
#print('Errors, first classifier:', err_c1)

#
#H_test = logsig(W@x_test_1D.T)
#G_test = logsig(V@H_test)

#    return [V, W, err_c1, Yhat]
