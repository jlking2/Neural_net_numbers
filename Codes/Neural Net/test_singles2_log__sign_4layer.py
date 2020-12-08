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

# Generate 1-degree polynomial X matrices
x_train_1D = np.hstack((x_train_0D,np.ones((y_length_train,1))))
x_test_1D = np.hstack((x_test_0D,np.ones((y_length_test,1))))

alpha_default = .1
L_default = 40
p_default = 100

y_length = len(y_train)
y_eval = np.zeros(y_length)
w_list = [""]
v_list = [""]
y_list = np.zeros((y_length,10))

for rr in range(0,10):
    for k in range(0,y_length):
        if y_train[k] == rr:
            y_eval[k] = 1
        elif y_train[k] != rr:
            y_eval[k] = 0       
                 
Y = y_eval.reshape(-1, 1)

n = np.shape(x_train_1D)[0] #examples    
m = np.shape(x_train_1D)[1] #features
p = p_default #nodes
q = 1 #number of classification problems - stick to 1 for now

W = np.random.randn(p, m);
V2 = np.random.randn(p, p); 
V1 = np.random.randn(p, p); 
U = np.random.randn(q, p);

for epoch in range(L_default):
    ind = np.random.permutation(n)
    for i in ind:
        # Forward-propagate
        H = logsig(W@(x_train_1D[[i],:]).T)
        G2 = logsig(V2@H)
        G1 = logsig(V1@G2)
        Yhat = logsig(U@G1)
        
        # Back-propagate
        beta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
        Unew = U-alpha_default*beta@G1.T    
                
        delta1 = np.multiply(U.T,np.multiply(G1,(1-G1)))@beta
        V1new = V1 - alpha_default*delta1@G2.T        

        delta2 = np.multiply(V1.T,np.multiply(G2,(1-G2)))@delta1
        V2new = V2 - alpha_default*delta2@G1.T
        
        gamma = np.multiply(V2.T,np.multiply(H,(1-H)))@delta2
        Wnew = W - alpha_default*gamma@x_train_1D[[i],:]
        
        U = Unew
        V2 = V2new
        V1 = V1new
        W = Wnew
        
    print(epoch)
        #time.sleep(5)
        
## Final predicted labels (on training data)
H_big = logsig(W@x_train_1D.T)
G2_big = logsig(V2@H_big)
G1_big = logsig(V1@G2_big)
Yhat_big = (logsig(U@G1_big)).T

err_c1 = np.sum(abs(np.round(Yhat_big[:,0])-Y[:,0]))
print('Errors, first classifier:', err_c1)

err_rate = err_c1/y_length_train
print('Errors, first classifier:', err_rate)

#train_output = [V, W, err_c1, Yhat]

#Eval to find max y
#y_fit = np.zeros(y_length)
#counter = 0
#for tt in range(0,y_length):
#    y_fit[tt] = np.argmax(y_list[tt,:])
#    if y_fit[tt] != y_train[tt]:
#        counter = counter + 1
#total_error = counter/y_length
#print(total_error)
#return(total_error)

#
##NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
#v_list.append(train_output[0])
#w_list.append(train_output[1])
#y_list[:,rr] = np.squeeze(train_output[3])
#    
#
#

#
#
##Eval to find max y
#y_fit = np.zeros(y_length)
#counter = 0
#for tt in range(0,y_length):
#    y_fit[tt] = np.argmax(y_list[tt,:])
#    if y_fit[tt] != y_train[tt]:
#        counter = counter + 1
#total_error = counter/y_length
#print(total_error)
#return(total_error)
#    
    
def NN_2layer_Relu(alpha, L, M, X, ydata_train): 
    Y = ydata_train.reshape(-1, 1)
    
    p = np.shape(X)[1] #features
    n = np.shape(X)[0] #examples
    q = np.shape(Y)[1] #number of classification problems
    
    ## initial weights
    V = np.random.randn(M+1, q); 
    W = np.random.randn(p, M);
    
    for epoch in range(L):
        ind = np.random.permutation(n)
        for i in ind:
            # Forward-propagate
            print(np.hstack((X[[i],:]@W,np.ones((1,1)) )))
            H = np.maximum(np.hstack((X[[i],:]@W,np.ones((1,1)))),0)
            print(np.shape(H))
            print('IT IS')
            time.sleep(5)
            Yhat = np.maximum(H@V,0)
            # Backpropagate
            delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
            Vnew = V-alpha*H.T@delta
            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
            Wnew = W - alpha*X[[i],:].T@gamma
            V = Vnew
            W = Wnew
        #print(epoch)
    
    ## Final predicted labels (on training data)
    H = np.maximum(np.hstack((X@W,np.ones((n,1)))),0)
    Yhat = np.maximum(H@V,0)
    
    err_c1 = np.sum(abs(np.round(Yhat[:,0])-Y[:,0]))
    #print('Errors, first classifier:', err_c1)
    
    return [V, W, err_c1, Yhat]
