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
#
#def NN_3layer_log(alpha, L, M, X, ydata_train): 
#    Y = y_eval.reshape(-1, 1)
#    
#    n = np.shape(x_train_1D)[0] #examples    
#    m = np.shape(x_train_1D)[1] #features
#    p = p_default #nodes
#    q = 1 #number of classification problems - stick to 1 for now
#    
#    W = np.random.randn(p, m);
#    V = np.random.randn(p, p); 
#    U = np.random.randn(q, p);
#    
#    for epoch in range(L_default):
#        ind = np.random.permutation(n)
#        for i in ind:
#            # Forward-propagate
#            H = logsig(W@(x_train_1D[[i],:]).T)
#            G = logsig(V@H)
#            Yhat = logsig(U@G)
#            
#            # Back-propagate
#            beta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
#            Unew = U-alpha_default*beta@G.T    
#            
#            delta = np.multiply(U.T,np.multiply(G,(1-G)))@beta
#            Vnew = V - alpha_default*delta@H.T
#            
#            gamma = np.multiply(V.T,np.multiply(H,(1-H)))@delta
#            Wnew = W - alpha_default*gamma@x_train_1D[[i],:]
#            
#            U = Unew
#            V = Vnew
#            W = Wnew
#            
#        print(epoch)
#            #time.sleep(5)
#    
#    H_big = logsig(W@x_train_1D.T)
#    G_big = logsig(V@H_big)
#    Yhat_big = (logsig(U@G_big)).T
#    
#    err_c1 = np.sum(abs(np.round(Yhat_big[:,0])-Y[:,0]))
#    print('Errors, first classifier:', err_c1)
#    
#    err_rate = err_c1/y_length_train
#    print('Errors, first classifier:', err_rate)
#    
#    train_output = [U, V, W, err_rate, Yhat]
#    return(train_output)
#        

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_train[1000:2000,:,:]
x_train = x_train[0:5000,:,:]

y_test = y_train[1000:2000]
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
L_default = 100 #60 is good 
p_default = 80 #80 is good
p_default2 = 70

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
p2 = p_default2
p3 = p_default #nodes
q = 10 #number of classification problems - stick to 1 for now

W = np.random.randn(p, m);
V2 = np.random.randn(p2, p+1); 
V1 = np.random.randn(p3, p2+1); 
U = np.random.randn(q, p3+1);

for epoch in range(L_default):
    ind = np.random.permutation(n)
    for i in ind:
        # Forward-propagate
        H = logsig(W@(x_train_1D[[i],:]).T)
        H = np.vstack((H,np.ones((1,1))))
        G2 = logsig(V2@H)
        G2 = np.vstack((G2,np.ones((1,1))))
        G1 = logsig(V1@G2)
        G1 = np.vstack((G1,np.ones((1,1))))
        Yhat = np.squeeze(logsig(U@G1))
        
        # Back-propagate
        beta = np.array([(Yhat-y_list[i,:])*Yhat*(1-Yhat)]).T
        Unew = U-alpha_default*beta@G1.T    
        #H[:-1,:] or H[1:,:]
        
        delta1 = np.multiply(U.T[:-1,:],np.multiply(G1[:-1,:],(1-G1[:-1,:])))@beta
        V1new = V1 - alpha_default*delta1@G2.T

        delta2 = np.multiply(V1.T[:-1,:],np.multiply(G2[:-1,:],(1-G2[:-1,:])))@delta1
        #delta2 = np.multiply(V1.T,np.multiply(G2,(1-G2)))@delta1
        V2new = V2 - alpha_default*delta2@G1.T
        
        gamma = np.multiply(V2.T[:-1,:],np.multiply(H[:-1,:],(1-H[:-1,:])))@delta2
        Wnew = W - alpha_default*gamma@x_train_1D[[i],:]
        
        U = Unew
        V1 = V1new
        V2 = V2new
        W = Wnew
        
    print(epoch)

H_big = logsig(W@x_train_1D.T)
H_big = np.vstack((H_big,np.ones((1,np.shape(H_big[1])[0]))))
G2_big = logsig(V2@H_big)
G2_big = np.vstack((G2_big,np.ones((1,np.shape(G2_big[1])[0]))))
G1_big = logsig(V1@G2_big)
G1_big = np.vstack((G1_big,np.ones((1,np.shape(G1_big[1])[0]))))
Yhat_big = (logsig(U@G1_big)).T

counter = 0
y_result = np.zeros((y_length,1))
for bb in range (0,y_length):
    y_result[bb] = np.argmax(Yhat_big[bb,:])
    if y_result[bb] != y_train[bb]:
        counter = counter + 1

err_rate = counter / y_length

print('Errors, classifier training:', err_rate)            

#
###Testing
#Yhat_test = np.zeros((y_length_test,10))
#for i in range(0,y_length_test):
#    # Forward-propagate
#    H = logsig(W@(x_test_1D[[i],:]).T)
#    G2 = logsig(V2@H)
#    G1 = logsig(V1@G2)
#    Yhat_test[i,:] = np.squeeze(logsig(U@G1))
#
#counter2 = 0
#y_result2 = np.zeros((y_length_test,1))
#for bb in range (0,y_length_test):
#    y_result2[bb] = np.argmax(Yhat_test[bb,:])
#    if y_result2[bb] != y_test[bb]:
#        counter2 = counter2 + 1
#
#err_rate2 = counter2 / y_length_test
#print('Errors, classifier testing:', err_rate2)
#
#
