# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:04:00 2020

@author: jlking2
"""

def logsig(_x):
    return 1/(1+np.exp(-_x))

#def Relu(_x):
#    if _x < 0 :
#        relu = 0
#    else:
#        relu = _x
#    return relu
#
#def Relu_mat(_x):
#    xtest = _x[0]
#    print(_x)
#    print('then')
#    print(xtest)
#    print(np.size(xtest))
#    relu = np.zeros((np.size(xtest),1))
#    for nn in range(0,np.size(_x)):    
#        print(xtest[nn])
#        if xtest[nn] < 0 :
#            relu[nn] = 0
#        else:
#            relu[nn] = xtest[nn]    
#    return relu


def NN_2layer_log(alpha, L, M, X, ydata_train): 
    ## format training data
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
            H = logsig(np.hstack((X[[i],:]@W,np.ones((1,1)) )))
            #print(np.shape(H))
            Yhat = logsig(H@V)
            # Backpropagate
            delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
            Vnew = V-alpha*H.T@delta
            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
            Wnew = W - alpha*X[[i],:].T@gamma
            V = Vnew
            W = Wnew
        #print(epoch)
    
    ## Final predicted labels (on training data)
    H = logsig(np.hstack((X@W,np.ones((n,1)))))
    Yhat = logsig(H@V)
    
    err_c1 = np.sum(abs(np.round(Yhat[:,0])-Y[:,0]))
    #print('Errors, first classifier:', err_c1)
    
    return [V, W, err_c1, Yhat]

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


#def NN_2layer_sign(alpha, L, M, Xdata_train, ydata_train): 
#    p = np.shape(Xdata_train)[1] #features
#    n = np.shape(Xdata_train)[0] #examples
#    
#    ## generate training data
#    X = Xdata_train
#    Y = ydata_train.reshape(-1, 1)
#    
#    ## Train NN
#    Xb = np.hstack((np.ones((n,1)), X))
#    q = np.shape(Y)[1] #number of classification problems
#    
#    ## initial weights
#    V = np.random.randn(M+1, q); 
#    W = np.random.randn(p+1, M);
#    
#    for epoch in range(L):
#        ind = np.random.permutation(n)
#        for i in ind:
#            # Forward-propagate
#            H = np.sign(np.hstack((np.ones((1,1)), Xb[[i],:]@W)))
#            #print(np.shape(H))
#            Yhat = np.sign(H@V)
#            # Backpropagate
#            delta = (Yhat-Y[[i],:])*Yhat*(1-Yhat)
#            Vnew = V-alpha*H.T@delta
#            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
#            Wnew = W - alpha*Xb[[i],:].T@gamma
#            V = Vnew
#            W = Wnew
#    
#    ## Final predicted labels (on training data)
#    H = np.sign(np.hstack((np.ones((n,1)), Xb@W)))
#    Yhat = np.sign(H@V)
#    
#    err_c1 = np.sum(abs(np.round(Yhat[:,0])-Y[:,0]))
#    #print('Errors, first classifier:', err_c1)
#    
#    return [V, W, err_c1, Yhat]

def Get_error_NN_log(alpha,L,M,x_train_1D,y_train):
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
        train_output = NN_2layer_log(alpha, L, M, x_train_1D, y_eval)
        #NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
        v_list.append(train_output[0])
        w_list.append(train_output[1])
        y_list[:,rr] = np.squeeze(train_output[3])
            
    #Eval to find max y
    y_fit = np.zeros(y_length)
    counter = 0
    for tt in range(0,y_length):
        y_fit[tt] = np.argmax(y_list[tt,:])
        if y_fit[tt] != y_train[tt]:
            counter = counter + 1
    total_error = counter/y_length
    print(total_error)
    return(total_error)
    
def Get_error_NN_Relu(alpha,L,M,x_train_1D,y_train):
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
        train_output = NN_2layer_Relu(alpha, L, M, x_train_1D, y_eval)
        #NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
        v_list.append(train_output[0])
        w_list.append(train_output[1])
        y_list[:,rr] = np.squeeze(train_output[3])
            
    #Eval to find max y
    y_fit = np.zeros(y_length)
    counter = 0
    for tt in range(0,y_length):
        y_fit[tt] = np.argmax(y_list[tt,:])
        if y_fit[tt] != y_train[tt]:
            counter = counter + 1
    total_error = counter/y_length
    print(total_error)
    return(total_error)

#def Get_error_NN_sign(alpha,L,M,x_train_0D,y_train):
#    y_length = len(y_train)
#    y_eval = np.zeros(y_length)
#    w_list = [""]
#    v_list = [""]
#    y_list = np.zeros((y_length,10))
#    
#    for rr in range(0,10):
#        for k in range(0,y_length):
#            if y_train[k] == rr:
#                y_eval[k] = 1
#            elif y_train[k] != rr:
#                y_eval[k] = 0
#        train_output = NN_2layer_sign(alpha, L, M, x_train_0D, y_eval)
#        #NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
#        v_list.append(train_output[0])
#        w_list.append(train_output[1])
#        y_list[:,rr] = np.squeeze(train_output[3])
#            
#    #Eval to find max y
#    y_fit = np.zeros(y_length)
#    counter = 0
#    for tt in range(0,y_length):
#        y_fit[tt] = np.argmax(y_list[tt,:])
#        if y_fit[tt] != y_train[tt]:
#            counter = counter + 1
#    total_error = counter/y_length
#    print(total_error)
#    return(total_error)
#    
#def Get_error_NN_Relu(alpha,L,M,x_train_0D,y_train):
#    y_length = len(y_train)
#    y_eval = np.zeros(y_length)
#    w_list = [""]
#    v_list = [""]
#    y_list = np.zeros((y_length,10))
#    
#    for rr in range(0,10):
#        for k in range(0,y_length):
#            if y_train[k] == rr:
#                y_eval[k] = 1
#            elif y_train[k] != rr:
#                y_eval[k] = 0
#        train_output = NN_2layer_Relu(alpha, L, M, x_train_0D, y_eval)
#        #NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
#        v_list.append(train_output[0])
#        w_list.append(train_output[1])
#        y_list[:,rr] = np.squeeze(train_output[3])
#            
#    #Eval to find max y
#    y_fit = np.zeros(y_length)
#    counter = 0
#    for tt in range(0,y_length):
#        y_fit[tt] = np.argmax(y_list[tt,:])
#        if y_fit[tt] != y_train[tt]:
#            counter = counter + 1
#    total_error = counter/y_length
#    print(total_error)
#    return(total_error)

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
L_default = 25
M_default = 30

#Default_eval = Get_error_NN_log(alpha_default,L_default,M_default,x_train_1D,y_train)
Default_eval_Relu = Get_error_NN_Relu(alpha_default,L_default,M_default,x_train_1D,y_train)
#Default_eval_sign = Get_error_NN_sign(alpha_default,L_default,M_default,x_train_0D,y_train)

print('done')
#alpha_list = [.1, .2, .4]
#M_list = [x for x in range(10,80,5)]
#L_list = [x for x in range(10,100,10)]
#
#alpha_err_rate_mat = np.zeros(len(alpha_list))
#M_err_rate_mat = np.zeros(len(M_list))
#L_err_rate_mat = np.zeros(len(L_list))
#
#for aa in range(0,len(alpha_list)):
#    alpha_err_rate_mat[aa] = Get_error_NN_log(alpha_list[aa],L_default,M_default,x_train_0D,y_train)
#    
#for mm in range(0,len(M_list)):
#    M_err_rate_mat[mm] = Get_error_NN_log(alpha_default,L_default,M_list[mm],x_train_0D,y_train)
#
#for ll in range(0,len(L_list)):
#    L_err_rate_mat[ll] = Get_error_NN_log(alpha_default,L_list[ll],M_default,x_train_0D,y_train)
#
#plt.figure(1)
#plt.scatter(alpha_list, alpha_err_rate_mat)
#plt.title('Predicted Labels, changing alpha')
#plt.show()
#
#plt.figure(2)
#plt.scatter(M_list, M_err_rate_mat)
#plt.title('Predicted Labels, changing M')
#plt.show()
#
#plt.figure(3)
#plt.scatter(L_list, L_err_rate_mat)
#plt.title('Predicted Labels, changing L')
#plt.show()
#
#
##alpha_list = .1
##M_list = [x for x in range(10,80,5)]
##L_list = [x for x in range(10,100,10)]
##
##alpha_err_rate_mat = np.zeros(len(alpha_list))
##M_err_rate_mat = np.zeros(len(M_list))
##L_err_rate_mat = np.zeros(len(L_list))
##
##for aa in range(0,len(alpha_list)):
##    alpha_err_rate_mat[aa] = Get_error_NN(alpha_list[aa],L_default,M_default,x_train_0D,y_train)
##    
##for mm in range(0,len(M_list)):
##    M_err_rate_mat[mm] = Get_error_NN(alpha_default,L_default,M_list[mm],x_train_0D,y_train)
##
##for ll in range(0,len(L_list)):
##    L_err_rate_mat[ll] = Get_error_NN(alpha_default,L_list[ll],M_default,x_train_0D,y_train)
##
##plt.figure(1)
##plt.scatter(alpha_list, alpha_err_rate_mat)
##plt.title('Predicted Labels, changing alpha')
##plt.show()
##
##plt.figure(2)
##plt.scatter(M_list, M_err_rate_mat)
##plt.title('Predicted Labels, changing M')
##plt.show()
##
##plt.figure(3)
##plt.scatter(L_list, L_err_rate_mat)
##plt.title('Predicted Labels, changing L')
##plt.show()
#
#
#
#
##test_err = Get_error_NN(.1,10,20,x_train_0D,y_train)
##print(test_err)
##
##y_length = len(y_train)
##y_eval = np.zeros(y_length)
##w_list = [""]
##v_list = [""]
##y_list = np.zeros((y_length,10))
##
##for rr in range(0,10):
##    for k in range(0,y_length):
##        if y_train[k] == rr:
##            y_eval[k] = 1
##        elif y_train[k] != rr:
##            y_eval[k] = 0
##    train_output = NN_2layer(.1, 10, 20, x_train_0D, y_eval)
##    #NN_2layer(alpha, L-epochs, M-nodes, Xdata_train, ydata_train): 
##    v_list.append(train_output[0])
##    w_list.append(train_output[1])
##    y_list[:,rr] = np.squeeze(train_output[3])
##        
###Eval to find max y
##y_fit = np.zeros(y_length)
##counter = 0
##for tt in range(0,y_length):
##    y_fit[tt] = np.argmax(y_list[tt,:])
##    if y_fit[tt] != y_train[tt]:
##        counter = counter + 1
##total_error = counter/y_length
##print(total_error)
#
#
##
##def get_error(x_test_mat, y_test_mat, w_big):
##    y_big = x_test_mat@w_big
##    y_length = np.shape(y_big)[0]
##    #Find the maximum for each fit
##    y_fit = np.zeros(y_length)
##    counter = 0
##    for qq in range(0,y_length):
##        y_fit[qq] = np.argmax(y_big[qq,:])
##        if y_fit[qq] != y_test_mat[qq]:
##            counter = counter + 1
##    Error_rate = 1 - (y_length - counter)/y_length    
##    return (Error_rate)
##
#
#    
#    ## Final predicted labels (on training data)
#    #H = logsig(np.hstack((np.ones((n,1)), Xb@W)))
#    #Yhat = logsig(H@V)
#
#
#
##(alpha/step size , L/ epochs, M/# of hidden nodes, Xdata, Ydata)
#
#
#
##
##p = np.shape(x_train_0D)[1] #features
##n = np.shape(x_train_0D)[0] #examples
##
#### generate training data
##X = x_train_0D
##Y = y_train.reshape(-1, 1)
##
#### Train NN
##Xb = np.hstack((np.ones((n,1)), X))
##q = np.shape(Y)[1] #number of classification problems
##M = 12 #number of hidden nodes
##
#### initial weights
##V = np.random.randn(M+1, q); 
##W = np.random.randn(p+1, M);
##
##alpha = 0.1 #step size
##L = 10 #number of epochs
##
##for epoch in range(L):
##    ind = np.random.permutation(n)
##    for i in ind:
##        # Forward-propagate
##        H = logsig(np.hstack((np.ones((1,1)), Xb[[i],:]@W)))
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









