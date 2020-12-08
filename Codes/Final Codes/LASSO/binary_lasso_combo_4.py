# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:21:57 2020

@author: jlking2
"""

def get_lasso_loop(x_train_mat, y_train, la_array):
    x_width = np.shape(x_train_mat)[1]
    y_length = len(y_train)
    y_eval = np.expand_dims(np.zeros(y_length),axis=1)
    #w_giant contains [elements,lambdas,numbers]    
    w_giant = np.zeros((x_width,len(la_array),10))
    for rr in range(0,10):
        number_eval = rr
        for k in range(0,y_length):
            if y_train[k] == number_eval:
                y_eval[k] = 1
            elif y_train[k] != number_eval:
                y_eval[k] = -1        
        w_giant[:,:,rr] = get_lasso(x_train_mat, y_eval, la_array)
    return w_giant

def get_lasso(A,d,la_array):
    max_iter = 10**4
    tol = 10**(-3)
    tau = 1/np.linalg.norm(A,2)**2
    n = A.shape[1]
    w = np.zeros((n,1))
    num_lam = len(la_array)
    X = np.zeros((n, num_lam))
    for i, each_lambda in enumerate(la_array):
        for j in range(max_iter):
            z = w - tau*(A.T@(A@w-d))
            w_old = w
            w = np.sign(z) * np.clip(np.abs(z)-tau*each_lambda/2, 0, np.inf)
            X[:, i:i+1] = w
            if np.linalg.norm(w - w_old) < tol:
                break
    return X

def get_w_Tik(x_train_mat, y_train, lambda_1):
    # ista_solve_hot: Iterative soft-thresholding for multiple values of
    # lambda with hot start for each case - the converged value for the previous
    # value of lambda is used as an initial condition for the current lambda.
    ## Make this into a function of lambda, x, y,... returns weights and errors
    ##GENERATE MATRICES
    x_width = np.shape(x_train_mat)[1]
    y_length = len(y_train)
    y_eval = np.zeros(y_length)
    w_big = np.zeros((x_width,10))
    
    for rr in range(0,10):
        number_eval = rr            
        for k in range(0,y_length):
            if y_train[k] == number_eval:
                y_eval[k] = 1
            elif y_train[k] != number_eval:
                y_eval[k] = -1        
        eyeye = lambda_1*np.eye(x_width)        
        w_big[:,rr] = np.linalg.inv(x_train_mat.T@x_train_mat+eyeye)@x_train_mat.T@y_eval
    return w_big

def get_error(x_test_mat, y_test_mat, w_big):
    y_big = x_test_mat@w_big
    y_length = np.shape(y_big)[0]
    #Find the maximum for each fit
    y_fit = np.zeros(y_length)
    counter = 0
    for qq in range(0,y_length):
        y_fit[qq] = np.argmax(y_big[qq,:])
        if y_fit[qq] != y_test_mat[qq]:
            counter = counter + 1
    Error_rate = 1 - (y_length - counter)/y_length    
    return (Error_rate)

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#x_test = x_train[30000:31000,:,:]
#x_train = x_train[0:20000,:,:]
#
#y_test = y_train[30000:31000]
#y_train = y_train[0:20000]

## Make function that generates X_data and w based on whether it's tik or not 
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
x_train_1D = np.zeros((y_length_train,785))
for k in range(0,y_length_train):
    xk_train_slice = x_train[k,:,:]
    x_train_1D[k,0:784] = xk_train_slice.flatten('C')
x_train_1D[:,784] = 1

x_test_1D = np.zeros((y_length_test,785))
for k in range(0,y_length_test):
    xk_test_slice = x_test[k,:,:]
    x_test_1D[k,0:784] = xk_test_slice.flatten('C')
x_train_1D[:,784] = 1

# Generate 2-degree polynomial X matrices
x_train_2D = np.zeros((y_length_train,785+784))
for k in range(0,y_length_train):
    xk_train_slice = x_train[k,:,:]
    xk_train_slice_2 = x_train[k,:,:]**2
    x_train_2D[k,0:784] = xk_train_slice.flatten('C')
    x_train_2D[k,784:2*784] = xk_train_slice_2.flatten('C')
x_train_2D[:,2*784] = 1

x_test_2D = np.zeros((y_length_test,785+784))
for k in range(0,y_length_test):
    xk_test_slice = x_test[k,:,:]
    xk_test_slice_2 = x_test[k,:,:]**2
    x_test_2D[k,0:784] = xk_test_slice.flatten('C')
    x_test_2D[k,784:2*784] = xk_test_slice_2.flatten('C')
x_test_2D[:,2*784] = 1

lambda_range1 = 1E-15*(10**np.array(range(1,10)))
lambda_range2 = 1E-6*(10**np.array(range(1,10)))
lambda_range3 = 1E3*(10**np.array(range(1,10)))
lambda_range = np.concatenate((lambda_range1,lambda_range2,lambda_range3))
len_range = len(lambda_range)

Err_0D = np.zeros(len_range)
Err_1D = np.zeros(len_range)
Err_2D = np.zeros(len_range)

w_giant_0D = get_lasso_loop(x_train_0D,y_train,lambda_range)
for m in range(0,len_range):
    w_biggo = w_giant_0D[:,m,:]
    Err_0D[m] = get_error(x_test_0D, y_test, w_biggo)
del(w_giant_0D,x_train_0D)

w_giant_1D = get_lasso_loop(x_train_1D,y_train,lambda_range)
for m in range(0,len_range):
    w_biggo = w_giant_1D[:,m,:]
    Err_1D[m] = get_error(x_test_1D, y_test, w_biggo)
del(w_giant_1D,x_train_1D)

w_giant_2D = get_lasso_loop(x_train_2D,y_train,lambda_range)
for m in range(0,len_range):
    w_biggo = w_giant_2D[:,m,:]
    Err_2D[m] = get_error(x_test_2D, y_test, w_biggo)
del(w_giant_2D,x_train_2D)

plt.figure(1)
plt.plot(lambda_range, Err_0D, 'bo')
plt.xlabel('Ridge Parameter, \u03BB')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(lambda_range, Err_1D, 'bo')
plt.xlabel('Ridge Parameter, \u03BB')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(lambda_range, Err_2D, 'bo')
plt.xlabel('Ridge Parameter, \u03BB')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.title('Error rate vs Ridge Parameter, \u03BB')
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(lambda_range, Err_0D, '-bo')
plt.plot(lambda_range, Err_1D, '-r^')
plt.plot(lambda_range, Err_2D, '-gs')
plt.xlabel('Lasso Parameter, \u03BB')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.title('Error rate vs Lasso Parameter, \u03BB')
plt.grid(True)
plt.legend(('0th degree','1st degree','2nd degree'))
plt.show()

plt.figure(5)
plt.plot(lambda_range, Err_0D, '-bo')
plt.plot(lambda_range, Err_1D, '-r^')
plt.plot(lambda_range, Err_2D, '-gs')
plt.xlabel('Lasso Parameter, \u03BB')
plt.ylabel('Error Rate')
plt.xscale('log')
plt.title('Error rate vs Lasso Parameter, \u03BB')
plt.grid(True)
plt.legend(('0th degree','1st degree','2nd degree'))
plt.xlim((1E-2,2E6))
plt.ylim((.23,.26))
plt.show()
