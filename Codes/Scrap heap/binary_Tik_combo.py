# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:21:57 2020

@author: jlking2
"""

def fit_eval(x_train_mat, y_train, lambda_1):
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
        
        #Based on several runs, lambda=1E-5 seems best
        eyeye = lambda_1*np.eye(x_width)
        
        w_hat = np.linalg.inv(x_train_mat.T@x_train_mat+eyeye)@x_train_mat.T@y_eval
        #y_train_hat = x_train_mat@w_hat
        
        #MAKE UN-SIGNED ERROR  
        #y_diff = y_train_hat - y_eval
        #err_y_diff_mat = np.linalg.norm(y_diff,2)
           
        #MAKE SIGNED ERROR
        #y_train_hat_sign = np.sign(y_train_hat)
        #y_diff_sign = np.sum(np.abs(y_train_hat_sign - y_eval))/2
        #y_diff_sign_mat = y_diff_sign
        
        #Compile weights for all numbers
        w_big[:,rr] = w_hat
        
    y_big = x_train_mat@w_big
    
    #Find the maximum for each fit 
    y_fit = np.zeros(y_length)
    counter = 0
    for qq in range(0,y_length):
        y_fit[qq] = np.argmax(y_big[qq,:])
        if y_fit[qq] != y_train[qq]:
            counter= counter + 1
    Error_rate = 1 - (y_length - counter)/y_length
    
    return (y_big, Error_rate)

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## Make function that generates X_data and w based on whether it's tik or not 
y_length = len(y_train)
y_eval = np.zeros(y_length)
    
# Generate 0-degree polynomial X matrices
x_train_0D = np.zeros((y_length,784))
for k in range(0,y_length):
    xk_train_slice = x_train[k,:,:]
    x_train_0D[k,:] = xk_train_slice.flatten('C')
x_width = np.shape(x_train_0D)[1]
w_big = np.zeros((x_width,10))

# Generate 1-degree polynomial X matrices
x_train_1D = np.zeros((y_length,785))
for k in range(0,y_length):
    xk_train_slice = x_train[k,:,:]
    x_train_1D[k,0:784] = xk_train_slice.flatten('C')
x_train_1D[:,784] = 1

# Generate 2-degree polynomial X matrices
x_train_2D = np.zeros((y_length,785+784))
for k in range(0,y_length):
    xk_train_slice = x_train[k,:,:]
    xk_train_slice_2 = x_train[k,:,:]**2
    x_train_2D[k,0:784] = xk_train_slice.flatten('C')
    x_train_2D[k,784:2*784] = xk_train_slice_2.flatten('C')
x_train_2D[:,2*784] = 1



lambda_1 = 1E-5 
(y_testo, Err_test) = fit_eval(x_train_0D, y_train, lambda_1)


#Find the maximum for each fit 
#y_fit = np.zeros(y_length)
#counter = 0
#for qq in range(0,y_length):
#    y_fit[qq] = np.argmax(y_big[qq,:])
#    if y_fit[qq] != y_train[qq]:
#        counter= counter + 1
#        
#Correct_fits =  y_length - counter       
#Success_rate = Correct_fits/y_length
#Error_rate = 1 - Success_rate

