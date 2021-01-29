#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:08:32 2021

@author: jimlee
"""


import csv
import numpy as np
import math 

# open the csv file
rawData = np.genfromtxt('train.csv', delimiter=',')
data = rawData[1:,3:] # data is ready, but need to be reorganized

# Test NR
print(data[10,0]) 
''' Here change the NaN term to 0'''
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if(math.isnan(data[i,j])):
            data[i,j] = 0
''' Here change the NaN term to 0'''           
# Check if NR changed to 0.0          
print(data[10,0])         


'''
    Now We Need To Change The Data To Some Form Like This:
    Let x be the feature vector(dim = 18x1)
    And x1_1_0 means that the feature vector on 1/1 0:00 and so on
    
    [ x1_1_0 ... x1_1_23 x1_2_0 ... x1_2_23 ... x12_20_0 ... x12_20_23]
    
    The dimension of the matrix must be 18x5760
'''
    

reorganizedData = np.zeros((18,5760))

startRowIndex = 0
startColumnIndex = 0

counter = 1
for i in range(data.shape[0]):
    if counter % 18 == 0:
        reorganizedData[:,startColumnIndex:startColumnIndex + 24] = data[startRowIndex:i + 1, :]
        startRowIndex = i + 1
        startColumnIndex = startColumnIndex + 24 
    counter += 1
        
'''Now We Have The ReorganizedData, We Have To Seperate the Train_x, Train_y from it'''

X = np.zeros((5652, 162)) # Train x
y_head = np.zeros((5652,1)) # Train y
      
for month in range(12):
    for hour in range(471):
        xi = []
        for i in range(hour,hour + 9):
            xi = np.append(xi,np.transpose(reorganizedData[:, month * 480 + i]))
            
        y_head[month * 471 + hour, 0] = reorganizedData[9, month * 480 + hour + 9]           
        X[month * 471 + hour,:] = xi
''' The training data need to be normalized'''

for row in range(X.shape[0]):
    X[row,:] = (X[row,:] - X[row,:].mean()) / math.sqrt(X[row,:].var())
                                                        
                                                        
''' Now we have successfully sample 5652 sets of training data. It's time to do the iteration'''
''' Define the parameter for training'''
lr = 1000

w = np.zeros((162,1))
prevGrad = np.zeros((162,1))
n = 1

for i in range(1000000000):
    y = np.dot(X,w)
    grad = 2 * (np.dot(np.transpose(X),y-y_head))
    prevGrad = prevGrad + grad**2
    #w = w - lr * grad / (np.sqrt(prevGrad / n))
    w = w - lr * grad / (np.sqrt(prevGrad))
    n = n + 1
    ''' Calculate the error'''
    if i % 1000 == 0:
        print(np.dot(np.transpose(y-y_head), (y-y_head)))
        
        
    
    




    