# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:30:01 2021

@author: User
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

f1= h5py.File(r'C:\Users\User\Desktop\3.2\EEE 443\Assignment 1\assign1_data1.h5','r+')
data = np.array(f1)
test1 = np.array(f1["testims"].value)
test2 = np.array(f1["testlbls"].value)
train1 = np.array(f1["trainims"].value)
train2 = np.array(f1["trainlbls"].value)


"""
x= test1[1]
x şimdi test1 in ilk image ı ve şekli 28 x 28
x = x.ravel()
ile x i 784 x 1 lik matrixe çevirdim.

x = test1[1]
y = test1[2]
r2 = np.corrcoef(test1[3],test1[3])
corr = np.cov(x,y)/ (np.std(x)*np.std(y))
a = np.correlate(x,y, mode= 'full')
"""