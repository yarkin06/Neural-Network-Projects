# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:07:23 2021

@author: User
"""

import numpy, random, os
lr = 0.1 #learning rate
bias = 1 #value of bias
weights = [random.random(),random.random(),random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
def Perceptron(input1, input2, input3, input4, output) :
   outputP = input1*weights[0]+input2*weights[1]+input3*weights[2]+input4*weights[3]+bias*weights[4]
   if outputP > 0 : #activation function (here Heaviside)
      outputP = 1
   else :
      outputP = 0
   error = output - outputP
   weights[0] += error * input1 * lr
   weights[1] += error * input2 * lr
   weights[2] += error * input3 * lr
   weights[3] += error * input4 * lr
   weights[4] += error * bias * lr


for i in range(200) :
   Perceptron(0,0,0,0,0) #False or false
   Perceptron(0,0,0,1,0) #False or false
   Perceptron(0,0,1,0,0) #False or false
   Perceptron(0,0,1,1,1) #False or false
   Perceptron(0,1,0,0,1) #False or false
   Perceptron(0,1,0,1,1) #False or false
   Perceptron(0,1,1,0,1) #False or false
   Perceptron(0,1,1,1,0) #False or false
   Perceptron(1,0,0,0,0) #False or false
   Perceptron(1,0,0,1,0) #False or false
   Perceptron(1,0,1,0,0) #False or false
   Perceptron(1,0,1,1,1) #False or false
   Perceptron(1,1,0,0,0) #False or false
   Perceptron(1,1,0,1,0) #False or false
   Perceptron(1,1,1,0,0) #False or false
   Perceptron(1,1,1,1,1) #True or true


x1 = int(input('X1= '))
x2 = int(input('X2= '))
x3 = int(input('X3= '))
x4 = int(input('X4= '))

outputP = x1*weights[0] + x2*weights[1] + x3*weights[2] + x4*weights[3] + bias*weights[4]
if outputP > 0 : #activation function
   outputP = 1
else :
   outputP = 0
print( "The result is : ", outputP)