# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:54:18 2021

@author: User
"""

#part b

import numpy as np

W_in=np.array([[0,-0.5,0.6,0.6,1], [-0.5,1.2,-0.5,0,1], [-0.5,1.2,0,-0.5,1], [0.3,0,0.4,0.5,1]])

X = np.array([
                [0,0,0,0,-1],
                [0,0,0,1,-1],
                [0,0,1,0,-1],
                [0,0,1,1,-1],
                [0,1,0,0,-1],
                [0,1,0,1,-1],
                [0,1,1,0,-1],
                [0,1,1,1,-1],
                [1,0,0,0,-1],
                [1,0,0,1,-1],
                [1,0,1,0,-1],
                [1,0,1,1,-1],
                [1,1,0,0,-1],
                [1,1,0,1,-1],
                [1,1,1,0,-1],
                [1,1,1,1,-1]
                ])
                
def uStep(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    y = x
    return y

h_out=uStep(np.matmul(W_in,X.T))

h_out = np.concatenate((h_out.T,-1*np.ones([16,1])), axis=1)

W_out=np.array([[1,1,1,1,0.5]])
out=uStep(np.matmul(W_out,h_out.T))

def XOR(x,y):
    return (x and (not y)) or ((not x) and y)
def logicFunction(x):
    return XOR(x[0] or (not x[1]), (not x[2]) or (not x[3]))


X_c = X[:,0:4]
out_logic = list()

for i in range(16):
    out_logic.append(logicFunction(X_c[i,:]))

print('Question 2 Part B')
print((out == out_logic).all())


#part c and d

W_in_new=np.array([[0,-0.25,0.25,0.25,0.375], [-0.25,0.25,-0.25,0,0.125], [-0.25,0.25,0,-0.25,0.125], [0.25,0,0.25,0.25,0.625]])
X_n= np.tile(X,(25,1))
std = 0.2
N = np.random.normal(0, std, 2000).reshape(400,5)
N[:,4] = 0

noisy_X_n= X_n + N

X_c_n = X_n[:,0:4]
out_logic_n=list()
for i in range(400):
    out_logic_n.append(logicFunction(X_c_n[i,:]))
    
W_out_n = W_out
    
    
#part c için accuracy
std_s = 0.1
small_noise=np.random.normal(0, std_s, 2000).reshape(400,5)
small_noise[:,4] = 0
small_noisy_X= X_n + small_noise

h_out_accuracy_s=uStep(np.matmul(W_in,small_noisy_X.T)) 
h_out_accuracy_s=np.concatenate((h_out_accuracy_s.T,-1*np.ones([400,1])), axis=1)                    
out_accuracy_s=uStep(np.matmul(W_out,h_out_accuracy_s.T))

h_out_accuracy_n=uStep(np.matmul(W_in_new,small_noisy_X.T)) 
h_out_accuracy_n=np.concatenate((h_out_accuracy_n.T,-1*np.ones([400,1])), axis=1)                    
out_accuracy_n=uStep(np.matmul(W_out_n,h_out_accuracy_n.T))

count_initial=0
for i in range(400):
    if(out_logic_n[i] == out_accuracy_s[0,i]):
        count_initial += 1
print('\nQuestion 2 Part C\nAccuracy for initial weighted network (small noise) = ' + str(count_initial/400*100) + "%")


count_after=0
for i in range(400):
    if(out_logic_n[i] == out_accuracy_n[0,i]):
        count_after += 1
print('Accuracy for new robust weighted network (small noise) = ' + str(count_after/400*100) + "%")

#part d accuracy

h_out_n = uStep(np.matmul(W_in_new,noisy_X_n.T))
h_out_n = np.concatenate((h_out_n.T,-1*np.ones([400,1])), axis=1)
out_n=uStep(np.matmul(W_out_n,h_out_n.T))

h_out_accuracy=uStep(np.matmul(W_in,noisy_X_n.T)) 
h_out_accuracy=np.concatenate((h_out_accuracy.T,-1*np.ones([400,1])), axis=1)                    
out_accuracy=uStep(np.matmul(W_out,h_out_accuracy.T))

count=0
for i in range(400):
    if(out_logic_n[i] == out_accuracy[0,i]):
        count += 1
print('\nQuestion 2 Part D\nAccuracy for initial weighted network = ' + str(count/400*100) + "%")

count_n=0
for i in range(400):
    if(out_logic_n[i] == out_n[0,i]):
        count_n += 1
print('Accuracy for new robust weighted network = ' + str(count_n/400*100) + "%")


