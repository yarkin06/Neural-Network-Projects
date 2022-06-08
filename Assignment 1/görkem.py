# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:34:52 2021

@author: User
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
file= "assign1_data1.h5"
f1= h5py.File(file,'r+')
data = np.array(f1)
test_ims = np.array(f1["testims"])
test_lbls = np.array(f1["testlbls"])
train_ims = np.array(f1["trainims"])
train_lbs = np.array(f1["trainlbls"])

# train_ims = f1['trainims'][()].astype('float64')
# train_lbs = f1['trainlbls'][()].astype('float64')
# #test data
# test_ims = f1['testims'][()].astype('float64')
# test_lbls = f1['testlbls'][()].astype('float64')

train_ims = train_ims.T
test_ims = test_ims.T


sample_ind = list()
train = train_lbs.size
count = 1
fig = plt.figure()
for i in range(train):
    if(count == train_lbs[i]):
        sub = plt.subplot(5, 6, count)
        plt.imshow(train_ims[:,:,i])
        sub.axis('off')
        #ax.autoscale(False)
        sample_ind.append(i)
        count += 1


#Within class
cor_mat = np.full([26,26],0).astype('float64')

for n in range(26):
    for m in range(26):
        coef = np.corrcoef(train_ims[:,:,n*200].ravel(),train_ims[:,:,m*200].ravel())
        cor_mat[n,m] = coef[1,0]

display(cor_mat)

# #Accross class
cor_mat2 = np.full([26,26],0).astype('float64')

for n in range(26):
    for m in range(26):
        coef2 = np.corrcoef(train_ims[:,:,n*199].ravel(),train_ims[:,:,m*200].ravel())
        cor_mat2[n,m] = coef2[1,0]
fig2 , sub2 = plt.subplots(2)
sub2[0].imshow(cor_mat)
sub2[1].imshow(cor_mat2)

display(cor_mat2)
"""From now on part B codes"""






"""
num_class = 26
corr_matrix = np.zeros(num_class**2).reshape(num_class,num_class)
for i in range(num_class):
    for j in range(num_class):
        corr_matrix[i,j] = np.corrcoef(train_ims[:,:,sample_ind[i]].flat, train_ims[:,:,sample_ind[j]].flat)[0,1]
cormat = pd.DataFrame(corr_matrix)
display(cormat)
#for i in range(25):
 #   if train_ims[]
"""
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
"""
random gaussian
mu = 0,
sigma = 0.1
random.gauss(mu,sigma)

sigmoid

def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)



"""