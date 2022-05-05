# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:59:45 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import h5py


filename = 'assign1_data1.h5'
f1 = h5py.File(filename,'r+')

testims = np.array(f1["testims"])
testlbls = np.array(f1["testlbls"])
trainims = np.array(f1["trainims"])
trainlbls = np.array(f1["trainlbls"])

trainims = trainims.T
testims = testims.T
trainsize = trainlbls.size
counting = 1
#sample_ind = list()
fig = plt.figure()
for i in range(trainsize):
    if(counting == trainlbls[i]):
          ax = plt.subplot(5, 7, counting)
          plt.imshow(trainims[:,:,i])
          ax.axis('off')
          #ax.autoscale(False)
          #sample_ind.append(i)
          counting += 1


co_matrix = np.zeros([26,26])
for a in range(26):
    for b in range(26):
        corrcoef = np.corrcoef(trainims[:,:,a*200].flat, trainims[:,:,b*200].flat)
        co_matrix[a,b]=corrcoef[1,0]
     
fig2 = plt.figure()
plt.imshow(co_matrix)
#display(co_matrix)

co_matrix_2 = np.zeros([26,26])  
for a in range(26):
    for b in range(26):
        corrcoef_2 = np.corrcoef(trainims[:,:,a*199].flat, trainims[:,:,b*200].flat)
        co_matrix_2[a,b]=corrcoef_2[1,0]
        
fig3 = plt.figure()
plt.imshow(co_matrix_2)
#display(co_matrix_2)

"""
partb
"""

one_encoder = np.zeros([26,5200])

for i in range(trainsize):
    one_encoder[int(trainlbls[i])-1,i] = 1

#print(one_encoder)

learning_rate = 0.06
learning_rate_up = 1
learning_rate_down = 0.0001
mean_3=0
std_3=0.01

def sigmoid(x):
    return 1/(1+np.exp(-x))

def norm(x):
    return x/np.max(x)

def RandWB(m,s):
    b = np.random.normal(m,s,26).reshape(26,1)
    w = np.random.normal(m,s,26*(28**2)).reshape(26,(28**2))
    return w,b

def ins(ims,onehot):
    
    rax = random.randint(0,5199)
    rag = trainims[:,:,rax].reshape(28**2,1)
    rag = norm(rag)
    #rag = rag/np.max(rag)
    ray = one_encoder[:,rax].reshape(26,1)
    return rax,rag,ray

def outa(w,b,i):
    return sigmoid(np.matmul(w,i)-b)

def mse(k):
    return np.sum((k)**2/(k.shape[0]))

def error(m,n):
    return m-n

def sigderiv(a,b):
    return (a*b*(1-b))

def updates(l,i):
    change = l*i
    return change

mseList = list()
weight_c,bias_c = RandWB(mean_3,std_3)

for i in range(10000):
    
    rax,rag,ray = ins(trainims,one_encoder)

    out_ne = outa(weight_c,bias_c,rag)

    erro=error(ray,out_ne)

    wupdate = -2*np.matmul(sigderiv(erro,out_ne),rag.T)

    #wupdate = -2*np.matmul(erro*out_ne*(1-out_ne),rag.T)
    
    bupdate = 2*(sigderiv(erro,out_ne))
    
    weight_c -= updates(learning_rate,wupdate)
    bias_c -= updates(learning_rate,bupdate)

    mseList.append(mse(error(ray,out_ne)))

figlet = plt.figure()
for i in range(26):
    ax2 = plt.subplot(4, 7, i+1)
    #weight_dig = weight_c[i,:].reshape(28,28)
    plt.imshow(weight_c[i,:].reshape(28,28))
    #plt.rcParams['figure.figsize'] = [8,8]
    ax2.axis('off')
    #ax2.autoscale(False)
    
"""part c """

weightsUp,biasUp = RandWB(mean_3,std_3)
weightsDown,biasDown = RandWB(mean_3,std_3)

mseListHi = list()
mseListLow = list()

for i in range(10000):
    rax,rag,ray = ins(trainims,one_encoder)
    out_ne = outa(weightsUp,biasUp,rag)
    erro = error(ray,out_ne)
    
    wupdate = -2*np.matmul(sigderiv(erro,out_ne),rag.T)
    bupdate = 2*(sigderiv(erro,out_ne))
    
    weightsUp -= updates(learning_rate_up,wupdate)
    biasUp -= updates(learning_rate_up,bupdate)
    
    mseListHi.append(mse(error(ray,out_ne)))

for i in range(10000):
    rax,rag,ray = ins(trainims,one_encoder)
    out_ne = outa(weightsDown,biasDown,rag)
    erro = error(ray,out_ne)
    
    wupdate = -2*np.matmul(sigderiv(erro,out_ne),rag.T)
    bupdate = 2*(sigderiv(erro,out_ne))
    
    weightsDown -= updates(learning_rate_down,wupdate)
    biasDown -= updates(learning_rate_down,bupdate)
    
    mseListLow.append(mse(error(ray,out_ne)))    

fig4 = plt.figure()
plt.plot(mseListHi)
plt.plot(mseListLow)
plt.plot(mseList)
plt.legend(["MSE for u="+str(learning_rate_up), "MSE for u="+str(learning_rate_down), "MSE for u="+str(learning_rate)])
plt.title("Mean Squared Errors for Different Learning Rates")
plt.xlabel("Iteration Number")
plt.ylabel("MSE")
plt.show()


""" part d """
testsize = testlbls.shape[0]
testims = testims.reshape(28**2,testsize)
testnorm = norm(testims)

def accuracies(w,b):
    
    bias_d = np.zeros([26,testsize])
    
    for i in range (testsize):
        bias_d[:,i] = b.flatten()
    guessino = outa(w,bias_d,testnorm)
    guessino_i = np.zeros(guessino.shape[1])
    
    for i in range (guessino.shape[1]):
        guessino_i[i] = np.argmax(guessino[:,i])+1
        
    counters = 0
    for i in range (guessino_i.shape[0]):
        if (guessino_i[i] == testlbls[i]):
            counters += 1
    
    accuracy = counters/testlbls.shape[0]*100
    return accuracy
        
print('Accuracy for learning rate =',learning_rate,':',accuracies(weight_c,bias_c),'%')
print('Accuracy for learning rate =',learning_rate_up,':',accuracies(weightsUp,biasUp),'%')
print('Accuracy for learning rate =',learning_rate_down,':',accuracies(weightsDown,biasDown),'%')
       
          