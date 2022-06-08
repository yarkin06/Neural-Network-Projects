# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:11:26 2021

@author: User
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py

def q2():

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


def q3():

    
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
    fig = plt.figure()
    for i in range(trainsize):
        if(counting == trainlbls[i]):
              ax = plt.subplot(5, 7, counting)
              plt.imshow(trainims[:,:,i])
              ax.axis('off')
              counting += 1
    
    
    co_matrix = np.zeros([26,26])
    for a in range(26):
        for b in range(26):
            corrcoef = np.corrcoef(trainims[:,:,a*200].flat, trainims[:,:,b*200].flat)
            co_matrix[a,b]=corrcoef[1,0]
         
    fig2 = plt.figure()
    plt.imshow(co_matrix)
    
    co_matrix_2 = np.zeros([26,26])  
    for a in range(26):
        for b in range(26):
            corrcoef_2 = np.corrcoef(trainims[:,:,a*199].flat, trainims[:,:,b*200].flat)
            co_matrix_2[a,b]=corrcoef_2[1,0]
            
    fig3 = plt.figure()
    plt.imshow(co_matrix_2)
    
    """
    partb
    """
    
    one_encoder = np.zeros([26,5200])
    
    for i in range(trainsize):
        one_encoder[int(trainlbls[i])-1,i] = 1
    
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
        
        bupdate = 2*(sigderiv(erro,out_ne))
        
        weight_c -= updates(learning_rate,wupdate)
        bias_c -= updates(learning_rate,bupdate)
    
        mseList.append(mse(error(ray,out_ne)))
    
    figlet = plt.figure()
    for i in range(26):
        ax2 = plt.subplot(4, 7, i+1)
        plt.imshow(weight_c[i,:].reshape(28,28))
        ax2.axis('off')
        
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

question = sys.argv[1]

def ayberk_yarkin_yildiz_21803386_hw1(question):
    
    if question == '2' :
        q2()
        
    elif question == '3' :
        q3()
      
ayberk_yarkin_yildiz_21803386_hw1(question)