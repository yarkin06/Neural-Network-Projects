# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:46:43 2021

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

filename2 = 'assign2_data2.h5'
f2 = h5py.File(filename2,'r+')

"""
data olanlar in, labels olanlar out
"""
words = np.array(f2['words'])
trainin = np.array(f2['trainx'])
trainout = np.array(f2['traind'])
valin = np.array(f2['valx'])
valout = np.array(f2['vald'])
testin = np.array(f2['testx'])
testout = np.array(f2['testd'])

class Network2:
    def __init__(self, dim, neuron, activ, std, mean=0):
        self.dim = dim
        self.neuron = neuron
        self.activ = activ
        if self.activ == 'sigmoid' or self.activ == 'softmax':
            self.bias = np.random.normal(mean,std,neuron).reshape(neuron,1)
            self.weight = np.random.normal(mean,std,dim*neuron).reshape(neuron,dim)
            self.param = np.concatenate((self.weight,self.bias), axis=1)
        elif self.activ == 'wordembed':
            self.d = neuron
            self.idim = dim
            self.weight = np.random.normal(mean,std,self.d*self.idim).reshape(self.d,self.idim)
        self.chan = None
        self.err = None
        self.prevAct = None
        self.prevchan = 0
        
    def activation(self,x):
        if (self.activ == 'sigmoid'):
            return np.exp(2*x)/(1+np.exp(2*x))
        elif (self.activ == 'softmax'):
            return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)),axis=0)
        elif (self.activ == 'wordembed'):
            return x
        
    def activations(self,x):
        if self.activ == 'sigmoid' or self.activ == 'softmax':
            if (x.ndim == 1):
                x=x.reshape(x.shape[0],1)
            self.prevAct = self.activation(np.matmul(self.param,np.r_[x, [np.ones(x.shape[1])*-1]]))
        elif self.activ == 'wordembed':
            nextlayer = np.zeros((x.shape[0],x.shape[1], self.idim))
            for k in range(nextlayer.shape[0]):
                nextlayer[k,:,:] = self.activation(np.matmul(x[k,:,:], self.weight))
            nextlayer = nextlayer.reshape((nextlayer.shape[0],nextlayer.shape[1]*nextlayer.shape[2]))
            self.prevAct = nextlayer.T
        return self.prevAct
    
    def derAct(self,a):
        if self.activ == 'sigmoid':
            return 2*(a*(1-a))
        elif self.activ == 'softmax':
            return a*(1-a)
        elif self.activ == 'wordembed':
            return np.ones(a.shape)
    
    def __repr__(self):
        return 'Input_Dim: '+str(self.dim)+', Neuron #: '+str(self.neuron)+ '\n Activation: '+ self.activ

def vec(x,y):
    nextt = np.zeros(y)
    nextt[x-1] = 1
    return nextt

def checkndim(x):
    if x.ndim == 1:
        x = x.reshape(x.shape[0],1)
    return x

def m1(x,y):
    nextt = np.zeros((x.shape[0],x.shape[1],y))
    for a in range(x.shape[0]):
        for b in range(x.shape[1]):
            nextt[a,b,:]=vec(x[a,b],y)
    return nextt

def graph2(x):
    plt.figure()
    plt.plot(x)
    #plt.plot(y)
    #plt.legend(['Validation Loss','Train Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Error')

def m2(x,y):
    nextt = np.zeros((x.shape[0],y))
    for a in range(x.shape[0]):
        nextt[a,:]=vec(x[a],y)
    return nextt

def pb(x,y,z,k,dici):
    randoma = np.random.permutation(len(testin))[0:5]
    tests = m1(x[randoma],dici)
    testsl = m2(y[randoma],dici)
    t10 = z.top10(tests, 10)
    #print(t10)
    for i in range(5):
        print('[' + str(i+1) + ']' + str(k[tests[i,0]-1].decode('utf-8')) + '  ' + str(k[tests[i,1]-1].decode('utf-8')) + '  ' + str(k[tests[i,2]-1].decode('utf-8')))
        print('True label= '+str(k[testsl[i,3]-1]))
        stri = 'The Top-10 Candidates are: {'
        for j in range(10):
            stri += (str(y[t10[j,i]].decode('utf-8'))) + '  '
        print(stri+'}')

   
class PropogateWords:
    def __init__(self):
        self.lays = []
    
    def appendLayers(self,lay):
        self.lays.append(lay)
        
    def forward(self,imp):
        nextt = imp
        for layers in self.lays:
            nextt = layers.activations(nextt)
        return nextt
    
    def predict(self,imp):
        nextt = self.forward(imp)
        if nextt.ndim == 1:
            return np.argmax(nextt)
        return np.argmax(nextt, axis=0)
    
    def romando(self,imp,out):
        randindex = np.random.permutation(len(imp))
        imp,out = imp[randindex],out[randindex]
        return imp,out
    
    def top10(self,imp,n):
        nextt = self.forward(imp)
        return np.argpartition(nextt, -n, axis=0)[-n:]
    
    def BP(self,imp,out,lrate,batch,momentum):
        network_output = self.forward(imp)
        for k in reversed(range(len(self.lays))):
            layers = self.lays[k]
            if layers == self.lays[-1]:
                layers.chan = out - network_output
            else:
                nextlayers = self.lays[k+1]
                nextlayers.weight = nextlayers.param[:,0:nextlayers.weight.shape[1]]
                layers.err = np.matmul(nextlayers.weight.T, nextlayers.chan)
                layers.chan = (layers.derAct(layers.prevAct)) * layers.err
                
        for k in range(len(self.lays)):
            layers = self.lays[k]
            if  k==0:
                checkndim(imp)
                useimp = imp
            else:
                useimp = np.r_[self.lays[k-1].prevAct, [np.ones(self.lays[k-1].prevAct.shape[1])*-1]]
                
            if layers.activ == 'sigmoid' or layers.activ == 'softmax':
                ch = (lrate*np.matmul(layers.chan, useimp.T))/batch
                layers.param = layers.param + (ch + momentum*layers.prevchan)
            elif layers.activ == 'wordembed':
                chan3 = layers.chan.reshape((3,batch,layers.idim))
                useimp = np.transpose(useimp, (1,2,0))
                ch = np.zeros((useimp.shape[1], chan3.shape[2]))
                for a in range(chan3.shape[0]):
                    ch = ch + lrate*np.matmul(useimp[a,:,:], chan3[a,:,:])
                ch = ch/batch
                layers.weight = layers.weight + (ch + momentum*layers.prevchan)
            layers.prevchan = ch
            
    def workout(self,imp,out,impT,outT,lrate,epoch,batch,momentum):
        entlist = []
        for x in range(epoch):
            print('\nEpoch',x)
                
            imp,out = self.romando(imp, out)
            bat = int(np.floor(len(imp)/batch))
                
            for j in range(bat):
                batimp = m1(imp[batch*j:batch*(j+1)],d)
                batout = m2(out[batch*j:batch*(j+1)],d).T
                self.BP(batimp,batout,lrate,batch,momentum)
                
            ov = self.forward(impT)
            CEerr = -np.sum(np.log(ov)*outT.T)/ov.shape[1]
            print('Cross-Entropy Error: ',CEerr)
            entlist.append(CEerr)
                
            av = np.sum(self.predict(impT) == np.argmax(outT.T, axis=0))
            print('Correct: ',av)
            print('Accuracy: ', (av/ov.shape[1])*100,'%')
                
        return entlist
        
    def __repr__(self):
        strr = ''
        for i, layers in enumerate(self.lays):
            strr += 'Layer ' + str(i) + ': ' + layers.__repr__() + '\n'
        return strr

""" part a"""
print('Question-2 Part-a')
p = 256
p2 = 128
p3 = 64
idim = 32
idim2 = 16
idim3 = 8
d = 250
lrate2 = 0.35
momentum2 = 0.85
batch2 = 250
epoch2 = 50

vin = m1(valin,d)
vout = m2(valout,d)
tin = m1(trainin,d)
tout = m2(trainout,d)


print('\nD = 32, P = 256')
goo2 = PropogateWords()
goo2.appendLayers(Network2(idim,d,'wordembed',0.2))
goo2.appendLayers(Network2(3*idim,p,'sigmoid',0.2))
goo2.appendLayers(Network2(p,d,'softmax',0.2))

# print('\nValidation')
err1 = goo2.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
# print('\nTrain')
# err11 = goo2.workout(trainin,trainout,tin,tout,lrate2,epoch2,batch2,momentum2)

# print('\nD = 16, P = 128')
# goo22 = PropogateWords()
# goo22.appendLayers(Network2(idim2,d,'wordembed',0.2))
# goo22.appendLayers(Network2(3*idim2,p2,'sigmoid',0.2))
# goo22.appendLayers(Network2(p2,d,'softmax',0.2))

# print('\nValidation')
# err2 = goo22.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
# print('\nTrain')
# err22 = goo22.workout(tin,tout,valin,valout,lrate2,epoch2,batch2,momentum2)

# print('\nD = 8, P = 64')
# goo23 = PropogateWords()
# goo23.appendLayers(Network2(idim3,d,'wordembed',0.2))
# goo23.appendLayers(Network2(3*idim3,p3,'sigmoid',0.2))
# goo23.appendLayers(Network2(p3,d,'softmax',0.2))

# print('\nValidation')
# err3 = goo23.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
# print('\nTrain')
# err33 = goo23.workout(tin,tout,valin,valout,lrate2,epoch2,batch2,momentum2)

# graph2(err1)
# plt.title('Cross-Entropy Error, D=32,P=256')
# plt.show()

# graph2(err2)
# plt.title('Cross-Entropy Error, D=16,P=128')
# plt.show()

# graph2(err3)
# plt.title('Cross-Entropy Error, D=8,P=64')
# plt.show()



""" part b"""

# pb(testin,testout,goo2,words,d)

randoma = np.random.permutation(len(testin))[0:5]
#     #tests = m1(testin[randoma],d)
# tests = testin[randoma]
# tests1 = m1(tests,d)
# testsl = testout[randoma]
# testsl = testsl.reshape(len(testsl),1)
# testsl1 = m2(testsl,d)
# t10 = goo2.top10(tests1, 10)
#     #print(t10)
# for i in range(5):
#     print('[' + str(i+1) + ']' + str(words[tests[i,0]-1].decode('utf-8')) + ', ' + str(words[tests[i,1]-1].decode('utf-8')) + ', ' + str(words[tests[i,2]-1].decode('utf-8')))
#     print('True label= ' + str(words[testsl[i,0]-1].decode('utf-8')))
#     stri = 'The Top-10 Candidates are: {'
#     for j in range(10):
#         stri += (str(words[t10[j,i]].decode('utf-8'))) + ', '
#     print(stri+'}')
    
    
tests = testin[randoma]
tests1 = m1(tests,d)
testsl = testout[randoma]
testsl = testsl.reshape(len(testsl),1)
t10 = goo2.top10(tests1, 10)
for i in range(5):
    print('[' + str(i+1) + ']' + str(words[tests[i,0]-1].decode('utf-8')) + ', ' + str(words[tests[i,1]-1].decode('utf-8')) + ', ' + str(words[tests[i,2]-1].decode('utf-8')))
    print('True label= ' + str(words[testsl[i,0]-1].decode('utf-8')))
    stri = 'The Top-10 Candidates are: {'
    for j in range(10):
        stri += (str(words[t10[j,i]].decode('utf-8'))) + ', '
    print(stri+'}')