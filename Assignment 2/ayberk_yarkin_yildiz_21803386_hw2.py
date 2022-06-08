# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:54:55 2021

@author: User
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py



def graph(x):
    plt.figure()
    plt.plot(x)
    plt.xlabel('Epochs')
    
def graphm(x,y,z):
    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.legend(['Optimal', 'High', 'Low'])
    plt.xlabel('Epochs')
    
def graphd(x,y):
    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.legend(['Momentum = 0','Momentum = 0.5'])
    plt.xlabel('Epochs')
    
def checkndim(x):
    if x.ndim == 1:
        x = x.reshape(x.shape[0],1)
    return x

def vec(x,y):
    nextt = np.zeros(y)
    nextt[x-1] = 1
    return nextt

def m1(x,y):
    nextt = np.zeros((x.shape[0],x.shape[1],y))
    for a in range(x.shape[0]):
        for b in range(x.shape[1]):
            nextt[a,b,:]=vec(x[a,b],y)
    return nextt

def graph2(x):
    plt.figure()
    plt.plot(x)
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Error')

def m2(x,y):
    nextt = np.zeros((x.shape[0],y))
    for a in range(x.shape[0]):
        nextt[a,:]=vec(x[a],y)
    return nextt

def pb(x,y,z,dici,k):
    randoma = np.random.permutation(len(x))[0:5]
    tests = x[randoma]
    tests1 = m1(tests,dici)
    testsl = k[randoma]
    testsl = testsl.reshape(len(testsl),1)
    t10 = z.top10(tests1, 10)
    for i in range(5):
        print('[' + str(i+1) + ']' + str(y[tests[i,0]-1].decode('utf-8')) + '  ' + str(y[tests[i,1]-1].decode('utf-8')) + '  ' + str(y[tests[i,2]-1].decode('utf-8')))
        print('True label= ' + str(y[testsl[i,0]-1].decode('utf-8')))
        stri = 'The Top-10 Candidates are: {'
        for j in range(10):
            stri += (str(y[t10[j,i]].decode('utf-8'))) + '  '
        print(stri+'}')


def q1():
    
    class Network:
        def __init__(self,dim,neuron,std,mean=0):
            self.dim = dim
            self.neuron = neuron
            self.bias = np.random.normal(mean,std,neuron).reshape(neuron,1)
            self.weight = np.random.normal(mean,std,dim*neuron).reshape(neuron,dim)
            self.param = np.concatenate((self.weight,self.bias), axis=1)
            self.chan = None
            self.err = None
            self.prevAct = None
            self.prevchan = 0
        
        def activation(self, x):
            if(x.ndim == 1):
                x = x.reshape(x.shape[0],1)
            self.prevAct = np.tanh(np.matmul(self.param, np.r_[x, [np.ones(x.shape[1])*-1]]))
            return self.prevAct
        
        def derAct(self,a):
            return 1-(a**2)
        
        def __repr__(self):
            return 'Input_Dim: '+str(self.dim)+', Neuron #: '+str(self.neuron)
        
        
    class PropogateLayers:
        def __init__(self):
            self.lays = []
        
        def appendLayers(self,lay):
            self.lays.append(lay)
        def forward(self,imp):
            nextt = imp
            for layers in self.lays:
                nextt = layers.activation(nextt)
            return nextt
        def accur(self,x):
            print('Accuracy:', str(np.sum(x.predict(testimsf.T/255).T == testlbls)/len(testlbls)*100) + "%")
        
        def predict(self,imp):
            nextt = self.forward(imp)
            nextt[nextt>=0] = 1
            nextt[nextt<0] = -1
            return nextt
        
        def romando(self,imp,out):
            randindex = np.random.permutation(len(imp))
            imp,out = imp[randindex],out[randindex]
            return imp,out
             
        def BP(self,imp,out,lrate,batch,momentum=0):
            network_output = self.forward(imp)
            for k in reversed(range(len(self.lays))):
                layers = self.lays[k]
                if(layers == self.lays[-1]):
                    layers.err = out - network_output
                    layers.chan = (layers.derAct(layers.prevAct)) * layers.err
                    
                else:
                    nextlayers = self.lays[k+1]
                    nextlayers.weight = nextlayers.param[:,0:nextlayers.weight.shape[1]]
                    layers.err = np.matmul(nextlayers.weight.T, nextlayers.chan)
                    layers.chan = (layers.derAct(layers.prevAct)) * layers.err
                    
            for k in range(len(self.lays)):
                layers = self.lays[k]
                if (k==0):
                    checkndim(imp)
                    useimp = np.r_[imp,[np.ones(imp.shape[1])*-1]]
                else:
                    useimp = np.r_[self.lays[k-1].prevAct,[np.ones(self.lays[k-1].prevAct.shape[1])*-1]]
                
                if momentum == 0:
                    layers.param = layers.param + (lrate*np.matmul(layers.chan, useimp.T))/batch
                else:
                    ch = (lrate*np.matmul(layers.chan, useimp.T))
                    layers.param = layers.param + momentum*layers.prevchan
                    layers.prevchan = ch
                
                
        def mserr(self, imp, out):
            mse = np.mean((out.T - self.forward(imp.T))**2, axis=1)
            return mse
        
        def mcerr(self, imp, out):
            mce = np.sum(self.predict(imp.T) == out.T)/len(out)*100
            return mce
        
        def workout(self, imp, out, impT, outT, lrate, epoch, batch, momentum = 0):
            mse_err = []
            mce_err = []
            mset_err = []
            mcet_err = []
            for x in range(epoch):
                print('Epoch',x)
                imp,out = self.romando(imp, out)
                bat = int(np.floor(len(imp)/batch))
                
                for s in range(bat):
                    self.BP(imp[batch*s:batch*(s+1)].T, out[batch*s:batch*(s+1)].T, lrate, batch)
                
                
                mse_err.append(self.mserr(imp, out))
                mce_err.append(self.mcerr(imp, out))
                mset_err.append(self.mserr(impT, outT))
                mcet_err.append(self.mcerr(impT, outT))
    
            return mse_err, mce_err, mset_err, mcet_err
        
        def __repr__(self):
            strr = ''
            for i, layers in enumerate(self.lays):
                strr += 'Layer ' + str(i) + ': ' + layers.__repr__() + '\n'
            return strr
    
    filename = 'assign2_data1.h5'
    f1 = h5py.File(filename,'r+')
    
    testims = np.array(f1['testims'])
    testlbls = np.array(f1['testlbls'])
    trainims = np.array(f1['trainims'])
    trainlbls = np.array(f1['trainlbls'])
    
    testlbls = testlbls.reshape((len(testlbls),1))
    trainlbls = trainlbls.reshape((len(trainlbls),1))
    
    trainimsf = trainims.reshape(trainims.shape[0],(trainims.shape[1])**2)
    testimsf = testims.reshape(testims.shape[0], (testims.shape[1])**2)
    
    """ part a"""
    print('\nQuestion-1 Part-a')   
    goo = PropogateLayers()
    goo.appendLayers(Network((trainims.shape[1])**2, 30, 0.01))
    goo.appendLayers(Network(30, 1, 0.01))
    
    print(goo)
    
    trainlbls[trainlbls==0] = -1
    testlbls = testlbls.astype(int)
    testlbls[testlbls == 0] = -1
    
    get_mse, get_mce, get_mset, get_mcet = goo.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)
    
    goo.accur(goo)    
    
    """ part c """
    
    print('\n\nQuestion-1 Part-c')  
    print('High Hidden Neurons Training')
    goohigh = PropogateLayers()
    goohigh.appendLayers(Network((trainims.shape[1])**2, 150, 0.01))
    goohigh.appendLayers(Network(150, 1, 0.01))
    print(goohigh)
    
    get_mseh, get_mceh, get_mseth, get_mceth = goohigh.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)
    
    print('Low Hidden Neurons Training')
    goolow = PropogateLayers()
    goolow.appendLayers(Network((trainims.shape[1])**2, 5, 0.01))
    goolow.appendLayers(Network(5, 1, 0.01))
    print(goolow)
                    
    get_msel, get_mcel, get_msetl, get_mcetl = goolow.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)
    
    """ part d """
    
    print('\nQuestion-1 Part-d')
    goohid = PropogateLayers()
    goohid.appendLayers(Network((trainims.shape[1])**2, 300, 0.01))
    goohid.appendLayers(Network(300, 30, 0.01))
    goohid.appendLayers(Network(30, 1, 0.01))
    print(goohid)
    
    get_msehid, get_mcehid, get_msethid, get_mcethid = goohid.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)
    
    goohid.accur(goohid)
    
    
    """ part e """
    
    print('\n\nQuestion-1 Part-e')
    goomom = PropogateLayers()
    goomom.appendLayers(Network((trainims.shape[1])**2, 300, 0.01))
    goomom.appendLayers(Network(300, 30, 0.01))
    goomom.appendLayers(Network(30, 1, 0.01))
    print(goomom)
    
    get_msemom, get_mcemom, get_msetmom, get_mcetmom = goomom.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50, 0.5)
    
    goomom.accur(goomom)
    
    """ plots """
    graph(get_mse)
    plt.title('Part-a Training Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graph(get_mset)
    plt.title('Part-a Testing Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graph(get_mce)
    plt.title('Part-a Training Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graph(get_mcet)
    plt.title('Part-a Testing Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graphm(get_mse,get_mseh,get_msel)
    plt.title('Part-c Training Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graphm(get_mset,get_mseth,get_msetl)
    plt.title('Part-c Testing Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graphm(get_mce,get_mceh,get_mcel)
    plt.title('Part-c Training Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graphm(get_mcet,get_mceth,get_mcetl)
    plt.title('Part-c Testing Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graph(get_msehid)
    plt.title('Part-d Training Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graph(get_msethid)
    plt.title('Part-d Testing Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graph(get_mcehid)
    plt.title('Part-d Training Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graph(get_mcethid)
    plt.title('Part-d Testing Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graphd(get_msehid,get_msemom)
    plt.title('Part-e Training Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graphd(get_msethid,get_msetmom)
    plt.title('Part-e Testing Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.show()
    
    graphd(get_mcehid,get_mcemom)
    plt.title('Part-e Training Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()
    
    graphd(get_mcethid,get_mcetmom)
    plt.title('Part-e Testing Mean Classification Error')
    plt.ylabel('Mean Classification Error')
    plt.show()


def q2():
    
    
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
    
    
    
    filename2 = 'assign2_data2.h5'
    f2 = h5py.File(filename2,'r+')
    
    words = np.array(f2['words'])
    trainin = np.array(f2['trainx'])
    trainout = np.array(f2['traind'])
    valin = np.array(f2['valx'])
    valout = np.array(f2['vald'])
    testin = np.array(f2['testx'])
    testout = np.array(f2['testd'])
    
    
    
    """ part a"""
    print('Question-2 Part-a')
    p = 256
    p2 = 128
    p3 = 64
    idim = 32
    idim2 = 16
    idim3 = 8
    d = 250
    lrate2 = 0.4
    momentum2 = 0.85
    batch2 = 250
    epoch2 = 50
    
    vin = m1(valin,d)
    vout = m2(valout,d)
    
    
    print('\nD = 32, P = 256')
    goo2 = PropogateWords()
    goo2.appendLayers(Network2(idim,d,'wordembed',0.1))
    goo2.appendLayers(Network2(3*idim,p,'sigmoid',0.1))
    goo2.appendLayers(Network2(p,d,'softmax',0.1))
    
    err1 = goo2.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
    
    print('\nD = 16, P = 128')
    goo22 = PropogateWords()
    goo22.appendLayers(Network2(idim2,d,'wordembed',0.1))
    goo22.appendLayers(Network2(3*idim2,p2,'sigmoid',0.1))
    goo22.appendLayers(Network2(p2,d,'softmax',0.1))
    
    err2 = goo22.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
    
    print('\nD = 8, P = 64')
    goo23 = PropogateWords()
    goo23.appendLayers(Network2(idim3,d,'wordembed',0.1))
    goo23.appendLayers(Network2(3*idim3,p3,'sigmoid',0.1))
    goo23.appendLayers(Network2(p3,d,'softmax',0.1))
    
    err3 = goo23.workout(trainin,trainout,vin,vout,lrate2,epoch2,batch2,momentum2)
    
    print('To execute the part b, after observation please close the plots')
    
    graph2(err1)
    plt.title('Cross-Entropy Error, D=32,P=256')
    plt.show()
    
    graph2(err2)
    plt.title('Cross-Entropy Error, D=16,P=128')
    plt.show()
    
    graph2(err3)
    plt.title('Cross-Entropy Error, D=8,P=64')
    plt.show()
    
    
    
    """ part b"""
    
    pb(testin,words,goo2,d,testout)



question = sys.argv[1]

def ayberk_yarkin_yildiz_21803386_hw2(question):
    
    if question == '1' :
        q1()
        
    elif question == '2' :
        q2()
      
ayberk_yarkin_yildiz_21803386_hw2(question)