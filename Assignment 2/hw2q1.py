# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:00:00 2021

@author: User
"""

#import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

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
        #for k in range(len(self.lays)-1,-1,-1):    
            layers = self.lays[k]
            if(layers == self.lays[-1]):
                """azalt alttakini öbürünn içine yaz"""
                layers.err = out - network_output
                layers.chan = (layers.derAct(layers.prevAct)) * layers.err
                
            else:
                nextlayers = self.lays[k+1]
                nextlayers.weight = nextlayers.param[:,0:nextlayers.weight.shape[1]]
                layers.err = np.matmul(nextlayers.weight.T, nextlayers.chan)
                layers.chan = (layers.derAct(layers.prevAct)) * layers.err
                
        for k in range(len(self.lays)):
            layers = self.lays[k]
            """ 0 ı silebiliriz veya len() dene:"""
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
            #print('Epoch',x)
            """bura değişti: ama ders notlarında random permutation var, geri eskisini yapabilirsin"""
            
            #randindex = np.random.shuffle(indexes)
            
            """ burayı ayrı fonksiyona çekebilirsin imp alıp onu shufflelayıp indexleri düzenleyecek şekilde output ve inputun"""
            imp,out = self.romando(imp, out)
            bat = int(np.floor(len(imp)/batch))
            #normlen = len(imp)/batch
            #bat = int(normlen - (normlen % 1))
            
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

""" part a"""
print('Question-1 Part-a')   
goo = PropogateLayers()
goo.appendLayers(Network((trainims.shape[1])**2, 30, 0.03))
goo.appendLayers(Network(30, 1, 0.03))

print(goo)

trainlbls[trainlbls==0] = -1
testlbls = testlbls.astype(int)
testlbls[testlbls == 0] = -1

get_mse, get_mce, get_mset, get_mcet = goo.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)

goo.accur(goo)

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

""" part c """

print('\n\nQuestion-1 Part-c')  
print('High Hidden Neurons Training')
goohigh = PropogateLayers()
goohigh.appendLayers(Network((trainims.shape[1])**2, 200, 0.03))
goohigh.appendLayers(Network(200, 1, 0.03))
print(goohigh)

get_mseh, get_mceh, get_mseth, get_mceth = goohigh.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)

print('Low Hidden Neurons Training')
goolow = PropogateLayers()
goolow.appendLayers(Network((trainims.shape[1])**2, 5, 0.03))
goolow.appendLayers(Network(5, 1, 0.03))
print(goolow)
                
get_msel, get_mcel, get_msetl, get_mcetl = goolow.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)

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


""" part d """

print('\nQuestion-1 Part-d')
goohid = PropogateLayers()
goohid.appendLayers(Network((trainims.shape[1])**2, 300, 0.03))
goohid.appendLayers(Network(300, 30, 0.03))
goohid.appendLayers(Network(30, 1, 0.03))
print(goohid)

get_msehid, get_mcehid, get_msethid, get_mcethid = goohid.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50)

goohid.accur(goohid)

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


""" part e """

print('\n\nQuestion-1 Part-e')
goomom = PropogateLayers()
goomom.appendLayers(Network((trainims.shape[1])**2, 300, 0.03))
goomom.appendLayers(Network(300, 30, 0.03))
goomom.appendLayers(Network(30, 1, 0.03))
print(goomom)

get_msemom, get_mcemom, get_msetmom, get_mcetmom = goomom.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.4, 300, 50, 0.5)

goomom.accur(goomom)

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
