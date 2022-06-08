# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:59:06 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:00:00 2021

@author: User
"""

#import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

"""
filename = 'assign2_data1.h5'
f1 = h5py.File(filename,'r+')

testims = np.array(f1["testims"])
testlbls = np.array(f1["testlbls"])
trainims = np.array(f1["trainims"])
trainlbls = np.array(f1["trainlbls"])

testlbls = testlbls.reshape((len(testlbls),1))
trainlbls = trainlbls.reshape((len(trainlbls),1))
"""

filename = 'assign2_data1.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # Get the data
    testims = f[list(f.keys())[0]].value
    testlbls = f[list(f.keys())[1]].value
    trainims = f[list(f.keys())[2]].value
    trainlbls = f[list(f.keys())[3]].value
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
    
        """
    def activation(self,a):
        if (a.ndim == 1):
            a = a.reshape(a.shape[0],1)
        #namito = a.shape[1]
        #namito = namito.reshape(a.shape[1],1)
        #fır = np.ones(namito)*-1
        #fır = fır.reshape(fır.shape[0],1,1)
        #temp = np.r_[a, [fır]]
        #self.prevAct = np.tanh(np.matmul(self.param,temp))
        self.prevAct = np.sinh(np.matmul(self.param,np.r_[a, [np.ones(a.shape[1])*-1]]))/np.cosh(np.matmul(self.weight,np.r_[a, [np.ones(a.shape[1])*-1]]))
        return self.prevAct
    
    """
    
    def activation(self, x):
        if(x.ndim == 1):
            x = x.reshape(x.shape[0],1)
        #tempInp = np.concatenate((x,np.ones(numSamples)*-1))
        self.prevAct = np.tanh(np.matmul(self.param, np.r_[x, [np.ones(x.shape[1])*-1]]))
        return self.prevAct
    
    def derAct(self,a):
        return 1-(a**2)
    
    def __repr__(self):
        return "Dimension Input: "+str(self.dim)+", # of Neurons:"+str(self.neuron)

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
    def predict(self,imp):
        nextt = self.forward(imp)
        nextt[nextt>=0] = 1
        nextt[nextt<0] = -1
        return nextt
    def BP(self,imp,out,lrate,batch):
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
            if (k==0):
                if(imp.ndim==1):
                    """ 0 ı silebiliriz veya len() dene:"""
                    imp = imp.reshape(imp.shape[0],1)
                useimp = np.r_[imp,[np.ones(imp.shape[1])*-1]]
            else:
                useimp = np.r_[self.lays[k-1].prevAct,[np.ones(self.lays[k-1].prevAct.shape[1])*-1]]
            layers.param = layers.param + (lrate*np.matmul(layers.chan, useimp.T))/batch
    
    def mserr(self, imp, out):
        mse = np.mean((out.T - self.forward(imp.T))**2, axis=1)
        return mse
    
    def mcerr(self, imp, out):
        mce = np.sum(self.predict(imp.T) == out.T)/len(out)*100
        return mce
    
    def workout(self, imp, out, impT, outT, lrate, epoch, batch):
        mse_err = []
        mce_err = []
        mset_err = []
        mcet_err = []
        for x in range(epoch):
            print("Epoch:",x)
            """bura değişti: ama ders notlarında random permutation var, geri eskisini yapabilirsin"""
            
            #randindex = np.random.shuffle(indexes)
            randindex = np.random.permutation(len(imp))
            """ burayı ayrı fonksiyona çekebilirsin imp alıp onu shufflelayıp indexleri düzenleyecek şekilde output ve inputun"""
            imp,out = imp[randindex],out[randindex]
            bat = int(np.floor(len(imp)/batch))
            #normlen = len(imp)/batch
            #bat = int(normlen - (normlen % 1))
            
            for s in range(bat):
                self.BP(imp[batch*s:batch*(s+1)].T, out[batch*s:batch*(s+1)].T, lrate, batch)
            
            
            mse_err.append(self.mserr(imp, out))
            mce_err.append(self.mcerr(imp, out))
            mset_err.append(self.mserr(impT, outT))
            mcet_err.append(self.mcerr(impT, outT))

            """
            mse = np.mean((out.T - self.forward(imp.T))**2, axis=1)
            mse_err.append(mse)
            mce = np.sum(self.predict(imp.T) == out.T)/len(out)*100
            mce_err.append(mce)
            mseT = np.mean((outT.T - self.forward(impT.T))**2, axis=1)
            mset_err.append(mseT)
            mceT = np.sum(self.predict(impT.T) == outT.T)/len(outT)*100
            mcet_err.append(mceT)
            """
        return mse_err, mce_err, mset_err, mcet_err
    
    def __repr__(self):
        strr = ""
        for i, layers in enumerate(self.lays):
            strr += "Layer " + str(i) + ": " + layers.__repr__() + "\n"
        return strr
    
goo = PropogateLayers()
goo.appendLayers(Network((trainims.shape[1])**2, 18, 0.02))
goo.appendLayers(Network(18, 1, 0.02))

print(goo)

trainlbls[trainlbls==0] = -1
testlbls = testlbls.astype(int)
testlbls[testlbls == 0] = -1

get_mse, get_mce, get_mset, get_mcet = goo.workout(trainimsf/255, trainlbls, testimsf/255, testlbls, 0.35, 300, 38)

print("Test Accuracy:", str(np.sum(goo.predict(testimsf.T/255).T == testlbls)/len(testlbls)*100) + "%")

plt.figure()
plt.plot(get_mse)
plt.title('MSE Over Training')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(get_mce)
plt.title('MCE Over Training')
plt.xlabel('Epoch')
plt.ylabel('MCE')
plt.show()

plt.figure()
plt.plot(get_mset)
plt.title('MSE Over Test')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(get_mcet)
plt.title('MCE Over Test')
plt.xlabel('Epoch')
plt.ylabel('MCE')
plt.show()
            
            
            
                
            