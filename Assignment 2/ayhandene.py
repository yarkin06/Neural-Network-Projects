# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:35:03 2021

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py


# ## Part A

# In[2]:


filename = 'assign2_data1.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # Get the data
    test_img = f[list(f.keys())[0]].value
    test_labels = f[list(f.keys())[1]].value
    train_img = f[list(f.keys())[2]].value
    train_labels = f[list(f.keys())[3]].value
test_labels = test_labels.reshape((len(test_labels),1))
train_labels = train_labels.reshape((len(train_labels),1))


# In[3]:


class Layer:
    def __init__(self, inputDim, numNeurons, std, beta=1, mean=0):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        
        self.beta = beta
        
        self.weights = np.random.normal(mean,std, inputDim*numNeurons).reshape(numNeurons, inputDim)
        self.biases = np.random.normal(mean,std, numNeurons).reshape(numNeurons,1)
        
        self.weightsE = np.concatenate((self.weights, self.biases), axis=1)
        
        self.delta = None
        self.error = None
        self.lastActiv = None
        
    def activtanh(self, x):
        if(x.ndim == 1):
            x = x.reshape(x.shape[0],1)
        numSamples = x.shape[1]
        tempInp = np.r_[x, [np.ones(numSamples)*-1]]
        self.lastActiv = np.tanh(self.beta*np.matmul(self.weightsE, tempInp))
        return self.lastActiv
    
    def tanhDer(self, x):
        return self.beta*(1-(x**2))
    
    def __repr__(self):
        return "Input Dim: " + str(self.inputDim) + ", Number of Neurons: " + str(self.numNeurons)


# In[4]:


class MLP:
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def forward(self, inp):
        out = inp
        for lyr in self.layers:
            #print("input", inp.shape)
            out = lyr.activtanh(out)
            #print("output", inp.shape)
        return out
    
    #only for binary classificaiton with 1 output neurons
    def prediction(self, inp):
        out = self.forward(inp)
        out[out>=0] = 1
        out[out<0] = -1
        return out 
    
    def backProp(self, inp, out, lrnRate, batchSize):
            
        net_out = self.forward(inp)
        #print('network out: ', net_out.shape)

        for i in reversed(range(len(self.layers))):
            lyr = self.layers[i]
            
            #outputLayer
            if(lyr == self.layers[-1]):
                lyr.error = out - net_out
                '''print("out\n",out)
                print("net out\n", net_out)
                print("error = out - net_out\n", lyr.error)''' 
                #print("derMatrix = dertanh-lastActiv-\n", derMatrix)
                lyr.delta = lyr.tanhDer(lyr.lastActiv) * lyr.error
            #hiddenLayer
            else:
                nextLyr = self.layers[i+1]
                nextLyr.weights = nextLyr.weightsE[:,0:nextLyr.weights.shape[1]]
                lyr.error = np.matmul(nextLyr.weights.T, nextLyr.delta)
                #print("Error: \n", lyr.error)
                #print("derMatrix \n", derMatrix)
                lyr.delta = lyr.tanhDer(lyr.lastActiv) * lyr.error
        
        #update weights
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            #print(inp)
            if(i == 0):
                if(inp.ndim == 1):
                    inp = inp.reshape(inp.shape[0],1)
                
                inputToUse = np.r_[inp, [np.ones(inp.shape[1])*-1]]
            else:
             
                inputToUse = np.r_[self.layers[i - 1].lastActiv, [np.ones(self.layers[i - 1].lastActiv.shape[1])*-1]]
            lyr.weightsE = lyr.weightsE+ (lrnRate* np.matmul(lyr.delta, inputToUse.T))/batchSize
            '''print("Layer:", i)
            print("delta: \n", lyr.delta)
            print("inputTouse.T: \n", inputToUse.T)
            print("Product: \n", np.matmul(lyr.delta, inputToUse.T))
            print("Weights: \n", lyr.weightsE)'''
        
            
    def train(self, inp, out, inpTest, outTest, lrnRate, epochNum, batchSize):
        mseList = []
        mceList = []
        mseTList = []
        mceTList = []
        
        for ep in range(epochNum):
            print('Epoch', ep)
            
            randomIndexes = np.random.permutation(len(inp))
            inp = inp[randomIndexes]
            out = out[randomIndexes]
            
            numBatches = int(np.floor(len(inp)/batchSize))
            
            for j in range(numBatches):
                self.backProp(inp[batchSize*j:batchSize*j+batchSize].T, out[batchSize*j:batchSize*j+batchSize].T, lrnRate, batchSize)
                
            mse = np.mean((out.T - self.forward(inp.T))**2, axis=1)
            mseList.append(mse)
            mce = np.sum(self.prediction(inp.T) == out.T)/len(out)*100
            mceList.append(mce)
            mseT = np.mean((outTest.T - self.forward(inpTest.T))**2, axis=1)
            mseTList.append(mseT)
            mceT = np.sum(self.prediction(inpTest.T) == outTest.T)/len(outTest)*100
            mceTList.append(mceT)
        return mseList, mceList, mseTList, mceTList
    
    def __repr__(self):
        retStr = ""
        for i, lyr in enumerate(self.layers):
            retStr += "Layer " + str(i) + ": " + lyr.__repr__() + "\n"
        return retStr


# In[5]:


print(test_img.shape,test_labels.shape)
print(train_img.shape, train_labels.shape)

inputWH = train_img.shape[1]

train_img_flat = train_img.reshape(train_img.shape[0], inputWH**2)
test_img_flat = test_img.reshape(test_img.shape[0], inputWH**2)

print(train_img_flat.shape, test_img_flat.shape)


# In[6]:


neuralNet = MLP()
neuralNet.addLayer(Layer(inputWH**2, 18, 0.02, 1))
neuralNet.addLayer(Layer(18, 1, 0.02, 1))

print(neuralNet)

train_labels[train_labels == 0] = -1

test_labels = test_labels.astype(int)
test_labels[test_labels == 0] = -1

#0.3 lr
#300 epoch
mses, mces, mseTs, mceTs = neuralNet.train(train_img_flat/255, train_labels, test_img_flat/255, test_labels,                                            0.35, 300, 38)



print("Test Accuracy:", str(np.sum(neuralNet.prediction(test_img_flat.T/255).T == test_labels)/len(test_labels)*100) + "%")

plt.figure()
plt.plot(mses)
plt.title('MSE Over Training')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(mces)
plt.title('MCE Over Training')
plt.xlabel('Epoch')
plt.ylabel('MCE')
plt.show()

plt.figure()
plt.plot(mseTs)
plt.title('MSE Over Test')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(mceTs)
plt.title('MCE Over Test')
plt.xlabel('Epoch')
plt.ylabel('MCE')
plt.show()