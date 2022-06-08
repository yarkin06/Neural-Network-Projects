#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import h5py


# In[2]:


def vector1H(x, maxInd):
    out = np.zeros(maxInd)
    out[x-1] = 1
    return out

def mat1H(x, maxInd):
    out = np.zeros((x.shape[0], x.shape[1], maxInd))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i,j,:] = vector1H(x[i,j], maxInd)
    return out

def mat1H2(y, maxInd):
    out = np.zeros((y.shape[0], maxInd))
    for i in range(y.shape[0]):
        out[i,:] = vector1H(y[i], maxInd)
    return out


# In[3]:


filename = 'assign2_data2.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # Get the data
    test_labels = f[list(f.keys())[0]].value
    test_data = f[list(f.keys())[1]].value
    train_labels = f[list(f.keys())[2]].value
    train_data = f[list(f.keys())[3]].value
    val_labels = f[list(f.keys())[4]].value
    val_data = f[list(f.keys())[5]].value
    wordDict = f[list(f.keys())[6]].value


# In[4]:


class Layer:
    def __init__(self, inputDim, numNeurons, activation, std, mean=0):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.activation = activation
        if self.activation == 'sigmoid' or self.activation == 'softmax':
            self.weights = np.random.normal(mean,std, inputDim*numNeurons).reshape(numNeurons, inputDim)
            self.biases = np.random.normal(mean,std, numNeurons).reshape(numNeurons,1)
            self.weightsE = np.concatenate((self.weights, self.biases), axis=1)
        elif self.activation == 'we':
            self.dictSize = numNeurons
            self.D = inputDim
            self.weights = np.random.normal(mean, std, dictSize*D).reshape((dictSize,D))
            
        
        self.delta = None
        self.error = None
        self.lastActiv = None
        
        self.prevUpdate = 0
        
        
    def actFcn(self,x):
        if(self.activation == 'sigmoid'):
            expx = np.exp(2*x)
            return expx/(1+expx)
        elif(self.activation == 'softmax'):
            expx = np.exp(x - np.max(x))
            return expx/np.sum(expx, axis=0)
        elif(self.activation == 'we'):
            return x
                    
    def activate(self, x):
        if self.activation == 'sigmoid' or self.activation == 'softmax':
            if(x.ndim == 1):
                x = x.reshape(x.shape[0],1)
            numSamples = x.shape[1]
            tempInp = np.r_[x, [np.ones(numSamples)*-1]]
            
            #print('hLayerW', self.weights.shape)
            #print('hLayerInp', tempInp.shape)
            
            self.lastActiv = self.actFcn(np.matmul(self.weightsE, tempInp))
        elif self.activation == 'we':
            #print('x: \n', x, x.shape)
            #print('w: \n', self.weights, self.weights.shape)
            layerOut = np.zeros((x.shape[0],x.shape[1], self.D))
            for m in range(layerOut.shape[0]):
                layerOut[m,:,:] = self.actFcn(np.matmul(x[m,:,:], self.weights))
            layerOut = layerOut.reshape((layerOut.shape[0], layerOut.shape[1] * layerOut.shape[2]))
            self.lastActiv = layerOut.T
            #print('weOut', layerOut.shape)
        return self.lastActiv
    
    def derActiv(self, x):
        if(self.activation == 'sigmoid'):
            return 2*(x*(1-x))
        elif(self.activation == 'softmax'):
            return x*(1-x)
        elif(self.activation == 'we'):
            return np.ones(x.shape)
    
    def __repr__(self):
        return "Input Dim: " + str(self.inputDim) + ", Number of Neurons: " + str(self.numNeurons) + "\n Activation: " + self.activation


# In[5]:


class WordNet:
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def forward(self, inp):
        out = inp
        for lyr in self.layers:
            out = lyr.activate(out)
        return out
    
    def prediction(self, inp):
        out = self.forward(inp)
        if(out.ndim == 1):
            return np.argmax(out)
        return np.argmax(out, axis=0)
    
    def predictionTopK(self, inp, k):
        out = self.forward(inp)
        #print(out)
        return np.argpartition(out, -k, axis=0)[-k:]
        
    
    def backProp(self, inp, out, lrnRate, momCoeff, batchSize):
        net_out = self.forward(inp)
        #print('network out: ', net_out.shape)
        for i in reversed(range(len(self.layers))):
            lyr = self.layers[i]
            #outputLayer
            if(lyr == self.layers[-1]):
                lyr.delta = out - net_out
            #hiddenLayer
            else:
                nextLyr = self.layers[i+1]
                nextLyr.weights = nextLyr.weightsE[:,0:nextLyr.weights.shape[1]]
                lyr.error = np.matmul(nextLyr.weights.T, nextLyr.delta)
                derMatrix = lyr.derActiv(lyr.lastActiv)
                lyr.delta = derMatrix * lyr.error
        
        #update weights
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            #write dynamic if later
            if(i == 0):
                if(inp.ndim == 1):
                    inp = inp.reshape(inp.shape[0],1)
                numSamples = inp.shape[1]
                inputToUse = inp
            else:
                numSamples = self.layers[i - 1].lastActiv.shape[1]
                inputToUse = np.r_[self.layers[i - 1].lastActiv, [np.ones(numSamples)*-1]]
            #print('delta of layer[' + str(i) + ']', lyr.delta.shape)
            #print('inputToUse.T of layer[' + str(i) + ']', inputToUse.T.shape)
            if(lyr.activation == 'sigmoid' or lyr.activation == 'softmax'):
                update =  (lrnRate * np.matmul(lyr.delta, inputToUse.T))/batchSize
                lyr.weightsE += update + momCoeff * lyr.prevUpdate
            elif(lyr.activation == 'we'):
                delta3d = lyr.delta.reshape((3,batchSize,lyr.D))
                inputToUse = np.transpose(inputToUse, (1,2,0))
                update = np.zeros((inputToUse.shape[1], delta3d.shape[2]))
                for i in range(delta3d.shape[0]):
                    update += lrnRate * np.matmul(inputToUse[i,:,:], delta3d[i,:,:])
                #mean the updates for each separate word
                #update = update/delta3d.shape[0]
                update = update/batchSize
                lyr.weights += update + momCoeff * lyr.prevUpdate
            lyr.prevUpdate = update
            
    def train(self, inp, out, inpTest, outTest, lrnRate, momCoeff, epochNum, batchSize):
        cveList = []
        
        for ep in range(epochNum):
            print('Epoch', ep)
            
            randomIndexes = np.random.permutation(len(inp))
            inp = inp[randomIndexes]
            out = out[randomIndexes]
            numBatches = int(np.floor(len(inp)/batchSize))
            
            for j in range(numBatches):
                batch_inp = inp[batchSize*j:batchSize*j+batchSize]
                batch_out = out[batchSize*j:batchSize*j+batchSize]
                
                batch_inp_1H = mat1H(batch_inp, dictSize)
                batch_out_1H = mat1H2(batch_out, dictSize).T
                
                self.backProp(batch_inp_1H, batch_out_1H, lrnRate, momCoeff, batchSize)
            
            valOutput = self.forward(inpTest)
            #print('Prediction Probs \n', valOutput)
            #print('Test \n', outTest)
            
            crossErr = - np.sum(np.log(valOutput) * outTest.T)/valOutput.shape[1]
            print('Cross-Entropy Error ', crossErr)
            cveList.append(crossErr)
            
            valAcc = np.sum(self.prediction(inpTest) == np.argmax(outTest.T, axis=0))
            print('Correct: ', valAcc)
            print('Accuracy: ', valAcc/valOutput.shape[1])
                
        return cveList
    
    def __repr__(self):
        retStr = ""
        for i, lyr in enumerate(self.layers):
            retStr += "Layer " + str(i) + ": " + lyr.__repr__() + "\n"
        return retStr


# In[6]:


P = 256
D = 32
dictSize = 250

nn = WordNet()
nn.addLayer(Layer(D, dictSize, 'we', 0.2))
nn.addLayer(Layer(3*D, P, 'sigmoid', 0.2))
nn.addLayer(Layer(P, dictSize, 'softmax', 0.2))

learnRate = 0.35
momCoeff = 0.85
batchSize = 250
epoch = 50

val_inp_1H = mat1H(val_data, dictSize)
val_labels_1H = mat1H2(val_labels, dictSize)

errors = nn.train(train_data, train_labels, val_inp_1H, val_labels_1H,                  learnRate, momCoeff,                  epoch, batchSize)


# In[16]:


plt.plot(errors)
plt.title('Cross-Entropy Error over Validation Set')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Error')
plt.show()


# In[26]:


num_samples = 5

random_sample_indexes = np.random.permutation(len(test_data))[0:num_samples]

test_samples = test_data[random_sample_indexes]
test_outputs = test_labels[random_sample_indexes]

test_samples_1H = mat1H(test_samples, dictSize)
test_labels_1H = mat1H2(test_outputs, dictSize)

top10predictions = nn.predictionTopK(test_samples_1H, 10)

print(top10predictions)

for i in range(num_samples):
    print('[' + str(i+1) + '] ' + str(wordDict[test_samples[i,0]-1].decode("utf-8"))  + ', ' +           str(wordDict[test_samples[i,1]-1].decode("utf-8"))           + ', ' + str(wordDict[test_samples[i,2]-1].decode("utf-8")))
    strin = 'The Top-K predictions are: {'
    for j in range(10):
        strin += (str(wordDict[top10predictions[j,i]].decode("utf-8"))) + ', '
    print(strin + '}')


# In[ ]:




