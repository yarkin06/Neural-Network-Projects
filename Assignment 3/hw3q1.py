# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:17:13 2021

@author: User
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
#import time


filename1 = "assign3_data1.h5"
f1 = h5py.File(filename1, 'r')
#data = f1['data'][()].astype('float64')
#f1.close()
data = np.array(f1['data'])

class AE():
    def initialize1(self, Lin, Lhid):
        
        Lout = Lin
        
        #re = (np.sqrt(6)/np.sqrt(Lin + Lhid))
        W1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(Lin + Lhid)),(np.sqrt(6)/np.sqrt(Lin + Lhid)), size=(Lin,Lhid))
        b1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(Lin + Lhid)),(np.sqrt(6)/np.sqrt(Lin + Lhid)), size=(1, Lhid))
        
        #ren = np.sqrt(6)/np.sqrt(Lhid + Lout)
        W2 = W1.T
        b2 = np.random.uniform(-np.sqrt(6)/np.sqrt(Lhid + Lout),np.sqrt(6)/np.sqrt(Lhid + Lout), size=(1,Lout))
        
        We = (W1, W2, b1, b2)
        momWe = (0,0,0,0)
        
        return We, momWe
    
    def workout1(self, data, params, lrate, mom, epoch, batch):
        Jl = []
        # if batch is None:
        #     batch = data.shape[0]
            
        Lin = params["Lin"]
        Lhid = params["Lhid"]
        We, momWe = self.initialize1(Lin, Lhid)
        
        # it = int(data.shape[0]/batch)
        for i in range(epoch):
            #times = time.time()
            
            Jt = 0
            go = 0
            stop = batch
            
            data = data[np.random.permutation(data.shape[0])]
            
            momWe = (0,0,0,0)
            
            for j in range(int(data.shape[0]/batch)):
                batching = data[go:stop]
                
                J, Jgrad, co = self.aeCost(We, batching, params)
                We, momWe = self.solver(Jgrad, co, We, momWe, lrate, mom)
                
                Jt = Jt + J
                go = stop
                stop = stop + batch
                
            # timel = (epoch-(i+1))*(time.time()-times)
            # if timel < 60:
            #     timel = round(timel)
            #     timen = "sec(s)"
            # else:
            #     timel = round(timel/60)
            #     timen = "min(s)"
                
            Jt = Jt/(int(data.shape[0]/batch))
            
            #print("Loss: {:.2f} [Epoch {} of {}, ETA: {} {}]".format(Jt, i+1, epoch, timel, timen))
            print("Epoch: {}, Loss: {:.3f}".format(i+1, Jt))
            #print("Epoch :",i+1,",","Loss :",Jt)
            #print("Epoch",i+1)
            Jl.append(Jt)
        
        return We, Jl
    
    def aeCost(self, We, data, params):
        
        nar = data.shape[0]
        
        W1, W2, b1, b2 = We
        
        rho = params["rho"]
        beta = params["beta"]
        lamda = params["lamda"]
        Lin = params["Lin"]
        Lhid = params["Lhid"]
        
        #u = data @ W1+b1
        h, hder = self.sigmoid(data @ W1+b1)
        
        #v = h @ W2+b2
        o, oder = self.sigmoid(h @ W2+b2)
        
        rhobe = h.mean(axis=0, keepdims=True)
        
        #loss = (0.5*(np.linalg.norm(data-o,axis=1)**2).sum())/nar
        #tako = 0.5*lamda*(np.sum(W1**2) + np.sum(W2**2))
        #kel = (rho*np.log(rho/rhobe) + (1-rho)*np.log((1-rho)/(1-rhobe))).sum()
        #kel = beta * (rho*np.log(rho/rhobe) + (1-rho)*np.log((1-rho)/(1-rhobe))).sum()
        
        J = ((0.5*(np.linalg.norm(data-o,axis=1)**2).sum())/nar) + (0.5*lamda*(np.sum(W1**2) + np.sum(W2**2))) + (beta * (rho*np.log(rho/rhobe) + (1-rho)*np.log((1-rho)/(1-rhobe))).sum())
        # derloss = (o-data)/nar
        # dertako2 = lamda * W2
        # dertako1 = lamda * W1
        # derkel = beta * ((1-rho)/(1-rhobe) - rho/rhobe)/nar
        
        co = (data, h, hder, oder)
        Jgrad = ((o-data)/nar, lamda * W2, lamda * W1, (beta * ((1-rho)/(1-rhobe) - rho/rhobe)/nar))
        
        return J, Jgrad, co
    
    def solver(self, Jgrad, co, We, momWe, lrate, mom):
        
        W1, W2, b1, b2 = We
        derW1, derW2 = 0,0
        
        # derW1 = 0
        # derW2 = 0
        # derb1 = 0
        # derb2 = 0
        
        data, h, hder, oder = co
        derloss, dertako2, dertako1, derkel = Jgrad
        
        # chan1 = derloss * oder
        # chan = hder*((derloss * oder) @ W2.T + derkel)

        derW2 = h.T @ (derloss * oder)+dertako2
        # derb2 = chan1.sum(axis=0, keepdims=True)
        
        derW1 = data.T @ (hder*((derloss * oder) @ W2.T + derkel)) + dertako1
        # derb1 = chan.sum(axis=0, keepdims=True)
        
        # derW2 = (derW1.T + derW2)/2
        # derW1 = derW2.T
        
        derWe = (((derW1.T + derW2)/2).T, (derW1.T + derW2)/2, (hder*((derloss * oder) @ W2.T + derkel)).sum(axis=0, keepdims=True), (derloss * oder).sum(axis=0, keepdims=True))
        
        We, momWe = self.modify(We, momWe, derWe, lrate, mom)
        
        return We, momWe
    
    def modify(self, We, momWe, derWe, lrate, mom):
        
        W1, W2, b1, b2 = We
        derW1, derW2, derb1, derb2 = derWe
        momW1, momW2, momb1, momb2 = momWe
        
        # momW1 = lrate*derW1 + mom*momW1
        # momW2 = lrate*derW2 + mom*momW2
        # momb1 = lrate*derb1 + mom*momb1
        # momb2 = lrate*derb2 + mom*momb2
        
        # W1 = W1 - momW1
        # W2 = W2 - momW2
        # b1 = b1 - momb1
        # b2 = b2 - momb2
        
        # momW1 = lrate*derW1 + mom*momW1
        # momW2 = lrate*derW2 + mom*momW2
        # momb1 = lrate*derb1 + mom*momb1
        # momb2 = lrate*derb2 + mom*momb2
        
        # W1 = W1 - (lrate*derW1 + mom*momW1)
        # W2 = W2 - (lrate*derW2 + mom*momW2)
        # b1 = b1 - (lrate*derb1 + mom*momb1)
        # b2 = b2 - (lrate*derb2 + mom*momb2)
        assert(W1 == W2.T).all()
        We = (W1 - (lrate*derW1 + mom*momW1),W2 - (lrate*derW2 + mom*momW2),b1 - (lrate*derb1 + mom*momb1),b2 - (lrate*derb2 + mom*momb2))
        momWe = (lrate*derW1 + mom*momW1,lrate*derW2 + mom*momW2,lrate*derb1 + mom*momb1,lrate*derb2 + mom*momb2)
        
        return We, momWe
    
    # def guess1(self, data, We):
        
    #     W1, W2, b1, b2 = We
        
    #     #u = data @ W1 + b1
    #     h = self.sigmoid(data @ W1 + b1)[0]
    #     #v = h @ W2 + b2
    #     o = self.sigmoid(h @ W2 + b2)[0]
        
    #     return o
    
    def sigmoid(self, x):
        k = np.exp(x)/(1 + np.exp(x))
        l = k * (1-k)
        return k,l
    

def norma(x):
    return (x-x.min())/(x.max()-x.min())

# def graph(w, d1, d2):
    
#     fig, ax = plt.subplots(d2, d1, figsize=(d1, d2))
#     c = 0  
#     for a in range(d2):
#         for b in range(d1):
#             ax[a,b].imshow(w[c], cmap='gray')
#             ax[a,b].axis('off')
#             c = c + 1
    
  
    
    #fig.subplots_adjust(wspace = 0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
    #fig.savefig(tit + ".png")
    #plt.close(fig)
    
def dis(lrate, mom, epoch, batch, rho, beta, lamda, Lin, Lhid, x, t, d):
    
    params = {"rho": rho, "beta": beta, "lamda": lamda, "Lin": Lin, "Lhid": Lhid}
    autoencoder = x()
    #wilk = autoencoder.workout1(trd, params, lrate, mom, epoch, batch)[0]
    wson = norma(autoencoder.workout1(t, params, lrate, mom, epoch, batch)[0][0]).T
    wson = wson.reshape((wson.shape[0], d, d))
    
    #tit = "rho={:.2f}, beta={:.2f}, lrate={:.2f}, momentum={:.2f}, lambda={}, batch={}, Lhid={}".format(rho, beta, lrate, mom, lamda, batch, Lhid)
    w_dimension = int(np.sqrt(wson.shape[0]))
    
    fig, ax = plt.subplots(w_dimension, w_dimension, figsize=(w_dimension, w_dimension))
    fig.suptitle("lambda={}, Lhid={}".format(lamda, Lhid))
    c = 0  
    for a in range(w_dimension):
        for b in range(w_dimension):
            ax[a,b].imshow(wson[c], cmap='gray')
            ax[a,b].axis('off')
            c = c + 1
    
    #graph(wson, w_dimension, w_dimension)

#%% Part a
print("Question 1 Part a")

datanew = 0.2126 * data[:, 0] + 0.7152 * data[:, 1] + 0.0722 * data[:, 2]

assert datanew.shape[1] == datanew.shape[2]
dimension = datanew.shape[1]
# datanew = np.reshape(datanew, (datanew.shape[0], dimension ** 2))

datanew = (np.reshape(datanew, (datanew.shape[0], dimension ** 2))) - (np.reshape(datanew, (datanew.shape[0], dimension ** 2))).mean(axis=1, keepdims = True)
# std = np.std(datanew)
datanew = 0.1 + 0.8*(norma(np.clip(datanew, - 3 * (np.std(datanew)), 3 * (np.std(datanew)))))
# datanew = norma(datanew)

# datanew = 0.1 + datanew*0.8
trd = datanew

datanew = np.reshape(datanew, (datanew.shape[0], dimension, dimension))
data = data.transpose((0,2,3,1))
fig1,ax1= plt.subplots(10,20,figsize=(20,10))
fig1.suptitle("RGB Images")
fig2,ax2= plt.subplots(10, 20, figsize=(20, 10))
fig2.suptitle("Grayscale Images")

for a in range (10):
    for b in range (20):
        c = np.random.randint(0, data.shape[0])
        plt.imshow(data[c].astype('float'))
        ax1[a,b].axis('off')
        
        plt.imshow(datanew[c], cmap='gray')
        ax2[a,b].axis("off")
        
#fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)


#fig1.savefig("q1a_rgb.png")
#fig2.savefig("q1a_gray.2.png")
#plt.close("all")  


#%% Part c
print("\nQuestion 1 Part c")
epoch = 200
mom = 0.8
rho = 0.03
beta = 3
lrate = 0.1
batch = 32
lamda = 5e-4
Lin = trd.shape[1]
Lhid = 64
#print("rho={:.2f}, beta={:.2f}, lrate={:.2f}, momentum={:.2f}, lambda={}, batch={}, Lhid={}".format(rho, beta, lrate, mom, lamda, batch, Lhid))
print("\nParameters: rho =",rho,",","beta =",beta,",","lrate =",lrate,",","momentum =",mom,",","lambda =",lamda,",","batch =",batch,",","Lin =",Lin,",","Lhid =",Lhid,",","epoch =",epoch,"\n")
# dis(lrate, mom, epoch, batch, rho, beta, lamda, Lin, Lhid, AE, trd, dimension)

#%% part d
print("\nQuestion 1 Part d")
Lhidl = 16
Lhidn = 49
Lhidh = 100
lamdal = 0
lamdan = 1e-5
lamdah = 1e-3

# print("\nlambda =",lamdal,",","Lhid =",Lhidl)
# dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidl, AE, trd, dimension)
# print("\nlambda =",lamdal,",","Lhid =",Lhidn)
# dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidn, AE, trd, dimension)
# print("\nlambda =",lamdal,",","Lhid =",Lhidh)
# dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidh, AE, trd, dimension)
# print("\nlambda =",lamdan,",","Lhid =",Lhidl)
# dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidl, AE, trd, dimension)
# print("\nlambda =",lamdan,",","Lhid =",Lhidn)
# dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidn, AE, trd, dimension)
# print("\nlambda =",lamdan,",","Lhid =",Lhidh)
# dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidh, AE, trd, dimension)
# print("\nlambda =",lamdah,",","Lhid =",Lhidl)
# dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidl, AE, trd, dimension)
# print("\nlambda =",lamdah,",","Lhid =",Lhidn)
# dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidn, AE, trd, dimension)
# print("\nlambda =",lamdah,",","Lhid =",Lhidh)
# dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidh, AE, trd, dimension)

