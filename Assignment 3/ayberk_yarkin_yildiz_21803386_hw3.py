# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:20:41 2021

@author: User
"""

import sys
import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

def getting(n, bi, tr1, tr2, l, m, b, e, te1, te2, no):
    net = n(bi,no)
    trloss, valoss, tracc, valacc = net.workout3(tr1,tr2,l,m,b,e).values()
    testacc = net.guess3(te1, te2, accur=True)
    return net, trloss, valoss, tracc, valacc, testacc

def guessing(n, tr1, tr2, te1, te2):
    trconf = n.guess3(tr1, tr2, accur=True, conf=True)
    teconf = n.guess3(te1, te2, accur=True, conf=True)
    return trconf, teconf

def lstmproc(w, b, zic, k):
    w = w + zic.T @ k
    b = b + k.sum(axis=0, keepdims=True)
    return w,b
    
def lstmdu(k, w, big):
    f = k @ w.T[:, :big]
    return f

def gruproc(a, d, h, w, u, b):
        w = w + a.T @ d
        u = u + h.T @ d
        b = b + d.sum(axis=0, keepdims=True)
        return w,u,b

def ifing1(h, b, n, k):
    if k > 0:
        hb = h[:, k-1, :]
    else:
        hb = np.zeros((n,b))
    return hb

def ifing2(h, k):
    if k > 0:
        hb = h[:, k-1, :]
    else:
        hb = 0
    return hb

def graphn(tr, val, te, dat, namee):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(str(namee)+"\nTraining Accuracy: {:.2f} | Validation Accuracy: {:.2f} | Testing Accuracy: {:.2f}\n ".format(tr[-1], val[-1], te))
    plt.plot(dat)
    plt.xlabel("Epoch")
    
def graphm(trconf, testconf):
    plt.figure(figsize=(20, 10), dpi=160)
    plt.subplot(1, 2, 1)
    sn.heatmap(trconf, annot=True, annot_kws={"size": 8}, xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], cmap=sn.cm.rocket_r, fmt='g')
    plt.title("Training Confusion Matrix")
    plt.ylabel("Real")
    plt.xlabel("Guess")
    plt.subplot(1, 2, 2)
    sn.heatmap(testconf, annot=True, annot_kws={"size": 8}, xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], cmap=sn.cm.rocket_r, fmt='g')
    plt.title("Testing Confusion Matrix")
    plt.ylabel("Real")
    plt.xlabel("Guess")

def calcul(a,b,f,k):
    accu = f(a,b,accur=True)
    loso = k(b, f(a,accur=False))
    return accu, loso

def norma(x):
    return (x-x.min())/(x.max()-x.min())
    
def dis(lrate, mom, epoch, batch, rho, beta, lamda, Lin, Lhid, x, t, d):
    params = {"rho": rho, "beta": beta, "lamda": lamda, "Lin": Lin, "Lhid": Lhid}
    autoencoder = x()
    wson = norma(autoencoder.workout1(t, params, lrate, mom, epoch, batch)[0][0]).T
    wson = wson.reshape((wson.shape[0], d, d))
    w_dimension = int(np.sqrt(wson.shape[0]))
    return w_dimension, wson
    
def imageshowc(w,d,lamda,Lhid):
    fig, ax = plt.subplots(d, d, figsize=(d, d))
    fig.suptitle("Question 1, Part c, lambda={}, Lhid={}".format(lamda, Lhid))
    c = 0  
    for a in range(d):
        for b in range(d):
            ax[a,b].imshow(w[c], cmap='gray')
            ax[a,b].axis('off')
            c = c + 1
    plt.show()

def imageshowd(w,d,lamda,Lhid):
    fig, ax = plt.subplots(d, d, figsize=(d, d))
    fig.suptitle("Question 1, Part d, lambda={}, Lhid={}".format(lamda, Lhid))
    c = 0  
    for a in range(d):
        for b in range(d):
            ax[a,b].imshow(w[c], cmap='gray')
            ax[a,b].axis('off')
            c = c + 1
    plt.show()

def q1():
    
    class AE():
        def initialize1(self, Lin, Lhid):
            Lout = Lin
            W1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(Lin + Lhid)),(np.sqrt(6)/np.sqrt(Lin + Lhid)), size=(Lin,Lhid))
            b1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(Lin + Lhid)),(np.sqrt(6)/np.sqrt(Lin + Lhid)), size=(1, Lhid))

            W2 = W1.T
            b2 = np.random.uniform(-np.sqrt(6)/np.sqrt(Lhid + Lout),np.sqrt(6)/np.sqrt(Lhid + Lout), size=(1,Lout))
            
            We = (W1, W2, b1, b2)
            momWe = (0,0,0,0)
            
            return We, momWe
        
        def solver(self, Jgrad, co, We, momWe, lrate, mom):
            
            W1, W2, b1, b2 = We
            derW1, derW2 = 0,0
            
            data, h, hder, oder = co
            derloss, dertako2, dertako1, derkel = Jgrad
    
            derW2 = h.T @ (derloss * oder)+dertako2
            derW1 = data.T @ (hder*((derloss * oder) @ W2.T + derkel)) + dertako1
            derWe = (((derW1.T + derW2)/2).T, (derW1.T + derW2)/2, (hder*((derloss * oder) @ W2.T + derkel)).sum(axis=0, keepdims=True), (derloss * oder).sum(axis=0, keepdims=True))
            
            We, momWe = self.modify(We, momWe, derWe, lrate, mom)
            
            return We, momWe
        
        def modify(self, We, momWe, derWe, lrate, mom):
            
            W1, W2, b1, b2 = We
            derW1, derW2, derb1, derb2 = derWe
            momW1, momW2, momb1, momb2 = momWe
            
            assert(W1 == W2.T).all()
            We = (W1 - (lrate*derW1 + mom*momW1),W2 - (lrate*derW2 + mom*momW2),b1 - (lrate*derb1 + mom*momb1),b2 - (lrate*derb2 + mom*momb2))
            momWe = (lrate*derW1 + mom*momW1,lrate*derW2 + mom*momW2,lrate*derb1 + mom*momb1,lrate*derb2 + mom*momb2)
            
            return We, momWe
        
        def workout1(self, data, params, lrate, mom, epoch, batch):
            Jl = []
                
            Lin = params["Lin"]
            Lhid = params["Lhid"]
            We, momWe = self.initialize1(Lin, Lhid)
            for i in range(epoch):
                
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
                    
                Jt = Jt/(int(data.shape[0]/batch))
                
                print("Epoch: {}, Loss: {:.3f}".format(i+1, Jt))
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

            h, hder = self.sigmoid(data @ W1+b1)

            o, oder = self.sigmoid(h @ W2+b2)
            
            rhobe = h.mean(axis=0, keepdims=True)
            
            J = ((0.5*(np.linalg.norm(data-o,axis=1)**2).sum())/nar) + (0.5*lamda*(np.sum(W1**2) + np.sum(W2**2))) + (beta * (rho*np.log(rho/rhobe) + (1-rho)*np.log((1-rho)/(1-rhobe))).sum())
            
            co = (data, h, hder, oder)
            Jgrad = ((o-data)/nar, lamda * W2, lamda * W1, (beta * ((1-rho)/(1-rhobe) - rho/rhobe)/nar))
            
            return J, Jgrad, co
        
        def sigmoid(self, x):
            k = np.exp(x)/(1 + np.exp(x))
            l = k * (1-k)
            return k,l
        
    filename1 = "assign3_data1.h5"
    f1 = h5py.File(filename1, 'r')
    data = np.array(f1['data'])
    
    """ Part a """    
    datanew = 0.2126 * data[:, 0] + 0.7152 * data[:, 1] + 0.0722 * data[:, 2]
    
    assert datanew.shape[1] == datanew.shape[2]
    dimension = datanew.shape[1]
    
    datanew = (np.reshape(datanew, (datanew.shape[0], dimension ** 2))) - (np.reshape(datanew, (datanew.shape[0], dimension ** 2))).mean(axis=1, keepdims = True)
    datanew = 0.1 + 0.8*(norma(np.clip(datanew, - 3 * (np.std(datanew)), 3 * (np.std(datanew)))))

    trd = datanew
    
    datanew = np.reshape(datanew, (datanew.shape[0], dimension, dimension))
    data = data.transpose((0,2,3,1))
    
    """ Part c """
    
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
    print("\nParameters: rho =",rho,",","beta =",beta,",","lrate =",lrate,",","momentum =",mom,",","lambda =",lamda,",","batch =",batch,",","Lin =",Lin,",","Lhid =",Lhid,",","epoch =",epoch,"\n")
    w_dimensioni, wsoni = dis(lrate, mom, epoch, batch, rho, beta, lamda, Lin, Lhid, AE, trd, dimension)

    """ Part d """
    
    print("\nQuestion 1 Part d")
    Lhidl = 16
    Lhidn = 49
    Lhidh = 100
    lamdal = 0
    lamdan = 1e-5
    lamdah = 1e-3
    
    print("\nlambda =",lamdal,",","Lhid =",Lhidl)
    w_dimension1, wson1 = dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidl, AE, trd, dimension)
    print("\nlambda =",lamdal,",","Lhid =",Lhidn)
    w_dimension2, wson2 = dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidn, AE, trd, dimension)
    print("\nlambda =",lamdal,",","Lhid =",Lhidh)
    w_dimension3, wson3 = dis(lrate, mom, epoch, batch, rho, beta, lamdal, Lin, Lhidh, AE, trd, dimension)
    print("\nlambda =",lamdan,",","Lhid =",Lhidl)
    w_dimension4, wson4 = dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidl, AE, trd, dimension)
    print("\nlambda =",lamdan,",","Lhid =",Lhidn)
    w_dimension5, wson5 = dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidn, AE, trd, dimension)
    print("\nlambda =",lamdan,",","Lhid =",Lhidh)
    w_dimension6, wson6 = dis(lrate, mom, epoch, batch, rho, beta, lamdan, Lin, Lhidh, AE, trd, dimension)
    print("\nlambda =",lamdah,",","Lhid =",Lhidl)
    w_dimension7, wson7 = dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidl, AE, trd, dimension)
    print("\nlambda =",lamdah,",","Lhid =",Lhidn)
    w_dimension8, wson8 = dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidn, AE, trd, dimension)
    print("\nlambda =",lamdah,",","Lhid =",Lhidh)
    w_dimension9, wson9 = dis(lrate, mom, epoch, batch, rho, beta, lamdah, Lin, Lhidh, AE, trd, dimension)
    
    
    """Plots"""
    print('\nTo continue executing, after observation please close the plots')
    print("\nQuestion 1 Part a Plot")
    fig1, ax1 = plt.subplots(10,20,figsize=(20,10))
    fig1.suptitle("Question 1, Part a, RGB Images")
    fig2, ax2 = plt.subplots(10, 20, figsize=(20, 10))
    fig2.suptitle("Question 1, Part a, Grayscale Images")
    
    for a in range (10):
        for b in range (20):
            c = np.random.randint(0, data.shape[0])
            ax2[a,b].imshow(datanew[c], cmap='gray')
            ax2[a,b].axis("off")
            
            ax1[a,b].imshow(data[c].astype('float'))
            ax1[a,b].axis('off')
            
    print('\nTo continue executing, after observation please close the plots')        
    plt.show()
    
    print("\nQuestion 1 Part c Plot")
    imageshowc(wsoni, w_dimensioni, lamda, Lhid)
    print("\nQuestion 1 Part d Plots")
    imageshowd(wson1, w_dimension1, lamdal, Lhidl)
    imageshowd(wson2, w_dimension2, lamdal, Lhidn)
    imageshowd(wson3, w_dimension3, lamdal, Lhidh)
    imageshowd(wson4, w_dimension4, lamdan, Lhidl)
    imageshowd(wson5, w_dimension5, lamdan, Lhidn)
    imageshowd(wson6, w_dimension6, lamdan, Lhidh)
    imageshowd(wson7, w_dimension7, lamdah, Lhidl)
    imageshowd(wson8, w_dimension8, lamdah, Lhidn)
    imageshowd(wson9, w_dimension9, lamdah, Lhidh)


def q3():
    class Network3():
    
        def __init__(self,big,num):
            
            self.num = num
            self.big = big
            self.laybig = len(big)-1
            self.percbig = None
            self.percpar = None
            self.flaypar = None
            self.percmom = None
            self.flaymom = None
            self.initialize3()
            
        def initialize3(self):
            num = self.num
            big = self.big
            laybig = self.laybig
            
            weights = []
            bias = []
            
            for i in range(1,laybig):
                weights.append(np.random.uniform(-(np.sqrt(6)/np.sqrt(big[1] + big[i+1])),(np.sqrt(6)/np.sqrt(big[1] + big[i+1])), size=(big[i],big[i+1])))
                bias.append(np.zeros((1, big[i+1])))
                
            self.percbig = len(weights)
            params = {"weights":weights, "bias":bias}
            mom = {"weights": [0]*self.percbig, "bias": [0]*self.percbig}
            self.percpar = params
            self.percmom = mom
            
            ne = big[0]
            he = big[1]
            ze = ne + he
            
            if num == 1:
                weightsih = np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)), (np.sqrt(6)/np.sqrt(ne+he)), size = (ne,he))
                weightshh = np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)), (np.sqrt(6)/np.sqrt(he+he)), size = (he,he))
                bias = np.zeros((1,he))
                
                params = {"weightsih": weightsih, "weightshh": weightshh, "bias": bias}
                
            if num == 2:
                weightsf = np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)), (np.sqrt(6)/np.sqrt(ze+he)), size = (ze,he))
                biasf = np.zeros((1,he))
                weightsi = np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)), (np.sqrt(6)/np.sqrt(ze+he)), size = (ze,he))
                biasi = np.zeros((1,he))
                weightsc = np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)), (np.sqrt(6)/np.sqrt(ze+he)), size = (ze,he))
                biasc = np.zeros((1,he))
                weightso = np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)), (np.sqrt(6)/np.sqrt(ze+he)), size = (ze,he))
                biaso = np.zeros((1,he))
               
                params = {"weightsf": weightsf, "biasf": biasf,"weightsi": weightsi, "biasi": biasi,"weightsc": weightsc, "biasc": biasc,"weightso": weightso, "biaso": biaso}
                
            if num == 3:
                weightsz = np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)), (np.sqrt(6)/np.sqrt(ne+he)), size=(ne, he))
                uzaz = np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)), (np.sqrt(6)/np.sqrt(he+he)), size=(he, he))
                biasz = np.zeros((1, he))
                weightsr = np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)), (np.sqrt(6)/np.sqrt(ne+he)), size=(ne, he))
                uzar = np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)), (np.sqrt(6)/np.sqrt(he+he)), size=(he, he))
                biasr = np.zeros((1, he))
                weightsh = np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)), (np.sqrt(6)/np.sqrt(ne+he)), size=(ne, he))
                uzah = np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)), (np.sqrt(6)/np.sqrt(he+he)), size=(he, he))
                biash = np.zeros((1, he))
                
                params = {"weightsz":weightsz, "uzaz": uzaz, "biasz": biasz, "weightsr":weightsr, "uzar": uzar, "biasr": biasr, "weightsh":weightsh, "uzah": uzah, "biash": biash}
                
            mom = dict.fromkeys(params.keys(), 0)
            self.flaypar = params
            self.flaymom = mom
            
        def modify(self, lrate, momc, gradflay, gradperc):
            
            flaypar = self.flaypar
            flaymom = self.flaymom
            percpar = self.percpar
            percmom = self.percmom
            
            for k in self.flaypar:
                flaymom[k] = lrate * gradflay[k] + momc * flaymom[k]
                flaypar[k] = flaypar[k] - (lrate * gradflay[k] + momc * flaymom[k])
                
            for i in range(self.percbig):
    
                percmom["weights"][i] = lrate*gradperc["weights"][i] + momc * percmom["weights"][i]
                percmom["bias"][i] = lrate * gradperc["bias"][i] + momc * percmom["bias"][i]
                
                percpar["weights"][i] = percpar["weights"][i] - (lrate*gradperc["weights"][i] + momc * percmom["weights"][i])
                percpar["bias"][i] = percpar["bias"][i] - (lrate * gradperc["bias"][i] + momc * percmom["bias"][i])
            
            self.flaypar = flaypar
            self.flaymom = flaymom
            self.percpar = percpar
            self.percmom = percmom
            
        def ileriperc(self,a,weights,bias,b):
            return self.activ((a @ weights + bias), b)
        
        def geriperc(self, weights, o, der, chan):
            
            dweights = o.T @ chan
            dbias = chan.sum(axis=0, keepdims=True)
            chan = der * (chan @ weights.T)
            return dweights, dbias, chan 
        
        def workout3(self, a, b, lrate, momc, batch, epoch):
            traininglossl, validationlossl, trainingaccuracyl, validationaccuracyl = [], [], [], []
            
            valbig = int(a.shape[0]/10)
            po = np.random.permutation(a.shape[0])
            vala = a[po][:valbig]
            valb = b[po][:valbig]
            a = a[po][valbig:]
            b = b[po][valbig:]
            it = int(a.shape[0]/batch)
            
            for i in range(epoch):           
                go = 0
                stop = batch
                po = np.random.permutation(a.shape[0])
                a = a[po]
                b = b[po]
                
                for j in range(it):
                    gues, o, der, h, hder, co = self.ilerigo(a[go:stop])
                    
                    chan = gues
                    chan[b[go:stop] ==1] = chan[b[go:stop] ==1]-1
                    chan = chan/batch
                    
                    gradflay, gradperc = self.gerigo(a[go:stop], o, der, chan, h, hder, co)
                    
                    self.modify(lrate, momc, gradflay, gradperc)
                    
                    go = stop
                    stop = stop + batch
    
                trainingaccuracy, trainingloss = calcul(a,b,self.guess3,self.CE)                
                validationaccuracy, validationloss = calcul(vala,valb,self.guess3,self.CE)
                    
                print("Epoch: %d | Training Loss: %.3f, Validation Loss: %.3f, Training Accuracy: %.3f, Validation Accuracy: %.3f"% (i + 1, trainingloss, validationloss, trainingaccuracy, validationaccuracy))
                traininglossl.append(trainingloss)
                validationlossl.append(validationloss)
                trainingaccuracyl.append(trainingaccuracy)
                validationaccuracyl.append(validationaccuracy)
                
                
                if i>15:
                    convergence = sum(validationlossl[-16:-1]) / len(validationlossl[-16:-1])
                    if (convergence - 0.001) < validationloss < (convergence + 0.001):
                        print("\nTraining stopped since validation C-E reached convergence.")
                        return {"traininglossl": traininglossl, "validationlossl": validationlossl, "trainingaccuracyl": trainingaccuracyl, "validationaccuracyl": validationaccuracyl}
            return {"traininglossl": traininglossl, "validationlossl": validationlossl, "trainingaccuracyl": trainingaccuracyl, "validationaccuracyl": validationaccuracyl}
        
        def activ(self, a, A):
            if A == "softmax":
                activ = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)
                deriv = None
                return activ, deriv
            
            if A == "tanh":
                activ = np.tanh(a)
                deriv = 1 - activ**2
                return activ, deriv
            
            if A == "relu":
                activ = a * (a>0)
                deriv = 1 * (a>0)
                return activ, deriv
            
            if A == "sigmoid":
                activ = np.exp(a)/(1 + np.exp(a))
                deriv = activ * (1-activ)
                return activ, deriv
        
        def ilerirnn(self, a, flaypar):
            
            ne, te, de = a.shape
            
            weightsih = flaypar["weightsih"]
            weightshh = flaypar["weightshh"]
            bias = flaypar["bias"]
            
            hbefore = np.zeros((ne, self.big[1]))
            h, hder = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
            
            for k in range(te):
                h[:, k, :], hder[:, k, :] = self.activ((a[:, k, :] @ weightsih + hbefore @ weightshh + bias), "tanh")
                hbefore = h[:, k, :]
                
            return h, hder
        
        def gerirnn(self, a, h, hder, chan, flaypar):
            
            ne, te, de = a.shape            
            weightshh = flaypar["weightshh"]            
            dweightsih, dweightshh, dbias = 0,0,0
            
            for k in reversed(range(te)):
                hbefore = ifing1(h, self.big[1], ne, k)
                hbeforeder = ifing2(hder, k)                
                dweightsih = dweightsih + a[:, k, :].T @ chan
                dweightshh = dweightshh + hbefore.T @ chan
                dbias = dbias + chan.sum(axis=0, keepdims=True)
                chan = hbeforeder * (chan@weightshh)
                
            return {"weightsih": dweightsih, "weightshh": dweightshh, "bias": dbias}
        
        def ilerigo(self,a):
            
            num = self.num
            percpar = self.percpar
            flaypar = self.flaypar
            o, der = [], []            
            h, hder, co = 0,0,0
            
            if num == 1:
                h, hder = self.ilerirnn(a, flaypar)
                o.append(h[:,-1,:])
                der.append(hder[:,-1,:])
            
            if num == 2:
                h, co = self.ilerilstm(a, flaypar)
                o.append(h)
                der.append(1)
                
            if num == 3:
                h, co = self.ilerigru(a,flaypar)
                o.append(h)
                der.append(1)
                
            for i in range(self.percbig-1):
                activ, deriv = self.ileriperc(o[-1], percpar["weights"][i], percpar["bias"][i], "relu")
                o.append(activ)
                der.append(deriv)
            
            gues = self.ileriperc(o[-1], percpar["weights"][-1], percpar["bias"][-1], "softmax")[0]
            
            return gues, o, der, h, hder, co
        
        def gerigo(self, a, o, der, chan, h, hder, co):
            
            num = self.num
            percpar = self.percpar
            flaypar = self.flaypar
            
            gradflay = dict.fromkeys(percpar.keys())
            gradperc = {"weights": [0] * self.percbig, "bias": [0]*self.percbig}
            
            for i in reversed(range(self.percbig)):
                gradperc["weights"][i], gradperc["bias"][i], chan = self.geriperc(percpar["weights"][i], o[i], der[i], chan)
                
            if num == 1:
                gradflay = self.gerirnn(a, h, hder, chan, flaypar)
            if num == 2:
                gradflay = self.gerilstm(co, flaypar, chan)
            if num == 3:
                gradflay = self.gerigru(a, co, flaypar, chan)
                
            return gradflay, gradperc
                    
        def ilerilstm(self, a, flaypar):
            
            ne, te, de = a.shape
            
            weightsi, biasi = flaypar["weightsi"], flaypar["biasi"]
            weightsf, biasf = flaypar["weightsf"], flaypar["biasf"]
            weightso, biaso = flaypar["weightso"], flaypar["biaso"]
            weightsc, biasc = flaypar["weightsc"], flaypar["biasc"]
            
            hbefore, cbefore = np.zeros((ne, self.big[1])), np.zeros((ne, self.big[1]))
            zi = np.empty((ne, te, de + self.big[1]))
            hfi = 0
            
            hii, hci, hoi, tanhci, ci, tanhcdi, hfdi, hidi, hcdi, hodi = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
            
            for k in range(te):
                zi[:, k, :] = np.column_stack((hbefore, a[:, k, :]))
                
                hfi, hfdi[:, k, :] = self.activ(zi[:, k, :] @ weightsf + biasf, "sigmoid")
                hii[:, k, :], hidi[:, k, :] = self.activ(zi[:, k, :] @ weightsi + biasi, "sigmoid")
                hci[:, k, :], hcdi[:, k, :] = self.activ(zi[:, k, :] @ weightsc + biasc, "tanh")
                hoi[:, k, :], hodi[:, k, :] = self.activ(zi[:, k, :] @ weightso + biaso, "sigmoid")
                
                ci[:, k, :] = hfi * cbefore + hii[:, k, :] * hci[:, k, :]
                tanhci[:, k, :], tanhcdi[:, k, :] = self.activ(ci[:, k, :], "tanh")
                hbefore = hoi[:, k, :] * tanhci[:, k, :]
                cbefore = ci[:, k, :]
                
                co = {"zi": zi, "ci": ci, "tanhci": (tanhci, tanhcdi), "hfdi": hfdi, "hii": (hii, hidi), "hci": (hci, hcdi), "hoi": (hoi, hodi)}
                
            return hbefore, co
        
        def gerilstm(self, co, flaypar, chan):
            
            weightsf = flaypar["weightsf"]
            weightsi = flaypar["weightsi"]
            weightsc = flaypar["weightsc"]
            weightso = flaypar["weightso"]
            
            zi = co["zi"]
            ci = co["ci"]
            tanhci, tanhcdi = co["tanhci"]
            hfdi = co["hfdi"]
            hii, hidi = co["hii"]
            hci, hcdi = co["hci"]
            hoi, hodi = co["hoi"]
            te = zi.shape[1]
            
            dweightsf, dweightsi, dweightsc, dweightso, dbiasf, dbiasi, dbiasc, dbiaso = 0,0,0,0,0,0,0,0
            
            for k in reversed(range(te)):
                cbefore = ifing2(ci, k)                    
                    
                dci = chan * hoi[:, k, :] * tanhcdi[:, k, :]
                dhfi = dci * cbefore * hfdi[:, k, :]
                dhii = dci * hci[:, k, :] * hidi[:, k, :]
                dhci = dci * hii[:, k, :] * hcdi[:, k, :]
                dhoi = chan * tanhci[:, k, :] * hodi[:, k, :]
                
                dweightsf, dbiasf = lstmproc(dweightsf, dbiasf, zi[:, k, :], dhfi)
                dweightsi, dbiasi = lstmproc(dweightsi, dbiasi, zi[:, k, :], dhii)
                dweightsc, dbiasc = lstmproc(dweightsc, dbiasc, zi[:, k, :], dhci)
                dweightso, dbiaso = lstmproc(dweightso, dbiaso, zi[:, k, :], dhoi)
                
                df = lstmdu(dhfi, weightsf, self.big[1])
                di = lstmdu(dhii, weightsi, self.big[1])
                dc = lstmdu(dhci, weightsc, self.big[1])
                do = lstmdu(dhoi, weightso, self.big[1])
                
                
                chan = (df + di + dc + do)
                
            return {"weightsf": dweightsf, "biasf": dbiasf, "weightsi": dweightsi, "biasi": dbiasi, "weightsc": dweightsc, "biasc": dbiasc, "weightso": dweightso, "biaso": dbiaso}
        
        def ilerigru(self, a, flaypar):
            weightsz = flaypar["weightsz"]
            weightsr = flaypar["weightsr"]
            weightsh = flaypar["weightsh"]
            
            uzaz = flaypar["uzaz"]
            uzar = flaypar["uzar"]
            uzah = flaypar["uzah"]
            
            biasz = flaypar["biasz"]
            biasr = flaypar["biasr"]
            biash = flaypar["biash"]
            
            ne, te, de = a.shape            
            hbefore = np.zeros((ne, self.big[1]))
            zi, zdi, ri, rdi, htider, htiderd, hi = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
            
            for k in range(te):
                zi[:, k, :], zdi[:, k, :] = self.activ(a[:, k, :] @ weightsz + hbefore @ uzaz + biasz, "sigmoid")
                ri[:, k, :], rdi[:, k, :] = self.activ(a[:, k, :] @ weightsr + hbefore @ uzar + biasr, "sigmoid")
                htider[:, k, :], htiderd[:, k, :] = self.activ(a[:, k, :] @ weightsh + (ri[:, k, :] * hbefore) @ uzah + biash, "tanh")
                hi[:, k, :] = (1 - zi[:, k, :]) * hbefore + zi[:, k, :] * htider[:, k, :]
                
                hbefore = hi[:, k, :]
                
            co = {"zi": (zi, zdi), "ri": (ri, rdi), "htider": (htider, htiderd), "hi": hi}
            
            return hbefore, co
        
        def gerigru(self, a, co, flaypar, chan):
            
            uzaz = flaypar["uzaz"]
            uzar = flaypar["uzar"]
            uzah = flaypar["uzah"]
            
            zi, zdi = co["zi"]
            ri, rdi = co["ri"]
            htider, htiderd = co["htider"]
            hi = co["hi"]
            
            ne, te, de = a.shape            
            dweightsz, dweightsr, dweightsh, duzaz, duzar, duzah, dbiasz, dbiasr, dbiash = 0,0,0,0,0,0,0,0,0
        
            for k in reversed(range(te)):                
                hbefore = ifing1(hi, self.big[1], ne, k)    
  
                dzi = chan * (htider[:, k, :] - hbefore) * zdi[:, k, :]
                dhtider = chan * zi[:, k, :] * htiderd[:, k, :]
                dri = (dhtider @ uzah.T) * hbefore * rdi[:, k, :]
                
                dweightsz, duzaz, dbiasz = gruproc(a[:, k, :], dzi, hbefore, dweightsz, duzaz, dbiasz)
                dweightsr, duzar, dbiasr = gruproc(a[:, k, :], dri, hbefore, dweightsr, duzar, dbiasr)
                dweightsh, duzah, dbiash = gruproc(a[:, k, :], dhtider, hbefore, dweightsh, duzah, dbiash)
                
                chan = (chan * (1 - zi[:, k, :])) + (dzi @ uzaz.T) + ((dhtider @ uzah.T) * (ri[:, k, :] + hbefore * (rdi[:, k, :] @ uzar.T)))
            
            return {"weightsz": dweightsz, "uzaz": duzaz, "biasz": dbiasz, "weightsr": dweightsr, "uzar": duzar, "biasr": dbiasr, "weightsh": dweightsh, "uzah": duzah, "biash": dbiash}
              
        def guess3(self, a, b=None, accur=True, conf=False):
            
            guessino = self.ilerigo(a)[0]
            
            if not accur:
                return guessino
            
            guessino = guessino.argmax(axis=1)
            b = b.argmax(axis=1)
            
            if not conf:
                return (guessino == b).mean() * 100
            
            cla = np.zeros((len(np.unique(b)),len(np.unique(b))))
            
            for k in range(len(b)):
                cla[b[k]][guessino[k]] = cla[b[k]][guessino[k]] + 1
                
            return cla
        
        def CE(self, d, y):
            return np.sum(np.log(y) * -d) / d.shape[0]
        
    filename3 = "assign3_data3.h5"
    f3 = h5py.File(filename3, 'r')
    
    trainx = np.array(f3['trX'])
    trainy = np.array(f3['trY'])
    testx = np.array(f3['tstX'])
    testy = np.array(f3['tstY'])
    
    
    """Part a"""

    print("Question 3 Part a")
    print("Recurrent Layer\n")
    epoch3rnn = 50
    lrate3rnn = 0.01
    batch3rnn = 32
    momc3rnn = 0.85
    bigrnn = [trainx.shape[2], 128, 32, 16, 6]
    
    net3rnn, traininglosslrnn, validationlosslrnn, trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn = getting(Network3, bigrnn, trainx, trainy, lrate3rnn, momc3rnn, batch3rnn, epoch3rnn, testx, testy, 1)    
    print("\nTest Accuracy: ", testaccuracyrnn, "\n\n")
    
    trainingconfrnn, testingconfrnn = guessing(net3rnn, trainx, trainy, testx, testy)    
    
    """Part b"""
    print("\nQuestion 3 Part b")
    print("LSTM Layer\n")
    
    epoch3lstm = 50
    lrate3lstm = 0.01
    batch3lstm = 32
    momc3lstm = 0.85
    biglstm = [trainx.shape[2], 128, 32, 16, 6]
   
    net3lstm, traininglossllstm, validationlossllstm, trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm = getting(Network3, biglstm, trainx, trainy, lrate3lstm, momc3lstm, batch3lstm, epoch3lstm, testx, testy, 2)
    print("\nTest Accuracy: ", testaccuracylstm, "\n\n")

    trainingconflstm, testingconflstm = guessing(net3lstm, trainx, trainy, testx, testy)
    
    """Part c"""
    print("\nQuestion 3 Part c")
    print("GRU Layer\n")
    
    epoch3gru = 50
    lrate3gru = 0.01
    batch3gru = 32
    momc3gru = 0.85
    biggru = [trainx.shape[2], 128, 32, 16, 6]
    
    net3gru, traininglosslgru, validationlosslgru, trainingaccuracylgru, validationaccuracylgru, testaccuracygru= getting(Network3, biggru, trainx, trainy, lrate3gru, momc3gru, batch3gru, epoch3gru, testx, testy, 3)    
    print("\nTest Accuracy: ", testaccuracygru, "\n\n")

    trainingconfgru, testingconfgru = guessing(net3gru, trainx, trainy, testx, testy)
        
    """ Plots"""
    print('To continue executing, after observation please close the plots')
    print("\nQuestion 3, Part a Plots")
    graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, traininglosslrnn, "RNN")
    plt.title("Training Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, validationlosslrnn, "RNN")
    plt.title("Validation Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, trainingaccuracylrnn, "RNN")
    plt.title("Training Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
    graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, validationaccuracylrnn, "RNN")
    plt.title("Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.show()    
    
    graphm(trainingconfrnn, testingconfrnn)
    plt.show()
    
    
    print("\nQuestion 3, Part b Plots")
    graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, traininglossllstm, "LSTM")
    plt.title("Training Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, validationlossllstm, "LSTM")
    plt.title("Validation Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, trainingaccuracyllstm, "LSTM")
    plt.title("Training Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
    graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, validationaccuracyllstm, "LSTM")
    plt.title("Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
    graphm(trainingconflstm, testingconflstm)
    plt.show()
    
    
    print("\nQuestion 3, Part c Plots")
    graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, traininglosslgru, "GRU")
    plt.title("Training Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, validationlosslgru, "GRU")
    plt.title("Validation Cross Entropy Loss")
    plt.ylabel("Loss")
    plt.show()
    
    graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, trainingaccuracylgru, "GRU")
    plt.title("Training Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
    graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, validationaccuracylgru, "GRU")
    plt.title("Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
    graphm(trainingconfgru, testingconfgru)
    plt.show()
    
question = sys.argv[1]

def ayberk_yarkin_yildiz_21803386_hw3(question):
    
    if question == '1' :
        q1()
        
    elif question == '3' :
        q3()
      
ayberk_yarkin_yildiz_21803386_hw3(question)