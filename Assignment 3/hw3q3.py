# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:16:28 2021

@author: User
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
# import time
# import torch

# device = torch.device('cuda')

filename3 = "assign3_data3.h5"
f3 = h5py.File(filename3, 'r')

trainx = np.array(f3['trX'])
trainy = np.array(f3['trY'])
testx = np.array(f3['tstX'])
testy = np.array(f3['tstY'])

# trainx = f3['trX'][()].astype('float64')
# testx = f3['tstX'][()].astype('float64')
# trainy = f3['trY'][()].astype('float64')
# testy = f3['tstY'][()].astype('float64')
# f3.close()

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
            # re = (np.sqrt(6)/np.sqrt(big[1] + big[i+1]))
            # re = np.sqrt(6 / (big[1]+big[i+1]))
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
            # re = (np.sqrt(6)/np.sqrt(ne+he))
            # re = np.sqrt(6 / (ne+he))
            weightsih = np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)), (np.sqrt(6)/np.sqrt(ne+he)), size = (ne,he))
            # re = (np.sqrt(6)/np.sqrt(he+he))
            # re = np.sqrt(6 / (he+he))
            weightshh = np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)), (np.sqrt(6)/np.sqrt(he+he)), size = (he,he))
            bias = np.zeros((1,he))
            
            params = {"weightsih": weightsih, "weightshh": weightshh, "bias": bias}
            
        if num == 2:
            # re = (np.sqrt(6)/np.sqrt(ze+he))
            # re = np.sqrt(6 / (ze+he))
            
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
            # ren = (np.sqrt(6)/np.sqrt(ne+he))
            # ren = np.sqrt(6 / (ne+he))
            # reh = (np.sqrt(6)/np.sqrt(he+he))
            # reh = np.sqrt(6 / (he+he))
            
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
        
    def workout3(self, a, b, lrate, momc, batch, epoch):
        
        # traininglossl = []
        # validationlossl = []
        # trainingaccuracyl = []
        # validationaccuracyl = []
        
        traininglossl, validationlossl, trainingaccuracyl, validationaccuracyl = [], [], [], []
        
        valbig = int(a.shape[0]/10)
        po = np.random.permutation(a.shape[0])
        vala = a[po][:valbig]
        valb = b[po][:valbig]
        a = a[po][valbig:]
        b = b[po][valbig:]
        
        # size = a.shape[0]
        it = int(a.shape[0]/batch)
        
        for i in range(epoch):           
            go = 0
            stop = batch
            po = np.random.permutation(a.shape[0])
            a = a[po]
            b = b[po]
            
            for j in range(it):
                
                # batcha = a[go:stop]
                # batchb = b[go:stop]
                
                gues, o, der, h, hder, co = self.ilerigo(a[go:stop])
                
                chan = gues
                chan[b[go:stop] ==1] = chan[b[go:stop] ==1]-1
                chan = chan/batch
                
                gradflay, gradperc = self.gerigo(a[go:stop], o, der, chan, h, hder, co)
                
                self.modify(lrate, momc, gradflay, gradperc)
                
                go = stop
                stop = stop + batch
            
            # # gues = self.guess3(a, accur=False)
            # trainingloss = self.CE(b, self.guess3(a, accur=False))
            
            # trainingaccuracy = self.guess3(a,b,accur=True)

            trainingaccuracy, trainingloss = calcul(a,b,self.guess3,self.CE)
            
            # validationaccuracy = self.guess3(vala, valb, accur=True)
            
            # # gues = self.guess3(vala, accur=False)
            # validationloss = self.CE(valb, self.guess3(vala, accur=False))
            
            validationaccuracy, validationloss = calcul(vala,valb,self.guess3,self.CE)
            
            # time_remain = (epoch-i-1)*(time.time()-time_start)
            
            # if time_remain <60:
            #     time_remain = round(time_remain)
            #     time_label = "seconds(s)"
                
            # else:
            #     time_remain = round(time_remain/60)
            #     time_label = "minute(s)"
                
            # print('Training Loss: %.2f, Validation Loss: %.2f, Training Accuracy: %.2f, Validation Accuracy: %.2f [Epoch: %d of %d, ETA: %d %s]'% (trainingloss, validationloss, trainingaccuracy, validationaccuracy, i + 1, epoch, time_remain, time_label))
            print("Epoch: %d | Training Loss: %.3f, Validation Loss: %.3f, Training Accuracy: %.3f, Validation Accuracy: %.3f"% (i + 1, trainingloss, validationloss, trainingaccuracy, validationaccuracy))
            traininglossl.append(trainingloss)
            validationlossl.append(validationloss)
            trainingaccuracyl.append(trainingaccuracy)
            validationaccuracyl.append(validationaccuracy)
            
            
            if i>15:
                # convergence = validationlossl[-16:-1]
                convergence = sum(validationlossl[-16:-1]) / len(validationlossl[-16:-1])
                
                # limiting = 0.001
                if (convergence - 0.001) < validationloss < (convergence + 0.001):
                    print("\nTraining stopped since validation C-E reached convergence.")
                    return {"traininglossl": traininglossl, "validationlossl": validationlossl, "trainingaccuracyl": trainingaccuracyl, "validationaccuracyl": validationaccuracyl}
        return {"traininglossl": traininglossl, "validationlossl": validationlossl, "trainingaccuracyl": trainingaccuracyl, "validationaccuracyl": validationaccuracyl}
    
    def ilerigo(self,a):
        
        num = self.num
        percpar = self.percpar
        flaypar = self.flaypar
        
        # o = []
        # der = []
        o, der = [], []
        # h = 0
        # hder = 0
        # co = 0
        
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
        # uv = a @ weights + bias
        return self.activ((a @ weights + bias), b)
    
    def geriperc(self, weights, o, der, chan):
        
        dweights = o.T @ chan
        dbias = chan.sum(axis=0, keepdims=True)
        chan = der * (chan @ weights.T)
        return dweights, dbias, chan
    
    def ilerirnn(self, a, flaypar):
        
        ne, te, de = a.shape
        # he = self.big[1]
        
        weightsih = flaypar["weightsih"]
        weightshh = flaypar["weightshh"]
        bias = flaypar["bias"]
        
        hbefore = np.zeros((ne, self.big[1]))
        
        h, hder = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
        # h = np.empty((ne, te, self.big[1]))
        # hder = np.empty((ne, te, self.big[1]))
        
        for k in range(te):
            # A = a[:, k, :]
            # uv = a[:, k, :] @ weightsih + hbefore @ weightshh + bias
            h[:, k, :], hder[:, k, :] = self.activ((a[:, k, :] @ weightsih + hbefore @ weightshh + bias), "tanh")
            hbefore = h[:, k, :]
            
        return h, hder
    
    def gerirnn(self, a, h, hder, chan, flaypar):
        
        ne, te, de = a.shape
        # he = self.big[1]
        
        weightshh = flaypar["weightshh"]
        
        # dweightsih = 0
        # dweightshh = 0
        # dbias = 0
        
        dweightsih, dweightshh, dbias = 0,0,0
        
        for k in reversed(range(te)):
            # A = a[:, k, :]
            
            hbefore = ifing1(h, self.big[1], ne, k)
            hbeforeder = ifing2(hder, k)
            
            # if k > 0 :
            #     hbefore = h[:, k-1, :]
            #     hbeforeder = hder[:, k-1, :]
            # else:
            #     hbefore = np.zeros((ne,self.big[1]))
            #     hbeforeder = 0
            
            dweightsih = dweightsih + a[:, k, :].T @ chan
            dweightshh = dweightshh + hbefore.T @ chan
            dbias = dbias + chan.sum(axis=0, keepdims=True)
            chan = hbeforeder * (chan@weightshh)
            
        return {"weightsih": dweightsih, "weightshh": dweightshh, "bias": dbias}
                
    def ilerilstm(self, a, flaypar):
        
        ne, te, de = a.shape
        # he = self.big[1]
        
        weightsi, biasi = flaypar["weightsi"], flaypar["biasi"]
        weightsf, biasf = flaypar["weightsf"], flaypar["biasf"]
        weightso, biaso = flaypar["weightso"], flaypar["biaso"]
        weightsc, biasc = flaypar["weightsc"], flaypar["biasc"]
        
        hbefore, cbefore = np.zeros((ne, self.big[1])), np.zeros((ne, self.big[1]))
        # cbefore = np.zeros((ne, he))
        zi = np.empty((ne, te, de + self.big[1]))
        # ci = np.empty((ne, te, he))
        # tanhci = np.empty((ne, te, he))
        hfi = 0
        
        hii, hci, hoi, tanhci, ci, tanhcdi, hfdi, hidi, hcdi, hodi = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
        
        # hii = np.empty((ne, te, he))
        # hci = np.empty((ne, te, he))
        # hoi = np.empty((ne, te, he))
        # tanhcdi = np.empty((ne, te, he))
        # hfdi = np.empty((ne, te, he))
        # hidi = np.empty((ne, te, he))
        # hcdi = np.empty((ne, te, he))
        # hodi = np.empty((ne, te, he))
        
        for k in range(te):
            zi[:, k, :] = np.column_stack((hbefore, a[:, k, :]))
            # zicuro = zi[:, k, :]
            
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
        
        # he = self.big[1]
        te = zi.shape[1]
        
        dweightsf, dweightsi, dweightsc, dweightso, dbiasf, dbiasi, dbiasc, dbiaso = 0,0,0,0,0,0,0,0
        # dweightsf = 0
        # dweightsi = 0
        # dweightsc = 0
        # dweightso = 0
        # dbiasf = 0
        # dbiasi = 0
        # dbiasc = 0
        # dbiaso = 0
        
        for k in reversed(range(te)):
            # zicuro = zi[:, k, :]
            
            cbefore = ifing2(ci, k)    
            
            # if k > 0:
            #     cbefore = ci[:, k-1, :]
            # else:
            #     cbefore = 0
                
                
            dci = chan * hoi[:, k, :] * tanhcdi[:, k, :]
            dhfi = dci * cbefore * hfdi[:, k, :]
            dhii = dci * hci[:, k, :] * hidi[:, k, :]
            dhci = dci * hii[:, k, :] * hcdi[:, k, :]
            dhoi = chan * tanhci[:, k, :] * hodi[:, k, :]
            
            # dweightsf = dweightsf + zi[:, k, :].T @ dhfi
            # dbiasf = dbiasf + dhfi.sum(axis=0, keepdims=True)
            
            # dweightsi = dweightsi + zi[:, k, :].T @ dhii
            # dbiasi = dbiasi + dhii.sum(axis=0, keepdims=True)
            
            # dweightsc = dweightsc + zi[:, k, :].T @ dhci
            # dbiasc = dbiasc + dhci.sum(axis=0, keepdims=True)
            
            # dweightso = dweightso + zi[:, k, :].T @ dhoi
            # dbiaso = dbiaso + dhoi.sum(axis=0, keepdims=True)
            
            dweightsf, dbiasf = lstmproc(dweightsf, dbiasf, zi[:, k, :], dhfi)
            dweightsi, dbiasi = lstmproc(dweightsi, dbiasi, zi[:, k, :], dhii)
            dweightsc, dbiasc = lstmproc(dweightsc, dbiasc, zi[:, k, :], dhci)
            dweightso, dbiaso = lstmproc(dweightso, dbiaso, zi[:, k, :], dhoi)
            
            # df = dhfi @ weightsf.T[:, :self.big[1]]
            # di = dhii @ weightsi.T[:, :self.big[1]]
            # dc = dhci @ weightsc.T[:, :self.big[1]]
            # do = dhoi @ weightso.T[:, :self.big[1]]
            
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
        # he = self.big[1]
        
        hbefore = np.zeros((ne, self.big[1]))
            
        zi, zdi, ri, rdi, htider, htiderd, hi = np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1])), np.empty((ne, te, self.big[1]))
            
        # zi = np.empty((ne, te, he))
        # zdi = np.empty((ne, te, he))
        # ri = np.empty((ne, te, he))
        # rdi = np.empty((ne, te, he))
        # htider = np.empty((ne, te, he))
        # htiderd = np.empty((ne, te, he))
        # hi = np.empty((ne, te, he))
        
        for k in range(te):
            # A = a[:, k, :]
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
        # he = self.big[1]
        
        dweightsz, dweightsr, dweightsh, duzaz, duzar, duzah, dbiasz, dbiasr, dbiash = 0,0,0,0,0,0,0,0,0
        
        # dweightsz = 0
        # duzaz = 0
        # dbiasz = 0
        # dweightsr = 0
        # duzar = 0
        # dbiasr = 0
        # dweightsh = 0
        # duzah = 0
        # dbiash = 0
    
        for k in reversed(range(te)):
            # A = a[:, k, :]
            
            hbefore = ifing1(hi, self.big[1], ne, k)    
            # if k>0:
            #     hbefore = hi[:, k-1, :]
                
            # else:
            #     hbefore = np.zeros((ne,self.big[1]))
                
            dzi = chan * (htider[:, k, :] - hbefore) * zdi[:, k, :]
            dhtider = chan * zi[:, k, :] * htiderd[:, k, :]
            dri = (dhtider @ uzah.T) * hbefore * rdi[:, k, :]
            
            # dweightsz += a[:, k, :].T @ dzi
            # duzaz += hbefore.T @ dzi
            # dbiasz += dzi.sum(axis=0, keepdims=True)
            
            # dweightsr += a[:, k, :].T @ dri
            # duzar += hbefore.T @dri
            # dbiasr += dri.sum(axis=0, keepdims=True)
            
            # dweightsh += a[:, k, :].T @ dhtider
            # duzah += hbefore.T @ dhtider
            # dbiash += dhtider.sum(axis=0, keepdims=True)
            
            dweightsz, duzaz, dbiasz = gruproc(a[:, k, :], dzi, hbefore, dweightsz, duzaz, dbiasz)
            dweightsr, duzar, dbiasr = gruproc(a[:, k, :], dri, hbefore, dweightsr, duzar, dbiasr)
            dweightsh, duzah, dbiash = gruproc(a[:, k, :], dhtider, hbefore, dweightsh, duzah, dbiash)
            
            # d1 = chan * (1 - zi[:, k, :])
            # d2 = dzi @ uzaz.T
            # d3 = (dhtider @ uzah.T) * (ri[:, k, :] + hbefore * (rdi[:, k, :] @ uzar.T))
            
            
            
            chan = (chan * (1 - zi[:, k, :])) + (dzi @ uzaz.T) + ((dhtider @ uzah.T) * (ri[:, k, :] + hbefore * (rdi[:, k, :] @ uzar.T)))
        
        return {"weightsz": dweightsz, "uzaz": duzaz, "biasz": dbiasz, "weightsr": dweightsr, "uzar": duzar, "biasr": dbiasr, "weightsh": dweightsh, "uzah": duzah, "biash": dbiash}
        
    
    def CE(self, d, y):
        return np.sum(np.log(y) * -d) / d.shape[0]
    
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
          
    def guess3(self, a, b=None, accur=True, conf=False):
        
        guessino = self.ilerigo(a)[0]
        
        if not accur:
            return guessino
        
        guessino = guessino.argmax(axis=1)
        b = b.argmax(axis=1)
        
        if not conf:
            return (guessino == b).mean() * 100
        
        # clno = len(np.unique(b))
        cla = np.zeros((len(np.unique(b)),len(np.unique(b))))
        
        for k in range(len(b)):
            cla[b[k]][guessino[k]] = cla[b[k]][guessino[k]] + 1
            
        return cla

# names = [1, 2, 3, 4, 5, 6]

#% Part a

print("Question 3 Part a")
print("Recurrent Layer\n")
epoch3rnn = 3
lrate3rnn = 0.01
batch3rnn = 32
momc3rnn = 0.85

bigrnn = [trainx.shape[2], 128, 32, 16, 6]
print("Parameters: lrate = {} | momentum = {} | batch = {} | hidden Layers = {}\n".format(lrate3rnn, momc3rnn, batch3rnn, bigrnn[2:-1]))

net3rnn, traininglosslrnn, validationlosslrnn, trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn = getting(Network3, bigrnn, trainx, trainy, lrate3rnn, momc3rnn, batch3rnn, epoch3rnn, testx, testy, 1)

# net3rnn = Network3(bigrnn, 1)
# traininglosslrnn, validationlosslrnn, trainingaccuracylrnn, validationaccuracylrnn = net3rnn.workout3(trainx, trainy, lrate3rnn, momc3rnn, batch3rnn, epoch3rnn).values()
# testaccuracyrnn = net3rnn.guess3(testx, testy, accur=True)

print("\nTest Accuracy: ", testaccuracyrnn, "\n\n")

graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, traininglosslrnn, "RNN")

# figrnn1 = plt.figure(figsize=(20, 10))
# figrnn1.suptitle("RNN\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylrnn[-1], validationaccuracylrnn[-1], testaccuracyrnn), fontsize=13)

# # plt.subplot(2, 2, 1)
# plt.plot(traininglosslrnn, "C2", label="Training Cross Entropy Loss")
plt.title("Training Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")

graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, validationlosslrnn, "RNN")


# figrnn2 = plt.figure(figsize=(20, 10))
# figrnn2.suptitle("RNN\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylrnn[-1], validationaccuracylrnn[-1], testaccuracyrnn), fontsize=13)

# # plt.subplot(2, 2, 2)
# plt.plot(validationlosslrnn, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")


graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, trainingaccuracylrnn, "RNN")

# figrnn3 = plt.figure(figsize=(20, 10))
# figrnn3.suptitle("RNN\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylrnn[-1], validationaccuracylrnn[-1], testaccuracyrnn), fontsize=13)

# # plt.subplot(2, 2, 3)
# plt.plot(trainingaccuracylrnn, "C2", label="Training Accuracy")
plt.title("Training Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")

graphn(trainingaccuracylrnn, validationaccuracylrnn, testaccuracyrnn, validationaccuracylrnn, "RNN")

# figrnn4 = plt.figure(figsize=(20, 10))
# figrnn4.suptitle("RNN\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylrnn[-1], validationaccuracylrnn[-1], testaccuracyrnn), fontsize=13)

# # plt.subplot(2, 2, 4)
# plt.plot(validationaccuracylrnn, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")

    
# plt.savefig("q3a.png", bbox_inches='tight')

trainingconfrnn, testingconfrnn = guessing(net3rnn, trainx, trainy, testx, testy)

# trainingconfrnn = net3rnn.guess3(trainx, trainy, accur=True, conf=True)
# testingconfrnn = net3rnn.guess3(testx, testy, accur=True, conf=True)

graphm(trainingconfrnn, testingconfrnn)

# plt.figure(figsize=(20, 10), dpi=160)

# plt.subplot(1, 2, 1)
# sn.heatmap(trainingconfrnn, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Training Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# plt.subplot(1, 2, 2)
# sn.heatmap(testingconfrnn, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Testing Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# # plt.savefig("q3a_confusion.png", bbox_inches='tight')


#% Part b
print("\nQuestion 3 Part b")
print("LSTM Layer\n")

epoch3lstm = 3
lrate3lstm = 0.01
batch3lstm = 32
momc3lstm = 0.85
biglstm = [trainx.shape[2], 128, 32, 16, 6]
print("Parameters: lrate = {} | momentum = {} | batch = {} | hidden layers = {}\n".format(lrate3lstm, momc3lstm, batch3lstm, biglstm[2:-1]))


net3lstm, traininglossllstm, validationlossllstm, trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm = getting(Network3, biglstm, trainx, trainy, lrate3lstm, momc3lstm, batch3lstm, epoch3lstm, testx, testy, 2)

# net3lstm = Network3(biglstm, 2)
# traininglossllstm, validationlossllstm, trainingaccuracyllstm, validationaccuracyllstm = net3lstm.workout3(trainx, trainy, lrate3lstm, momc3lstm, batch3lstm, epoch3lstm).values()
# testaccuracylstm = net3lstm.guess3(testx, testy, accur=True)

print("\nTest Accuracy: ", testaccuracylstm, "\n\n")


graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, traininglossllstm, "LSTM")

# figlstm1 = plt.figure(figsize=(20, 10))
# figlstm1.suptitle("LSTM\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracyllstm[-1], validationaccuracyllstm[-1], testaccuracylstm), fontsize=13)

# # plt.subplot(2, 2, 1)
# plt.plot(traininglossllstm, "C2", label="Training Cross Entropy Loss")
plt.title("Training Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")

graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, validationlossllstm, "LSTM")

# figlstm2 = plt.figure(figsize=(20, 10))
# figlstm2.suptitle("LSTM\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracyllstm[-1], validationaccuracyllstm[-1], testaccuracylstm), fontsize=13)

# # plt.subplot(2, 2, 2)
# plt.plot(validationlossllstm, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")


graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, trainingaccuracyllstm, "LSTM")

# figlstm3 = plt.figure(figsize=(20, 10))
# figlstm3.suptitle("LSTM\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracyllstm[-1], validationaccuracyllstm[-1], testaccuracylstm), fontsize=13)

# # plt.subplot(2, 2, 3)
# plt.plot(trainingaccuracyllstm, "C2", label="Training Accuracy")
plt.title("Training Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")


graphn(trainingaccuracyllstm, validationaccuracyllstm, testaccuracylstm, validationaccuracyllstm, "LSTM")

# figlstm4 = plt.figure(figsize=(20, 10))
# figlstm4.suptitle("LSTM\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracyllstm[-1], validationaccuracyllstm[-1], testaccuracylstm), fontsize=13)

# # plt.subplot(2, 2, 4)
# plt.plot(validationaccuracyllstm, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# plt.savefig("q3b.png", bbox_inches='tight')

# trainingconflstm = net3lstm.guess3(trainx, trainy, accur=True, conf=True)
# testingconflstm = net3lstm.guess3(testx, testy, accur=True, conf=True)

trainingconflstm, testingconflstm = guessing(net3lstm, trainx, trainy, testx, testy)

graphm(trainingconflstm, testingconflstm)

# plt.figure(figsize=(20, 10), dpi=160)

# plt.subplot(1, 2, 1)
# sn.heatmap(trainingconflstm, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Training Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# plt.subplot(1, 2, 2)
# sn.heatmap(testingconflstm, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Testing Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# # plt.savefig("q3b_confusion.png", bbox_inches='tight')


#% Part c
print("\nQuestion 3 Part c")
print("GRU Layer\n")

epoch3gru = 3
lrate3gru = 0.01
batch3gru = 32
momc3gru = 0.85
biggru = [trainx.shape[2], 128, 32, 16, 6]
print("Parameters: lrate = {} | momentum = {} | batch = {} | hidden layers = {}\n".format(lrate3gru, momc3gru, batch3gru, biggru[2:-1]))


net3gru, traininglosslgru, validationlosslgru, trainingaccuracylgru, validationaccuracylgru, testaccuracygru= getting(Network3, biggru, trainx, trainy, lrate3gru, momc3gru, batch3gru, epoch3gru, testx, testy, 3)


# net3gru = Network3(biggru, 3)
# traininglosslgru, validationlosslgru, trainingaccuracylgru, validationaccuracylgru = net3gru.workout3(trainx, trainy, lrate3gru, momc3gru, batch3gru, epoch3gru).values()
# testaccuracygru = net3gru.guess3(testx, testy, accur=True)

print("\nTest Accuracy: ", testaccuracygru, "\n\n")

graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, traininglosslgru, "GRU")


# figgru1 = plt.figure(figsize=(20, 10))
# figgru1.suptitle("GRU\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylgru[-1], validationaccuracylgru[-1], testaccuracygru), fontsize=13)

# # plt.subplot(2, 2, 1)
# plt.plot(traininglosslgru, "C2", label="Training Cross Entropy Loss")
plt.title("Training Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")


graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, validationlosslgru, "GRU")

# figgru2 = plt.figure(figsize=(20, 10))
# figgru2.suptitle("GRU\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylgru[-1], validationaccuracylgru[-1], testaccuracygru), fontsize=13)

# # plt.subplot(2, 2, 2)
# plt.plot(validationlosslgru, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
# plt.xlabel("Epoch")
plt.ylabel("Loss")


graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, trainingaccuracylgru, "GRU")

# figgru3 = plt.figure(figsize=(20, 10))
# figgru3.suptitle("GRU\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylgru[-1], validationaccuracylgru[-1], testaccuracygru), fontsize=13)

# # plt.subplot(2, 2, 3)
# plt.plot(trainingaccuracylgru, "C2", label="Training Accuracy")
plt.title("Training Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")


graphn(trainingaccuracylgru, validationaccuracylgru, testaccuracygru, validationaccuracylgru, "GRU")

# figgru4 = plt.figure(figsize=(20, 10))
# figgru4.suptitle("GRU\n"
#               "Training Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Testing Accuracy: {:.1f}\n "
#               .format(trainingaccuracylgru[-1], validationaccuracylgru[-1], testaccuracygru), fontsize=13)

# # plt.subplot(2, 2, 4)
# plt.plot(validationaccuracylgru, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
# plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# plt.savefig("q3c.png", bbox_inches='tight')

# trainingconfgru = net3gru.guess3(trainx, trainy, accur=True, conf=True)
# testingconfgru = net3gru.guess3(testx, testy, accur=True, conf=True)

trainingconfgru, testingconfgru = guessing(net3gru, trainx, trainy, testx, testy)

graphm(trainingconfgru, testingconfgru)


# plt.figure(figsize=(20, 10), dpi=160)

# plt.subplot(1, 2, 1)
# sn.heatmap(trainingconfgru, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Training Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# plt.subplot(1, 2, 2)
# sn.heatmap(testingconfgru, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
# plt.title("Testing Confusion Matrix")
# plt.ylabel("Real")
# plt.xlabel("Guess")
# # plt.savefig("q3c_confusion.png", bbox_inches='tight')
