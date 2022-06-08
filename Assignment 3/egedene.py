# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:17:13 2021

@author: User
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
import time


filename = "assign3_data1.h5"
f1 = h5py.File(filename, 'r')
#FIXME
""" önceki ödevlerdeki gibi çekebilirsin datayı:"""
data = f1['data'][()].astype('float64')
f1.close()

#FIXME
""" object silinebilir"""


class AE(object):
    def initialize(self, Lin, Lhid):

        Lout = Lin

        re = np.sqrt(6) / np.sqrt(Lin + Lhid)
        W1 = np.random.uniform(-re, re, size=(Lin, Lhid))
        b1 = np.random.uniform(-re, re, size=(1, Lhid))

        ren = np.sqrt(6) / np.sqrt(Lhid + Lout)
        W2 = W1.T
        b2 = np.random.uniform(-ren, ren, size=(1, Lout))

        We = (W1, W2, b1, b2)
        momWe = (0, 0, 0, 0)

        return We, momWe

    def workout(self, data, params, lrate=0.1, mom=0.9, epoch=10, batch=None):
        Jl = []
        if batch is None:
            batch = data.shape[0]

        Lin = params["Lin"]
        Lhid = params["Lhid"]
        We, momWe = self.initialize(Lin, Lhid)

        it = int(data.shape[0] / batch)
        for i in range(epoch):
            times = time.time()

            Jt = 0
            go = 0
            stop = batch

            po = np.random.permutation(data.shape[0])
            data = data[po]

            momWe = (0, 0, 0, 0)

            for j in range(it):
                batching = data[go:stop]

                J, Jgrad, co = self.aeCost(We, batching, params)
                We, momWe = self.solver(Jgrad, co, We, momWe, lrate, mom)

                Jt = Jt + J
                go = stop
                stop = stop + batch

            timel = (epoch - (i + 1)) * (time.time() - times)
            if timel < 60:
                timel = round(timel)
                timen = "sec(s)"
            else:
                timel = round(timel / 60)
                timen = "min(s)"

            Jt = Jt / it

            print("Loss: {:.2f} [Epoch {} of {}, ETA: {} {}]".format(Jt, i + 1, epoch, timel, timen))
            Jl.append(Jt)

        print("\n")

        return We, Jl

    def aeCost(self, We, data, params):

        nar = data.shape[0]

        W1, W2, b1, b2 = We

        rho = params["rho"]
        beta = params["beta"]
        lamda = params["lamda"]
        Lin = params["Lin"]
        Lhid = params["Lhid"]

        u = data @ W1 + b1
        h, hder = self.sigmoid(u)

        v = h @ W2 + b2
        o, oder = self.sigmoid(v)

        rhobe = h.mean(axis=0, keepdims=True)

        loss = (0.5 * (np.linalg.norm(data - o, axis=1) ** 2).sum()) / nar
        tako = 0.5 * lamda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        kel = rho * np.log(rho / rhobe) + (1 - rho) * np.log((1 - rho) / (1 - rhobe))
        kel = beta * kel.sum()

        J = loss + tako + kel

        derloss = (o - data) / nar
        dertako2 = lamda * W2
        dertako1 = lamda * W1
        derkel = beta * ((1 - rho) / (1 - rhobe) - rho / rhobe) / nar

        co = (data, h, hder, oder)
        Jgrad = (derloss, dertako2, dertako1, derkel)

        return J, Jgrad, co

    def solver(self, Jgrad, co, We, momWe, lrate, mom):

        W1, W2, b1, b2 = We
        # FIXME
        """ tek sırada yazabilirsin hepsini = 0 diye: """
        derW1 = 0
        derW2 = 0
        derb1 = 0
        derb2 = 0

        data, h, hder, oder = co
        derloss, dertako2, dertako1, derkel = Jgrad

        chan = derloss * oder

        derW2 = h.T @ chan + dertako2
        derb2 = chan.sum(axis=0, keepdims=True)

        chan = hder * (chan @ W2.T + derkel)

        derW1 = data.T @ chan + dertako1
        derb1 = chan.sum(axis=0, keepdims=True)

        derW2 = (derW1.T + derW2) / 2
        derW1 = derW2.T

        derWe = (derW1, derW2, derb1, derb2)

        We, momWe = self.modify(We, momWe, derWe, lrate, mom)

        return We, momWe

    def modify(self, We, momWe, derWe, lrate, mom):

        W1, W2, b1, b2 = We
        derW1, derW2, derb1, derb2 = derWe
        momW1, momW2, momb1, momb2 = momWe

        momW1 = lrate * derW1 + mom * momW1
        momW2 = lrate * derW2 + mom * momW2
        momb1 = lrate * derb1 + mom * momb1
        momb2 = lrate * derb2 + mom * momb2

        W1 = W1 - momW1
        W2 = W2 - momW2
        b1 = b1 - momb1
        b2 = b2 - momb2
        assert (W1 == W2.T).all()
        We = (W1, W2, b1, b2)
        momWe = (momW1, momW2, momb1, momb2)

        return We, momWe

    def guess(self, data, We):

        W1, W2, b1, b2 = We

        u = data @ W1 + b1
        h = self.sigmoid(u)[0]
        v = h @ W2 + b2
        o = self.sigmoid(v)[0]

        return o

    def sigmoid(self, x):
        """
        Sigmoid function
        @param X: input
        @return: output and derivative
        """
        k = 1 / (1 + np.exp(-x))
        l = k * (1 - k)
        return k, l


def norma(x):
    """
    Normalizes given input
    @param X: input
    @return: normalized X
    """

    return (x - x.min())/(x.max() - x.min())


def plot(w, tit, d1, d2):
    """
    A function which plots the weights for Q1
    @param W: Weight
    @param name: filename
    @param dim1: width
    @param dim2: height
    """

    fig, ax = plt.subplots(d2, d1, figsize=(d1, d2), dpi=320, facecolor='w', edgecolor='k')
    c = 0
    for a in range(d2):
        for b in range(d1):
            ax[a, b].imshow(w[c], cmap='gray')
            ax[a, b].axis("off")
            c = c + 1

    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
    fig.savefig(tit + ".png")
    plt.close(fig)

# convert to grayscale using the luminosity model
datanew = 0.2126 * data[:, 0] + 0.7152 * data[:, 1] + 0.0722 * data[:, 2]

# normalize
assert datanew.shape[1] == datanew.shape[2]
dimension = datanew.shape[1]
datanew = np.reshape(datanew, (datanew.shape[0], dimension ** 2))  # flatten

datanew = datanew - datanew.mean(axis=1, keepdims=True)  # differentiate per image
std = np.std(datanew)  # find std
datanew = np.clip(datanew, - 3 * std, 3 * std)  # clip -+3 std
datanew = norma(datanew)  # normalize to 0 - 1

datanew = 0.1 + datanew * 0.8  # map to 0.1 - 0.9
trd = datanew

# plot 200 random images
datanew = np.reshape(datanew, (datanew.shape[0], dimension, dimension))  # reshape for imshow
data = data.transpose((0, 2, 3, 1))
fig1, ax1 = plt.subplots(10, 20, figsize=(20, 10))
fig2, ax2 = plt.subplots(10, 20, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')

for a in range(10):
    for b in range(20):
        c = np.random.randint(0, data.shape[0])

        ax1[a, b].imshow(data[c].astype('float'))
        ax1[a, b].axis("off")

        ax2[a, b].imshow(datanew[c], cmap='gray')
        ax2[a, b].axis("off")

fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
fig1.savefig("q1a_rgb.png")
fig2.savefig("q1a_gray.2.png")
plt.close("all")

lrate = 0.075
mom = 0.85
epoch = 200
batch = 32
rho = 0.025
beta = 2
lamda = 5e-4
Lin = trd.shape[1]
Lhid = 64

params = {"rho": rho, "beta": beta, "lamda": lamda, "Lin": Lin, "Lhid": Lhid}
autoencoder = AE()
wilk = autoencoder.workout(trd, params, lrate, mom, epoch, batch)[0]
wson = norma(wilk[0]).T
wson = wson.reshape((W.shape[0], dimension, dimension))

name = "rho={:.2f}, beta={:.2f}, lrate={:.2f}, momentum={:.2f}, lambda={}, batch={}, Lhid={}".format(rho, beta, lrate, mom, lamda, batch, Lhid)
w_dimension = int(np.sqrt(wson.shape[0]))
plot(wson, tit + "weights", w_dimension, w_dimension)