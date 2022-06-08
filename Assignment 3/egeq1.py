# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:25:21 2021

@author: User
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sn
import time
#import sys


########################################
#  QUESTION 1
########################################


class Q1AutoEncoder(object):
    """
    Autoencoder class for Question 1
    """


    def init_params(self, Lin, Lhid):
        """
        A function which initializes the weights following the assignment requirements
        @param Lin: The input layer size, 256
        @param Lhid: The hidden layer size
        @return: The initialized weights and their corresponding momentum values
        """

        Lout = Lin

        r = np.sqrt(6 / (Lin + Lhid))
        W1 = np.random.uniform(-r, r, size=(Lin, Lhid))
        b1 = np.random.uniform(-r, r, size=(1, Lhid))

        r = np.sqrt(6 / (Lhid + Lout))
        W2 = W1.T
        b2 = np.random.uniform(-r, r, size=(1, Lout))

        We = (W1, W2, b1, b2)
        mWe = (0, 0, 0, 0)

        return We, mWe


    def train(self, data, params, eta=0.1, alpha=0.9, epoch=10, batch_size=None):
        """
        The training function. Runs epochs and trains the given data. For this question this is
        used for the autoencoder.
        @param data: the training data
        @param params: the required parameters, given in the assignment
        @param eta: learning ratet
        @param alpha: momentum multiplier
        @param epoch: the epoch number for training
        @param batch_size: batch size for SGD
        @return: the weights (exttracted features) and the loss
        """

        J_list = []
        if batch_size is None:
            batch_size = data.shape[0]

        Lin = params["Lin"]
        Lhid = params["Lhid"]
        We, mWe = self.init_params(Lin,  Lhid)

        iter_per_epoch = int(data.shape[0] / batch_size)

        for i in range(epoch):

            time_start = time.time()

            J_total = 0

            start = 0
            end = batch_size

            p = np.random.permutation(data.shape[0])
            data = data[p]

            mWe = (0, 0, 0, 0)

            for j in range(iter_per_epoch):

                batchData = data[start:end]

                J, Jgrad, cache = self.aeCost(We, batchData, params)
                We, mWe = self.solver(Jgrad, cache, We, mWe, eta, alpha)

                J_total += J
                start = end
                end += batch_size

            time_remain = (epoch - i - 1) * (time.time() - time_start)
            if time_remain < 60:
                time_remain = round(time_remain)
                time_label = "second(s)"
            else:
                time_remain = round(time_remain / 60)
                time_label = "minute(s)"

            J_total = J_total/iter_per_epoch

            print("Loss: {:.2f} [Epoch {} of {}, ETA: {} {}]".format(J_total, i+1, epoch, time_remain, time_label))
            J_list.append(J_total)

        print("\n")

        return We, J_list


    def aeCost(self, We, data, params):
        """
        This function finds the first error gradients and does forward pass
        @param We: Weights
        @param data: training data, this comes in as the batch data
        @param params: the parameters
        @return: returns the error gradients and derivative and other variables via cahce
        """

        N = data.shape[0]

        W1, W2, b1, b2 = We

        rho = params["rho"]
        beta = params["beta"]
        lmb = params["lmb"]
        Lin = params["Lin"]
        Lhid = params["Lhid"]

        u = data @ W1 + b1
        h, h_drv = self.sigmoid(u)  # N x Lhid
        # h, h_drv = self.tanh(u)  # N x Lhid
        v = h @ W2 + b2
        o, o_drv = self.sigmoid(v)  # N x Lin
        # o, o_drv = self.tanh(v)  # N x Lin

        rho_b = h.mean(axis=0, keepdims=True)  # 1 x Lhid



        loss = 0.5/N * (np.linalg.norm(data - o, axis=1) ** 2).sum()
        tykhonov = 0.5 * lmb * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        KL = rho * np.log(rho/rho_b) + (1 - rho) * np.log((1 - rho)/(1 - rho_b))
        KL = beta * KL.sum()

        J = loss + tykhonov + KL
        #FIXME
        dloss = -(data - o)/N
        dtyk2 = lmb * W2
        dtyk1 = lmb * W1
        dKL = beta * (- rho/rho_b + (1-rho)/(1 - rho_b))/N

        cache = (data, h, h_drv, o_drv)
        Jgrad = (dloss, dtyk2, dtyk1, dKL)


        return J, Jgrad, cache


    def solver(self, Jgrad, cache, We, mWe, eta, alpha):
        """
        Finds weight updates and updates them
        @param Jgrad: Error gradients
        @param cache: cache of variables coming from aeCost, needed for updates
        @param We: weights
        @param mWe: corresponding momentum terms
        @param eta: learning ratte
        @param alpha: momentum multiplier
        @return:
        """

        W1, W2, b1, b2 = We

        dW1 = 0
        dW2 = 0
        db1 = 0
        db2 = 0

        data, h, h_drv, o_drv = cache
        dloss, dtyk2, dtyk1, dKL = Jgrad

        delta = dloss * o_drv


        dW2 = h.T @ delta + dtyk2
        db2 = delta.sum(axis=0, keepdims=True)

        delta = h_drv * (delta @ W2.T + dKL)

        dW1 = data.T @ delta + dtyk1
        db1 = delta.sum(axis=0, keepdims=True)

        # FIXME
        dW2 = (dW1.T + dW2)/2
        dW1 = dW2.T

        dWe = (dW1, dW2, db1, db2)

        We, mWe = self.update(We, mWe, dWe, eta, alpha)

        return We, mWe


    def update(self, We, mWe, dWe, eta, alpha):
        """
        Updates weights
        @param We: weights
        @param mWe:momentum terms
        @param dWe: updates
        @param eta: learning rate
        @param alpha: mometum multiplier
        @return:updated weights and momentum terms
        """

        W1, W2, b1, b2 = We
        dW1, dW2, db1, db2 = dWe
        mW1, mW2, mb1, mb2 = mWe

        mW1 = eta * dW1 + alpha * mW1
        mW2 = eta * dW2 + alpha * mW2
        mb1 = eta * db1 + alpha * mb1
        mb2 = eta * db2 + alpha * mb2

        W1 -= mW1
        W2 -= mW2
        b1 -= mb1
        b2 -= mb2
        assert (W1 == W2.T).all()
        We = (W1, W2, b1, b2)
        mWe = (mW1, mW2, mb1, mb2)

        return We, mWe


    def predict(self, data, We):
        """
        Predicts the output, aka does forward pass
        @param data: input data
        @param We: weights
        @return: output
        """

        W1, W2, b1, b2 = We

        u = data @ W1 + b1
        h = self.sigmoid(u)[0]
        v = h @ W2 + b2
        o = self.sigmoid(v)[0]
        return o


    def sigmoid(self, X):
        """
        Sigmoid function
        @param X: input
        @return: output and derivative
        """
        a = 1 / (1 + np.exp(-X))
        d = a * (1 - a)
        return a, d


def normalize(X):
    """
    Normalizes given input
    @param X: input
    @return: normalized X
    """

    return (X - X.min())/(X.max() - X.min())


def plot(W, name, dim1, dim2):
    """
    A function which plots the weights for Q1
    @param W: Weight
    @param name: filename
    @param dim1: width
    @param dim2: height
    """

    fig, ax = plt.subplots(dim2, dim1, figsize=(dim1, dim2), dpi=320, facecolor='w', edgecolor='k')
    k = 0
    for i in range(dim2):
        for j in range(dim1):
            ax[i, j].imshow(W[k], cmap='gray')
            ax[i, j].axis("off")
            k += 1

    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
    fig.savefig(name + ".png")
    plt.close(fig)


filename = "assign3_data1.h5"
h5 = h5py.File(filename, 'r')
data = h5['data'][()].astype('float64')
h5.close()

# convert to grayscale using the luminosity model
data_n = 0.2126 * data[:, 0] + 0.7152 * data[:, 1] + 0.0722 * data[:, 2]

# normalize
assert data_n.shape[1] == data_n.shape[2]
dim = data_n.shape[1]
data_n = np.reshape(data_n, (data_n.shape[0], dim ** 2))  # flatten

data_n = data_n - data_n.mean(axis=1, keepdims=True)  # differentiate per image
std = np.std(data_n)  # find std
data_n = np.clip(data_n, - 3 * std, 3 * std)  # clip -+3 std
data_n = normalize(data_n)  # normalize to 0 - 1

data_n = 0.1 + data_n * 0.8  # map to 0.1 - 0.9
trainData = data_n

# plot 200 random images
data_n = np.reshape(data_n, (data_n.shape[0], dim, dim))  # reshape for imshow
data = data.transpose((0, 2, 3, 1))
fig1, ax1 = plt.subplots(10, 20, figsize=(20, 10))
fig2, ax2 = plt.subplots(10, 20, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')

for i in range(10):
    for j in range(20):
        k = np.random.randint(0, data.shape[0])

        ax1[i, j].imshow(data[k].astype('float'))
        ax1[i, j].axis("off")

        ax2[i, j].imshow(data_n[k], cmap='gray')
        ax2[i, j].axis("off")

fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
fig1.savefig("q1a_rgb.png")
fig2.savefig("q1a_gray.2.png")
plt.close("all")

eta = 0.075
alpha = 0.85
epoch = 200
batch_size = 32
rho = 0.025
beta = 2
lmb = 5e-4
Lin = trainData.shape[1]
Lhid = 64

params = {"rho": rho, "beta": beta, "lmb": lmb, "Lin": Lin, "Lhid": Lhid}
ae = Q1AutoEncoder()
w = ae.train(trainData, params, eta, alpha, epoch, batch_size)[0]
W = normalize(w[0]).T
W = W.reshape((W.shape[0], dim, dim))

name = "rho={:.2f}, beta={:.2f}, eta={:.2f}, alpha={:.2f}, lambda={}, batch={}, Lhid={}".format(rho, beta, eta, alpha, lmb, batch_size, Lhid)
wdim = int(np.sqrt(W.shape[0]))
plot(W, name + "weights", wdim, wdim)