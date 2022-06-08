# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 15:53:45 2021

@author: User
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import time


class Q3NET(object):
    """
    Network class for Q3. Implements RNN, LSTM and GRU.
    """


    def __init__(self, size, qPart):

        self.qPart = qPart
        self.size = size
        self.layer_size = len(size) - 1
        self.mlp_layer_size = None
        self.mlp_params, self.first_layer_params, self.mlp_momentum, self.first_layer_momentum = None, None, None, None
        self.init_params()


    def init_params(self):
        """
        Initializes parameters for MLP layers and also thte first layer.
        Uses Xavier Distribution for initialization.
        """


        qPart = self.qPart
        size = self.size
        layer_size = self.layer_size

        W = []
        b = []
        for i in range(1, layer_size):
            # Xavier Uniform
            r = np.sqrt(6 / (size[i] + size[i + 1]))
            W.append(np.random.uniform(-r, r, size=(size[i], size[i + 1])))
            b.append(np.zeros((1, size[i + 1])))

        self.mlp_layer_size = len(W)
        params = {"W": W, "b": b}
        momentum = {"W": [0] * self.mlp_layer_size, "b": [0] * self.mlp_layer_size}
        self.mlp_params = params
        self.mlp_momentum = momentum

        N = size[0]
        H = size[1]
        Z = N + H

        if qPart == 1:
            r = np.sqrt(6 / (N + H))
            Wih = np.random.uniform(-r, r, size=(N, H))
            r = np.sqrt(6 / (H + H))
            Whh = np.random.uniform(-r, r, size=(H, H))
            b = np.zeros((1, H))

            params = {"Wih": Wih, "Whh": Whh, "b": b}

        if qPart == 2:
            r = np.sqrt(6 / (Z + H))

            Wf = np.random.uniform(-r, r, size=(Z, H))
            Wi = np.random.uniform(-r, r, size=(Z, H))
            Wc = np.random.uniform(-r, r, size=(Z, H))
            Wo = np.random.uniform(-r, r, size=(Z, H))

            bf = np.zeros((1, H))
            bi = np.zeros((1, H))
            bc = np.zeros((1, H))
            bo = np.zeros((1, H))

            params = {"Wf": Wf, "bf": bf,
                      "Wi": Wi, "bi": bi,
                      "Wc": Wc, "bc": bc,
                      "Wo": Wo, "bo": bo}

        if qPart == 3:
            rN = np.sqrt(6 / (N + H))
            rH = np.sqrt(6 / (H + H))

            Wz = np.random.uniform(-rN, rN, size=(N, H))
            Uz = np.random.uniform(-rH, rH, size=(H, H))
            bz = np.zeros((1, H))

            Wr = np.random.uniform(-rN, rN, size=(N, H))
            Ur = np.random.uniform(-rH, rH, size=(H, H))
            br = np.zeros((1, H))

            Wh = np.random.uniform(-rN, rN, size=(N, H))
            Uh = np.random.uniform(-rH, rH, size=(H, H))
            bh = np.zeros((1, H))

            params = {"Wz": Wz, "Uz": Uz, "bz": bz,
                      "Wr": Wr, "Ur": Ur, "br": br,
                      "Wh": Wh, "Uh": Uh, "bh": bh}


        momentum = dict.fromkeys(params.keys(), 0)
        self.first_layer_params = params
        self.first_layer_momentum = momentum


    def train(self, X, Y, eta, alpha, batch_size, epoch):
        """
        Training function. Calls forward and backward pass. Uses SGD and trains with mini-batch. Ya'know, the regular stuff.
        @param X: Training data
        @param Y: Training labels
        @param eta: learning rate
        @param alpha: momentum multiplier
        @param batch_size: self-explanatory
        @param epoch: self-explanatory
        @return Loss Metrics
        """

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []

        # create validation set
        val_size = int(X.shape[0] / 10)
        p = np.random.permutation(X.shape[0])
        valX = X[p][:val_size]
        valY = Y[p][:val_size]
        X = X[p][val_size:]
        Y = Y[p][val_size:]

        sample_size = X.shape[0]
        iter_per_epoch = int(sample_size / batch_size)

        for i in range(epoch):

            time_start = time.time()

            start = 0
            end = batch_size
            p = np.random.permutation(X.shape[0])
            X = X[p]
            Y = Y[p]

            for j in range(iter_per_epoch):

                batchX = X[start:end]
                batchY = Y[start:end]

                # forward
                pred, o, drv, h, h_drv, cache = self.forward_pass(batchX)

                # error gradient at last layer
                delta = pred
                delta[batchY == 1] -= 1
                delta = delta / batch_size

                # backward
                fl_grads, mlp_grads = self.backward_pass(batchX, o, drv, delta, h, h_drv, cache)

                # update
                self.update_params(eta, alpha, fl_grads, mlp_grads)

                start = end
                end += batch_size

            # epoch end

            # train loss
            pred = self.predict(X, acc=False)
            train_loss = self.cross_entropy(Y, pred)

            # train acc
            train_acc = self.predict(X, Y, acc=True)

            # val acc
            val_acc = self.predict(valX, valY, acc=True)

            # val loss
            pred = self.predict(valX, acc=False)
            val_loss = self.cross_entropy(valY, pred)

            # create time remaining
            time_remain = (epoch - i - 1) * (time.time() - time_start)

            if time_remain < 60:
                time_remain = round(time_remain)
                time_label = "second(s)"
            else:
                time_remain = round(time_remain/60)
                time_label = "minute(s)"

            print('Train Loss: %.2f, Val Loss: %.2f, Train Acc: %.2f, Val Acc: %.2f [Epoch: %d of %d, ETA: %d %s]'
                  % (train_loss, val_loss, train_acc, val_acc, i + 1, epoch, time_remain, time_label))

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)


            # stop if the cross entropy of validation set converged
            if i > 15:
                conv = val_loss_list[-16:-1]
                conv = sum(conv) / len(conv)

                limit = 0.02
                if (conv - limit) < val_loss < (conv + limit):
                    print("\nTraining stopped since validation C-E reached convergence.")
                    return {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list,
                            "train_acc_list": train_acc_list, "val_acc_list": val_acc_list}


        return {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list,
                "train_acc_list": train_acc_list, "val_acc_list": val_acc_list}


    def forward_pass(self, X):
        """
        Forward pass.
        @param X: Input data
        @return: Stuff needed to update the weights, such as derivatives and activations through the forward pass.
        """

        qPart = self.qPart
        mlp_p = self.mlp_params
        fl_p = self.first_layer_params

        o = []
        drv = []

        h = 0
        h_drv = 0
        cache = 0

        # first layer
        if qPart == 1:
            h, h_drv = self.forward_recurrent(X, fl_p)
            o.append(h[:, -1, :])
            drv.append(h_drv[:, -1, :])
        if qPart == 2:
            h, cache = self.forward_lstm(X, fl_p)
            o.append(h)
            drv.append(1)
        if qPart == 3:
            h, cache = self.forward_gru(X, fl_p)
            o.append(h)
            drv.append(1)

        # relu layers
        for i in range(self.mlp_layer_size - 1):
            activation, derivative = self.forward_perceptron(o[-1], mlp_p["W"][i], mlp_p["b"][i], "relu")
            o.append(activation)
            drv.append(derivative)

        # output layer
        pred = self.forward_perceptron(o[-1], mlp_p["W"][-1], mlp_p["b"][-1], "softmax")[0]

        return pred, o, drv, h, h_drv, cache


    def backward_pass(self, X, o, drv, delta, h=None, h_drv=None, cache=None):
        """
        The backward pass function which calls network layers to obtain the gradients.
        @param X: training data
        @param o: activations of mlp layers
        @param drv: derivatives of mlp layers
        @param delta: The first error gradient for the backward pass, this getts updated through the layers and time
        @param h: only required for recurrent first layer, the activations for all time samples
        @param h_drv: only required for recurrent first layer, the derivatives of activations for all time samples
        @param cache: needed parameters to find the first layer updates
        @return: first layer gradients, mlp gradients
        """
        qPart = self.qPart
        fl_p = self.first_layer_params
        mlp_p = self.mlp_params

        fl_grads = dict.fromkeys(fl_p.keys())
        mlp_grads = {"W": [0] * self.mlp_layer_size, "b": [0] * self.mlp_layer_size}

        # backpropagation until recurrent
        for i in reversed(range(self.mlp_layer_size)):
            mlp_grads["W"][i], mlp_grads["b"][i], delta = self.backward_perceptron(mlp_p["W"][i], o[i], drv[i], delta)

        # backpropagation through time
        if qPart == 1:
            fl_grads = self.backward_recurrent(X, h, h_drv, delta, fl_p)
        if qPart == 2:
            fl_grads = self.backward_lstm(cache, fl_p, delta)
        if qPart == 3:
            fl_grads = self.backward_gru(X, cache, fl_p, delta)

        return fl_grads, mlp_grads


    def update_params(self, eta, alpha, fl_grads, mlp_grads):
        """
        Updates parameters for first and mlp layers.
        @param eta: learning rate
        @param alpha: momentum multiplier
        @param fl_grads: first layer gradients
        @param mlp_grads: mlp gradients
        """

        # obtain
        fl_p = self.first_layer_params
        fl_m = self.first_layer_momentum
        mlp_p = self.mlp_params
        mlp_m = self.mlp_momentum

        # first layer
        for p in self.first_layer_params:
            fl_m[p] = eta * fl_grads[p] + alpha * fl_m[p]
            fl_p[p] -= fl_m[p]

        # mlp layers
        for i in range(self.mlp_layer_size):
            mlp_m["W"][i] = eta * mlp_grads["W"][i] + alpha * mlp_m["W"][i]
            mlp_m["b"][i] = eta * mlp_grads["b"][i] + alpha * mlp_m["b"][i]
            mlp_p["W"][i] -= mlp_m["W"][i]
            mlp_p["b"][i] -= mlp_m["b"][i]

        # update
        self.first_layer_params = fl_p
        self.first_layer_momentum = fl_m
        self.mlp_params = mlp_p
        self.mlp_momentum = mlp_m


    def forward_perceptron(self, X, W, b, a):
        """
        Finds the activation and derivative for MLP layers
        @param X: input data
        @param W: weight
        @param b: bias
        @param a: the fundtion type (relu, sigmoid, tanh, softmax)
        @return: the activation and its derivative
        """
        u = X @ W + b
        return self.activation(u, a)


    def backward_perceptron(self, W, o, drv, delta):
        """
        Finds the gradients of MLP layers
        @param W: weight
        @param o: past output, input of this layer
        @param drv: past output derivative, derivative of the input to this layer
        @param delta: the error gradient from the previous layer
        @return: gradients and the updated delta term
        """
        dW = o.T @ delta
        db = delta.sum(axis=0, keepdims=True)
        delta = drv * (delta @ W.T)
        return dW, db, delta


    def forward_recurrent(self, X, fl_p):
        """
        Forward pass for the recurrent layer. Not very different from MLP except for the fact
        that its done over 150 time samples.
        @param X: input data
        @param fl_p: first layer parameters
        @return: the activations and their derivatives needed for backward pass
        """

        N, T, D = X.shape
        H = self.size[1]

        Wih = fl_p["Wih"]
        Whh = fl_p["Whh"]
        b = fl_p["b"]

        h_prev = np.zeros((N, H))
        h = np.empty((N, T, H))
        h_drv = np.empty((N, T, H))

        for t in range(T):
            x = X[:, t, :]
            u = x @ Wih + h_prev @ Whh + b
            h[:, t, :], h_drv[:, t, :] = self.activation(u, "tanh")
            h_prev = h[:, t, :]

        return h, h_drv


    def backward_recurrent(self, X, h, h_drv, delta, fl_p):
        """
        Backwards pass for recurrent layer. This is very similar to the MLP backward propagation
        as it can be seen and the way delta is updated is also the same. BPTT is done to find the
        gradients, this means the gradients are summed up through 150 time steps.
        @param X: input data
        @param h: the 150 time sampled activations
        @param h_drv: their derivatives
        @param delta: the error gradient coming from the previous layer
        @param fl_p: first layer parameters
        @return: gradients
        """

        N, T, D = X.shape
        H = self.size[1]

        Whh = fl_p["Whh"]

        dWih = 0
        dWhh = 0
        db = 0

        for t in reversed(range(T)):
            x = X[:, t, :]

            if t > 0:
                h_prev = h[:, t - 1, :]
                h_prev_derv = h_drv[:, t - 1, :]
            else:
                h_prev = np.zeros((N, H))
                h_prev_derv = 0

            dWih += x.T @ delta
            dWhh += h_prev.T @ delta
            db += delta.sum(axis=0, keepdims=True)
            delta = h_prev_derv * (delta @ Whh)

        return {"Wih": dWih, "Whh": dWhh, "b": db}


    def forward_lstm(self, X, fl_p):
        """
        Forward pass of LSTM. It might look a bit confusing but its not so different from the
        mathematical equations of the gates.
        @param X: input data
        @param fl_p: first layer parameters
        @return: the final h value (this is needed for th error gradient calculations
        of the first MLP layer) and cache which conttains needed variables for this layers backward pass.
        """

        N, T, D = X.shape
        H = self.size[1]

        Wf, bf = fl_p["Wf"], fl_p["bf"]
        Wi, bi = fl_p["Wi"], fl_p["bi"]
        Wc, bc = fl_p["Wc"], fl_p["bc"]
        Wo, bo = fl_p["Wo"], fl_p["bo"]

        h_prev = np.zeros((N, H))
        c_prev = np.zeros((N, H))
        z = np.empty((N, T, D + H))
        c = np.empty((N, T, H))
        tanhc = np.empty((N, T, H))
        hf = 0
        hi = np.empty((N, T, H))
        hc = np.empty((N, T, H))
        ho = np.empty((N, T, H))
        tanhc_d = np.empty((N, T, H))
        hf_d = np.empty((N, T, H))
        hi_d = np.empty((N, T, H))
        hc_d = np.empty((N, T, H))
        ho_d = np.empty((N, T, H))

        for t in range(T):
            z[:, t, :] = np.column_stack((h_prev, X[:, t, :]))
            z_cur = z[:, t, :]

            hf, hf_d[:, t, :] = self.activation(z_cur @ Wf + bf, "sigmoid")
            hi[:, t, :], hi_d[:, t, :] = self.activation(z_cur @ Wi + bi, "sigmoid")
            hc[:, t, :], hc_d[:, t, :] = self.activation(z_cur @ Wc + bc, "tanh")
            ho[:, t, :], ho_d[:, t, :] = self.activation(z_cur @ Wo + bo, "sigmoid")

            c[:, t, :] = hf * c_prev + hi[:, t, :] * hc[:, t, :]
            tanhc[:, t, :], tanhc_d[:, t, :] = self.activation(c[:, t, :], "tanh")
            h_prev = ho[:, t, :] * tanhc[:, t, :]
            c_prev = c[:, t, :]

            cache = {"z": z,
                     "c": c,
                     "tanhc": (tanhc, tanhc_d),
                     "hf_d": hf_d,
                     "hi": (hi, hi_d),
                     "hc": (hc, hc_d),
                     "ho": (ho, ho_d)}

        return h_prev, cache


    def backward_lstm(self, cache, fl_p, delta):
        """
        Backward propagation for LSTM.
        @param cache: has the needed variables for gradients
        @param fl_p: first layer parameters
        @param delta: error gradient from upper layer
        @return: gradients
        """
        # unpack variables
        Wf = fl_p["Wf"]
        Wi = fl_p["Wi"]
        Wc = fl_p["Wc"]
        Wo = fl_p["Wo"]

        z = cache["z"]
        c = cache["c"]
        tanhc, tanhc_d = cache["tanhc"]
        hf_d = cache["hf_d"]
        hi, hi_d = cache["hi"]
        hc, hc_d = cache["hc"]
        ho, ho_d = cache["ho"]

        H = self.size[1]
        T = z.shape[1]

        # initialize gradients to zero
        dWf = 0
        dWi = 0
        dWc = 0
        dWo = 0
        dbf = 0
        dbi = 0
        dbc = 0
        dbo = 0

        # BPTT starts
        for t in reversed(range(T)):

            z_cur = z[:, t, :]

            # if t = 0, c = 0
            if t > 0:
                c_prev = c[:, t - 1, :]
            else:
                c_prev = 0

            # firs find all 4 "gate gradients"
            # finding these first reduces clutter.
            dc = delta * ho[:, t, :] * tanhc_d[:, t, :]
            dhf = dc * c_prev * hf_d[:, t, :]
            dhi = dc * hc[:, t, :] * hi_d[:, t, :]
            dhc = dc * hi[:, t, :] * hc_d[:, t, :]
            dho = delta * tanhc[:, t, :] * ho_d[:, t, :]

            # add to all weights their respective values at that time
            dWf += z_cur.T @ dhf
            dbf += dhf.sum(axis=0, keepdims=True)

            dWi += z_cur.T @ dhi
            dbi += dhi.sum(axis=0, keepdims=True)

            dWc += z_cur.T @ dhc
            dbc += dhc.sum(axis=0, keepdims=True)

            dWo += z_cur.T @ dho
            dbo += dho.sum(axis=0, keepdims=True)

            # update the error gradient.
            # since weights are multiplied with a sttacked version of x, h(t-1)
            # we take only he weights of the previous layer by [:, :H]
            dxf = dhf @ Wf.T[:, :H]
            dxi = dhi @ Wi.T[:, :H]
            dxc = dhc @ Wc.T[:, :H]
            dxo = dho @ Wo.T[:, :H]

            delta = (dxf + dxi + dxc + dxo)  # we add them up

        grads = {"Wf": dWf, "bf": dbf,
                 "Wi": dWi, "bi": dbi,
                 "Wc": dWc, "bc": dbc,
                 "Wo": dWo, "bo": dbo}

        return grads


    def forward_gru(self, X, fl_p):
        """
        Forward pass for GRU. Again, tthe same as the respective mathematical equations listed out for GRU.
        @param X: input data
        @param fl_p: first layer parameters.
        @return: the final activation value and cache for backprop
        """

        Wz = fl_p["Wz"]
        Wr = fl_p["Wr"]
        Wh = fl_p["Wh"]

        Uz = fl_p["Uz"]
        Ur = fl_p["Ur"]
        Uh = fl_p["Uh"]

        bz = fl_p["bz"]
        br = fl_p["br"]
        bh = fl_p["bh"]

        N, T, D = X.shape
        H = self.size[1]

        h_prev = np.zeros((N, H))

        z = np.empty((N, T, H))
        z_d = np.empty((N, T, H))
        r = np.empty((N, T, H))
        r_d = np.empty((N, T, H))
        h_tilde = np.empty((N, T, H))
        h_tilde_d = np.empty((N, T, H))
        h = np.empty((N, T, H))

        for t in range(T):
            x = X[:, t, :]
            z[:, t, :], z_d[:, t, :] = self.activation(x @ Wz + h_prev @ Uz + bz, "sigmoid")
            r[:, t, :], r_d[:, t, :] = self.activation(x @ Wr + h_prev @ Ur + br, "sigmoid")
            h_tilde[:, t, :], h_tilde_d[:, t, :] = self.activation(x @ Wh + (r[:, t, :] * h_prev) @ Uh + bh, "tanh")
            h[:, t, :] = (1 - z[:, t, :]) * h_prev + z[:, t, :] * h_tilde[:, t, :]

            h_prev = h[:, t, :]

        cache = {"z": (z, z_d),
                 "r": (r, r_d),
                 "h_tilde": (h_tilde, h_tilde_d),
                 "h": h}

        return h_prev, cache


    def backward_gru(self, X, cache, fl_p, delta):
        """
        Backpropagation for GRU. Much easier than LSTM although the errror gradient updates di get a bit confusing.
        I did all of these by hand btw, using the chain rule and the derivative of multiplications.
        @param X: input data
        @param cache: variables needed for backprop from the forward pass
        @param fl_p: first layer parameters
        @param delta: error gradient from upper layer
        @return: gradients
        """

        # unpack
        Uz = fl_p["Uz"]
        Ur = fl_p["Ur"]
        Uh = fl_p["Uh"]

        z, z_d = cache["z"]
        r, r_d = cache["r"]
        h_tilde, h_tilde_d = cache["h_tilde"]
        h = cache["h"]

        H = self.size[1]
        N, T, D = X.shape

        # initialize to zero since we are doing BPTT
        dWz = 0
        dUz = 0
        dbz = 0
        dWr = 0
        dUr = 0
        dbr = 0
        dWh = 0
        dUh = 0
        dbh = 0

        for t in reversed(range(T)):
            x = X[:, t, :]

            # if t = 0 we want h(t-1) = 0
            if t > 0:
                h_prev = h[:, t - 1, :]
            else:
                h_prev = np.zeros((N, H))

            # similar to LSTM we find some intermediate values for each gate
            # dE/dz is named as dz for example, this is true for all naming
            dz = delta * (h_tilde[:, t, :] - h_prev) * z_d[:, t, :]
            dh_tilde = delta * z[:, t, :] * h_tilde_d[:, t, :]
            dr = (dh_tilde @ Uh.T) * h_prev * r_d[:, t, :]

            # add to the sum of gradients
            dWz += x.T @ dz
            dUz += h_prev.T @ dz
            dbz += dz.sum(axis=0, keepdims=True)

            dWr += x.T @ dr
            dUr += h_prev.T @ dr
            dbr += dr.sum(axis=0, keepdims=True)

            dWh += x.T @ dh_tilde
            dUh += h_prev.T @ dh_tilde
            dbh += dh_tilde.sum(axis=0, keepdims=True)

            # update delta, this step uses chain rule and derivative of multiplication, at the end it simplifies to
            #the sum of these three terms
            d1 = delta * (1 - z[:, t, :])
            d2 = dz @ Uz.T
            d3 = (dh_tilde  @ Uh.T) * (r[:, t, :] + h_prev * (r_d[:, t, :] @ Ur.T))

            delta = d1 + d2 + d3


        grads = {"Wz": dWz, "Uz": dUz, "bz": dbz,
                 "Wr": dWr, "Ur": dUr, "br": dbr,
                 "Wh": dWh, "Uh": dUh, "bh": dbh}

        return grads


    def cross_entropy(self, desired, output):
        """
        Cross entropy error
        @param desired: labels
        @param output: predictions
        @return: cross entropy error
        """
        return np.sum(- desired * np.log(output)) / desired.shape[0]


    def activation(self, X, a):
        """
        Function which outputs activation and derivative calculations
        @param X: input data
        @param a: activation type
        @return: activation, its derivative
        """

        if a == "tanh":
            activation = np.tanh(X)
            derivative = 1 - activation ** 2
            return activation, derivative

        if a == "sigmoid":
            activation = 1 / (1 + np.exp(-X))
            derivative = activation * (1 - activation)
            return activation, derivative

        if a == "relu":
            activation = X * (X > 0)
            derivative = 1 * (X > 0)
            return activation, derivative

        if a == "softmax":
            activation = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
            derivative = None
            return activation, derivative


    def predict(self, X, Y=None, acc=True, confusion = False):
        """
        The predict function, which does forward pass and then either returns the raw prediction or with labels
        returns the accuracy, or can also create the confusion matrix.
        @param X: Input matrix, these are not labels just data
        @param Y: Labels, ground truth, actual values
        @param acc: If true, computes the argmax version of prediction
        @param confusion: If true, gives the confusion matrix
        @return: prediction, accuracy or confusion matrix
        """
        pred = self.forward_pass(X)[0]

        if not acc:
            return pred

        pred = pred.argmax(axis=1)
        Y = Y.argmax(axis=1)

        if not confusion:
            return (pred == Y).mean() * 100 #accuracy

        K = len(np.unique(Y))  # Number of classes
        c = np.zeros((K, K))

        for i in range(len(Y)):
            c[Y[i]][pred[i]] += 1

        return c

filename = "assign3_data3.h5"
h5 = h5py.File(filename, 'r')
trX = h5['trX'][()].astype('float64')
tstX = h5['tstX'][()].astype('float64')
trY = h5['trY'][()].astype('float64')
tstY = h5['tstY'][()].astype('float64')
h5.close()

print("\n!!DISCLAIMER!!\nThe code runs slow in the training stage, hence the epoch number has ben selected as 10 for all layers.\n"
      "The plotted graphs in the report show the whole 50 epoch run. If you desire to see the whole run, just change the epoch variable to 50.\n\n")

alpha = 0.85
eta = 0.01
epoch = 10
batch_size = 32
size = [trX.shape[2], 128, 32, 16, 6]

print("Recurrent Layer\n")
nn = Q3NET(size, 1)
train_loss_list, val_loss_list, train_acc_list, val_acc_list = nn.train(trX, trY, eta, alpha, batch_size, epoch).values()
tst_acc = nn.predict(tstX, tstY, acc=True)

print("\nTest Accuracy: ", tst_acc, "\n\n")

fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')
fig.suptitle("RNN\nLearning Rate {} | Momentum = {} | Batch Size  = {} | Hidden Layers = {}\n"
             "Train Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Test Accuracy: {:.1f}\n "
             .format(eta, alpha, batch_size, size[2:-1], train_acc_list[-1], val_acc_list[-1], tst_acc), fontsize=13)

plt.subplot(2, 2, 1)
plt.plot(train_loss_list, "C2", label="Train Cross Entropy Loss")
plt.title("Train Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 2)
plt.plot(val_loss_list, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 3)
plt.plot(train_acc_list, "C2", label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.subplot(2, 2, 4)
plt.plot(val_acc_list, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("q3a.png", bbox_inches='tight')

train_confusion = nn.predict(trX, trY, acc=True, confusion=True)
test_confusion = nn.predict(tstX, tstY, acc=True, confusion=True)

plt.figure(figsize=(20, 10), dpi=160)

names = [1, 2, 3, 4, 5, 6]

plt.subplot(1, 2, 1)
sn.heatmap(train_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Train Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.subplot(1, 2, 2)
sn.heatmap(test_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Test Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.savefig("q3a_confusion.png", bbox_inches='tight')

##############################

alpha = 0.85
eta = 0.01
epoch = 10
batch_size = 32
size = [trX.shape[2], 128, 32, 16, 6]

print("\nLSTM Layer\n")

nn = Q3NET(size, 2)
train_loss_list, val_loss_list, train_acc_list, val_acc_list = nn.train(trX, trY, eta, alpha, batch_size, epoch).values()
tst_acc = nn.predict(tstX, tstY, acc=True)

print("\nTest Accuracy: ", tst_acc, "\n\n")

fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')
fig.suptitle("LSTM\nLearning Rate {} | Momentum = {} | Batch Size  = {} | Hidden Layers = {}\n"
             "Train Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Test Accuracy: {:.1f}\n "
             .format(eta, alpha, batch_size, size[2:-1], train_acc_list[-1], val_acc_list[-1], tst_acc), fontsize=13)

plt.subplot(2, 2, 1)
plt.plot(train_loss_list, "C2", label="Train Cross Entropy Loss")
plt.title("Train Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 2)
plt.plot(val_loss_list, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 3)
plt.plot(train_acc_list, "C2", label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.subplot(2, 2, 4)
plt.plot(val_acc_list, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("q3b.png", bbox_inches='tight')

train_confusion = nn.predict(trX, trY, acc=True, confusion=True)
test_confusion = nn.predict(tstX, tstY, acc=True, confusion=True)

plt.figure(figsize=(20, 10), dpi=160)

plt.subplot(1, 2, 1)
sn.heatmap(train_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Train Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.subplot(1, 2, 2)
sn.heatmap(test_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Test Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.savefig("q3b_confusion.png", bbox_inches='tight')

##############################

alpha = 0.85
eta = 0.01
epoch = 10
batch_size = 32
size = [trX.shape[2], 128, 32, 16, 6]

print("\nGRU Layer\n")

nn = Q3NET(size, 3)
train_loss_list, val_loss_list, train_acc_list, val_acc_list = nn.train(trX, trY, eta, alpha, batch_size, epoch).values()
tst_acc = nn.predict(tstX, tstY, acc=True)

print("\nTest Accuracy: ", tst_acc, "\n\n")

fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')
fig.suptitle("GRU\nLearning Rate {} | Momentum = {} | Batch Size  = {} | Hidden Layers = {}\n"
             "Train Accuracy: {:.1f} | Validation Accuracy: {:.1f} | Test Accuracy: {:.1f}\n "
             .format(eta, alpha, batch_size, size[2:-1], train_acc_list[-1], val_acc_list[-1], tst_acc), fontsize=13)

plt.subplot(2, 2, 1)
plt.plot(train_loss_list, "C2", label="Train Cross Entropy Loss")
plt.title("Train Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 2)
plt.plot(val_loss_list, "C3", label="Validation Cross Entropy Loss")
plt.title("Validation Cross Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(2, 2, 3)
plt.plot(train_acc_list, "C2", label="Train Accuracy")
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.subplot(2, 2, 4)
plt.plot(val_acc_list, "C3", label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("q3c.png", bbox_inches='tight')

train_confusion = nn.predict(trX, trY, acc=True, confusion=True)
test_confusion = nn.predict(tstX, tstY, acc=True, confusion=True)

plt.figure(figsize=(20, 10), dpi=160)

plt.subplot(1, 2, 1)
sn.heatmap(train_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Train Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.subplot(1, 2, 2)
sn.heatmap(test_confusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
plt.title("Test Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Prediction")
plt.savefig("q3c_confusion.png", bbox_inches='tight')

