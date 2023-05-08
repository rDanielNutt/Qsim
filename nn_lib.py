import numpy as np
import cupy as cp
import math
import progressbar
from npy_append_array import NpyAppendArray as npy

import os
from cupyx.scipy.signal import fftconvolve


def onehot(labels):
    enc = np.zeros([labels.shape[0], np.max(labels)+1])
    enc[range(labels.size), labels.reshape([-1])] = 1
    return enc

def sigmoid(x):
    e = cp.exp(x)
    a = 1 / (1 + (1/e))
    da = e / cp.square(e + 1)
    return a, da

def tanh(x):
    a = cp.tanh(x)
    da = 1 / cp.square(cp.cosh(x))
    return a, da

def softmax(x):
    e = cp.exp(x)
    esum = cp.sum(e, axis=1, keepdims=True)
    a = e / esum
    da = ((esum - e) * e) / cp.square(esum)
    return a, da

def relu(x):
    da = cp.where(x > 0, 1, 0)
    a = x * da
    return a, da

def comp_norm(x):
    asum = cp.sqrt(cp.nansum(cp.square(cp.abs(x)), axis=(1,2,3), keepdims=True) * 1e-4)
    a = x / asum
    da = ((1+1j) / asum) - (((1e-4 + 1e-4j) * cp.square(x)) / cp.square(asum))
    return a, da

def none(x):
    return x, 1

activation_funcs = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softmax': softmax,
    'relu': relu,
    'norm': comp_norm,
    'none': none,
}


def mse(true, pred):
    loss = cp.mean(cp.abs(pred - true))
    dloss = pred - true
    return loss, dloss

def cat_crossentropy(true, pred):
    loss = cp.mean(-(true * cp.log(pred)) + ((1 - true) * cp.log(1 - pred)))
    dloss = - (pred - true) / (cp.square(pred) - pred)
    return loss, dloss

def wave_loss(true, pred):
    loss = cp.mean(cp.abs(true) * cp.abs(pred - true))
    dloss = cp.abs(true) * (pred - true)
    return loss, dloss

loss_funcs = {
    'mse': mse,
    'catcrossentropy': cat_crossentropy,
    'wave': wave_loss,
}


class BaseConv2D:
    
    def __init__(self, n_kernels, kernel_size, padding='valid', activation='sigmoid', input_size=(), lr=1e-2, dtype=cp.float64, **kwargs):
        
        self.pad_type = padding
        self.activate = activation_funcs[activation]
        self.lr = lr
        self.n_kernels = n_kernels

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size, 1)
        elif isinstance(input_size, tuple):
            self.input_size = (*input_size, *([1]*(3-len(input_size))))

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, self.input_size[2])
        elif len(kernel_size) == 1:
            self.kernel_size = (kernel_size[0], 1, self.input_size[2])
        elif len(kernel_size) == 2:
            self.kernel_size = (*kernel_size, self.input_size[2])

        self.weights = cp.random.rand(1, *self.kernel_size, self.n_kernels).astype(dtype) / 100
        self.bias = cp.zeros([1, 1, 1, self.n_kernels]).astype(dtype)

        if self.pad_type == 'same':
            pre0 = math.ceil((self.kernel_size[0] - 1) / 2)
            post0 = math.floor((self.kernel_size[0] - 1) / 2)
            pre1 = math.ceil((self.kernel_size[1] - 1) / 2)
            post1 = math.floor((self.kernel_size[1] - 1) / 2)
        elif self.pad_type == 'full':
            pre0 = post0 = self.kernel_size[0] - 1
            pre1 = post1 = self.kernel_size[1] - 1
        else:
            pre0 = post0 = pre1 = post1 = 0

        self.pad = ((0,0), (pre0, post0), (pre1, post1), (0,0))
        self.output_size = (self.input_size[0] - self.kernel_size[0] + pre0+post0+1, self.input_size[1] - self.kernel_size[1] + pre1+post1+1, self.n_kernels)
        self.dact = cp.empty([])

    def forward(self, x):
        self.x = cp.pad(x, self.pad, 'constant', constant_values=0)[:,:,:,:,cp.newaxis]
        act, self.dact = self.activate(cp.sum(fftconvolve(self.x, self.weights, mode='valid', axes=(1, 2)), axis=3) + self.bias)
        return act
    
    def backprop(self, grad):
        grad = (self.dact * grad)[:,:,:,cp.newaxis]
        grad = cp.where(cp.isfinite(grad), grad, 0)
        wgrad = cp.sum(fftconvolve(cp.flip(self.x, axis=(1,2)), grad, mode='valid', axes=(1, 2)), axis=0, keepdims=True)
        bgrad = cp.sum(grad, axis=(0, 1, 2))

        p0 = self.pad[1]
        p1 = self.pad[2]
        grad = cp.sum(fftconvolve(grad, cp.flip(self.weights, axis=(1, 2)), mode='full', axes=(1, 2)), axis=4)

        # self.bias -= (self.lr * bgrad)
        self.weights -= (self.lr * wgrad)

        return grad[:, p0[0]:grad.shape[1]-p0[1], p1[0]:grad.shape[2]-p1[1], :]


class BaseFlatten:
    def __init__(self, input_size, **kwargs):
        self.input_size = input_size
        self.output_size = np.prod(input_size)
    
    def forward(self, x):
        return x.reshape([-1, self.output_size])

    def backprop(self, grad):
        return grad.reshape([-1, *self.input_size])


class BaseDense:
    def __init__(self, n_neurons, input_size, lr, activation, dtype):
        
        self.weights = cp.random.rand(input_size, n_neurons).astype(dtype) / 10
        self.bias = cp.zeros([1, n_neurons]).astype(dtype)
        
        self.lr = lr
        self.activate = activation_funcs[activation]
        self.output_size = n_neurons
        self.x = cp.empty([])
        self.dact = cp.empty([])

    def forward(self, x):
        self.x = x.copy()
        act, self.dact = self.activate(cp.dot(self.x, self.weights) + self.bias)
        return act
    
    def backprop(self, grad):
        grad = self.dact * grad
        grad = cp.where(cp.isfinite(grad), grad, 0)
        wgrad = cp.dot(self.x.T, grad)
        bgrad = cp.sum(grad, axis=0, keepdims=True)

        grad = cp.dot(grad, self.weights.T)

        self.weights -= (self.lr * wgrad)
        self.bias -= (self.lr * bgrad)

        return grad

def Flatten(input_size=()):
    class Flatten(BaseFlatten):
        def __init__(self, input_size=input_size, **kwargs):
            super().__init__(input_size, **kwargs)
    
    return Flatten

def Dense(n_neurons=0, input_size=(), lr=1e-2, activation='sigmoid'):
    class Dense(BaseDense):
        def __init__(self, n_neurons=n_neurons, input_size=input_size, lr=lr, activation=activation):
            super().__init__(n_neurons, input_size, lr, activation)
    
    return Dense

def Conv2D(n_kernels=0, kernel_size=0, padding='valid', activation='sigmoid', input_size=()):
    class Conv2D(BaseConv2D):
        def __init__(self, n_kernels=n_kernels, kernel_size=kernel_size, padding=padding, activation=activation, input_size=input_size, **kwargs):
            super().__init__(n_kernels, kernel_size, padding, activation, input_size, **kwargs)

    return Conv2D

layer_funcs = {
    'Flatten': Flatten,
    'Dense': Dense,
    'Conv2D': Conv2D,
}

class Sequential:
    def __init__(self, layers=[], loss='mse', lr=1e-2, dtype=cp.float64):
        self.loss = loss_funcs[loss]
        self.lr = lr
        self.dtype = dtype

        if len(layers) > 0:
            self.layers = [layers[0](lr=self.lr, dtype=self.dtype)]
            output_size = self.layers[0].output_size

            for layer in layers[1:]:
                self.layers.append(layer(input_size=output_size, lr=self.lr, dtype=self.dtype))
                output_size = self.layers[-1].output_size
        else:
            self.layers = []

        self.history = []

    def save(self, path, name):
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(f'{path}{name}/'):
            os.mkdir(f'{path}{name}/')

        with npy(f'{path}{name}/history.npy', delete_if_exists=True) as h:
            h.append(np.array(self.history))

        for i, layer in enumerate(self.layers):
            l_type = str(type(layer).__name__)
            try:
                with npy(f'{path}{name}/{i}_{l_type}_weights.npy', delete_if_exists=True) as w:
                    w.append(layer.weights)
                with npy(f'{path}{name}/{i}_{l_type}_bias.npy', delete_if_exists=True) as b:
                    b.append(layer.bias)
            except AttributeError:
                continue

    def load_weights(self, path, name):
        for i, layer in enumerate(self.layers):
            l_type = str(type(layer).__name__)
            try:
                layer.weights = cp.load(f'{path}{name}/{i}_{l_type}_weights.npy')
                layer.bias = cp.load(f'{path}{name}/{i}_{l_type}_bias.npy')
            except AttributeError:
                continue

    def add(self, layer):
        if len(self.layers) > 0:
            output_size = self.layers[-1].output_size
            self.layers.append(layer(input_size=output_size, lr=self.lr, dtype=self.dtype))
        else:
            self.layers.append(layer(lr=self.lr, dtype=self.dtype))
    
    def set(self, loss=None, lr=None):
        if loss is not None:
            self.loss = loss_funcs[loss]
        if lr is not None:
            self.lr = lr

        for layer in self.layers:
            layer.lr = self.lr

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

    def train_batch(self, x, y, epochs=1):
        for epoch in range(1, epochs+1):
            pred = self.predict(x)
            loss, grad = self.loss(y, pred)

            if not cp.all(cp.isfinite(loss)):
                raise Exception('Training Error: Non-finite values present in loss') 
            if not cp.all(cp.isfinite(grad)):
                raise Exception('Training Error: Non-finite values present in grad')

            self.backprop(grad)
            self.history.append(loss.get())
    

    def train(self, x, y, batch_size=None, epochs=1, loss=None, lr=None):
        self.set(loss, lr)

        if batch_size is not None:
            batches = [range(i, i+batch_size) for i in range(0, y.shape[0]-batch_size, batch_size)]
            batches += [range(y.shape[0]-(y.shape[0] % batch_size), y.shape[0])]*(y.shape[0] % batch_size > 0)
        else:
            batches = [range(y.shape[0])]

        bar = progressbar.ProgressBar(maxval=len(batches), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch: >5}/{epochs: <3}:\n')

            bar.start()
            for i, batch in enumerate(batches):
                pred = self.predict(cp.array(x[batch]))
                
                loss, grad = self.loss(cp.array(y[batch]), pred)
                self.backprop(grad)

                bar.update(i+1)
            bar.finish()

            self.history.append(loss)
            print(f'loss[{loss:.3f}]\n')



