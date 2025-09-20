"""
DON.py

Deep Operator Network (DeepONet) and related neural architectures for operator learning.
Includes generic feedforward networks, standard DeepONet, and recurrent DeepONet implementations.

Author: Jari Beysen
"""

# Imports
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class NeuralNet(nn.Module):
    """
    Generic feedforward neural network.
    """
    def __init__(self, indim, layers, outdim, acfuncs, device=None, norm=(0, 1)):
        super().__init__()
        self.device = device
        self.dtype = torch.float32
        self.indim = indim
        self.layers = layers
        self.outdim = outdim
        self.acfuncs = acfuncs
        self.norm = norm

        # Build layers
        mods = [nn.Linear(indim, layers[0], bias=True)]
        mods += [nn.Linear(layers[i], layers[i + 1], bias=True) for i in range(len(layers) - 1)]
        mods += [nn.Linear(layers[-1], outdim, bias=True)]
        self.mods = nn.ModuleList(mods)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        """
        for layer in self.mods:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def scale(self, x):
        """
        Scale input using normalization parameters.
        """
        return (x - self.norm[0]) / self.norm[1]

    def forward(self, x):
        """
        Forward pass through the network.
        """
        for i, layer in enumerate(self.mods):
            x = self.acfuncs[i](layer(x))
        return x

    def predict(self, x):
        """
        Predict output for given input.
        """
        xt = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.forward(xt).detach().cpu().numpy()


class DeepONet(nn.Module):
    """
    Standard Deep Operator Network (DeepONet).
    """
    def __init__(self, branch, trunk, psize, outdim, loss_func=None, device=None):
        super().__init__()
        self.device = device
        self.loss_func = loss_func or self.default_loss
        self.dtype = torch.float32
        self.branch = branch.to(device)
        self.trunk = trunk.to(device)
        self.psize = psize
        self.outdim = outdim
        self.bias = nn.Parameter(torch.zeros(1, outdim), requires_grad=True)

    def readout(self, b, t):
        """
        Readout operation combining branch and trunk outputs.
        """
        b = b.view(b.size(0), self.psize, self.outdim)
        t = t.view(t.size(0), self.psize, self.outdim)
        return (b * t).sum(1) + self.bias

    def forward(self, x, u):
        """
        Forward pass for DeepONet.
        """
        b = self.branch(u.to(self.device))
        t = self.trunk(x.to(self.device))
        return self.readout(b, t)

    def predict(self, x, u):
        """
        Predict output for given input and function.
        """
        xt = torch.tensor(x, dtype=self.dtype, device=self.device)
        ut = torch.tensor(u, dtype=self.dtype, device=self.device)
        return self.forward(xt, ut).detach().cpu().numpy()

    def default_loss(self, y, yp):
        """
        Normalized mean squared error loss.
        """
        y, yp = y.to(self.device), yp.to(self.device)
        nmse = torch.mean((y - yp) ** 2) / (torch.std(y) ** 2)
        return nmse

    def loss(self, y, yp):
        """
        Compute loss.
        """
        return self.loss_func(y, yp)


class RecurrentDeepONet(nn.Module):
    """
    Recurrent Deep Operator Network for sequence prediction.
    """
    def __init__(self, branch, trunk, psize, outdim, loss_func=None, device=None):
        super().__init__()
        self.device = device
        self.loss_func = loss_func or self.default_loss
        self.dtype = torch.float32
        self.branch = branch.to(device)
        self.trunk = trunk.to(device)
        self.psize = psize
        self.outdim = outdim
        self.bias = nn.Parameter(torch.zeros(1, outdim), requires_grad=True)

    def readout(self, b, t):
        """
        Readout operation combining branch and trunk outputs.
        """
        b = b.view(b.size(0), self.psize, self.outdim)
        t = t.view(t.size(0), self.psize, self.outdim)
        return (b * t).sum(1) + self.bias

    def forward(self, y, u):
        """
        Forward pass for sequence prediction.
        """
        y, u = y.to(self.device), u.to(self.device)
        yf = y.reshape(-1, y.shape[2])
        uf = u.reshape(-1, u.shape[2])
        b = self.branch(uf)
        t = self.trunk(yf)
        return self.readout(b, t).reshape(y.shape[0], y.shape[1], -1)

    def predict(self, y, u):
        """
        Predict output for given input sequences.
        """
        y = torch.tensor(y, dtype=self.dtype, device=self.device)
        u = torch.tensor(u, dtype=self.dtype, device=self.device)
        return self.forward(y, u).detach().cpu().numpy()

    def predict_n(self, y, u, n, printt=False):
        """
        Predict n steps ahead in sequence.
        """
        yp = torch.zeros(y.shape[0], n, y.shape[2], device=self.device)
        yp[:, :y.shape[1]] = y

        start_time = time.monotonic()
        for i in range(y.shape[1], n):
            yp[:, i:i + 1] = self.forward(yp[:, i - 1:i], u[:, i - 1:i])
        elapsed_time = time.monotonic() - start_time

        if printt:
            print(f"Prediction completed in {elapsed_time:.6f} seconds.")

        return yp.cpu().detach().numpy()

    def default_loss(self, y, yp):
        """
        Normalized mean squared error loss.
        """
        y, yp = y.to(self.device), yp.to(self.device)
        nmse = torch.mean((y - yp) ** 2) / (torch.std(y) ** 2)
        return nmse

    def loss(self, y, yp):
        """
        Compute loss.
        """
        return self.loss_func(y, yp)
