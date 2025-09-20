"""
Extreme Learning DeepONet Models

Implements Extreme Learning Machine (ELM) layers and DeepONet architectures for fast regression and sequence modeling.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Allow duplicate OpenMP libraries (for some environments)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ExtremeLearning(nn.Module):
    """
    Extreme Learning Machine layer with randomized, fixed weights.
    """
    def __init__(self, indim, outdim, c=1, s=1, acfunc=nn.Tanh(), norm=(0, 1), device=None):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.af = acfunc
        self.norm = norm
        self.c = c
        self.s = s
        self.device = device

        self.R = self._init_R(indim, outdim, c)
        self.b = torch.rand(1, outdim, device=self.device) * 2 - 1

    def _init_R(self, indim, outdim, c):
        """
        Random sparse weight initialization with c active connections per output neuron.
        """
        res = torch.zeros(outdim, indim)
        for i in range(outdim):
            idx = np.random.choice(indim, min(c, indim), replace=False)
            res[i, idx] = torch.rand(len(idx)) * 2 / np.sqrt(len(idx)) - 1
        return res.T.to(self.device)

    def scale(self, x):
        """
        Normalize input x using self.norm.
        """
        return (x - self.norm[0]) / self.norm[1]

    def forward(self, x):
        """
        Forward pass through the ELM layer.
        """
        x = x.to(self.device)
        y = self.scale(x)
        y = torch.matmul(y, self.R) + self.b
        return self.af(self.s * y)


class _BaseExtremONet(nn.Module):
    """
    Base class for Extreme Learning DeepONet architectures.
    """
    def __init__(self, outdim, psize, trunk, branch, loss_func=None, device=None):
        super().__init__()
        self.outdim = outdim
        self.psize = psize
        self.trunk = trunk
        self.branch = branch
        self.device = device

        # Output weights (learned via regression)
        self.B = nn.Parameter(torch.zeros(1, outdim), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(psize, outdim), requires_grad=False)

        self.loss_func = loss_func or self.default_loss

    def convolve(self, b, t):
        """
        Elementwise product for feature combination.
        """
        return b * t

    def default_loss(self, y, yp):
        """
        Normalized mean squared error (NMSE).
        """
        y, yp = y.to(self.device), yp.to(self.device)
        return torch.mean((y - yp) ** 2) / (torch.std(y) ** 2)

    def loss(self, y, yp):
        """
        Compute loss between targets and predictions.
        """
        return self.loss_func(y, yp)


class ExtremONet(_BaseExtremONet):
    """
    Standard DeepONet with Extreme Learning Machine layers.
    """
    def forward(self, x, u):
        """
        Forward pass: combine trunk and branch features.
        """
        x, u = x.to(self.device), u.to(self.device)
        b = self.branch(u)
        t = self.trunk(x)
        return self.convolve(b, t)

    def predict(self, x, u, printt=False):
        """
        Predict output for given input x and u.
        """
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        u = torch.tensor(u, dtype=torch.float32, device=self.device)

        start = time.monotonic()
        yp = torch.matmul(self.forward(x, u), self.A) + self.B
        elapsed = time.monotonic() - start

        if printt:
            print(f"Prediction completed in {elapsed:.6f} seconds.")

        return yp.detach().cpu().numpy()


class RecurrentExtremONet(_BaseExtremONet):
    """
    Recurrent DeepONet for sequence modeling.
    """
    def forward(self, y, u):
        """
        Forward pass over sequences.
        """
        B, T, Dy = y.shape
        yf = y.reshape(B * T, Dy).to(self.device)
        uf = u.reshape(B * T, u.shape[2]).to(self.device)

        b = self.branch(uf)
        t = self.trunk(yf)
        return self.convolve(b, t).reshape(B, T, -1)

    def predict(self, y, u):
        """
        Predict output sequence for given input sequences y and u.
        """
        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        yp = self.forward(y, u) @ self.A + self.B
        return yp.detach().cpu().numpy()

    def predict_n(self, y, u, n, printt=False):
        """
        Autoregressive prediction for n steps.
        """
        yp = np.zeros((y.shape[0], n, y.shape[2]))
        yp[:, :y.shape[1]] = y

        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        yp = torch.tensor(yp, dtype=torch.float32, device=self.device)

        start = time.monotonic()
        for i in range(y.shape[1], n):
            step = self.forward(yp[:, i - 1:i], u[:, i - 1:i])[:, 0]
            yp[:, i] = step @ self.A + self.B
        elapsed = time.monotonic() - start

        if printt:
            print(f"Prediction completed in {elapsed:.6f} seconds.")

        return yp.detach().cpu().numpy()
