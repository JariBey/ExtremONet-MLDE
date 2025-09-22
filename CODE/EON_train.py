"""
EON_train.py

Training utilities for ExtremONet models using ridge regression with scalar regularization search.

Author: Jari Beysen
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Allow duplicate MKL libraries (fixes some Windows issues)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Constants
DTYPE = torch.float32

def train_EON(
    model: nn.Module,
    x,
    u,
    y,
    bounds: list = [-10, 10],
    iters: int = 200,
    tp: float = 0.2,
    verbose: bool = True,
    rnn: bool = False,
):
    """
    Train the readout (A, B) of an ExtremONet using ridge regression.

    Args:
        model: ExtremONet or RecurrentExtremONet (expects .forward(x, u) giving features h)
        x, u, y: numpy arrays or torch tensors (converted inside)
        bounds: [low, high] search interval for log10 regularization exponent
        iters: max iterations for the scalar minimizer
        tp: fraction of samples used as validation set
        verbose: print progress/results
        rnn: if True, treat h and y as (B, T, D) and collapse to (B*T, D)

    Returns:
        (train_history, val_history): NMSE values for train and val across tried lambdas
    """
    device = getattr(model, "device", torch.device("cpu"))

    # Convert inputs to tensors on device
    def to_tensor(arr):
        return arr if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=DTYPE, device=device)

    x_t, u_t, y_t = map(to_tensor, (x, u, y))

    start_time = time.time()

    # Compute features
    with torch.no_grad():
        h = model.forward(x_t, u_t)

    # Collapse batch/time dims if recurrent
    if rnn:
        h = h.reshape(-1, h.shape[-1])
        y_t = y_t.reshape(-1, y_t.shape[-1])

    # Train/val split
    n_samples = h.shape[0]
    idx = np.random.permutation(n_samples)
    n_val = int(np.ceil(tp * n_samples))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    h_tr, y_tr = h[train_idx], y_t[train_idx]
    h_val, y_val = h[val_idx], y_t[val_idx]

    # Augment with bias column
    def augment(H):
        ones = torch.ones(H.shape[0], 1, device=device, dtype=DTYPE)
        return torch.cat([H, ones], dim=1)

    H_tr_aug = augment(h_tr)
    H_val_aug = augment(h_val)

    # Precompute matrices
    HtH = H_tr_aug.T @ H_tr_aug
    HtY = H_tr_aug.T @ y_tr

    Ahist, Bhist, val_hist, tr_hist, l_hist = [], [], [], [], []

    def solve_ridge_for_l(l):
        """
        Solve ridge regression for regularization exponent l.
        Returns A_aug, A, B, train NMSE, val NMSE.
        """
        lam = 10.0 ** l
        reg = lam * torch.eye(HtH.shape[0], device=device, dtype=DTYPE)
        try:
            A_aug = torch.linalg.solve(HtH + reg, HtY)
        except RuntimeError:
            A_aug = torch.inverse(HtH + reg) @ HtY

        A, B = A_aug[:-1, :], A_aug[-1:, :]
        with torch.no_grad():
            y_val_pred = H_val_aug @ A_aug
            val_loss = model.loss(y_val, y_val_pred).item()
            y_tr_pred = H_tr_aug @ A_aug
            tr_loss = model.loss(y_tr, y_tr_pred).item()
        return A_aug, A, B, tr_loss, val_loss

    def objective(l_scalar):
        """
        Objective for scalar minimizer: validation NMSE.
        """
        A_aug, A, B, tr_loss, val_loss = solve_ridge_for_l(l_scalar)
        Ahist.append(A.cpu())
        Bhist.append(B.cpu())
        tr_hist.append(tr_loss)
        val_hist.append(val_loss)
        l_hist.append(l_scalar)
        return val_loss

    # Minimize validation NMSE over log10(lambda)
    result = minimize_scalar(
        objective,
        bounds=tuple(bounds),
        method="bounded",
        options={"maxiter": iters, "xatol": 1e-2},
    )

    # Select best by min validation loss
    min_idx = int(np.argmin(val_hist))
    best_A_aug = Ahist[min_idx].to(device=device, dtype=DTYPE)
    best_A = best_A_aug
    best_B = Bhist[min_idx].to(device=device, dtype=DTYPE)

    # Assign to model
    with torch.no_grad():
        if hasattr(model, "A"):
            model.A.data = best_A
        else:
            model.A = nn.Parameter(best_A, requires_grad=False)
        if hasattr(model, "B"):
            model.B.data = best_B
        else:
            model.B = nn.Parameter(best_B, requires_grad=False)
        model.valloss = val_hist[min_idx]

    if verbose:
        duration = time.time() - start_time
        print("-" * 100)
        print(f"Selected log10(lambda) = {l_hist[min_idx]:.4f}")
        print(f"Validation NMSE = {val_hist[min_idx]:.6e}, Train NMSE = {tr_hist[min_idx]:.6e}")
        print(f"Training completed in {duration:.2f} s")

    return tr_hist, val_hist

