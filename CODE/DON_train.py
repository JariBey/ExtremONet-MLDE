"""
DON_train.py

Training routine for DeepONet models with early stopping and logging.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial

# Allow duplicate OpenMP libraries (for some environments)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Constants
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_NPRINT = 100
DEFAULT_STOP_CRITERION = 1e-10
DEFAULT_N_CRITERION = 1000

def slope_condition(losses: np.ndarray) -> float:
    """
    Compute log10 slope over last half vs first half of recent losses.
    """
    n = len(losses)
    return np.sum(
        np.log10(losses[int(n / 2):] / losses[: int(n / 2)])
    ) / (n / 2)

def train_DON(
    model,
    optimizer,
    scheduler,
    train_data,
    iterations: int,
    test_split: float = DEFAULT_TEST_SPLIT,
    device=None,
    nprint: int = DEFAULT_NPRINT,
    stop_criterion: float = DEFAULT_STOP_CRITERION,
    n_criterion: int = DEFAULT_N_CRITERION,
):
    """
    Train a DeepONet model with early stopping.

    Args:
        model: DeepONet or RecurrentDeepONet instance
        optimizer: torch optimizer
        scheduler: learning rate scheduler
        train_data: tuple (x, u, y) with training data
        iterations: number of training epochs
        test_split: fraction of data used for testing
        device: torch device
        nprint: print progress every nprint iterations
        stop_criterion: slope threshold for early stopping
        n_criterion: number of epochs used to compute slope

    Returns:
        train_losses, test_losses (lists of floats)
    """
    x, u, y = train_data
    x = torch.tensor(x, dtype=model.dtype, device=model.device)
    u = torch.tensor(u, dtype=model.dtype, device=model.device)
    y = torch.tensor(y, dtype=model.dtype, device=model.device)

    # Store normalization stats in branch and trunk
    model.branch.norm = [torch.mean(u).item(), torch.std(u).item()]
    model.trunk.norm = [torch.mean(x).item(), torch.std(x).item()]

    # Train/test split
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(test_split * num_samples))
    train_idx, test_idx = indices[split:], indices[:split]
    x_train, u_train, y_train = x[train_idx], u[train_idx], y[train_idx]
    x_test, u_test, y_test = x[test_idx], u[test_idx], y[test_idx]

    train_losses, test_losses = [], []
    best_model_state = None
    best_train_loss, best_test_loss = float("inf"), float("inf")

    start_time = time.time()

    for epoch in range(iterations):
        # Training step
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train, u_train)
        train_loss = model.loss(y_train, y_pred)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test, u_test)
            test_loss = model.loss(y_test, y_test_pred)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        # Save best model
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_train_loss = train_loss.item()
            best_model_state = model.state_dict()

        # Early stopping
        if epoch > n_criterion:
            recent_losses = np.array(test_losses[-n_criterion:])
            slope = slope_condition(recent_losses)
            if slope > -stop_criterion:
                print(f"Early stopping at epoch {epoch}, slope={slope:.3e}")
                break

        # Logging
        if (epoch * nprint)%iterations == 0 or epoch == iterations - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (iterations - epoch - 1)
            print("-" * 80)
            print(
                f"Epoch {epoch + 1}/{iterations} | "
                f"Train Loss: {train_loss.item():.6e} | "
                f"Test Loss: {test_loss.item():.6e} | "
                f"ETA: {eta:.1f}s"
            )
            if epoch > n_criterion:
                print(f"log10 slope: {slope:.3e}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if (epoch*nprint)%iterations != 0:
        print("-" * 80)
        print(
            f"Lowest Test Loss: {best_test_loss:.6e}, "
            f"Corresponding Train Loss: {best_train_loss:.6e}"
        )

    return train_losses, test_losses
