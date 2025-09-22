"""
test_center.py

Main experiment orchestration for EON/DON experiments.

- Structured imports and constants
- Minimal docstrings for all functions
- Consistent formatting and whitespace
- Removed duplicate code and grouped related logic

Assumes project-specific modules (EON, DON, ODE, PDE, EON_train, DON_train) are available.

Author: Jari Beysen
"""

# ----------------------
# Imports
# ----------------------
from typing import Callable, Iterable, List, Tuple, Any
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

# Project imports (assumed available in PYTHONPATH)
from EON import *
from DON import *
from ODE import *
from PDE import *
from EON_train import *
from DON_train import *

# ----------------------
# Constants
# ----------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------
# Data Loading
# ----------------------
ODE_PEND_TRAIN, ODE_PEND_TEST = open_data("train_data_ODE_ut.pkl"), open_data("test_data_ODE_ut.pkl")
ODE_L63_TRAIN, ODE_L63_TEST = open_data("train_data_ODE_L63.pkl"), open_data("test_data_ODE_L63.pkl")
ODE_PEND_PARAMS = open_data("ODE_params_ut.pkl")
ODE_L63_PARAMS = open_data("ODE_params_L63.pkl")

PDE_DIFF_TRAIN, PDE_DIFF_TEST = open_data("train_data_PDE_diff.pkl"), open_data("test_data_PDE_diff.pkl")
PDE_KG_TRAIN, PDE_KG_TEST = open_data("train_data_PDE_KG.pkl"), open_data("test_data_PDE_KG.pkl")
PDE_DIFF_PARAMS = open_data("PDE_params_diff.pkl")
PDE_KG_PARAMS = open_data("PDE_params_KG.pkl")

PDE_DIFFR_TRAIN, PDE_DIFFR_TEST = open_data("train_data_PDE_diffR.pkl"), open_data("test_data_PDE_diffR.pkl")
PDE_DIFFR_PARAMS = open_data("PDE_params_diffR.pkl")

# ----------------------
# Utility Functions
# ----------------------

def repeater(func: Callable, repeats: int, r: Iterable, vn: str, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repeat func over r for repeats, aggregate mean/std/min/max."""
    results = []
    for _ in range(repeats):
        run = [func(*args, vn=vn, value=v, **kwargs) for v in r]
        results.append(run)
    arr = np.array(results)
    return np.mean(arr, axis=0), np.std(arr, axis=0), np.min(arr, axis=0), np.max(arr, axis=0)

def textify(data: np.ndarray, filename: str, rank_names: List[str] = None) -> None:
    """Write numpy array to text file with indentation and optional rank names."""
    def write_array(f, array, indent=0, rank=0):
        if array.ndim == 1:
            f.write(" " * indent + "[" + ", ".join(f"{x:.6f}" for x in array) + "]\n")
        else:
            f.write(" " * indent + "[\n")
            for i, sub in enumerate(array):
                if rank_names and rank < len(rank_names):
                    f.write(" " * (indent + 2) + f"# {rank_names[rank]}: {i}\n")
                write_array(f, sub, indent + 2, rank + 1)
            f.write(" " * indent + "]\n")
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    with open(filename, "w") as f:
        f.write(f"# Shape: {data.shape}\n")
        write_array(f, data)

def NMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean squared error."""
    return np.mean((y_true - y_pred) ** 2) / (np.mean(y_true ** 2) + 1e-12)

def double_roller(x: List[np.ndarray]) -> np.ndarray:
    """Stack and expand arrays for result aggregation."""
    return np.vstack(np.expand_dims([np.vstack(np.expand_dims(xi, 0)) for xi in x], 0))

def flatten_grid(grid: Iterable) -> np.ndarray:
    """Flatten meshgrid into 2D array (points as rows)."""
    return np.array(grid).reshape(len(grid), -1).T

def unflatten_result(flattened: np.ndarray, axis: int = None) -> np.ndarray:
    """Unflatten mesh result into square grid (assumes perfect square)."""
    n = int(np.sqrt(flattened.shape[0]))
    return flattened.reshape((n, n, flattened.shape[1]))

# ----------------------
# Test Routines
# ----------------------

def general_test(dtr, dte, vn: str, value: Any, nsens: int = 120):
    """Generic EON evaluation for width, st, sb, cb, sensors, data, OOD."""
    width, st, sb, cb = 1000, 10, 0.1, 1
    # Select train/test splits and parameters
    if vn == 'width':
        width = int(value)
    elif vn == 'st':
        st = float(value)
    elif vn == 'sb':
        sb = float(value)
    elif vn == 'cb':
        cb = float(value)
    ttr, ytr, utr = dtr
    tte, yte, ute = dte
    if vn == 'sensors':
        if int(value) == 0:
            utr, ute = np.ones((utr.shape[0], 1)), np.ones((ute.shape[0], 1))
        else:
            randinds = np.random.choice(range(utr.shape[1]), int(value), replace=False)
            utr, ute = utr[:, randinds], ute[:, randinds]
    elif vn == 'data':
        randinds = np.random.choice(range(utr.shape[0]), int(value), replace=False)
        ttr, utr, ytr = ttr[randinds], utr[randinds], ytr[randinds]
    elif vn.startswith('OOD_'):
        ttr, ytr, utr = dtr[value[0]]
        tte, yte, ute = dte[value[1]]
    utr = utr[:, :nsens]
    ute = ute[:, :nsens]
    trunk = ExtremeLearning(1, width, c=1, s=st, acfunc=nn.Tanh(), norm=[np.mean(ttr), np.std(ttr)], device=DEVICE).to(DEVICE)
    branch = ExtremeLearning(utr.shape[1], width, c=cb, s=sb, acfunc=nn.Tanh(), norm=[np.mean(utr), np.std(utr)], device=DEVICE).to(DEVICE)
    EON = ExtremONet(ytr.shape[1], width, trunk, branch, device=DEVICE).to(DEVICE)
    tr_start = time.time()
    train_EON(EON, ttr, utr, ytr, iters=100)
    tr_time = time.time() - tr_start
    pred_start = time.time()
    yte_pred = EON.predict(tte, ute)
    pred_time = time.time() - pred_start
    ytr_pred = EON.predict(ttr, utr)
    del trunk, branch, EON
    torch.cuda.empty_cache()
    return NMSE(ytr, ytr_pred), NMSE(yte, yte_pred), tr_time, pred_time

def general_test_DON(dtr, dte, vn: str, value: Any, nsens: int = 120):
    """Generic DON evaluation for OOD experiments."""
    iters = 1_000_000
    layers = [500, 500, 500, 20, 1e-3]
    afs = [nn.Tanh() if i != len(layers[:-2]) else (lambda x: x) for i in range(len(layers[:-2]) + 1)]
    ttr, ytr, utr = dtr[value[0]]
    tte, yte, ute = dte[value[1]]
    trunk_norm = [np.mean(ttr), np.std(ttr)]
    branch_norm = [np.mean(utr), np.std(utr)]
    utr = utr[:, :nsens]
    ute = ute[:, :nsens]
    trunk = NeuralNet(ttr.shape[1], layers[:-2], ytr.shape[1] * layers[-2], afs, device=DEVICE, norm=trunk_norm).to(DEVICE)
    branch = NeuralNet(utr.shape[1], layers[:-2], ytr.shape[1] * layers[-2], afs, device=DEVICE, norm=branch_norm).to(DEVICE)
    DON_model = DeepONet(branch, trunk, layers[-2], ytr.shape[1], device=DEVICE).to(DEVICE)
    optim = AdamW(DON_model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.LinearLR(optim, layers[-1], layers[-1], iters)
    tr_start = time.time()
    train_DON(DON_model, optim, scheduler, [ttr, utr, ytr], iters, stop_criterion=0, n_criterion=1000, nprint=10)
    tr_time = time.time() - tr_start
    pred_start = time.time()
    yte_pred = DON_model.predict(tte, ute)
    pred_time = time.time() - pred_start
    ytr_pred = DON_model.predict(ttr, utr)
    del trunk, branch, DON_model
    torch.cuda.empty_cache()
    return NMSE(ytr, ytr_pred), NMSE(yte, yte_pred), tr_time, pred_time

def recurrent_test(dtr, dte, vn, value, nsens: int = 120):
    """Recurrent EON evaluation for OOD2 experiments."""
    ttr, ytr, utr = dtr[value[0]]
    tte, yte, ute = dte[value[0]][value[1]]
    utr = utr[:, :nsens]
    ute = ute[:, :nsens]
    st, sb, ct, cb = 0.1, 0.1, 1, 1
    width = 1000
    trunk = ExtremeLearning(ytr.shape[2], width, c=ct, s=st, acfunc=nn.Tanh(), norm=[np.mean(ytr), np.std(ytr)], device=DEVICE).to(DEVICE)
    branch = ExtremeLearning(utr.shape[2], width, c=cb, s=sb, acfunc=nn.Tanh(), norm=[np.mean(utr), np.std(utr)], device=DEVICE).to(DEVICE)
    EON = RecurrentExtremONet(ytr.shape[2], width, trunk, branch, device=DEVICE).to(DEVICE)
    tr_start = time.time()
    train_EON(EON, ytr[:, :-1], utr[:, :-1], ytr[:, 1:], iters=100, rnn=True)
    tr_time = time.time() - tr_start
    pred_start = time.time()
    yte_pred = EON.predict(yte[:, :-1], ute[:, :-1])
    pred_time = time.time() - pred_start
    ytr_pred = EON.predict(ytr[:, :-1], utr[:, :-1])
    del trunk, branch, EON
    torch.cuda.empty_cache()
    return NMSE(yte[:, 1:], yte_pred), NMSE(ytr[:, 1:], ytr_pred), tr_time, pred_time

def recurrent_test_DON(dtr, dte, vn, value, nsens: int = 120):
    """Recurrent DON evaluation for OOD2 experiments."""
    iters = 1_000_000
    ttr, ytr, utr = dtr[value[0]]
    tte, yte, ute = dte[value[0]][value[1]]
    utr = utr[:, :, :nsens]
    ute = ute[:, :, :nsens]
    layers = [500, 500, 500, 20, 1e-3]
    afs = [nn.Tanh() if i != len(layers[:-2]) else (lambda x: x) for i in range(len(layers[:-2]) + 1)]
    trunk = NeuralNet(ytr.shape[2], layers[:-2], ytr.shape[2] * layers[-2], afs, device=DEVICE, norm=[np.mean(ytr), np.std(ytr)]).to(DEVICE)
    branch = NeuralNet(utr.shape[2], layers[:-2], ytr.shape[2] * layers[-2], afs, device=DEVICE, norm=[np.mean(utr), np.std(utr)]).to(DEVICE)
    DON_model = RecurrentDeepONet(branch, trunk, layers[-2], ytr.shape[2], device=DEVICE).to(DEVICE)
    optim = AdamW(DON_model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.LinearLR(optim, layers[-1], layers[-1], iters)
    tr_start = time.time()
    train_DON(DON_model, optim, scheduler, [ytr[:, :-1], utr[:, :-1], ytr[:, 1:]], iters, stop_criterion=0, n_criterion=1000, nprint=10)
    tr_time = time.time() - tr_start
    pred_start = time.time()
    yte_pred = DON_model.predict(yte[:, :-1], ute[:, :-1])
    pred_time = time.time() - pred_start
    ytr_pred = DON_model.predict(ytr[:, :-1], utr[:, :-1])
    del trunk, branch, DON_model
    torch.cuda.empty_cache()
    return NMSE(yte[:, 1:], yte_pred), NMSE(ytr[:, 1:], ytr_pred), tr_time, pred_time

# ----------------------
# Main Experiment Orchestration
# ----------------------

def run_experiments(testlist: List[str]):
    """Run selected experiments from testlist."""
    repeats = 100
    ntest = 30

    def run_and_save(test_name, values, train_data, test_data, filename):
        res1 = repeater(general_test, repeats, values, test_name, train_data[0], train_data[1])
        res2 = repeater(general_test, repeats, values, test_name, test_data[0], test_data[1])
        res3 = repeater(general_test, repeats, values, test_name, PDE_KG_TRAIN, PDE_KG_TEST)
        res4 = repeater(general_test, repeats, values, test_name, PDE_DIFF_TRAIN, PDE_DIFF_TEST)
        res = double_roller([res1, res2, res3, res4])
        textify(res, f'{filename}.txt', rank_names=['system', 'mean/std/min/max', 'value', 'tre/tee/trt/tet'])
        save_data(res, f'{filename}.pkl')

    if 'width' in testlist:
        values = np.linspace(1, 1000, ntest).astype(int)
        run_and_save('width', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'width_test')

    if 'sensors' in testlist:
        values = np.linspace(0, 120, ntest).astype(int)
        run_and_save('sensors', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'sensors_test')

    if 'data' in testlist:
        values = np.linspace(10, 3000, ntest).astype(int)
        run_and_save('data', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'data_test')

    if 'st' in testlist:
        values = np.linspace(1e-5, 10, ntest)
        run_and_save('st', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'st_test')

    if 'sb' in testlist:
        values = np.linspace(1e-5, 10, ntest)
        run_and_save('sb', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'sb_test')

    if 'cb' in testlist:
        values = np.linspace(1, 120, ntest).astype(int)
        run_and_save('cb', values, (ODE_L63_TRAIN, ODE_L63_TEST), (ODE_PEND_TRAIN, ODE_PEND_TEST), 'cb_test')

    if any(x.startswith('OOD') for x in testlist):
        repeats = 10

    if 'OOD1' in testlist:
        values1, values2, _ = open_data('OOD_grid.pkl')
        vg = np.meshgrid(range(len(values1)), range(len(values2)))
        vgf = flatten_grid(vg)
        # DON
        dtr, dte = open_data('OOD_ODE.pkl')
        res1 = repeater(general_test_DON, repeats, vgf, 'OOD_ODE', dtr, dte)
        res1 = [unflatten_result(r) for r in res1]
        dtr, dte = open_data('OOD_PDE.pkl')
        res2 = repeater(general_test_DON, repeats, vgf, 'OOD_PDE', dtr, dte)
        res2 = [unflatten_result(r) for r in res2]
        res = double_roller([res1, res2])
        textify(res, 'OOD1_testDON.txt', rank_names=['system', 'mean/std/min/max', 'valuex', 'valuey', 'tre/tee/trt/tet'])
        save_data(res, 'OOD1_testDON.pkl')
        # EON
        dtr, dte = open_data('OOD_ODE.pkl')
        res1 = repeater(general_test, repeats, vgf, 'OOD_ODE', dtr, dte)
        res1 = [unflatten_result(r) for r in res1]
        dtr, dte = open_data('OOD_PDE.pkl')
        res2 = repeater(general_test, repeats, vgf, 'OOD_PDE', dtr, dte)
        res2 = [unflatten_result(r) for r in res2]
        res = double_roller([res1, res2])
        textify(res, 'OOD1_testEON.txt', rank_names=['system', 'mean/std/min/max', 'valuex', 'valuey', 'tre/tee/trt/tet'])
        save_data(res, 'OOD1_testEON.pkl')

    if 'OOD2' in testlist:
        values1, values2, _ = open_data('OOD_gridD.pkl')
        vg = np.meshgrid(range(len(values1)), range(len(values2)))
        vgf = flatten_grid(vg)
        dtr, dte = open_data('OOD_PDED.pkl')
        res1 = repeater(recurrent_test_DON, repeats, vgf, '', dtr, dte)
        res1 = [unflatten_result(r) for r in res1]
        res = double_roller([res1])
        textify(res, 'OOD2_testDON.txt', rank_names=['system', 'mean/std/min/max', 'valuex', 'valuey', 'tre/tee/trt/tet'])
        save_data(res, 'OOD2_testDON.pkl')
        dtr, dte = open_data('OOD_PDED.pkl')
        res1 = repeater(recurrent_test, repeats, vgf, '', dtr, dte)
        res1 = [unflatten_result(r) for r in res1]
        res = double_roller([res1])
        textify(res, 'OOD2_testEON.txt', rank_names=['system', 'mean/std/min/max', 'valuex', 'valuey', 'tre/tee/trt/tet'])
        save_data(res, 'OOD2_testEON.pkl')

# ----------------------
# Manual Entrypoint
# ----------------------

# Example usage: manually specify tests to run
if __name__ == '__main__':
    # Directly specify which tests to run here
    tests_to_run = ['width', 'sensors', 'data', 'st', 'sb', 'cb', 'OOD1', 'OOD2']
    run_experiments(tests_to_run)

