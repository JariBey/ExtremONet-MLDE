#!/usr/bin/env python3
"""
Benchmarking script for comparing ExtremONet (EON) and DeepONet (DON) models
on ODE and PDE datasets. Loads data, runs repeated experiments, aggregates
results, and saves timing and performance summaries.

Assumes project-local modules (EON, DON, PDE, EON_train, DON_train) are available.

Author: Jari Beysen
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from EON import *
from DON import *
from PDE import *
from EON_train import *
from DON_train import *

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

TEXT_TIMES_PATH = "time_comp.txt"
TEXT_LOWEST_PATH =  "general_comp.txt"
DATATRAIN_PATH =  "datatrain.pkl"
DATATIME_PATH =  "datatime.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data from a pickle file using the project's open_data function."""
    return open_data(path)

def safe_empty_cache() -> None:
    """Empty CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def pad_with_final_value(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad a 1D array to target_len by repeating the final value."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("pad_with_final_value expects a 1D array")
    if len(arr) >= target_len:
        return arr.copy()
    pad_width = target_len - len(arr)
    return np.concatenate([arr, np.full(pad_width, arr[-1])])

def pad_and_get_min(data_list: List[np.ndarray]) -> np.ndarray:
    """Pad each 1D array in data_list to the maximum length and return per-array minima."""
    if not data_list:
        return np.array([])
    max_len = max(len(d) for d in data_list)
    padded = np.vstack([pad_with_final_value(np.asarray(d), max_len) for d in data_list])
    return np.min(padded, axis=1)

# ---------------------------------------------------------------------------
# Model Test Functions
# ---------------------------------------------------------------------------
def test_DON(
    ttr: np.ndarray, utr: np.ndarray, ytr: np.ndarray,
    tte: np.ndarray, ute: np.ndarray,
    layers: List[float], device: torch.device = device
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Train and time a DeepONet (DON) model."""
    iters = 1000000
    afs = [
        (nn.Tanh() if i != len(layers[:-2]) else (lambda x: x))
        for i in range(len(layers[:-2]) + 1)
    ]
    Trunk = NeuralNet(
        indim=ttr.shape[1],
        layers=layers[:-2],
        outdim=ytr.shape[1] * int(layers[-2]),
        acfuncs=afs,
        device=device,
        norm=[np.mean(ttr), np.std(ttr)],
    ).to(device)
    Branch = NeuralNet(
        indim=utr.shape[1],
        layers=layers[:-2],
        outdim=ytr.shape[1] * int(layers[-2]),
        acfuncs=afs,
        device=device,
        norm=[np.mean(utr), np.std(utr)],
    ).to(device)
    DON_model = DeepONet(Branch, Trunk, layers[-2], ytr.shape[1], device=device).to(device)
    print(f"[DON] Total trainable parameters: {sum(p.numel() for p in DON_model.parameters() if p.requires_grad)}")
    optim = AdamW(DON_model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.LinearLR(optim, start_factor=layers[-1], end_factor=layers[-1], total_iters=iters)
    t0 = time.time()
    trhist, valhist = train_DON(DON_model, optim, scheduler, [ttr, utr, ytr], iters, stop_criterion=0, n_criterion=1000, nprint=10)
    trtime = time.time() - t0
    t0 = time.time()
    _ = DON_model.predict(tte, ute)
    predtime = time.time() - t0
    del Trunk, Branch, DON_model
    safe_empty_cache()
    return np.asarray(trhist), np.asarray(valhist), trtime, predtime

def test_EON(
    ttr: np.ndarray, utr: np.ndarray, ytr: np.ndarray,
    tte: np.ndarray, ute: np.ndarray,
    width: int, device: torch.device = device
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Train and time an ExtremONet (EON) model."""
    trunk = ExtremeLearning(
        indim=1,
        outdim=width,
        c=1,
        s=10,
        acfunc=nn.Tanh(),
        norm=[np.mean(ttr), np.std(ttr)],
        device=device,
    ).to(device)
    branch = ExtremeLearning(
        indim=utr.shape[1],
        outdim=width,
        c=1,
        s=0.1,
        acfunc=nn.Tanh(),
        norm=[np.mean(utr), np.std(utr)],
        device=device,
    ).to(device)
    EON_model = ExtremONet(ytr.shape[1], width, trunk, branch, device=device).to(device)
    print(f"[EON]")
    t0 = time.time()
    trhist, valhist = train_EON(EON_model, ttr, utr, ytr, iters=100)
    trtime = time.time() - t0
    t0 = time.time()
    _ = EON_model.predict(tte, ute)
    predtime = time.time() - t0
    del trunk, branch, EON_model
    safe_empty_cache()
    return np.asarray(trhist), np.asarray(valhist), trtime, predtime

def testloop(
    ttr: np.ndarray, utr: np.ndarray, ytr: np.ndarray,
    tte: np.ndarray, ute: np.ndarray, yte: np.ndarray,
    repeats: int, layers_list: List[List[float]], widths: List[int]
) -> Tuple:
    """Run repeated experiments for each (layers, width) configuration."""
    Dtrhist, Dvalhist, Dtrtime, Dpredtime = [], [], [], []
    Etrhist, Evalhist, Etrtime, Epredtime = [], [], [], []
    Dtrhist_std, Dvalhist_std, Dtrtime_std, Dpredtime_std = [], [], [], []
    Etrhist_std, Evalhist_std, Etrtime_std, Epredtime_std = [], [], [], []

    for layers, width in zip(layers_list, widths):
        sDtrhist, sDvalhist, sDtrtime, sDpredtime = [], [], [], []
        sEtrhist, sEvalhist, sEtrtime, sEpredtime = [], [], [], []
        for _ in range(repeats):
            d_trhist, d_valhist, d_trtime, d_predtime = test_DON(ttr, utr, ytr, tte, ute, layers)
            e_trhist, e_valhist, e_trtime, e_predtime = test_EON(ttr, utr, ytr, tte, ute, width)
            sDtrhist.append(d_trhist)
            sDvalhist.append(d_valhist)
            sDtrtime.append(d_trtime)
            sDpredtime.append(d_predtime)
            sEtrhist.append(e_trhist)
            sEvalhist.append(e_valhist)
            sEtrtime.append(e_trtime)
            sEpredtime.append(e_predtime)
        def pad_group(hist_list: List[np.ndarray]) -> np.ndarray:
            if not hist_list:
                return np.array([])
            max_len = max(len(h) for h in hist_list)
            return np.vstack([pad_with_final_value(h, max_len) for h in hist_list])
        Dtr_padded = pad_group(sDtrhist)
        Dval_padded = pad_group(sDvalhist)
        Estr_padded = pad_group(sEtrhist)
        Eval_padded = pad_group(sEvalhist)
        if Dtr_padded.size:
            Dtrhist.append(np.mean(Dtr_padded, axis=0))
            Dvalhist.append(np.mean(Dval_padded, axis=0))
            Dtrhist_std.append(np.std(Dtr_padded, axis=0))
            Dvalhist_std.append(np.std(Dval_padded, axis=0))
        else:
            Dtrhist.append(np.array([])); Dvalhist.append(np.array([]))
            Dtrhist_std.append(np.array([])); Dvalhist_std.append(np.array([]))
        Dtrtime.append(float(np.mean(sDtrtime)))
        Dpredtime.append(float(np.mean(sDpredtime)))
        Dtrtime_std.append(float(np.std(sDtrtime)))
        Dpredtime_std.append(float(np.std(sDpredtime)))
        if Estr_padded.size:
            Etrhist.append(np.mean(Estr_padded, axis=0))
            Evalhist.append(np.mean(Eval_padded, axis=0))
            Etrhist_std.append(np.std(Estr_padded, axis=0))
            Evalhist_std.append(np.std(Eval_padded, axis=0))
        else:
            Etrhist.append(np.array([])); Evalhist.append(np.array([]))
            Etrhist_std.append(np.array([])); Evalhist_std.append(np.array([]))
        Etrtime.append(float(np.mean(sEtrtime)))
        Epredtime.append(float(np.mean(sEpredtime)))
        Etrtime_std.append(float(np.std(sEtrtime)))
        Epredtime_std.append(float(np.std(sEpredtime)))
    return (
        Dtrhist, Dvalhist, Dtrtime, Dpredtime, Etrhist, Evalhist, Etrtime, Epredtime,
        Dtrhist_std, Dvalhist_std, Dtrtime_std, Dpredtime_std,
        Etrhist_std, Evalhist_std, Etrtime_std, Epredtime_std
    )

# ---------------------------------------------------------------------------
# Saving Helpers
# ---------------------------------------------------------------------------
def save_time_text(data_dict: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Save the time metrics to a human-readable text file."""
    with path.open("w") as fp:
        for category, metrics in data_dict.items():
            fp.write(f"Category: {category}\n")
            for metric_name, values in metrics.items():
                fp.write(f"  Metric: {metric_name}\n")
                try:
                    for i, v in enumerate(values):
                        fp.write(f"    Run {i + 1}: {v}\n")
                except TypeError:
                    fp.write(f"    Value: {values}\n")
            fp.write("\n")

def save_pickle(obj: Any, path: Path) -> None:
    """Save object to pickle file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)

# ---------------------------------------------------------------------------
# Main Flow
# ---------------------------------------------------------------------------
def main():
    """Main benchmarking workflow."""
    start_time = time.time()
    # Load datasets
    ODE_param_ttr, ODE_param_ytr, ODE_param_utr = load_data("train_data_ODE_L63.pkl")
    ODE_GRF_ttr, ODE_GRF_ytr, ODE_GRF_utr = load_data("train_data_ODE_ut.pkl")
    PDE_param_ttr, PDE_param_ytr, PDE_param_utr = load_data("train_data_PDE_KG.pkl")
    PDE_GRF_ttr, PDE_GRF_ytr, PDE_GRF_utr = load_data("train_data_PDE_diff.pkl")
    ODE_param_tte, ODE_param_yte, ODE_param_ute = load_data("test_data_ODE_L63.pkl")
    ODE_GRF_tte, ODE_GRF_yte, ODE_GRF_ute = load_data("test_data_ODE_ut.pkl")
    PDE_param_tte, PDE_param_yte, PDE_param_ute = load_data("test_data_PDE_KG.pkl")
    PDE_GRF_tte, PDE_GRF_yte, PDE_GRF_ute = load_data("test_data_PDE_diff.pkl")
    # Experiment configuration
    nrepeat = 10
    layers = [
        [100, 10, 2e-2],
        [200, 200, 10, 5e-3],
        [500, 500, 500, 20, 1e-3],
    ]
    widths = [100, 300, 1000]
    # Run experiments
    print("Running ODE_param benchmark...")
    dth_ODE_param, dvh_ODE_param, dtt_ODE_param, dpt_ODE_param, \
    eth_ODE_param, evh_ODE_param, ett_ODE_param, ept_ODE_param, \
    dth_ODE_param_std, dvh_ODE_param_std, dtt_ODE_param_std, dpt_ODE_param_std, \
    eth_ODE_param_std, evh_ODE_param_std, ett_ODE_param_std, ept_ODE_param_std = testloop(
        ODE_param_ttr, ODE_param_utr, ODE_param_ytr,
        ODE_param_tte, ODE_param_ute, ODE_param_yte,
        nrepeat, layers, widths
    )
    print("Running ODE_GRF benchmark...")
    dth_ODE_GRF, dvh_ODE_GRF, dtt_ODE_GRF, dpt_ODE_GRF, \
    eth_ODE_GRF, evh_ODE_GRF, ett_ODE_GRF, ept_ODE_GRF, \
    dth_ODE_GRF_std, dvh_ODE_GRF_std, dtt_ODE_GRF_std, dpt_ODE_GRF_std, \
    eth_ODE_GRF_std, evh_ODE_GRF_std, ett_ODE_GRF_std, ept_ODE_GRF_std = testloop(
        ODE_GRF_ttr, ODE_GRF_utr, ODE_GRF_ytr,
        ODE_GRF_tte, ODE_GRF_ute, ODE_GRF_yte,
        nrepeat, layers, widths
    )
    print("Running PDE_param benchmark...")
    dth_PDE_param, dvh_PDE_param, dtt_PDE_param, dpt_PDE_param, \
    eth_PDE_param, evh_PDE_param, ett_PDE_param, ept_PDE_param, \
    dth_PDE_param_std, dvh_PDE_param_std, dtt_PDE_param_std, dpt_PDE_param_std, \
    eth_PDE_param_std, evh_PDE_param_std, ett_PDE_param_std, ept_PDE_param_std = testloop(
        PDE_param_ttr, PDE_param_utr, PDE_param_ytr,
        PDE_param_tte, PDE_param_ute, PDE_param_yte,
        nrepeat, layers, widths
    )
    print("Running PDE_GRF benchmark...")
    dth_PDE_GRF, dvh_PDE_GRF, dtt_PDE_GRF, dpt_PDE_GRF, \
    eth_PDE_GRF, evh_PDE_GRF, ett_PDE_GRF, ept_PDE_GRF, \
    dth_PDE_GRF_std, dvh_PDE_GRF_std, dtt_PDE_GRF_std, dpt_PDE_GRF_std, \
    eth_PDE_GRF_std, evh_PDE_GRF_std, ett_PDE_GRF_std, ept_PDE_GRF_std = testloop(
        PDE_GRF_ttr, PDE_GRF_utr, PDE_GRF_ytr,
        PDE_GRF_tte, PDE_GRF_ute, PDE_GRF_yte,
        nrepeat, layers, widths
    )
    total_elapsed = time.time() - start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    # Save timing data
    data_dict = {
        "ODE_param": {
            "EON Train Time": ett_ODE_param,
            "EON Train Time Std": ett_ODE_param_std,
            "DON Train Time": dtt_ODE_param,
            "DON Train Time Std": dtt_ODE_param_std,
            "EON Prediction Time": ept_ODE_param,
            "EON Prediction Time Std": ept_ODE_param_std,
            "DON Prediction Time": dpt_ODE_param,
            "DON Prediction Time Std": dpt_ODE_param_std,
        },
        "ODE_GRF": {
            "EON Train Time": ett_ODE_GRF,
            "EON Train Time Std": ett_ODE_GRF_std,
            "DON Train Time": dtt_ODE_GRF,
            "DON Train Time Std": dtt_ODE_GRF_std,
            "EON Prediction Time": ept_ODE_GRF,
            "EON Prediction Time Std": ept_ODE_GRF_std,
            "DON Prediction Time": dpt_ODE_GRF,
            "DON Prediction Time Std": dpt_ODE_GRF_std,
        },
        "PDE_param": {
            "EON Train Time": ett_PDE_param,
            "EON Train Time Std": ett_PDE_param_std,
            "DON Train Time": dtt_PDE_param,
            "DON Train Time Std": dtt_PDE_param_std,
            "EON Prediction Time": ept_PDE_param,
            "EON Prediction Time Std": ept_PDE_param_std,
            "DON Prediction Time": dpt_PDE_param,
            "DON Prediction Time Std": dpt_PDE_param_std,
        },
        "PDE_GRF": {
            "EON Train Time": ett_PDE_GRF,
            "EON Train Time Std": ett_PDE_GRF_std,
            "DON Train Time": dtt_PDE_GRF,
            "DON Train Time Std": dtt_PDE_GRF_std,
            "EON Prediction Time": ept_PDE_GRF,
            "EON Prediction Time Std": ept_PDE_GRF_std,
            "DON Prediction Time": dpt_PDE_GRF,
            "DON Prediction Time Std": dpt_PDE_GRF_std,
        },
    }
    save_time_text(data_dict, TEXT_TIMES_PATH)
    print(f"Data saved to {TEXT_TIMES_PATH}")
    # Save lowest values summary
    lowest_values_data = {
        "ODE_param": {
            "EON": pad_and_get_min(eth_ODE_param),
            "EON Std": pad_and_get_min(eth_ODE_param_std),
            "DON": pad_and_get_min(dth_ODE_param),
            "DON Std": pad_and_get_min(dth_ODE_param_std),
        },
        "ODE_GRF": {
            "EON": pad_and_get_min(eth_ODE_GRF),
            "EON Std": pad_and_get_min(eth_ODE_GRF_std),
            "DON": pad_and_get_min(dth_ODE_GRF),
            "DON Std": pad_and_get_min(dth_ODE_GRF_std),
        },
        "PDE_param": {
            "EON": pad_and_get_min(eth_PDE_param),
            "EON Std": pad_and_get_min(eth_PDE_param_std),
            "DON": pad_and_get_min(dth_PDE_param),
            "DON Std": pad_and_get_min(dth_PDE_param_std),
        },
        "PDE_GRF": {
            "EON": pad_and_get_min(eth_PDE_GRF),
            "EON Std": pad_and_get_min(eth_PDE_GRF_std),
            "DON": pad_and_get_min(dth_PDE_GRF),
            "DON Std": pad_and_get_min(dth_PDE_GRF_std),
        },
    }
    with TEXT_LOWEST_PATH.open("w") as fp:
        for key, metrics in lowest_values_data.items():
            fp.write(f"Category: {key}\n")
            for model, values in metrics.items():
                vals_list = np.asarray(values).tolist()
                fp.write(f"  Model: {model}, Lowest Values: {vals_list}\n")
            fp.write("\n")
    print(f"Lowest values data saved to {TEXT_LOWEST_PATH}")
    # Save train/val histories and timings
    datatrain = [
        (eth_ODE_param, evh_ODE_param, dth_ODE_param, dvh_ODE_param,
         eth_ODE_param_std, evh_ODE_param_std, dth_ODE_param_std, dvh_ODE_param_std),
        (eth_ODE_GRF, evh_ODE_GRF, dth_ODE_GRF, dvh_ODE_GRF,
         eth_ODE_GRF_std, evh_ODE_GRF_std, dth_ODE_GRF_std, dvh_ODE_GRF_std),
        (eth_PDE_param, evh_PDE_param, dth_PDE_param, dvh_PDE_param,
         eth_PDE_param_std, evh_PDE_param_std, dth_PDE_param_std, dvh_PDE_param_std),
        (eth_PDE_GRF, evh_PDE_GRF, dth_PDE_GRF, dvh_PDE_GRF,
         eth_PDE_GRF_std, evh_PDE_GRF_std, dth_PDE_GRF_std, dvh_PDE_GRF_std),
    ]
    save_pickle(datatrain, DATATRAIN_PATH)
    print(f"datatrain saved to {DATATRAIN_PATH}")
    datatime = [
        (dtt_ODE_param, ett_ODE_param, dpt_ODE_param, ept_ODE_param, dtt_ODE_param_std, ett_ODE_param_std, dpt_ODE_param_std, ept_ODE_param_std),
        (dtt_ODE_GRF, ett_ODE_GRF, dpt_ODE_GRF, ept_ODE_GRF, dtt_ODE_GRF_std, ett_ODE_GRF_std, dpt_ODE_GRF_std, ept_ODE_GRF_std),
        (dtt_PDE_param, ett_PDE_param, dpt_PDE_param, ept_PDE_param, dtt_PDE_param_std, ett_PDE_param_std, dpt_PDE_param_std, ept_PDE_param_std),
        (dtt_PDE_GRF, ett_PDE_GRF, dpt_PDE_GRF, ept_PDE_GRF, dtt_PDE_GRF_std, ett_PDE_GRF_std, dpt_PDE_GRF_std, ept_PDE_GRF_std),
    ]
    save_pickle(datatime, DATATIME_PATH)
    print(f"datatime saved to {DATATIME_PATH}")

if __name__ == "__main__":
    main()

