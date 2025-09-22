"""
General plot script for comparing EON and DON models.
Generates bar and error plots for generalization error, training error, and timing.

Author: Jari Beysen
"""

# Imports
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from skopt import gp_minimize
from skopt.space import Real, Integer

# Local imports
from EON import *
from DON import *
from PDE import *
from EON_train import *
from DON_train import *

# Constants and configuration
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
})

ECAI = False
WIDTH = 6.0017 if not ECAI else 3.40457
BASE_COLORS = {'ab': 'blue', 'cd': 'red'}
LINESTYLES = {'a': '-', 'b': '-.', 'c': '-', 'd': '-.'}
MOSAIC = """
    ab
    cd
"""

def generate_shades(base_color, num_shades=3):
    """Generate shades of a base color."""
    rgb = to_rgb(base_color)
    return [tuple(x * f for x in rgb) for f in np.linspace(0.2, 1, num_shades)]

ab_shades = generate_shades(BASE_COLORS['ab'])
cd_shades = generate_shades(BASE_COLORS['cd'])

def get_generalization_error(train, val):
    """Compute generalization error."""
    min_tr_index = np.argmin(train)
    min_tr_value = train[min_tr_index]
    min_val_value = val[min_tr_index]
    return np.abs(min_tr_value - min_val_value) / min_tr_value

def set_log_ticks(ax, y_min, y_max):
    """Set log scale ticks for an axis."""
    exp_min = int(np.floor(np.log10(y_min))) if y_min > 0 else 0
    exp_max = int(np.ceil(np.log10(y_max)))
    ticks = [10.0**i for i in np.arange(exp_min, exp_max + 1)[::2]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{i}}}$" for i in np.arange(exp_min, exp_max + 1)[::2]])

# Load data
with open("datatrain.pkl", "rb") as f:
    datatrain = pickle.load(f)
with open("datatime.pkl", "rb") as f:
    datatime = pickle.load(f)

# Process generalization and training errors
DONGEN, EONGEN, EONTRAIN, DONTRAIN = [], [], [], []
for (eth, evh, dth, dvh, eths, evhs, dths, dvhs) in datatrain:
    donsub, eonsub, dontrainsub, eontrainsub = [], [], [], []
    for (a, b, c, d, ass, bs, cs, ds) in zip(eth, evh, dth, dvh, eths, evhs, dths, dvhs):
        eonsub.append(float(get_generalization_error(a, b)))
        donsub.append(float(get_generalization_error(c, d)))
        eontrainsub.append(np.min(a))
        dontrainsub.append(np.min(c))
    EONGEN.append(eonsub)
    DONGEN.append(donsub)
    EONTRAIN.append(eontrainsub)
    DONTRAIN.append(dontrainsub)

print(f'EON gen error = {EONGEN}, mean test = {np.median(EONGEN)}')
print(f'DON gen error = {DONGEN}, mean test = {np.median(DONGEN)}')

# Prepare arrays for plotting
EONGEN_arr = np.array(EONGEN)
DONGEN_arr = np.array(DONGEN)
EONTRAIN_arr = np.array(EONTRAIN)
DONTRAIN_arr = np.array(DONTRAIN)
n_groups = EONGEN_arr.shape[1] if EONGEN_arr.ndim > 1 else len(EONGEN_arr)
x = np.arange(n_groups)
w = 0.4

if EONGEN_arr.ndim > 1:
    eon_vals = np.mean(EONGEN_arr, axis=0)
    don_vals = np.mean(DONGEN_arr, axis=0)
    eon_train_vals = np.mean(EONTRAIN_arr, axis=0)
    don_train_vals = np.mean(DONTRAIN_arr, axis=0)
else:
    eon_vals = EONGEN_arr
    don_vals = DONGEN_arr
    eon_train_vals = EONTRAIN_arr
    don_train_vals = DONTRAIN_arr

eon_colors = ab_shades if len(ab_shades) >= n_groups else ['blue'] * n_groups
don_colors = cd_shades if len(cd_shades) >= n_groups else ['red'] * n_groups

# Plot: Training and generalization error comparison
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * .25), sharex=True)

# Training error
axs[0].bar(x - w/2, eon_train_vals, w, label='EON', color=eon_colors[:n_groups], alpha=0.7)
axs[0].bar(x + w/2, don_train_vals, w, label='DON', color=don_colors[:n_groups], alpha=0.7)
axs[0].set_ylabel(r'$\mathcal{L}_{train}$')
axs[0].set_yscale('log')
axs[0].text(-0.05, 1.115, r"$\mathbf{a)}$", transform=axs[0].transAxes, ha='right', va='bottom')
axs[0].set_xlabel("Model", labelpad=2)
axs[0].set_yticks([1e0, 1e-1, 1e-2])
axs[0].grid()
axs[0].set_axisbelow(True)

# Generalization error
axs[1].bar(x - w/2, eon_vals, w, label='EON', color=eon_colors[:n_groups], alpha=0.7)
axs[1].bar(x + w/2, don_vals, w, label='DON', color=don_colors[:n_groups], alpha=0.7)
axs[1].set_ylabel(r'$(\mathcal{L}_{test}-\mathcal{L}_{train})/\mathcal{L}_{train}$')
axs[1].set_xticks(x)
axs[1].set_xticklabels([f'({i+1})' for i in x])
axs[1].text(-0.05, 1.115, r"$\mathbf{b)}$", transform=axs[1].transAxes, ha='right', va='bottom')
axs[1].set_xlabel("Model", labelpad=2)
axs[1].grid()
axs[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig('Gencomp.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

# Plot: Error curves for each model
fig, axs = plt.subplot_mosaic(MOSAIC, figsize=(WIDTH, WIDTH * .6), sharey=True)

for (key, ax), (eth, evh, dth, dvh, eths, evhs, dths, dvhs) in zip(axs.items(), datatrain):
    y_min, y_max = 1e-6, 1e2
    ax.set_yscale('log')
    ax.set_ylim((y_min, y_max))
    set_log_ticks(ax, y_min, y_max)
    ax_top = ax.twiny()
    ax_top.set_yscale('log')
    ax_top.set_ylim((y_min, y_max))
    set_log_ticks(ax_top, y_min, y_max)
    n_points = 10
    offset = 1
    for (a, b, c, d, ass, bs, cs, ds, shade1, shade2) in zip(eth, evh, dth, dvh, eths, evhs, dths, dvhs, ab_shades, cd_shades):
        idxs_a = np.linspace(0, len(a) - 1, n_points) - offset * .5
        idxs_b = np.linspace(0, len(b) - 1, n_points) - offset * 0.25
        idxs_c = 10**np.linspace(0, np.log10(len(c) - 1), n_points) + offset * 0.25
        idxs_d = 10**np.linspace(0, np.log10(len(d) - 1), n_points) + offset * .5
        ax.errorbar(idxs_a, a[idxs_a.astype(int)], yerr=ass[idxs_a.astype(int)], fmt='-', color=shade1, capsize=0, linewidth=1, markersize=1, zorder=4)
        ax.errorbar(idxs_b, b[idxs_b.astype(int)], yerr=bs[idxs_b.astype(int)], fmt=':', color=shade1, capsize=0, linewidth=1, markersize=1, zorder=4)
        ax_top.errorbar(idxs_c, c[idxs_c.astype(int)], yerr=cs[idxs_c.astype(int)], fmt='-', color=shade2, capsize=0, linewidth=1, markersize=1, zorder=4)
        ax_top.errorbar(idxs_d, d[idxs_d.astype(int)], yerr=ds[idxs_d.astype(int)], fmt=':', color=shade2, capsize=0, linewidth=1, markersize=1, zorder=4)
    if key == 'a':
        ax.set_ylabel(r"$\mathcal{L}$", labelpad=1)
    if key == 'c':
        ax.set_xlabel("iterations", labelpad=1)
        ax.set_ylabel(r"$\mathcal{L}$", labelpad=1)
    if key == 'd':
        ax.set_xlabel("iterations", labelpad=1)
    ax_top.set_xscale('log')
    ax_top.text(-0.05, 1.115, f"$\mathbf{{{key})}}$", transform=ax_top.transAxes, ha='right', va='bottom')
    ax.grid()
    ax.set_axisbelow(True)

plt.tight_layout()
if not ECAI:
    fig.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.15, wspace=0.25, hspace=0.5)
    plt.savefig('general_comp.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
else:
    fig.subplots_adjust(left=0.15, right=0.95, top=0.87, bottom=0.18, wspace=0.25, hspace=.6)
    plt.savefig('general_comp2.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

print('-' * 95)

# Plot: Timing comparison
fig, axs = plt.subplot_mosaic(MOSAIC, figsize=(WIDTH, WIDTH * 0.4), sharex=True, sharey=True)
exp_min = int(np.floor(np.log10(np.min(datatime))))
exp_max = int(np.ceil(np.log10(np.max(datatime))))
for (key, ax), (ttd, tte, ptd, pte, ttds, ttes, ptds, ptes) in zip(axs.items(), datatime):
    ax.grid()
    ax.set_axisbelow(True)
    y_min, y_max = 1e-2, 1e4
    x_labels = [r'(1)', r'(2)', r'(3)', r'(1)', r'(2)', r'(3)']
    pos = np.arange(6)
    width = 0.4
    ax.bar(pos - width/2, np.concatenate((ttd, ptd)), width=width, alpha=0.7, color=[*cd_shades, *cd_shades], linewidth=1, label='DON')
    ax.bar(pos + width/2, np.concatenate((tte, pte)), width=width, alpha=0.7, color=[*ab_shades, *ab_shades], linewidth=1, label='EON')
    ax.errorbar(pos - width/2, np.concatenate((ttd, ptd)), yerr=np.concatenate((ttds, ptds)), fmt='none', ecolor='black', elinewidth=1, capsize=2, zorder=10)
    ax.errorbar(pos + width/2, np.concatenate((tte, pte)), yerr=np.concatenate((ttes, ptes)), fmt='none', ecolor='black', elinewidth=1, capsize=2, zorder=10)
    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim((y_min, y_max))
    ax.set_yscale('log')
    set_log_ticks(ax, y_min, y_max)
    ax.text(-0.05, 1.115, f"$\mathbf{{{key})}}$", transform=ax.transAxes, ha='right', va='bottom')
    if key in ('a', 'b'):
        ax.text(.35, 1, "Training", transform=ax.transAxes, ha='right', va='bottom')
        ax.text(.87, 1, "Prediction", transform=ax.transAxes, ha='right', va='bottom')
    if key == 'c':
        ax.set_ylabel(r"Time (s)", labelpad=1)
    if key in ('c', 'd'):
        ax.set_xlabel("Model", labelpad=2)
    ax.vlines(x=2.5, ymin=y_min, ymax=y_max, color='black', linestyles='--', linewidth=1.5)

plt.tight_layout()
if not ECAI:
    fig.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.15, wspace=0.15, hspace=0.3)
    plt.savefig('time_comp.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
else:
    fig.subplots_adjust(left=0.15, right=0.95, top=0.87, bottom=0.15, wspace=0.25, hspace=0.4)
    plt.savefig('time_comp2.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

plt.show()

