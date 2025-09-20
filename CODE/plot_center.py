"""
plot_center.py

Visualization and analysis utilities for EXTREMONET experiments.
Includes plotting functions for general metrics, out-of-distribution (OOD) tests,
and curve fitting for error/time relations.

Author: Jari Beysen
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

from EON import *
from DON import *
from ODE import *
from PDE import *

# Constants and Style
ECAI = False
NT_TEST = 50

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

WIDTH = 6.0017 if not ECAI else 3.40457

# Utility Functions
# ------------------------------------------------------------------------------------------------------------

def calculate_max(*data):
    """Return the maximum value from nested arrays."""
    all_data = np.concatenate([np.concatenate([np.array(di).flatten() for di in d]) for d in data])
    return np.max(all_data)

def calculate_min(*data):
    """Return the minimum value from nested arrays."""
    all_data = np.concatenate([np.concatenate([np.array(di).flatten() for di in d]) for d in data])
    return np.min(all_data)

def set_log_ticks(ax, y_min, y_max, max_ticks=3):
    """Set log-scale ticks and labels for an axis."""
    exp_min = int(np.floor(np.log10(y_min))) if y_min > 0 else 0
    exp_max = int(np.ceil(np.log10(y_max)))
    n_ticks = min(max_ticks, exp_max - exp_min + 1)
    if n_ticks <= 1:
        ticks = [10.0 ** exp_min]
        labels = [f"$10^{{{exp_min}}}$"]
    else:
        exponents = np.linspace(exp_min, exp_max, n_ticks, dtype=int)
        ticks = [10.0 ** i for i in exponents]
        labels = [f"$10^{{{i}}}$" for i in exponents]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.get_yaxis().set_minor_locator(ticker.NullLocator())

def propgate_error(xs, xerrs):
    """Propagate error for mean of xs with uncertainties xerrs."""
    xs_arr = np.array(xs)
    xerrs_arr = np.array(xerrs)
    finite = np.isfinite(xs_arr) & np.isfinite(xerrs_arr)
    xs_clean = xs_arr[finite]
    xerrs_clean = xerrs_arr[finite]
    xm = np.mean(xs_clean)
    xerr = np.sqrt(np.mean(xerrs_clean ** 2) + np.std(xs_clean) ** 2)
    return xm, xerr

def propgate_error_min(xs, xerrs, ys, yerrs):
    """Propagate error for mean of xs-ys with uncertainties."""
    xs_arr = np.array(xs)
    xerrs_arr = np.array(xerrs)
    ys_arr = np.array(ys)
    yerrs_arr = np.array(yerrs)
    finite = np.isfinite(xs_arr) & np.isfinite(xerrs_arr) & np.isfinite(ys_arr) & np.isfinite(yerrs_arr)
    xs_clean = xs_arr[finite]
    xerrs_clean = xerrs_arr[finite]
    ys_clean = ys_arr[finite]
    yerrs_clean = yerrs_arr[finite]
    xm = np.mean(xs_clean - ys_clean)
    xerr = np.sqrt(np.mean(xerrs_clean ** 2) + np.mean(yerrs_clean ** 2))
    return xm, xerr

def propgate_error_general(xs, xerrs, ys, yerrs):
    """Propagate error for mean of (ys-xs)/xs with uncertainties."""
    xs_arr = np.array(xs)
    xerrs_arr = np.array(xerrs)
    ys_arr = np.array(ys)
    yerrs_arr = np.array(yerrs)
    finite = np.isfinite(xs_arr) & np.isfinite(xerrs_arr) & np.isfinite(ys_arr) & np.isfinite(yerrs_arr)
    xs_clean = xs_arr[finite]
    xerrs_clean = xerrs_arr[finite]
    ys_clean = ys_arr[finite]
    yerrs_clean = yerrs_arr[finite]
    xm = np.mean((ys_clean - xs_clean) / xs_clean)
    xerr = xm * np.sqrt(np.mean((xerrs_clean / ys_clean) ** 2) + np.mean((xs_clean * yerrs_clean / (ys_clean ** 2)) ** 2))
    return xm, xerr

# Plotting Functions
# ------------------------------------------------------------------------------------------------------------

def general_plotter(xdata, ydata, name, xlabel, ylabel, ylabeltime, logscale=False, halfsize=True):
    """
    Plot general metrics and timing for a given experiment.
    """
    trdata = ydata[:, :, :, 0]
    tedata = ydata[:, :, :, 1]
    trtime = ydata[:, :, :, 2]
    tetime = ydata[:, :, :, 3]
    mosaic = "ab\ncd"
    figsize = (WIDTH * .48, WIDTH * 0.3) if halfsize else (WIDTH, WIDTH * 0.5)

    # Error plots
    fig, axs = plt.subplot_mosaic(mosaic, figsize=figsize, sharex=True, sharey=False)
    for (key, ax), traind, testd in zip(axs.items(), trdata, tedata):
        y_max = calculate_max(np.array(traind[3]), np.array(testd[3]))
        y_min = calculate_min(np.array(traind[2]), np.array(testd[2]))
        ax.set_yscale("log")
        ax.set_ylim((y_min, y_max))
        set_log_ticks(ax, y_min, y_max)
        ax.plot(xdata, traind[0], color="blue", label="Mean")
        ax.fill_between(xdata, np.array(traind[0]) - np.array(traind[1]), 
                        np.array(traind[0]) + np.array(traind[1]), color="blue", alpha=0.2, label="Std")
        ax.plot(xdata, testd[0], color="black", label="Mean")
        ax.fill_between(xdata, np.array(testd[0]) - np.array(testd[1]), 
                        np.array(testd[0]) + np.array(testd[1]), color="black", alpha=0.2, label="Std")
        ax.plot(xdata, traind[2], color="blue", linestyle="--", label="Max")
        ax.plot(xdata, traind[3], color="blue", linestyle=":", label="Min")
        ax.plot(xdata, testd[2], color="black", linestyle="--", label="Max")
        ax.plot(xdata, testd[3], color="black", linestyle=":", label="Min")
        ax.text(-0.05, 1.115, f"$\mathbf{{{key})}}$", transform=ax.transAxes, ha='right', va='bottom')
        if logscale:
            ax.set_xscale('log')
        if key == "c":
            ax.set_ylabel(ylabel, labelpad=1)
            ax.set_xlabel(xlabel, labelpad=1)
        if key == "d":
            ax.set_xlabel(xlabel, labelpad=1)
        if key == "a":
            ax.set_ylabel(ylabel, labelpad=1)
            legend_size = 4 if halfsize else 6
            ncol = 2 if halfsize else 4
            ax.legend(fontsize=legend_size, frameon=False, handlelength=1, handletextpad=0.3, borderpad=0.1, fancybox=False, ncol=ncol)
    plt.tight_layout()
    if ECAI:
        fig.subplots_adjust(left=0.14, right=0.99, top=0.87, bottom=0.18, wspace=0.3, hspace=.5)
        plt.savefig(name + "2.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.15, wspace=0.35, hspace=0.6 if halfsize else 0.45)
        plt.savefig(name + ".svg", dpi=300, bbox_inches="tight", pad_inches=0.05)

    # Timing plots
    fig, axs = plt.subplot_mosaic(mosaic, figsize=figsize, sharex=True, sharey=False)
    for (key, ax), traind, testd in zip(axs.items(), trtime, tetime):
        y_max = calculate_max(np.array(traind[3]), np.array(testd[3]))
        y_min = calculate_min(np.array(traind[2]), np.array(testd[2]))
        ax.set_yscale("log")
        ax.set_ylim((y_min, y_max))
        set_log_ticks(ax, y_min, y_max)
        ax.plot(xdata, traind[0], color="blue", label="Mean")
        ax.fill_between(xdata, np.array(traind[0]) - np.array(traind[1]), 
                        np.array(traind[0]) + np.array(traind[1]), color="blue", alpha=0.2, label="Std")
        ax.plot(xdata, testd[0], color="black", label="Mean")
        ax.fill_between(xdata, np.array(testd[0]) - np.array(testd[1]), 
                        np.array(testd[0]) + np.array(testd[1]), color="black", alpha=0.2, label="Std")
        ax.plot(xdata, traind[2], color="blue", linestyle="--", label="Max")
        ax.plot(xdata, traind[3], color="blue", linestyle=":", label="Min")
        ax.plot(xdata, testd[2], color="black", linestyle="--", label="Max")
        ax.plot(xdata, testd[3], color="black", linestyle=":", label="Min")
        ax.text(-0.05, 1.115, f"$\mathbf{{{key})}}$", transform=ax.transAxes, ha='right', va='bottom')
        if key == "c":
            ax.set_ylabel(ylabeltime, labelpad=1)
            ax.set_xlabel(xlabel, labelpad=1)
        if key == "d":
            ax.set_xlabel(xlabel, labelpad=1)
        if key == "a":
            ax.set_ylabel(ylabeltime, labelpad=1)
            legend_size = 4 if halfsize else 6
            ncol = 2 if halfsize else 4
            ax.legend(fontsize=legend_size, frameon=False, handlelength=1, handletextpad=0.3, borderpad=0.1, fancybox=False, ncol=ncol)
    plt.tight_layout()
    if ECAI:
        fig.subplots_adjust(left=0.14, right=.99, top=0.87, bottom=0.18, wspace=0.35, hspace=.5)
        plt.savefig(name + "2_time.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)
    else:
        fig.subplots_adjust(left=0.1, right=0.95, top=0.87, bottom=0.15, wspace=0.35, hspace=0.6 if halfsize else 0.45)
        plt.savefig(name + "_time.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)

def OOD1_plotter(lrange, ydata, name, xlabel1, ylabel1, xlabel2, ylabel2):
    """
    Plot OOD1 experiment results.
    """
    colors = [(0.0, 0.0, 1), (1.0, 1.0, 1.0), (1, 0.0, 0.0)]
    cmap_custom = LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    trdata = ydata[:, :, :, :, 0]
    tedata = ydata[:, :, :, :, 1]
    trgrido = np.mean(trdata[0, 0], axis=0)
    trgridp = np.mean(trdata[1, 0], axis=0)
    tegrido = tedata[0, 0] / trgrido
    tegridp = tedata[1, 0] / trgridp
    trgrido_std = np.mean(trdata[0, 1], axis=0)
    trgridp_std = np.mean(trdata[1, 1], axis=0)
    trgrido_min = np.min(trdata[0, 2], axis=0)
    trgridp_min = np.min(trdata[1, 2], axis=0)
    trgrido_max = np.max(trdata[0, 3], axis=0)
    trgridp_max = np.max(trdata[1, 3], axis=0)
    X, Y = np.meshgrid(lrange, lrange)
    fig1, axs1 = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.45), sharex=False, sharey=False)
    vmax = np.abs(max(np.log10(np.max(np.abs(tegrido))), np.log10(np.max(tegridp))))
    vmin = -vmax
    contour1 = axs1[0].contourf(X, Y, np.log10(tegrido), levels=len(lrange), cmap=cmap_custom, vmin=vmin, vmax=vmax)
    axs1[0].plot(lrange, lrange, linestyle=':', color='black')
    axs1[0].text(-0.05, 1.115, r"$\mathbf{a)}$", transform=axs1[0].transAxes, ha='right', va='bottom')
    axs1[0].set_yscale('log')
    axs1[0].set_xscale('log')
    axs1[0].set_ylabel(ylabel1, labelpad=1)
    axs1[0].set_xlabel(xlabel1, labelpad=1)
    contour2 = axs1[1].contourf(X, Y, np.log10(tegridp), levels=len(lrange), cmap=cmap_custom, vmin=vmin, vmax=vmax)
    axs1[1].plot(lrange, lrange, linestyle=':', color='black')
    axs1[1].text(-0.05, 1.115, r"$\mathbf{b)}$", transform=axs1[1].transAxes, ha='right', va='bottom')
    axs1[1].set_yscale('log')
    axs1[1].set_xscale('log')
    axs1[1].set_xlabel(xlabel1, labelpad=1)
    fig1.colorbar(contour2, ax=axs1[1], orientation='vertical', fraction=0.05, pad=0.02)
    fig1.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.2)
    plt.savefig(name + '1.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

    fig2, axs2 = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.3), sharex=True, sharey=True)
    axs2[0].plot(lrange, trgrido, color='blue', label="Mean")
    axs2[0].fill_between(lrange, trgrido - trgrido_std, trgrido + trgrido_std, color="blue", alpha=0.2, label="Std")
    axs2[0].plot(lrange, trgrido_max, color="blue", linestyle="--", label="Max")
    axs2[0].plot(lrange, trgrido_min, color="blue", linestyle=":", label="Min")
    axs2[0].set_yscale('log')
    axs2[0].set_xscale('log')
    axs2[0].text(-0.05, 1.115, r"$\mathbf{a)}$", transform=axs2[0].transAxes, ha='right', va='bottom')
    axs2[0].set_ylabel(ylabel2, labelpad=1)
    axs2[0].set_xlabel(xlabel2, labelpad=1)
    axs2[0].legend(fontsize=6, frameon=False, handlelength=1, handletextpad=0.3, borderpad=0.1, fancybox=False, ncol=2)
    axs2[1].plot(lrange, trgridp, color='blue')
    axs2[1].fill_between(lrange, trgridp - trgridp_std, trgridp + trgridp_std, color="blue", alpha=0.2, label="Std")
    axs2[1].plot(lrange, trgridp_max, color="blue", linestyle="--", label="Max")
    axs2[1].plot(lrange, trgridp_min, color="blue", linestyle=":", label="Min")
    axs2[1].set_yscale('log')
    axs2[1].set_xscale('log')
    axs2[1].text(-0.05, 1.115, r"$\mathbf{b)}$", transform=axs2[1].transAxes, ha='right', va='bottom')
    axs2[1].set_xlabel(xlabel2, labelpad=1)
    fig2.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.2)
    plt.savefig(name + '2.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

def OOD2_plotter(lrange1, lrange2, ydata, name, xlabel1, ylabel1, xlabel2, ylabel2):
    """
    Plot OOD2 experiment results.
    """
    colors = [(0.0, 0.0, 1), (1.0, 1.0, 1.0), (1, 0.0, 0.0)]
    cmap_custom = LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    tedata = ydata[:, :, :, :, 0]
    trdata = ydata[:, :, :, :, 1]
    gridp = tedata[0, 0] / np.mean(trdata[0, 0], axis=0)
    trgridp = np.mean(trdata[0, 0], axis=0)
    trgridp_std = np.mean(trdata[0, 1], axis=0)
    trgridp_min = np.min(trdata[0, 2], axis=0)
    trgridp_max = np.max(trdata[0, 3], axis=0)
    X, Y = np.meshgrid(lrange1, lrange2)
    fig1, axs1 = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.45))
    vmax = np.max(np.abs(np.log10(gridp)))
    vmin = -vmax
    axs1[0].plot(lrange1, trgridp, color='blue', label="Mean")
    axs1[0].fill_between(lrange1, trgridp - trgridp_std, trgridp + trgridp_std, color="blue", alpha=0.2, label="Std")
    axs1[0].plot(lrange1, trgridp_max, color="blue", linestyle="--", label="Max")
    axs1[0].plot(lrange1, trgridp_min, color="blue", linestyle=":", label="Min")
    axs1[0].set_yscale('log')
    axs1[0].set_xscale('log')
    axs1[0].text(-0.05, 1.115, r"$\mathbf{a)}$", transform=axs1[0].transAxes, ha='right', va='bottom')
    axs1[0].set_ylabel(ylabel1, labelpad=1)
    axs1[0].set_xlabel(xlabel1, labelpad=1)
    axs1[0].legend(fontsize=6, frameon=False, handlelength=1, handletextpad=0.3, borderpad=0.1, fancybox=False, ncol=2)
    contour1 = axs1[1].contourf(X, Y, np.log10(gridp), levels=len(lrange1), cmap=cmap_custom, vmax=vmax, vmin=vmin)
    axs1[1].text(-0.05, 1.115, r"$\mathbf{b)}$", transform=axs1[1].transAxes, ha='right', va='bottom')
    axs1[1].set_xscale('log')
    axs1[1].set_ylabel(ylabel2, labelpad=1)
    axs1[1].set_xlabel(xlabel2, labelpad=1)
    axs1[1].get_yaxis().set_minor_locator(ticker.NullLocator())
    fig1.colorbar(contour1, ax=axs1[1], orientation='vertical', fraction=0.05, pad=0.02)
    fig1.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.2)
    plt.savefig(name + '.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)

# Analysis Functions
# ------------------------------------------------------------------------------------------------------------

def find_error_relation(x, data):
    """Fit error relation and print fit parameters."""
    print('------------------------------------------------------------------------------------------------------------')
    print('Error relation')
    def xexpon(x, a, b, c):
        return a * x**b + c
    params_tr, covs_tr, params_te, covs_te = [], [], [], []
    for ind, system in enumerate(data):
        ytr = system[0, :, 0]
        ytrstd = system[1, :, 0]
        yte = system[0, :, 1]
        ytestd = system[1, :, 1]
        fit_tr, cov_tr = curve_fit(xexpon, x, ytr, sigma=ytrstd, maxfev=100000, p0=[1, -1, 0])
        fit_te, cov_te = curve_fit(xexpon, x, yte, sigma=ytestd, maxfev=100000, p0=[1, -1, 0])
        params_tr.append(fit_tr)
        covs_tr.append(np.diag(cov_tr))
        params_te.append(fit_te)
        covs_te.append(np.diag(cov_te))
        brat = propgate_error_min(fit_te[1], cov_te[1, 1], fit_tr[1], cov_tr[1, 1])
        crat = propgate_error_general(fit_tr[2], cov_tr[2, 2], fit_te[2], cov_te[2, 2])
        print('------------------------------------------------------------------------------------------------------------')
        print(f"System: {ind}")
        print(f"Training Xexponent fit:  a = {fit_tr[0]:.2e}, b = {fit_tr[1]:.2e}, c = {fit_tr[2]:.2e}")
        print(f"Training Xexponent fit cov:  a = {cov_tr[0,0]:.2e}, b = {cov_tr[1,1]:.2e}, c = {cov_tr[2,2]:.2e}")
        print(f"Testing Xexponent fit:  a = {fit_te[0]:.2e}, b = {fit_te[1]:.2e}, c = {fit_te[2]:.2e}")
        print(f"Testing Xexponent fit cov:  a = {cov_te[0,0]:.2e}, b = {cov_te[1,1]:.2e}, c = {cov_te[2,2]:.2e}")
        print(f'bte-btr = {brat[0]:.2e} ± {brat[1]:.2e}')
        print(f'geninf = {crat[0]:.2e} ± {crat[1]:.2e}')
    params_tr = np.array(params_tr)
    covs_tr = np.array(covs_tr)
    params_te = np.array(params_te)
    covs_te = np.array(covs_te)
    mean_a_tr, std_a_tr = propgate_error(params_tr[:, 0], covs_tr[:, 0])
    mean_b_tr, std_b_tr = propgate_error(params_tr[:, 1], covs_tr[:, 1])
    mean_c_tr, std_c_tr = propgate_error(params_tr[:, 2], covs_tr[:, 2])
    mean_a_te, std_a_te = propgate_error(params_te[:, 0], covs_te[:, 0])
    mean_b_te, std_b_te = propgate_error(params_te[:, 1], covs_te[:, 1])
    mean_c_te, std_c_te = propgate_error(params_te[:, 2], covs_te[:, 2])
    print('------------------------------------------------------------------------------------------------------------')
    print(f"Mean Training Xexponent a:  {mean_a_tr:.2e} ± {std_a_tr:.2e}")
    print(f"Mean Training Xexponent b:  {mean_b_tr:.2e} ± {std_b_tr:.2e}")
    print(f"Mean Training Xexponent c:  {mean_c_tr:.2e} ± {std_c_tr:.2e}")
    print(f"Mean Testing Xexponent a:   {mean_a_te:.2e} ± {std_a_te:.2e}")
    print(f"Mean Testing Xexponent b:   {mean_b_te:.2e} ± {std_b_te:.2e}")
    print(f"Mean Testing Xexponent c:   {mean_c_te:.2e} ± {std_c_te:.2e}")

def find_time_relation(x, data):
    """Fit time relation and print fit parameters."""
    print('------------------------------------------------------------------------------------------------------------')
    print('Time relation')
    def xexpon(x, a, b, c):
        return a * x**b + c
    b_values_xexpon_tr, cov_values_xexpon_tr = [], []
    b_values_xexpon_te, cov_values_xexpon_te = [], []
    for ind, system in enumerate(data):
        ytr = system[0, :, 2]
        ytrstd = system[1, :, 2]
        yte = system[0, :, 3]
        ytestd = system[1, :, 3]
        fit2_tr, cov2_tr = curve_fit(xexpon, x, ytr, sigma=ytrstd, maxfev=100000, p0=[1, 1, 0])
        b_values_xexpon_tr.append(fit2_tr[1])
        cov_values_xexpon_tr.append(cov2_tr[1, 1])
        fit2_te, cov2_te = curve_fit(xexpon, x, yte, sigma=ytestd, maxfev=100000, p0=[1, 1, 0])
        b_values_xexpon_te.append(fit2_te[1])
        cov_values_xexpon_te.append(cov2_te[1, 1])
        print('------------------------------------------------------------------------------------------------------------')
        print(f"System: {ind}")
        print(f"Training Xexponent fit:  b = {fit2_tr[1]:.2e}")
        print(f"Training Xexponent fit cov:  b = {cov2_tr[1,1]:.2e}")
        print(f"Testing Xexponent fit:  b = {fit2_te[1]:.2e}")
        print(f"Testing Xexponent fit cov:  b = {cov2_te[1,1]:.2e}")
    mean_b_xexpon_tr, std_b_xexpon_tr = propgate_error(b_values_xexpon_tr, cov_values_xexpon_tr)
    mean_b_xexpon_te, std_b_xexpon_te = propgate_error(b_values_xexpon_te, cov_values_xexpon_te)
    print('------------------------------------------------------------------------------------------------------------')
    print(f"Mean Training Xexponent b:  {mean_b_xexpon_tr:.2e} ± {std_b_xexpon_tr:.2e}")
    print(f"Mean Testing Xexponent b:   {mean_b_xexpon_te:.2e} ± {std_b_xexpon_te:.2e}")

def find_error_ratio_relation(x, data):
    """Fit error ratio relation and print fit parameters."""
    print('------------------------------------------------------------------------------------------------------------')
    print('Error ratio relation')
    def xexpon(x, a, b, c):
        return a * x**b + c
    b_values_xexpon_tr, cov_values_xexpon_tr = [], []
    for ind, system in enumerate(data):
        ytr = system[0, :, 0]
        ytrstd = system[1, :, 0]
        yte = system[0, :, 1]
        ytestd = system[1, :, 1]
        fit2_tr, cov2_tr = curve_fit(
            xexpon, x, yte / ytr,
            sigma=np.abs(yte / ytr) * np.sqrt((ytrstd / ytr) ** 2 + (ytestd / yte) ** 2),
            maxfev=100000, p0=[1, 1, 0]
        )
        b_values_xexpon_tr.append(fit2_tr[1])
        cov_values_xexpon_tr.append(cov2_tr[1, 1])
        print('------------------------------------------------------------------------------------------------------------')
        print(f"System: {ind}")
        print(f"Ratio Xexponent fit:  b = {fit2_tr[1]:.2e}")
        print(f"Ratio Xexponent fit cov:  b = {cov2_tr[1,1]:.2e}")
    mean_b_xexpon_tr, std_b_xexpon_tr = propgate_error(b_values_xexpon_tr, cov_values_xexpon_tr)
    print('------------------------------------------------------------------------------------------------------------')
    print(f"Mean ratio Xexponent b:  {mean_b_xexpon_tr:.2e} ± {std_b_xexpon_tr:.2e}")

def find_grid_linear(x1, x2, y):
    """Fit grid-linear relation and print fit parameters."""
    print('------------------------------------------------------------------------------------------------------------')
    print('grind-linear relation')
    def linear_func(X, a, b, c, d):
        x, y = X.T
        return a * x + b * y + c * x * y + d
    X1, X2 = np.meshgrid(x1, x2)
    x = np.hstack([X1.ravel().reshape(-1, 1), X2.ravel().reshape(-1, 1)])
    yr = y[0, :, :, 1] / y[0, :, :, 0]
    yrstd = np.abs(yr) * np.sqrt((y[1, :, :, 0] / y[0, :, :, 0]) ** 2 + (y[1, :, :, 1] / y[0, :, :, 1]) ** 2)
    y_flat = yr.ravel()
    ystd_flat = yrstd.ravel()
    popt, pcov = curve_fit(linear_func, x, y_flat, sigma=ystd_flat, maxfev=100000, p0=[-1, 0, 0, 1])
    print('------------------------------------------------------------------------------------------------------------')
    print('Grid linear relation')
    print(f"Linear fit:  a = {popt[0]:.2e} +- {np.sqrt(pcov[0,0]):.2e}")
    print(f"Linear fit:  b = {popt[1]:.2e} +- {np.sqrt(pcov[1,1]):.2e}")
    print(f"Linear fit:  c = {popt[2]:.2e} +- {np.sqrt(pcov[2,2]):.2e}")
    print(f"Linear fit:  d = {popt[3]:.2e} +- {np.sqrt(pcov[3,3]):.2e}")

# Main Execution
# ------------------------------------------------------------------------------------------------------------

print('WIDTH')
data = open_data("width_test.pkl")
values = np.linspace(1, 1000, NT_TEST).astype(int)
general_plotter(values, data, 'Width_comp', r'Width ($m$)', r'$\mathcal{L}$', r'Time ($s$)')
find_error_relation(values, data)
find_time_relation(values, data)

data = open_data("sensors_test.pkl")
values = np.linspace(0, 120, NT_TEST).astype(int)
general_plotter(values, data, 'Sensors_comp', r'Sensors ($r$)', r'$\mathcal{L}$', r'Time ($s$)')

print('DATA')
data = open_data("data_test.pkl")
values = np.linspace(100, 3000, NT_TEST)
general_plotter(values, data, 'Data_comp', r'$N$', r'$\mathcal{L}$', r'Time ($s$)')
find_error_ratio_relation(values, data)

data = open_data("st_test.pkl")
values = 10 ** np.linspace(-3, 2, NT_TEST)
general_plotter(values, data, 'St_comp', r'$\sigma_t$', r'$\mathcal{L}$', r'Time ($s$)', logscale=True)

data = open_data("sb_test.pkl")
values = 10 ** np.linspace(-3, 1, NT_TEST)
general_plotter(values, data, 'Sb_comp', r'$\sigma_b$', r'$\mathcal{L}$', r'Time ($s$)', logscale=True)

data = open_data("cb_test.pkl")
values = np.arange(0, 30, 1)
general_plotter(values, data, 'Cb_comp', r'$c_b$', r'$\mathcal{L}$', r'Time ($s$)')

data = open_data("OOD1_testEON.pkl")
values = 10 ** np.linspace(-2, 0, 6)
OOD1_plotter(values, data[:2], 'OOD1_compEON', r'$l$', r'$l$', r'$l$', r'$\mathcal{L}$')

data = open_data("OOD2_testEON.pkl")
values1 = 10 ** np.linspace(-2, 0, 6)
values2 = np.linspace(1e-10, 1, 6)
OOD2_plotter(values1, values2, data, 'OOD3_compEON', r'$l$', r'$\mathcal{L}$', r'$l$', r'$d$')
find_grid_linear(values1, values2, data[0])

data = open_data("OOD1_testDON.pkl")
values = 10 ** np.linspace(-2, 0, 6)
OOD1_plotter(values, data[:2], 'OOD1_compDON', r'$l$', r'$l$', r'$l$', r'$\mathcal{L}$')

data = open_data("OOD2_testDON.pkl")
values1 = 10 ** np.linspace(-2, 0, 6)
values2 = np.linspace(1e-10, 1, 6)
OOD2_plotter(values1, values2, data, 'OOD3_compDON', r'$l$', r'$\mathcal{L}$', r'$l$', r'$d$')
find_grid_linear(values1, values2, data[0])

plt.show()
