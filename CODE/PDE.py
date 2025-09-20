"""
PDE Timeseries Generation Module

This module provides utilities for generating timeseries data from PDEs using Gaussian random fields (GRF)
and parameterized fields. It supports parallel sample generation, sensor extraction, and data saving/loading.

Author: Jari Beysen
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import pickle

# Constants
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Kernel and Field Functions ---

def rbf_kernel(x1, x2, l):
    """Radial basis function (RBF) kernel."""
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * l ** 2))

def generate_grf(domain, l, num_samples=1):
    """Generate samples from a Gaussian random field (GRF) over a domain."""
    n = len(domain)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = rbf_kernel(domain[i], domain[j], l)
    mean = np.zeros(n)
    samples = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    return samples

def interpolate_grf(domain, grf_sample):
    """Interpolate GRF sample over domain."""
    return interp1d(domain, grf_sample, kind='cubic', fill_value="extrapolate")

def poly_field(x, a):
    """Polynomial field."""
    return np.sum([x ** i * a[i] for i in range(len(a))], axis=0)

def exp_field(x, a):
    """Exponential field."""
    return np.sum([(np.exp(-x ** 2 * i) - 1) * a[i] for i in range(len(a))], axis=0)

def sin_field(x, a):
    """Sinusoidal field."""
    return np.sum([np.sin(x * a[i] * np.pi) for i in range(len(a))], axis=0)

def sinc_field(x, a):
    """Sinc field."""
    return np.sinc(1 / a[0] * np.pi * x)

# --- PDE Definitions ---

def diffusion_pde(t, y, dx, u_interp):
    """Diffusion PDE with quadratic reaction and external field."""
    D = 0.01
    k = -0.1
    n = len(y)
    d2y_dx2 = np.zeros(n)
    d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx ** 2
    u_values = u_interp(np.linspace(0, 1, n))
    du_dt = D * d2y_dx2 + k * y ** 2 + u_values
    return du_dt

def diffusion_pde_drift(t, y, dx, u_interp):
    """Diffusion PDE with time-dependent drift in the external field."""
    D = 0.01
    k = -0.1
    n = len(y)
    d2y_dx2 = np.zeros(n)
    d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx ** 2
    u_values = u_interp(np.linspace(0, 1, n), t)
    du_dt = D * d2y_dx2 + k * y ** 2 + u_values
    return du_dt

def klein_gordon_pde(t, state, dx, u_interp):
    """Klein-Gordon PDE with periodic boundary conditions."""
    n = len(state) // 2
    u = state[:n]
    v = state[n:]
    d2u_dx2 = np.empty_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / dx ** 2
    d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / dx ** 2
    dudt = v
    dvdt = d2u_dx2 - u_interp(u)
    return np.concatenate([dudt, dvdt])

# --- Utility Functions ---

def GetSensors(U, sensor_locs):
    """Extract sensor values from field U at specified locations."""
    return U(sensor_locs).flatten()

def isnotnan(x):
    """Check if each row in x does not contain NaN or infinite values."""
    return [not (np.any(np.isnan(xi)) or np.any(np.isneginf(xi)) or np.any(np.isposinf(xi))) for xi in x]

def y0_func(x, center, dx=1, dy=0.1):
    """Initial condition function (Gaussian bump)."""
    return dy * np.exp(-(x - center) ** 2 / (2 * dx ** 2))

def save_data(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def open_data(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# --- Timeseries Generation Functions ---

def GenerateTimeseries_PDE_GRF(diffeq, y0, Trange, l, sensor_locs, x_grid, num_samples, save_path=None, T_init=[], drift=0):
    """
    Generate timeseries data for a PDE with GRF forcing.
    """
    def generate_sample():
        if len(T_init) == 0:
            t_random = np.random.uniform(Trange[0], Trange[1], Trange[2])
            T = np.sort(np.unique(np.concatenate(([0], t_random))))
        else:
            T = T_init
        if drift == 0:
            grf_sample = generate_grf(x_grid, l, num_samples=1)[0]
            u_interp = interpolate_grf(x_grid, grf_sample)
        else:
            grf_sample1 = generate_grf(x_grid, l, num_samples=1)[0]
            u_interp1 = interpolate_grf(x_grid, grf_sample1)
            grf_sample2 = generate_grf(x_grid, l, num_samples=1)[0]
            u_interp2 = interpolate_grf(x_grid, grf_sample2)
            u_interp = lambda x, t: np.sqrt(drift * t / T[-1]) * u_interp1(x) + np.sqrt((1 - drift * t / T[-1])) * u_interp2(x)
        sol = solve_ivp(diffeq, (T[0], T[-1]), y0, t_eval=T, args=(x_grid[1] - x_grid[0], u_interp))
        sol = sol.y.T[1:]
        T = T[1:]
        if len(T_init) == 0:
            sensor_vals = np.repeat(GetSensors(u_interp, sensor_locs).reshape(1, -1), Trange[2], axis=0)
            return T.reshape(-1, 1), sol, sensor_vals
        elif len(T) != 0 and drift != 0:
            sensor_vals = np.vstack([GetSensors(lambda x: u_interp(x, ti), sensor_locs).reshape(1, -1) for ti in T])
            return np.expand_dims(T.reshape(-1, 1), 0), np.expand_dims(sol, 0), np.expand_dims(sensor_vals, 0)
        else:
            sensor_vals = np.repeat(GetSensors(u_interp, sensor_locs).reshape(1, -1), len(T), axis=0)
            return np.expand_dims(T.reshape(-1, 1), 0), np.expand_dims(sol, 0), np.expand_dims(sensor_vals, 0)

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    valid_rows = np.array(isnotnan(ts_array)) & np.array(isnotnan(ys_array)) & np.array(isnotnan(us_array))
    ts_array = ts_array[valid_rows]
    ys_array = ys_array[valid_rows]
    us_array = us_array[valid_rows]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

def GenerateTimeseries_PDE_param(diffeq, y0, Trange, paramr, field, sensor_locs, x_grid, num_samples, save_path=None):
    """
    Generate timeseries data for a PDE with parameterized field forcing.
    """
    def generate_sample():
        params = [np.random.uniform(p[0], p[1], 1) for p in paramr]
        u_interp = lambda x: field(x, params)
        t_random = np.random.uniform(Trange[0], Trange[1], Trange[2])
        T = np.sort(np.unique(np.concatenate(([0], t_random))))
        sol_obj = solve_ivp(
            diffeq,
            (T[0], T[-1]),
            y0,
            t_eval=T,
            args=(x_grid[1] - x_grid[0], u_interp),
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )
        sol = sol_obj.sol(T).T[1:]
        T = T[1:]
        sensor_vals = GetSensors(u_interp, sensor_locs)
        return T.reshape(-1, 1), sol, np.repeat(sensor_vals.reshape(1, -1), Trange[2], axis=0)

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    valid_ts_rows = ~np.any(np.isnan(ts_array) | np.isposinf(ts_array) | np.isneginf(ts_array), axis=1)
    valid_ys_rows = ~np.any(np.isnan(ys_array) | np.isposinf(ys_array) | np.isneginf(ys_array), axis=1)
    valid_us_rows = ~np.any(np.isnan(us_array) | np.isposinf(us_array) | np.isneginf(us_array), axis=1)
    valid_rows = valid_ts_rows & valid_ys_rows & valid_us_rows
    ts_array = ts_array[valid_rows]
    ys_array = ys_array[valid_rows]
    us_array = us_array[valid_rows]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

# --- Main Execution ---

if __name__ == "__main__":
    # Grids and sensor locations
    x_grid_diff = np.linspace(0, 1, 100)
    x_grid_KG = np.linspace(0, 2, 200)
    sensor_locs_diff = np.random.uniform(0, 1, 600)
    sensor_locs_KG = np.random.uniform(-10, 10, 600)

    # Initial conditions
    y0_diff = y0_func(x_grid_diff, .75, .1)
    y0_KG = np.concatenate((
        y0_func(x_grid_KG[:len(x_grid_KG) // 2], 1 / np.sqrt(2), .1, 1),
        y0_func(x_grid_KG[len(x_grid_KG) // 2:], 1, .5, 0)
    ))

    # Time ranges
    Trange_diff = [0, 1, 10]
    Trange_diffR = np.linspace(0, 1, 100)
    Trange_KG = [0, 3, 10]

    # Sample counts and parameter ranges
    num_train_samples = 1000
    num_test_samples = 10000
    l = 0.1
    prange = [[0, 1] for _ in range(3)]

    # Generate and save datasets
    GenerateTimeseries_PDE_GRF(diffusion_pde, y0_diff, Trange_diff, l, sensor_locs_diff, x_grid_diff, num_train_samples, save_path='train_data_PDE_diff.pkl')
    GenerateTimeseries_PDE_GRF(diffusion_pde, y0_diff, Trange_diff, l, sensor_locs_diff, x_grid_diff, num_test_samples, save_path='test_data_PDE_diff.pkl')
    GenerateTimeseries_PDE_GRF(diffusion_pde, y0_diff, [Trange_diff[0], Trange_diff[1], 1000], l, sensor_locs_diff, x_grid_diff, 1, save_path='example_PDE_diff.pkl')

    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, Trange_KG, prange, sin_field, sensor_locs_KG, x_grid_KG, num_train_samples, save_path='train_data_PDE_KG.pkl')
    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, Trange_KG, prange, sin_field, sensor_locs_KG, x_grid_KG, num_test_samples, save_path='test_data_PDE_KG.pkl')
    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, [Trange_KG[0], Trange_KG[1], 1000], prange, sin_field, sensor_locs_KG, x_grid_KG, 1, save_path='example_PDE_KG.pkl')

    save_data([Trange_diff, sensor_locs_diff, y0_diff, x_grid_diff, l], 'PDE_params_diff.pkl')
    save_data([Trange_diffR, sensor_locs_diff, y0_diff, x_grid_diff, l], 'PDE_params_diffR.pkl')
    save_data([Trange_KG, sensor_locs_KG, y0_KG, x_grid_KG, prange], 'PDE_params_KG.pkl')
