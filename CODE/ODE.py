"""
ODE Timeseries Generation with Gaussian Random Fields and Parameterized Inputs.

This module provides utilities to generate time series data for ODEs with
Gaussian random field (GRF) or parameterized inputs, including sensor sampling,
data saving/loading, and example usage for pendulum and Lorenz63 systems.

Author: Jari Beysen
"""

import os
import pickle
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# Environment settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Constants
SAMPLES = 1000
NUM_TEST_SAMPLES = 10000
NTEST = 6
TRANGE_PEND = [0, 1, 10]
TRANGE_L63 = [0, 1, 10]
LPARAM_RANGE = [[0, 10.0], [0, 28.0], [0, 8.0 / 3.0]]
GRID_T = np.linspace(TRANGE_PEND[0], TRANGE_PEND[1], 100)
LT = 0.1

def rbf_kernel(x1, x2, l):
    """Radial basis function kernel."""
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * l ** 2))

def generate_grf(domain, l, num_samples=1):
    """Generate Gaussian random field samples over a domain."""
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
    return interp1d(domain, grf_sample, kind='quadratic', fill_value="extrapolate")

def ut_ode(y, t, u):
    """Pendulum ODE with control input u(t)."""
    theta, omega = y
    u_value = u(t)
    dtheta_dt = omega
    domega_dt = -np.sin(theta) + u_value
    return np.array([dtheta_dt, domega_dt])

def L63f(params):
    """Lorenz63 system function generator."""
    sigma, rho, beta = params
    return lambda y: np.array([
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2]
    ])

def Lorenz63(y, t, f):
    """Lorenz63 ODE wrapper."""
    return np.array(f(y))

def GetSensors(u, sensor_locs):
    """Sample sensor values from input function u at sensor locations."""
    return u(sensor_locs).flatten()

def isnotnan(x):
    """Check for non-NaN and finite entries in array."""
    return [not (np.any(np.isnan(xi)) or np.any(np.isneginf(xi)) or np.any(np.isposinf(xi))) for xi in x]

def GenerateTimeseries_ODE_GRF(diffeq, y0, Trange, l, sensor_locs, num_samples, grid, save_path=None, t_grid_init=None, drift=0):
    """
    Generate time series data for ODEs with GRF input.
    """
    t_grid_init = t_grid_init if t_grid_init is not None else []
    def generate_sample():
        if not t_grid_init:
            t_grid = np.append([0], np.random.uniform(Trange[0], Trange[1], Trange[2]))
            t_grid.sort()
        else:
            t_grid = t_grid_init
        if drift == 0:
            grf_sample = generate_grf(grid, l, num_samples=1)[0]
            u_interp = interpolate_grf(grid, grf_sample)
        else:
            grf_sample1 = generate_grf(grid, l, num_samples=1)[0]
            u_interp1 = interpolate_grf(grid, grf_sample1)
            grf_sample2 = generate_grf(grid, l, num_samples=1)[0]
            u_interp2 = interpolate_grf(grid, grf_sample2)
            u_interp = lambda t: np.sqrt(drift * t / t_grid[-1]) * u_interp1(t) + np.sqrt((1 - drift * t / t_grid[-1])) * u_interp2(t)
        f = lambda t, y: diffeq(y, t, u_interp)
        sol = solve_ivp(f, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, method="RK45")
        t_eval = sol.t[1:]
        y_eval = sol.y.T[1:]
        sensor_vals = GetSensors(u_interp, sensor_locs)
        sensor_vals_rep = np.repeat(sensor_vals.reshape(1, -1), len(t_eval), axis=0)
        if not t_grid_init:
            return t_eval.reshape(-1, 1), y_eval, sensor_vals_rep
        else:
            return np.expand_dims(t_eval.reshape(-1, 1), 0), np.expand_dims(y_eval, 0), np.expand_dims(sensor_vals_rep, 0)

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    mask = np.array(isnotnan(ts_array)) & np.array(isnotnan(ys_array)) & np.array(isnotnan(us_array))
    ts_array = ts_array[mask]
    ys_array = ys_array[mask]
    us_array = us_array[mask]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

def GenerateTimeseries_ODE_param(diffeq, u, y0, Trange, param_range, sensor_locs, num_samples, save_path=None):
    """
    Generate time series data for ODEs with parameter input.
    """
    def generate_sample():
        t_grid = np.append([0], np.random.uniform(Trange[0], Trange[1], Trange[2]))
        t_grid.sort()
        param = np.array([np.random.uniform(r[0], r[1]) for r in param_range])
        usamp = u(param)
        f = lambda t, y: diffeq(y, t, usamp)
        sol = solve_ivp(f, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, method="RK45")
        t_eval = sol.t[1:]
        y_eval = sol.y.T[1:]
        sensor_vals = GetSensors(usamp, sensor_locs.T)
        sensor_vals_rep = np.repeat(sensor_vals.reshape(1, -1), len(t_eval), axis=0)
        return t_eval.reshape(-1, 1), y_eval, sensor_vals_rep

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    mask = np.array(isnotnan(ts_array)) & np.array(isnotnan(ys_array)) & np.array(isnotnan(us_array))
    ts_array = ts_array[mask]
    ys_array = ys_array[mask]
    us_array = us_array[mask]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

def y0_func(scale=1, dim=1):
    """Generate random initial condition."""
    return np.random.normal(0, scale, dim)

def save_data(data, filename):
    """Save data to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def open_data(filename):
    """Load data from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Sensor locations
    sensor_locs_ut = np.random.uniform(0, 1, (600, 1))
    sensor_locs_L63 = np.random.uniform(-30, 30, (200, 3))
    values1 = 10 ** np.linspace(-2, 0, NTEST)
    values2 = 10 ** np.linspace(-2, 0, NTEST)
    y0pend = y0_func(.1, 2)
    y0L63 = y0_func(1, 3)

    # Generate datasets
    GenerateTimeseries_ODE_GRF(ut_ode, y0pend, TRANGE_PEND, LT, sensor_locs_ut, SAMPLES, GRID_T, save_path='train_data_ODE_ut.pkl')
    GenerateTimeseries_ODE_GRF(ut_ode, y0pend, TRANGE_PEND, LT, sensor_locs_ut, NUM_TEST_SAMPLES, GRID_T, save_path='test_data_ODE_ut.pkl')
    GenerateTimeseries_ODE_GRF(ut_ode, y0pend, [TRANGE_PEND[0], TRANGE_PEND[1], 1000], LT, sensor_locs_ut, 1, GRID_T, save_path='example_ODE_ut.pkl')

    GenerateTimeseries_ODE_param(Lorenz63, L63f, y0L63, TRANGE_L63, LPARAM_RANGE, sensor_locs_L63, SAMPLES, save_path='train_data_ODE_L63.pkl')
    GenerateTimeseries_ODE_param(Lorenz63, L63f, y0L63, TRANGE_L63, LPARAM_RANGE, sensor_locs_L63, NUM_TEST_SAMPLES, save_path='test_data_ODE_L63.pkl')
    GenerateTimeseries_ODE_param(Lorenz63, L63f, y0L63, [TRANGE_L63[0], TRANGE_L63[1], 1000], LPARAM_RANGE, sensor_locs_L63, 1, save_path='example_ODE_L63.pkl')

    dtr = [GenerateTimeseries_ODE_GRF(ut_ode, y0pend, TRANGE_PEND, i, sensor_locs_ut, SAMPLES, GRID_T) for i in values1]
    dte = [GenerateTimeseries_ODE_GRF(ut_ode, y0pend, TRANGE_PEND, i, sensor_locs_ut, SAMPLES, GRID_T) for i in values2]

    save_data([TRANGE_PEND, sensor_locs_ut, y0pend, GRID_T, LT], 'ODE_params_ut.pkl')
    save_data([TRANGE_L63, sensor_locs_L63, y0L63, LPARAM_RANGE], 'ODE_params_L63.pkl')
    save_data([values1, values2, SAMPLES], 'OOD_grid.pkl')
    save_data([dtr, dte], 'OOD_ODE.pkl')

