"""The data-generating process and the Monte Carlo machinery.

Model:  X ~ Uniform(0, 1),   Y = f(X) + eps,   eps ~ Normal(0, sigma^2)
with f(x) = sin(2 pi x).

Because we *know* f and sigma, we can do something you can't do with real
data: draw many independent training sets, refit the model on each, and
measure exactly how much of the error is bias, variance or irreducible noise.
"""

import numpy as np


def true_f(x):
    return np.sin(2 * np.pi * np.asarray(x, float))


def sample(n, sigma, rng):
    """Draw n points (x, y) from the model."""
    x = rng.uniform(0, 1, n)
    y = true_f(x) + rng.normal(0, sigma, n)
    return x, y


def bias_variance(make_model, n_sims, n_train, sigma, rng, grid):
    """Monte Carlo bias-variance decomposition.

    For each of n_sims simulations: draw a fresh training set, fit
    ``make_model()``, and predict on ``grid``. Then, averaging over the grid,

        bias^2   = mean_x ( E[f_hat(x)] - f(x) )^2
        variance = mean_x Var[f_hat(x)]

    and the expected test MSE is bias^2 + variance + sigma^2.

    Returns a dict with bias2, variance, noise, expected_mse, test_mse
    (an empirical check using fresh noisy targets), train_mse and the raw
    prediction matrix ``preds`` of shape (n_sims, len(grid)).
    """
    grid = np.asarray(grid, float)
    preds = np.empty((n_sims, grid.size))
    train_mse = np.empty(n_sims)
    for s in range(n_sims):
        x, y = sample(n_train, sigma, rng)
        model = make_model().fit(x, y)
        preds[s] = model.predict(grid)
        train_mse[s] = np.mean((model.predict(x) - y) ** 2)

    f = true_f(grid)
    bias2 = np.mean((preds.mean(axis=0) - f) ** 2)
    variance = np.mean(preds.var(axis=0))
    y_test = f + rng.normal(0, sigma, preds.shape)  # fresh noisy test targets
    return {
        "bias2": bias2,
        "variance": variance,
        "noise": sigma**2,
        "expected_mse": bias2 + variance + sigma**2,
        "test_mse": np.mean((preds - y_test) ** 2),
        "train_mse": train_mse.mean(),
        "preds": preds,
    }


def kfold_cv_mse(make_model, x, y, folds):
    """K-fold cross-validation MSE. ``folds`` is a list of index arrays."""
    all_idx = np.concatenate(folds)
    errors = []
    for held_out in folds:
        train = np.setdiff1d(all_idx, held_out)
        model = make_model().fit(x[train], y[train])
        errors.append(np.mean((model.predict(x[held_out]) - y[held_out]) ** 2))
    return float(np.mean(errors))
