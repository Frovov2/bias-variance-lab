"""Bias-Variance Lab: small from-scratch models and Monte Carlo helpers."""

from .models import BaggedTrees, PolynomialRegression, RegressionTree
from .simulation import bias_variance, kfold_cv_mse, sample, true_f

__all__ = [
    "PolynomialRegression",
    "RegressionTree",
    "BaggedTrees",
    "true_f",
    "sample",
    "bias_variance",
    "kfold_cv_mse",
]
