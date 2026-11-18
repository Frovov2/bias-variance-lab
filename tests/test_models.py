import numpy as np

from biasvariance import (
    BaggedTrees,
    PolynomialRegression,
    RegressionTree,
    bias_variance,
    kfold_cv_mse,
    sample,
)


def test_polynomial_interpolates_with_enough_degrees():
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    model = PolynomialRegression(degree=4).fit(x, y)
    assert np.allclose(model.predict(x), y)


def test_huge_ridge_penalty_gives_the_mean():
    rng = np.random.default_rng(0)
    x, y = sample(40, 0.3, rng)
    model = PolynomialRegression(degree=8, alpha=1e10).fit(x, y)
    assert np.allclose(model.predict(x), y.mean(), atol=1e-6)


def test_deep_tree_memorises_training_data():
    rng = np.random.default_rng(1)
    x, y = sample(25, 0.3, rng)
    assert np.allclose(RegressionTree().fit(x, y).predict(x), y)


def test_stump_has_two_values():
    rng = np.random.default_rng(2)
    x, y = sample(50, 0.3, rng)
    preds = RegressionTree(max_depth=1).fit(x, y).predict(np.linspace(0, 1, 100))
    assert len(np.unique(preds)) == 2


def test_bagging_shapes_and_reproducibility():
    rng = np.random.default_rng(3)
    x, y = sample(30, 0.3, rng)
    grid = np.linspace(0, 1, 20)
    a = BaggedTrees(10, seed=5).fit(x, y)
    b = BaggedTrees(10, seed=5).fit(x, y)
    assert a.predict_each(grid).shape == (10, 20)
    assert np.allclose(a.predict(grid), b.predict(grid))


def test_decomposition_matches_simulated_test_error():
    rng = np.random.default_rng(4)
    r = bias_variance(lambda: PolynomialRegression(3), 400, 30, 0.3, rng, np.linspace(0, 1, 200))
    assert r["bias2"] >= 0 and r["variance"] >= 0
    assert abs(r["expected_mse"] - r["test_mse"]) / r["expected_mse"] < 0.05


def test_cross_validation_runs():
    rng = np.random.default_rng(5)
    x, y = sample(30, 0.3, rng)
    folds = np.array_split(rng.permutation(30), 5)
    assert kfold_cv_mse(lambda: PolynomialRegression(3), x, y, folds) > 0
