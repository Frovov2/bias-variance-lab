"""Three regression models written from scratch with NumPy.

* PolynomialRegression - least squares (optionally ridge) on polynomial features
* RegressionTree       - CART-style tree for a single feature, squared-error splits
* BaggedTrees          - bootstrap aggregation of RegressionTree

Each model has the same tiny interface: ``fit(x, y)`` returns ``self`` and
``predict(x)`` returns an array of predictions.
"""

import numpy as np


class PolynomialRegression:
    """Polynomial regression on x in [0, 1].

    Features are Legendre polynomials of (2x - 1). They span exactly the same
    space as 1, x, x^2, ... but are far better conditioned, so high degrees
    (e.g. 15) don't blow up numerically.

    alpha = 0  -> ordinary least squares
    alpha > 0  -> ridge: minimise ||y - X b||^2 + alpha * ||b[1:]||^2
                  (the intercept is not penalised)
    """

    def __init__(self, degree, alpha=0.0):
        self.degree = degree
        self.alpha = alpha

    def _features(self, x):
        return np.polynomial.legendre.legvander(2 * np.asarray(x, float) - 1, self.degree)

    def fit(self, x, y):
        X = self._features(x)
        y = np.asarray(y, float)
        if self.alpha == 0:
            self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        else:
            penalty = np.eye(X.shape[1])
            penalty[0, 0] = 0.0
            self.coef_ = np.linalg.solve(X.T @ X + self.alpha * penalty, X.T @ y)
        return self

    def predict(self, x):
        return self._features(x) @ self.coef_


class RegressionTree:
    """Regression tree for one feature.

    At each node we try every split point between consecutive sorted x values
    and keep the one with the smallest total squared error (sum of squared
    deviations from each side's mean). Leaves predict the mean of their y's.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        order = np.argsort(x)
        self.root_ = self._grow(x[order], y[order], depth=0)
        return self

    def _grow(self, x, y, depth):
        n = len(y)
        m = self.min_samples_leaf
        leaf = ("leaf", y.mean())
        if n < 2 * m or (self.max_depth is not None and depth >= self.max_depth):
            return leaf

        # Squared error of every candidate split, using running sums:
        # SSE = sum(y^2) - (sum y)^2 / n
        csum, csum2 = np.cumsum(y), np.cumsum(y**2)
        total, total2 = csum[-1], csum2[-1]
        k = np.arange(1, n)  # number of points going left
        left_sse = csum2[:-1] - csum[:-1] ** 2 / k
        right_sse = (total2 - csum2[:-1]) - (total - csum[:-1]) ** 2 / (n - k)
        sse = left_sse + right_sse

        valid = (k >= m) & (n - k >= m) & (x[1:] > x[:-1])  # can't split tied x's
        if not valid.any():
            return leaf
        sse = np.where(valid, sse, np.inf)
        i = int(np.argmin(sse))
        if sse[i] >= total2 - total**2 / n - 1e-12:  # split doesn't help
            return leaf

        threshold = (x[i] + x[i + 1]) / 2
        return (
            "split",
            threshold,
            self._grow(x[: i + 1], y[: i + 1], depth + 1),
            self._grow(x[i + 1 :], y[i + 1 :], depth + 1),
        )

    def predict(self, x):
        x = np.asarray(x, float)
        out = np.empty_like(x)
        self._predict(self.root_, x, np.arange(len(x)), out)
        return out

    def _predict(self, node, x, idx, out):
        if node[0] == "leaf":
            out[idx] = node[1]
            return
        _, threshold, left, right = node
        goes_left = x[idx] <= threshold
        self._predict(left, x, idx[goes_left], out)
        self._predict(right, x, idx[~goes_left], out)


class BaggedTrees:
    """Bagging: fit each tree on a bootstrap resample, average the predictions."""

    def __init__(self, n_trees=50, max_depth=None, min_samples_leaf=1, seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        n = len(x)
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)  # sample n points with replacement
            tree = RegressionTree(self.max_depth, self.min_samples_leaf)
            self.trees_.append(tree.fit(x[idx], y[idx]))
        return self

    def predict_each(self, x):
        """Predictions of every individual tree, shape (n_trees, len(x))."""
        return np.array([tree.predict(x) for tree in self.trees_])

    def predict(self, x):
        return self.predict_each(x).mean(axis=0)
