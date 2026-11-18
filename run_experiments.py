"""Run every experiment and write the figures to ./figures.

    python run_experiments.py                 # defaults
    python run_experiments.py --sims 500      # smoother curves, slower
    python run_experiments.py --noise 0.5 --n-train 50 --seed 7
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from biasvariance import (  # noqa: E402
    BaggedTrees,
    PolynomialRegression,
    RegressionTree,
    bias_variance,
    kfold_cv_mse,
    sample,
    true_f,
)

GRID = np.linspace(0, 1, 300)
DEGREES = list(range(0, 16))
LAMBDAS = np.logspace(-8, 2, 41)
TREE_COUNTS = [1, 2, 5, 10, 20, 50, 100]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def draw_truth_and_data(ax, x, y):
    ax.plot(GRID, true_f(GRID), "k--", lw=1.5, label="true f(x)")
    ax.scatter(x, y, s=16, color="0.35", zorder=3, label="training data")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")


# 1. Under- vs over-fitting on a single dataset ---------------------------------
def polynomial_fits(rng, args, out):
    x, y = sample(args.n_train, args.noise, rng)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (d, verdict) in zip(axes, [(1, "underfits"), (3, "about right"), (15, "overfits")]):
        draw_truth_and_data(ax, x, y)
        ax.plot(GRID, PolynomialRegression(d).fit(x, y).predict(GRID), color="C0", lw=2, label="fit")
        ax.set_title(f"Degree {d}: {verdict}")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, out / "01_polynomial_fits.png")


# 2. Bias-variance decomposition across model complexity ------------------------
def polynomial_bias_variance(rng, args, out):
    rows = []
    for d in DEGREES:
        r = bias_variance(lambda: PolynomialRegression(d), args.sims, args.n_train, args.noise, rng, GRID)
        rows.append(r)

    get = lambda key: np.array([r[key] for r in rows])  # noqa: E731
    best = DEGREES[int(np.argmin(get("expected_mse")))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(DEGREES, get("train_mse"), "o-", color="C0", label="training error")
    ax1.plot(DEGREES, get("test_mse"), "o-", color="C3", label="test error")
    ax1.axhline(args.noise**2, color="0.5", ls=":", label=r"noise $\sigma^2$")
    ax1.set(yscale="log", xlabel="polynomial degree (model complexity)", ylabel="MSE",
            title="Training error keeps falling, test error doesn't")
    ax1.legend()

    ax2.plot(DEGREES, get("bias2"), "o-", color="C0", label=r"bias$^2$")
    ax2.plot(DEGREES, get("variance"), "o-", color="C1", label="variance")
    ax2.axhline(args.noise**2, color="0.5", ls=":", label=r"noise $\sigma^2$")
    ax2.plot(DEGREES, get("expected_mse"), "k-", lw=2, label=r"bias$^2$ + var + $\sigma^2$")
    ax2.plot(DEGREES, get("test_mse"), "x", color="C3", ms=7, label="simulated test MSE")
    ax2.axvline(best, color="C2", ls="--", lw=1)
    ax2.set(yscale="log", xlabel="polynomial degree (model complexity)",
            title=f"Bias-variance trade-off (best degree = {best})")
    ax2.legend(fontsize=8)
    save(fig, out / "02_bias_variance_tradeoff.png")

    print(f"\n  {'degree':>6} {'bias^2':>10} {'variance':>10} {'test MSE':>10}")
    for d, r in zip(DEGREES, rows):
        print(f"  {d:>6} {r['bias2']:>10.4f} {r['variance']:>10.4f} {r['expected_mse']:>10.4f}")
    print()


# 3. Ridge regression with lambda chosen by cross-validation --------------------
def ridge_cross_validation(rng, args, out, degree=12, k=5):
    x, y = sample(args.n_train, args.noise, rng)
    folds = np.array_split(rng.permutation(len(x)), k)

    cv = [kfold_cv_mse(lambda: PolynomialRegression(degree, lam), x, y, folds) for lam in LAMBDAS]
    true_err = [
        np.mean((PolynomialRegression(degree, lam).fit(x, y).predict(GRID) - true_f(GRID)) ** 2) + args.noise**2
        for lam in LAMBDAS
    ]
    best = LAMBDAS[int(np.argmin(cv))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(LAMBDAS, cv, "o-", ms=3, color="C0", label=f"{k}-fold CV estimate")
    ax1.plot(LAMBDAS, true_err, "-", color="C3", label="true test MSE (known f)")
    ax1.axvline(best, color="C2", ls="--", lw=1, label=rf"CV choice $\lambda$ = {best:.1e}")
    ax1.set(xscale="log", yscale="log", xlabel=r"penalty strength $\lambda$", ylabel="MSE",
            title=f"Ridge on a degree-{degree} polynomial")
    ax1.legend(fontsize=8)

    draw_truth_and_data(ax2, x, y)
    for lam, colour, name in [(LAMBDAS[0], "C1", "tiny"), (best, "C2", "CV-chosen"), (LAMBDAS[-1], "C4", "huge")]:
        fit = PolynomialRegression(degree, lam).fit(x, y).predict(GRID)
        ax2.plot(GRID, fit, color=colour, lw=2, label=rf"{name} $\lambda$={lam:.0e}")
    ax2.set_title("Penalty pulls an overfit model back")
    ax2.legend(fontsize=8, loc="lower left")
    save(fig, out / "03_ridge_cross_validation.png")


# 4. Trees vs bagging ------------------------------------------------------------
def trees_and_bagging(rng, args, out):
    n_sims = max(10, args.sims // 4)  # trees are slower than polynomials
    seed = lambda: int(rng.integers(2**31))  # noqa: E731

    x, y = sample(args.n_train, args.noise, rng)
    one_tree = RegressionTree().fit(x, y)
    forest = BaggedTrees(100, seed=seed()).fit(x, y)

    # Test MSE as we average more and more bootstrap trees
    max_b = max(TREE_COUNTS)
    mse = np.zeros(len(TREE_COUNTS))
    for _ in range(n_sims):
        xs, ys = sample(args.n_train, args.noise, rng)
        each = BaggedTrees(max_b, seed=seed()).fit(xs, ys).predict_each(GRID)
        running_mean = np.cumsum(each, axis=0) / np.arange(1, max_b + 1)[:, None]
        y_test = true_f(GRID) + rng.normal(0, args.noise, GRID.size)
        mse += [np.mean((running_mean[b - 1] - y_test) ** 2) for b in TREE_COUNTS]
    mse /= n_sims

    # Bias / variance of three models
    models = {
        "single deep tree": lambda: RegressionTree(),
        "bagged (50 trees)": lambda: BaggedTrees(50, seed=seed()),
        "polynomial deg 3": lambda: PolynomialRegression(3),
    }
    bv = {name: bias_variance(m, n_sims, args.n_train, args.noise, rng, GRID) for name, m in models.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    draw_truth_and_data(axes[0], x, y)
    axes[0].step(GRID, one_tree.predict(GRID), where="mid", color="C1", lw=1.5, label="1 deep tree")
    axes[0].plot(GRID, forest.predict(GRID), color="C0", lw=2, label="bagged, 100 trees")
    axes[0].set(ylabel="y", title="One tree is jumpy; averaging smooths it")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].plot(TREE_COUNTS, mse, "o-", color="C0", label="test MSE")
    axes[1].axhline(args.noise**2, color="0.5", ls=":", label=r"noise $\sigma^2$")
    axes[1].set(xscale="log", xlabel="number of bootstrap trees", ylabel="MSE",
                title="More trees, lower error (then it plateaus)")
    axes[1].legend(fontsize=8)

    names = list(bv)
    pos = np.arange(len(names))
    axes[2].bar(pos - 0.2, [bv[n]["bias2"] for n in names], 0.4, color="C0", label=r"bias$^2$")
    axes[2].bar(pos + 0.2, [bv[n]["variance"] for n in names], 0.4, color="C1", label="variance")
    axes[2].set_xticks(pos, names, fontsize=8)
    axes[2].set(title="Bagging mainly cuts variance")
    axes[2].legend(fontsize=8)
    save(fig, out / "04_trees_and_bagging.png")

    for n in names:
        print(f"  {n:<18} bias^2={bv[n]['bias2']:.4f}  variance={bv[n]['variance']:.4f}")


# 5. Animation: watching variance happen -----------------------------------------
def variance_animation(rng, args, out, frames=30, degrees=(3, 15)):
    datasets = [sample(args.n_train, args.noise, rng) for _ in range(frames)]
    fits = {d: [PolynomialRegression(d).fit(x, y).predict(GRID) for x, y in datasets] for d in degrees}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def draw(i):
        for ax, d in zip(axes, degrees):
            ax.clear()
            for j in range(i):
                ax.plot(GRID, fits[d][j], color="C0", alpha=0.18, lw=1)
            draw_truth_and_data(ax, *datasets[i])
            ax.plot(GRID, fits[d][i], color="C3", lw=2.2)
            ax.set_title(f"degree {d}  |  training set {i + 1}/{frames}")
        axes[0].set_ylabel("y")

    anim = FuncAnimation(fig, draw, frames=frames)
    path = out / "05_variance_animation.gif"
    anim.save(path, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims", type=int, default=200, help="Monte Carlo repetitions (default 200)")
    p.add_argument("--n-train", type=int, default=30, help="training-set size (default 30)")
    p.add_argument("--noise", type=float, default=0.3, help="noise std sigma (default 0.3)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--out", type=Path, default=Path("figures"), help="output folder")
    p.add_argument("--no-gif", action="store_true", help="skip the animation")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"sims={args.sims}  n_train={args.n_train}  noise={args.noise}  seed={args.seed}")

    steps = [
        ("Polynomial fits", polynomial_fits),
        ("Bias-variance trade-off", polynomial_bias_variance),
        ("Ridge + cross-validation", ridge_cross_validation),
        ("Trees and bagging", trees_and_bagging),
    ]
    if not args.no_gif:
        steps.append(("Variance animation", variance_animation))

    for name, step in steps:
        t0 = time.perf_counter()
        print(f"[{name}]")
        step(rng, args, args.out)
        print(f"  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
