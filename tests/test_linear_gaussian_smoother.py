#!/usr/bin/env python3
"""Linear-Gaussian smoothing with closed-form answers.

The other examples validate the library against a *second implementation*
written here.  This one validates it against **Bayes' rule written out in closed
form**, which is a different and stronger kind of check: there is no filter
recursion anywhere in the reference.

Model -- a scalar random walk observed in noise, the standard smoothing
example::

    x[0] ~ N(m0, P0)                (a proper prior)
    x[n] = x[n-1] + w[n],  w ~ N(0, q)     n = 1 .. N
    y[n] = x[n] + v[n],    v ~ N(0, r)     n = 0 .. N

Two closed forms are used
-------------------------
1. *Steady state (the algebraic Riccati equation).*  The one-step prediction
   variance of the filter obeys v = v r / (v + r) + q, i.e.

       v^2 - q v - q r = 0   =>   v = ( q + sqrt(q^2 + 4 q r) ) / 2 ,
       K  = v / (v + r) .

   ``KF``, ``SquareRootKF`` and ``UDKF`` must approach these as the recursion
   runs.  This is algebra, not a reference code.

2. *The joint Gaussian posterior.*  The precision matrix of
   (x[0], ..., x[N]) given (y[0], ..., y[N]) is exactly

       Lambda = T / q + I / r + e0 e0^T / P0 ,
       b      = ( m0/P0 + y[0]/r , y[1]/r , ... , y[N]/r )^T

   where T is the path-graph Laplacian (T_00 = T_NN = 1, T_ii = 2 otherwise,
   T_{i,i+-1} = -1), coming from sum_n (x[n] - x[n-1])^2 / (2q).  Then

       smoothed mean = Lambda^{-1} b ,   smoothed covariance = Lambda^{-1} .

   ``rts_smooth`` and ``EnKS`` must reproduce this to machine precision (and in
   the L -> infinity limit respectively).  Note there is no forward/backward
   recursion on the right-hand side -- it is the definition of the answer.

Run standalone (writes figures with --figures)::

    python3 tests/test_linear_gaussian_smoother.py
    python3 tests/test_linear_gaussian_smoother.py --figures
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _estimation_loader import Harness, load_lib  # noqa: E402


# ==========================================================================
# The model
# ==========================================================================

N_STATES = 31        # x[0] .. x[N_STATES-1]
Q_VAR = 1.0          # process noise variance, q
R_VAR = 2.0          # measurement noise variance, r
M0 = 0.0             # prior mean
P0_VAR = 5.0         # prior variance
SEED = 20261006

H = np.array([[1.0]])
R = np.array([[R_VAR]])
F = np.array([[1.0]])
Q = np.array([[Q_VAR]])


def steady_state_prior_variance(q=Q_VAR, r=R_VAR):
    """v, the steady one-step prediction variance: v^2 = q v + q r."""
    return 0.5 * (q + math.sqrt(q * q + 4.0 * q * r))


def steady_state_gain(q=Q_VAR, r=R_VAR):
    v = steady_state_prior_variance(q, r)
    return v / (v + r)


def path_laplacian(n):
    """T: x^T T x = sum_{i=1..n-1} (x[i] - x[i-1])^2."""
    T = np.zeros((n, n))
    for i in range(n - 1):
        T[i, i] += 1.0
        T[i + 1, i + 1] += 1.0
        T[i, i + 1] -= 1.0
        T[i + 1, i] -= 1.0
    return T


def joint_posterior(y, m0=M0, P0=P0_VAR, q=Q_VAR, r=R_VAR):
    """The smoothed mean and covariance of (x[0], ..., x[N]) given all y.

    Bayes' rule for the joint Gaussian: precision and linear term assembled
    from the prior, the random-walk transitions and the observations.  No
    filtering or smoothing recursion is involved.
    """
    n = len(y)
    Lam = path_laplacian(n) / q + np.eye(n) / r
    Lam[0, 0] += 1.0 / P0
    b = np.asarray(y, dtype=float) / r
    b[0] += m0 / P0
    mean = np.linalg.solve(Lam, b)
    cov = np.linalg.inv(Lam)
    return mean, cov


def simulate(n=N_STATES, seed=SEED, q=Q_VAR, r=R_VAR, m0=M0, P0=P0_VAR):
    """One realization: the true trajectory and the observations."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = m0 + math.sqrt(P0) * rng.standard_normal()
    for i in range(1, n):
        x[i] = x[i - 1] + math.sqrt(q) * rng.standard_normal()
    y = x + math.sqrt(r) * rng.standard_normal(n)
    return {"x_true": x, "y": y, "n": n}


def as_mats(ys):
    """The scalar sequences in the shapes the library wants."""
    yv = [np.array([v]) for v in ys]
    Hv = [H] * len(ys)
    Rv = [R] * len(ys)
    Fv = [F] * len(ys)
    Qv = [Q] * len(ys)
    return yv, Hv, Rv, Fv, Qv


# ==========================================================================
# Figures
# ==========================================================================

FIG_TITLES = {
    "fig_smooth": "Linear-Gaussian smoothing: filter, smoother and the joint posterior",
    "fig_var": "Smoothing variance against the filter variance",
}


def make_figures(outdir, run, filtered, smoothed, post_mean, post_cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as plt

    os.makedirs(outdir, exist_ok=True)
    n = run["n"]
    t = np.arange(n)
    xs = np.arange(n)
    f_sd = np.sqrt([P[0, 0] for P in filtered["P_posterior"]])
    s_sd = np.sqrt(np.diag(post_cov))
    written = []

    fig, ax = plt.subplots()
    ax.plot(t, run["x_true"], "k-", lw=1.0, label="true state")
    ax.plot(t, run["y"], "o", ms=3, alpha=0.4, label="observations")
    ax.plot(t, filtered["x_posterior_arr"], "s--", ms=3, lw=0.9, label="filtered")
    ax.plot(xs, post_mean, "^-", ms=3, lw=0.9, label="smoothed (joint posterior)")
    ax.fill_between(t, post_mean - 2 * s_sd, post_mean + 2 * s_sd,
                    alpha=0.15, color="C2", label=r"smoother $\pm 2\sigma$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$x[n]$")
    ax.set_title(FIG_TITLES["fig_smooth"])
    ax.legend(fontsize=8)
    p = os.path.join(outdir, "linear_gaussian_smooth.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots()
    ax.plot(t, f_sd, "s--", ms=3, lw=0.9, label="filtered $\\sqrt{P_{n|n}}$")
    ax.plot(xs, s_sd, "^-", ms=3, lw=0.9, label="smoothed $\\sqrt{P_{n|N}}$")
    ax.set_xlabel(r"$n$")
    ax.set_title(FIG_TITLES["fig_var"])
    ax.legend()
    p = os.path.join(outdir, "linear_gaussian_var.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    return written


# ==========================================================================
# Tests
# ==========================================================================

HARNESS = Harness()


def test_precision_is_the_path_laplacian():
    # Sanity check on the closed form itself: the transition part of the
    # precision reproduces sum (x[i]-x[i-1])^2 exactly.
    n = 6
    T = path_laplacian(n)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    quad = float(x @ T @ x)
    ref = float(np.sum(np.diff(x) ** 2))
    HARNESS.check_close(quad, ref, 1e-12, "x^T T x == sum (x[i] - x[i-1])^2")


def test_steady_state_matches_the_algebraic_riccati_solution():
    # KF, SquareRootKF and UDKF must all converge to the ARE solution -- algebra,
    # not a reference implementation.
    lib = load_lib()
    q, r = Q_VAR, R_VAR
    v_inf = steady_state_prior_variance(q, r)
    k_inf = steady_state_gain(q, r)
    # Verify the ARE residual before using it as a target.
    HARNESS.check(abs(v_inf * v_inf / (v_inf + r) - q) < 1e-12,
                  "the target v satisfies the algebraic Riccati equation",
                  f"{v_inf}")

    yv, Hv, Rv, Fv, Qv = as_mats(np.zeros(2000))   # the y values do not matter
    for name in ("KF", "SquareRootKF", "UDKF"):
        f = getattr(lib, name)(1)
        f.initialize(np.array([0.0]), np.array([[P0_VAR]]))
        rec = f.batch(yv, Hv, Rv, Fv, Qv, lib.RECORD_P_PRIOR)
        v_end = float(rec.P_prior[-1][0, 0])
        HARNESS.check(abs(v_end - v_inf) / v_inf < 1e-3,
                      f"{name}: prior variance -> the ARE solution",
                      f"{v_end:.6f} vs {v_inf:.6f}")


def test_the_gain_converges_to_the_steady_state_gain():
    lib = load_lib()
    k_inf = steady_state_gain()
    f = lib.KF(1)
    yv, Hv, Rv, Fv, Qv = as_mats(np.zeros(2000))   # the y values do not matter
    f.initialize(np.array([0.0]), np.array([[P0_VAR]]))
    rec = f.batch(yv, Hv, Rv, Fv, Qv, lib.RECORD_P_PRIOR)
    v = float(rec.P_prior[-1][0, 0])
    HARNESS.check_close(v / (v + R_VAR), k_inf, 1e-4,
                        "K -> the steady-state ARE gain")


def test_rts_smoother_matches_the_joint_gaussian_posterior():
    # The headline check: the smoother equals Bayes' rule written out in closed
    # form.  No forward/backward recursion appears in the reference.
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    mu = np.array([M0])
    PI = np.array([[P0_VAR]])
    ref_mean, ref_cov = joint_posterior(run["y"])

    for name in ("KF", "SquareRootKF", "UDKF"):
        f = getattr(lib, name)(1)
        f.initialize(mu, PI)
        rec = f.batch(yv, Hv, Rv, Fv, Qv)
        sm = lib.rts_smooth(rec, Fv, Qv)
        got_mean = np.array([v[0] for v in sm.x_smoothed])
        got_cov = np.array([P[0, 0] for P in sm.P_smoothed])
        HARNESS.check(np.max(np.abs(got_mean - ref_mean)) < 1e-9,
                      f"{name} + rts_smooth matches the joint posterior mean",
                      f"{np.max(np.abs(got_mean - ref_mean)):.2e}")
        HARNESS.check(np.max(np.abs(got_cov - np.diag(ref_cov))) < 1e-9,
                      f"{name} + rts_smooth matches the joint posterior variance",
                      f"{np.max(np.abs(got_cov - np.diag(ref_cov))):.2e}")


def test_enks_converges_to_the_joint_gaussian_posterior():
    # The Monte Carlo smoother reaches the same closed form as L -> infinity.
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    ref_mean, ref_cov = joint_posterior(run["y"])
    ks = lib.EnKS(1, 20000, 5)
    ks.initialize(np.array([M0]), np.array([[P0_VAR]]))
    sm = ks.smooth(yv, Hv, Rv, Fv, Qv)
    got_mean = np.array([v[0] for v in sm.x_smoothed])
    HARNESS.check(np.max(np.abs(got_mean - ref_mean)) < 0.15,
                  "EnKS (L=20000) approaches the joint posterior mean",
                  f"{np.max(np.abs(got_mean - ref_mean)):.4f}")


def test_smoother_variance_is_below_filter_variance():
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    f = lib.KF(1)
    f.initialize(np.array([M0]), np.array([[P0_VAR]]))
    rec = f.batch(yv, Hv, Rv, Fv, Qv)
    sm = lib.rts_smooth(rec, Fv, Qv)
    worst = max(float(rec.P_posterior[i][0, 0] - sm.P_smoothed[i][0, 0])
                for i in range(run["n"]))
    HARNESS.check(worst >= -1e-12,
                  "smoothing never increases the variance (P_{n|N} <= P_{n|n})",
                  f"max (P_filter - P_smooth) = {-worst:.3e}")


def test_the_three_exact_filters_give_one_smoother():
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    means, covs = {}, {}
    for name in ("KF", "SquareRootKF", "UDKF"):
        f = getattr(lib, name)(1)
        f.initialize(np.array([M0]), np.array([[P0_VAR]]))
        rec = f.batch(yv, Hv, Rv, Fv, Qv)
        sm = lib.rts_smooth(rec, Fv, Qv)
        means[name] = np.array([v[0] for v in sm.x_smoothed])
        covs[name] = np.array([P[0, 0] for P in sm.P_smoothed])
    HARNESS.check(np.max(np.abs(means["KF"] - means["SquareRootKF"])) < 1e-12
                  and np.max(np.abs(means["KF"] - means["UDKF"])) < 1e-12,
                  "KF, SquareRootKF and UDKF give the same smoothed mean")
    HARNESS.check(np.max(np.abs(covs["KF"] - covs["SquareRootKF"])) < 1e-12
                  and np.max(np.abs(covs["KF"] - covs["UDKF"])) < 1e-12,
                  "KF, SquareRootKF and UDKF give the same smoothed variance")


def test_smoothing_beats_filtering_on_the_estimate():
    # On this model the smoothed estimate is closer to the truth than the
    # filtered one at almost every step -- the whole point of a smoother.
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    f = lib.KF(1)
    f.initialize(np.array([M0]), np.array([[P0_VAR]]))
    rec = f.batch(yv, Hv, Rv, Fv, Qv)
    sm = lib.rts_smooth(rec, Fv, Qv)
    f_err = np.array([abs(rec.x_posterior[i][0] - run["x_true"][i])
                      for i in range(run["n"])])
    s_err = np.array([abs(sm.x_smoothed[i][0] - run["x_true"][i])
                      for i in range(run["n"])])
    HARNESS.check(float(np.mean(s_err)) < 0.85 * float(np.mean(f_err)),
                  "smoothing reduces the estimation error",
                  f"{np.mean(s_err):.4f} vs {np.mean(f_err):.4f}")


def test_figures_are_written():
    import tempfile
    lib = load_lib()
    run = simulate()
    yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
    f = lib.KF(1)
    f.initialize(np.array([M0]), np.array([[P0_VAR]]))
    rec = f.batch(yv, Hv, Rv, Fv, Qv)
    sm = lib.rts_smooth(rec, Fv, Qv)
    ref_mean, ref_cov = joint_posterior(run["y"])
    filtered = {"x_posterior_arr": np.array([v[0] for v in rec.x_posterior]),
                "P_posterior": rec.P_posterior}
    smoothed = {"x": np.array([v[0] for v in sm.x_smoothed])}
    with tempfile.TemporaryDirectory() as d:
        written = make_figures(d, run, filtered, smoothed, ref_mean, ref_cov)
        HARNESS.check(len(written) == 2, "both figures are written")
        HARNESS.check(all(os.path.getsize(p) > 0 for p in written),
                      "every figure is non-empty")


TESTS = [
    test_precision_is_the_path_laplacian,
    test_steady_state_matches_the_algebraic_riccati_solution,
    test_the_gain_converges_to_the_steady_state_gain,
    test_rts_smoother_matches_the_joint_gaussian_posterior,
    test_enks_converges_to_the_joint_gaussian_posterior,
    test_smoother_variance_is_below_filter_variance,
    test_the_three_exact_filters_give_one_smoother,
    test_smoothing_beats_filtering_on_the_estimate,
    test_figures_are_written,
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--figures":
        lib = load_lib()
        outdir = argv[1] if len(argv) > 1 else os.path.join("figures",
                                                            "linear_gaussian_smoother")
        run = simulate()
        yv, Hv, Rv, Fv, Qv = as_mats(run["y"])
        f = lib.KF(1)
        f.initialize(np.array([M0]), np.array([[P0_VAR]]))
        rec = f.batch(yv, Hv, Rv, Fv, Qv)
        sm = lib.rts_smooth(rec, Fv, Qv)
        ref_mean, ref_cov = joint_posterior(run["y"])
        filtered = {"x_posterior_arr": np.array([v[0] for v in rec.x_posterior]),
                    "P_posterior": rec.P_posterior}
        smoothed = {"x": np.array([v[0] for v in sm.x_smoothed])}
        for p in make_figures(outdir, run, filtered, smoothed, ref_mean, ref_cov):
            print("wrote", p)
        return 0

    print(__doc__.splitlines()[0])
    for t in TESTS:
        HARNESS.run(t.__name__, t)
    return HARNESS.summary(len(TESTS))


if __name__ == "__main__":
    sys.exit(main())
