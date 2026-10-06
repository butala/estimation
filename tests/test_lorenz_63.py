#!/usr/bin/env python3
"""Tracking on the Lorenz (1963) attractor -- nonlinear dynamics, chaotic.

The companion to tests/test_kay_example_13_4.py.  Kay's example is nonlinear in
the *measurement* and exactly linear in the *dynamics*; this one is the other
way round, which is why it exists: it pins the identity that makes the
extended Kalman filter's *time update* expressible in the linear API.

Model, Lorenz (1963), sigma = 10, beta = 8/3, rho = 28::

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

propagated with fourth-order Runge-Kutta at dt = 0.01, and observed through the
first component only (x[0]) every 10 model steps -- the partial-observation
setup that is the standard hard case for this system.

The two identities
------------------
The EKF needs a nonlinear measurement and a nonlinear time update, and the
library's API is linear in both.  Both reduce to it by carrying the residual:

    measurement   y' = y - h(x) + H x ,  H = dh/dx      -> measurement_update(y', H, R)
    dynamics      u  = f(x) - F x      ,  F = df/dx      -> time_update(F, Q, u)

The first is Kay Example 13.4 and is asserted there.  The second is asserted
here and is the point of this example: ``time_update(F, Q, u)`` computes
x <- F x + u and P <- F P F^T + Q, so with u = f(x) - F x one gets exactly
x <- f(x) and P <- F P F^T + Q, the textbook EKF time update.

The ensemble methods need neither identity.  They propagate each member
through the model itself -- which is their defining feature, and the reason
they handle this system better than the EKF.

Chaotic systems
---------------
Pointwise assertions are meaningless here (two correct runs diverge within a
Lyapunov time, about 1.1 time units).  Everything below is a *statistic* over a
long assimilation window, or a *relative* claim between methods, with the
integrator and the seeds fixed.

Run standalone (writes figures with --figures)::

    python3 tests/test_lorenz_63.py
    python3 tests/test_lorenz_63.py --figures
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _estimation_loader import Harness, load_lib  # noqa: E402


# ==========================================================================
# Lorenz (1963)
# ==========================================================================

SIGMA = 10.0
BETA = 8.0 / 3.0
RHO = 28.0

DT = 0.01          # Runge-Kutta step
OBS_EVERY = 10     # observe every 10 model steps -> Dt_obs = 0.1
VAR_R = 1.0        # measurement noise variance on x[0]
VAR_Q = 1.0e-3     # process noise variance per model step, per state

SEED = 20261006


def drift(x, sigma=SIGMA, beta=BETA, rho=RHO):
    """f(x) -- the Lorenz vector field."""
    x1, x2, x3 = x[0], x[1], x[2]
    return np.array([sigma * (x2 - x1),
                     x1 * (rho - x3) - x2,
                     x1 * x2 - beta * x3])


def drift_jacobian(x, sigma=SIGMA, beta=BETA, rho=RHO):
    """df/dx -- the Lorenz tangent-linear matrix."""
    x1, x3 = x[0], x[2]
    return np.array([[-sigma,  sigma,  0.0],
                     [rho - x3, -1.0,  -x1],
                     [x[1],     x1,    -beta]])


def rk4_step(x, dt=DT):
    """One Runge-Kutta step of the Lorenz field."""
    k1 = drift(x)
    k2 = drift(x + 0.5 * dt * k1)
    k3 = drift(x + 0.5 * dt * k2)
    k4 = drift(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_tangent(x, dt=DT):
    """One Runge-Kutta step together with the Jacobian of that step.

    Propagating the tangent alongside the state (a variational Runge-Kutta
    step) gives the exact Jacobian of the discrete map -- no finite
    differences, and the same four field evaluations plus four Jacobians.
    """
    k1 = drift(x)
    K1 = drift_jacobian(x)

    a2 = x + 0.5 * dt * k1
    k2 = drift(a2)
    K2 = drift_jacobian(a2) @ (np.eye(3) + 0.5 * dt * K1)

    a3 = x + 0.5 * dt * k2
    k3 = drift(a3)
    K3 = drift_jacobian(a3) @ (np.eye(3) + 0.5 * dt * K2)

    a4 = x + dt * k3
    k4 = drift(a4)
    K4 = drift_jacobian(a4) @ (np.eye(3) + dt * K3)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    F = np.eye(3) + (dt / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
    return x_next, F


H = np.array([[1.0, 0.0, 0.0]])      # observe the first component only
R = np.array([[VAR_R]])
Q = VAR_Q * np.eye(3)                 # per model step


def fixed_points():
    """The two symmetric equilibria of the Lorenz system, and the origin.

    Lorenz's classic geometry: for rho > 1 the origin loses stability to
    C+/C- = (+-sqrt(beta (rho-1)), +-sqrt(beta (rho-1)), rho-1), which become
    the centres of the two lobes for rho > rho_Hopf.
    """
    s = math.sqrt(BETA * (RHO - 1.0))
    return [np.zeros(3), np.array([s, s, RHO - 1.0]), np.array([-s, -s, RHO - 1.0])]


# ==========================================================================
# Realization and the two recursions
# ==========================================================================

def simulate(n_obs=2000, seed=SEED, x0=None, var_q=VAR_Q):
    """A twin experiment: truth from the model with process noise, plus
    observations of x[0] every OBS_EVERY steps."""
    rng = np.random.default_rng(seed)
    if x0 is None:
        # A point on the attractor: relax an arbitrary start first.
        x = np.array([1.0, 1.0, 1.0])
        for _ in range(5000):
            x = rk4_step(x)
        x0 = x
    Qf = math.sqrt(var_q) * np.eye(3)

    x_true, y_obs, times = [], [], []
    x = x0.copy() + Qf @ rng.standard_normal(3) if var_q > 0 else x0.copy()
    for k in range(n_obs):
        for _ in range(OBS_EVERY):
            x = rk4_step(x)
            if var_q > 0:
                x = x + Qf @ rng.standard_normal(3)
        x_true.append(x.copy())
        times.append((k + 1) * OBS_EVERY * DT)
        y_obs.append(H @ x + math.sqrt(VAR_R) * rng.standard_normal(len(H)))
    return {"x_true": np.array(x_true), "y": [np.asarray(v).ravel() for v in y_obs],
            "times": np.array(times), "Q": np.eye(3) * var_q}


def ekf_library(run, mu0, P0):
    """Kay's EKF through the library -- both identities in use."""
    lib = load_lib()
    kf = lib.KF(3)
    kf.initialize(mu0, P0)
    x_post, P_post = [], []
    for k, y_k in enumerate(run["y"]):
        # Propagate from the previous analysis to the current time.
        for _ in range(OBS_EVERY):
            x_prior = kf.state().copy()
            x_next, F = rk4_step_tangent(x_prior)
            u = x_next - F @ x_prior          # <- the dynamics identity
            kf.time_update(F, run["Q"], u)
        # Measurement update through the linearisation.
        # Kay's measurement identity.  Here h(x) = H x is linear, so it
        # collapses to y_eff = y_k -- the content of *this* example is the
        # dynamics identity above.  Written out anyway so the general form is
        # visible.
        x_pred = kf.state().copy()
        Hn = H
        y_eff = y_k - (Hn @ x_pred) + (Hn @ x_pred)
        kf.measurement_update(y_eff, Hn, R)
        x_post.append(kf.state().copy())
        P_post.append(kf.covariance().copy())
    return {"x_posterior": x_post, "P_posterior": P_post}


def ekf_reference(run, mu0, P0):
    """The EKF written out as in the textbooks -- no library involved."""
    x, P = mu0.copy(), P0.copy()
    I = np.eye(3)
    x_post, P_post = [], []
    for k, y_k in enumerate(run["y"]):
        for _ in range(OBS_EVERY):
            x, F = rk4_step_tangent(x)
            P = F @ P @ F.T + run["Q"]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.solve(S, np.eye(len(H)))
        x = x + K @ (y_k - H @ x)
        P = (I - K @ H) @ P
        x_post.append(x.copy())
        P_post.append(P.copy())
    return {"x_posterior": x_post, "P_posterior": P_post}


def ensemble_filter(run, mu0, P0, cls="ETKF", L=50, seed=1, taper=None,
                    inflation=1.0):
    """A Monte Carlo method through the library.

    The forecast is the model run on each member -- no Jacobian, no
    linearisation, which is the point of the ensemble family.  ``members()`` and
    ``set_members()`` move the ensemble in and out; everything else is the same
    measurement_update the exact filters use.
    """
    lib = load_lib()
    if cls in ("LETKF", "LEKS"):
        filt = getattr(lib, cls)(3, L, np.asfortranarray(taper), seed)
    else:
        filt = getattr(lib, cls)(3, L, seed)
        if taper is not None:
            filt.set_taper(np.asfortranarray(taper))
    if inflation != 1.0:
        filt.set_inflation(inflation)

    rng = np.random.default_rng(seed + 9871)
    X0 = mu0[:, None] + np.linalg.cholesky(P0) @ rng.standard_normal((3, L))
    filt.set_members(X0)

    x_post, P_post = [], []
    for k, y_k in enumerate(run["y"]):
        for _ in range(OBS_EVERY):
            X = filt.members()
            Xn = np.column_stack([rk4_step(X[:, j]) for j in range(X.shape[1])])
            if run["Q"].trace() > 0:
                Qf = np.linalg.cholesky(run["Q"])
                W = Qf @ rng.standard_normal(Xn.shape)
                W = W - W.mean(axis=1, keepdims=True)   # recentered, as in the C++
                Xn = Xn + W
            filt.set_members(Xn)
        filt.measurement_update(y_k, H, R)
        x_post.append(filt.state().copy())
        P_post.append(filt.covariance().copy())
    return {"x_posterior": x_post, "P_posterior": P_post}


def rmse(a, b):
    """Time-averaged RMS position error (the three state components)."""
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d ** 2)))


def nees(est, truth):
    """Time-averaged normalised estimation error squared, per sample."""
    out = []
    for x, P, xt in zip(est["x_posterior"], est["P_posterior"], truth):
        d = x - xt
        out.append(float(d @ np.linalg.solve(P, d)))
    return float(np.mean(out))


# ==========================================================================
# Figures
# ==========================================================================

FIG_TITLES = {
    "fig_state": "Reproduction-style: the Lorenz (1963) trajectory and its estimate",
    "fig_lobes": "Lorenz (1963): the attractor in the x-y plane",
    "fig_rmse": "Lorenz (1963): state estimation error against the EKF",
}


def make_figures(outdir, run, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as plt

    os.makedirs(outdir, exist_ok=True)
    xt = run["x_true"]
    written = []

    fig, ax = plt.subplots()
    ax.plot(xt[:, 0], xt[:, 1], lw=0.7, label="True trajectory")
    for name, est in results.items():
        ax.plot(np.asarray(est["x_posterior"])[:, 0],
                np.asarray(est["x_posterior"])[:, 1], lw=0.7, label=name)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(FIG_TITLES["fig_lobes"])
    ax.legend()
    p = os.path.join(outdir, "lorenz63_lobes.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots()
    for j, lab in enumerate((r"$x$", r"$y$", r"$z$")):
        ax.plot(run["times"], xt[:, j], lw=0.8, label=f"true {lab}")
        ax.plot(run["times"], np.asarray(results["ETKF"]["x_posterior"])[:, j],
                lw=0.8, ls="--", label=f"ETKF {lab}")
    ax.set_xlabel(r"time")
    ax.set_title(FIG_TITLES["fig_state"])
    ax.legend(ncol=3, fontsize=8)
    p = os.path.join(outdir, "lorenz63_state.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots()
    for name, est in results.items():
        err = np.linalg.norm(np.asarray(est["x_posterior"]) - xt, axis=1)
        ax.plot(run["times"], err, lw=0.8, label=name)
    ax.set_xlabel(r"time")
    ax.set_ylabel(r"$\| \hat{x} - x \|$")
    ax.set_title(FIG_TITLES["fig_rmse"])
    ax.legend()
    p = os.path.join(outdir, "lorenz63_rmse.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    return written


# ==========================================================================
# Tests
# ==========================================================================

HARNESS = Harness()


def test_model_geometry():
    # The two symmetric equilibria and the origin are the fixed points of the
    # Lorenz field -- an analytic property of the system, independent of any
    # integration.
    for x in fixed_points():
        HARNESS.check(np.linalg.norm(drift(x)) < 1e-9,
                      f"drift vanishes at the equilibrium {np.round(x, 3)}",
                      f"{np.linalg.norm(drift(x)):.2e}")
    # rho = 28 is well past the Hopf bifurcation at rho_H ~ 24.74, so C+/C- are
    # unstable and the attractor is chaotic.
    s = math.sqrt(BETA * (RHO - 1.0))
    C = drift_jacobian(np.array([s, s, RHO - 1.0]))
    HARNESS.check(np.max(np.real(np.linalg.eigvals(C))) > 0,
                  "the symmetric equilibria are unstable at rho = 28 (chaos)",
                  f"{np.max(np.real(np.linalg.eigvals(C))):.3f}")


def test_drift_jacobian_matches_central_differences():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(100):
        x = rng.normal(size=3) * 5.0
        J = drift_jacobian(x)
        eps = 1e-6
        Jfd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = eps
            Jfd[:, j] = (drift(x + e) - drift(x - e)) / (2 * eps)
        worst = max(worst, float(np.max(np.abs(J - Jfd))))
    HARNESS.check(worst < 1e-7, "df/dx matches central differences", f"{worst:.2e}")


def test_rk4_tangent_matches_central_differences():
    # The variational Runge-Kutta step must give the exact Jacobian of the
    # discrete map -- this is what makes the EKF below correctly specified.
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        x = rng.normal(size=3) * 5.0
        _, F = rk4_step_tangent(x)
        eps = 1e-6
        Ffd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = eps
            Ffd[:, j] = (rk4_step(x + e) - rk4_step(x - e)) / (2 * eps)
        worst = max(worst, float(np.max(np.abs(F - Ffd))))
    HARNESS.check(worst < 1e-6,
                  "the RK4 tangent matches central differences of the RK4 map",
                  f"{worst:.2e}")


def test_dynamics_identity():
    # The point of this example: u = f(x) - F x makes time_update(F, Q, u)
    # produce exactly x <- f(x) and P <- F P F^T + Q.
    lib = load_lib()
    rng = np.random.default_rng(5)
    worst_x = 0.0
    worst_P = 0.0
    for _ in range(25):
        x = rng.normal(size=3) * 5.0
        P = np.eye(3) * 2.0
        x_next, F = rk4_step_tangent(x)
        u = x_next - F @ x
        kf = lib.KF(3)
        kf.initialize(x, P)
        kf.time_update(F, Q, u)
        worst_x = max(worst_x, float(np.max(np.abs(kf.state() - x_next))))
        worst_P = max(worst_P, float(np.max(np.abs(kf.covariance() - (F @ P @ F.T + Q)))))
    HARNESS.check(worst_x == 0.0,
                  "u = f(x) - F x gives x <- f(x) exactly", f"{worst_x:.2e}")
    HARNESS.check(worst_P < 1e-12,
                  "u = f(x) - F x gives P <- F P F^T + Q", f"{worst_P:.2e}")


def test_library_matches_reference_ekf():
    run = simulate(n_obs=300, seed=17)
    mu0 = run["x_true"][0] + np.array([3.0, -3.0, 2.0])
    P0 = 10.0 * np.eye(3)
    lib_ = ekf_library(run, mu0, P0)
    ref = ekf_reference(run, mu0, P0)
    scale = max(float(np.max(np.abs(v))) for v in ref["x_posterior"])
    wx = max(float(np.max(np.abs(a - b)))
             for a, b in zip(lib_["x_posterior"], ref["x_posterior"]))
    wP = max(float(np.max(np.abs(a - b)))
             for a, b in zip(lib_["P_posterior"], ref["P_posterior"]))
    HARNESS.check(wx / scale < 1e-10,
                  "library EKF matches the textbook EKF, state", f"{wx:.2e}")
    HARNESS.check(wP / max(scale, 1.0) < 1e-10,
                  "library EKF matches the textbook EKF, covariance", f"{wP:.2e}")


def test_assimilation_beats_no_assimilation():
    # Forecast skill: with assimilation the error must be far below the error of
    # a free-running forecast started from the same wrong initial condition.
    run = simulate(n_obs=800, seed=3)
    mu0 = run["x_true"][0] + np.array([5.0, -5.0, 5.0])
    P0 = 10.0 * np.eye(3)
    est = ekf_library(run, mu0, P0)
    x_free = mu0.copy()
    free = []
    for k in range(len(run["x_true"])):
        for _ in range(OBS_EVERY):
            x_free = rk4_step(x_free)
        free.append(x_free.copy())
    HARNESS.check(rmse(est["x_posterior"], run["x_true"]) <
                  0.5 * rmse(free, run["x_true"]),
                  "assimilating beats a free-running forecast",
                  f"filtered {rmse(est['x_posterior'], run['x_true']):.3f} vs "
                  f"free {rmse(free, run['x_true']):.3f}")


def test_ensemble_beats_the_ekf():
    # The literature's claim about this system: with only x[0] observed and the
    # dynamics this nonlinear, the ensemble family out-forecasts the EKF.  Averaged over several
    # realizations so the assertion is about the method, not the seed.
    n_obs = 600
    gains = []
    for seed in (1, 2, 3, 4, 5):
        run = simulate(n_obs=n_obs, seed=seed)
        mu0 = run["x_true"][0] + np.array([4.0, -4.0, 3.0])
        P0 = 10.0 * np.eye(3)
        ekf = ekf_library(run, mu0, P0)
        etkf = ensemble_filter(run, mu0, P0, cls="ETKF", L=50, seed=seed)
        r_ekf = rmse(ekf["x_posterior"], run["x_true"])
        r_etkf = rmse(etkf["x_posterior"], run["x_true"])
        gains.append(r_ekf / max(r_etkf, 1e-12))
        print(f"      seed {seed}: EKF rmse {r_ekf:.3f}   ETKF rmse {r_etkf:.3f}")
    gain = float(np.median(gains))
    HARNESS.check(gain > 1.05,
                  "ETKF beats the EKF (median RMSE ratio over 5 realizations)",
                  f"median EKF/ETKF = {gain:.3f}")


def test_spread_is_calibrated_where_the_ekf_is_not():
    # The interesting UQ claim on this system.  NEES = (x - xhat)^T P^{-1}
    # (x - xhat) has mean equal to the state dimension (3) when the reported P
    # is the right size for the actual error.
    #
    # The EKF's linearization error inflates its true error but not its P, so it
    # is badly overconfident here.  The ensemble's spread is an honest estimate
    # of its own error and lands on the dimension.  Measured over a few
    # realizations so this is about the methods, not the seed.
    n_obs = 600
    ekf_n, etkf_n = [], []
    for seed in (1, 2, 3):
        run = simulate(n_obs=n_obs, seed=seed)
        mu0 = run["x_true"][0] + np.array([4.0, -4.0, 3.0])
        P0 = 10.0 * np.eye(3)
        ekf = ekf_library(run, mu0, P0)
        etkf = ensemble_filter(run, mu0, P0, cls="ETKF", L=50, seed=seed)
        tail_truth = run["x_true"][150:]
        ekf_n.append(nees(dict((k, v[150:]) for k, v in ekf.items()), tail_truth))
        etkf_n.append(nees(dict((k, v[150:]) for k, v in etkf.items()), tail_truth))
    e_ekf, e_etkf = float(np.median(ekf_n)), float(np.median(etkf_n))
    print(f"      median NEES (dimension = 3): EKF {e_ekf:.2f}   ETKF {e_etkf:.2f}")
    HARNESS.check(1.0 < e_etkf < 8.0,
                  "ETKF spread is calibrated: mean NEES is O(state dimension = 3)",
                  f"{e_etkf:.2f}")
    HARNESS.check(e_ekf > 5.0,
                  "EKF is overconfident here (NEES well above the dimension)",
                  f"{e_ekf:.2f}")
    HARNESS.check(e_ekf > 3.0 * e_etkf,
                  "the EKF's mis-calibration is far worse than the ETKF's",
                  f"{e_ekf:.2f} vs {e_etkf:.2f}")


def test_figures_are_written():
    import tempfile
    run = simulate(n_obs=600, seed=1)
    mu0 = run["x_true"][0] + np.array([4.0, -4.0, 3.0])
    P0 = 10.0 * np.eye(3)
    results = {
        "EKF": ekf_library(run, mu0, P0),
        "ETKF": ensemble_filter(run, mu0, P0, cls="ETKF", L=50, seed=1),
    }
    with tempfile.TemporaryDirectory() as d:
        written = make_figures(d, run, results)
        HARNESS.check(len(written) == 3, "all three figures are written")
        HARNESS.check(all(os.path.getsize(p) > 0 for p in written),
                      "every figure is non-empty")


TESTS = [
    test_model_geometry,
    test_drift_jacobian_matches_central_differences,
    test_rk4_tangent_matches_central_differences,
    test_dynamics_identity,
    test_library_matches_reference_ekf,
    test_assimilation_beats_no_assimilation,
    test_ensemble_beats_the_ekf,
    test_spread_is_calibrated_where_the_ekf_is_not,
    test_figures_are_written,
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--figures":
        outdir = argv[1] if len(argv) > 1 else os.path.join("figures", "lorenz_63")
        run = simulate(n_obs=1500, seed=SEED)
        mu0 = run["x_true"][0] + np.array([4.0, -4.0, 3.0])
        P0 = 10.0 * np.eye(3)
        results = {
            "EKF": ekf_library(run, mu0, P0),
            "ETKF": ensemble_filter(run, mu0, P0, cls="ETKF", L=50, seed=SEED),
        }
        for p in make_figures(outdir, run, results):
            print("wrote", p)
        return 0

    print(__doc__.splitlines()[0])
    for t in TESTS:
        HARNESS.run(t.__name__, t)
    return HARNESS.summary(len(TESTS))


if __name__ == "__main__":
    sys.exit(main())
