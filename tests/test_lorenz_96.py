#!/usr/bin/env python3
"""Tracking on the Lorenz (1996) model -- the assimilation benchmark.

Where tests/test_lorenz_63.py is about *nonlinear dynamics* (3 states), this
one is about **localization and inflation** (40 states on a ring).  Lorenz-96 is
the field's standard assimilation testbed -- it ships with DART, with PDAF and
with DataAssimilationBenchmarks.jl -- precisely because it is the smallest
model with realistic spatial coupling, which is what makes covariance
localization necessary.

Model, Lorenz (1996), F = 8, N = 40, indices cyclic::

    dx[i]/dt = ( x[i+1] - x[i-2] ) x[i-1] - x[i] + F

propagated with fourth-order Runge-Kutta at dt = 0.05.  Assimilation every 5
model steps (Dt_obs = 0.25) of every other state, so the observation network is
sparse -- the setting where a small ensemble's spurious long-range correlations
do real damage.

What is tested here that nothing else tests
-------------------------------------------
* ``set_taper`` for *efficacy* rather than semantics.  Everything else asserts
  what a taper is (ones == none, zero taper, ``LETKF == ETKF + set_taper``).
  Here, at N = 40 with L = 20, the unlocalised ensemble is degraded by
  spurious correlations and the localised one is not.  That is the entire
  reason the knob exists.
* ``set_inflation`` for efficacy, likewise.
* that smoothing helps on a high-dimensional chaotic problem (deferred, for a
  narrower reason than it used to be: the EnKS forward pass propagates through
  the LINEAR time_update F x + u, and this model's dynamics are nonlinear, so
  the smoother cannot run the model on each member the way ensemble_filter
  below does.  The Bryson-Frazier form itself needs only the innovation
  covariance inverse and imposes no condition on L versus N -- see
  src/smoother.hpp.  Smoothing is covered in depth by
  tests/test_linear_gaussian_smoother.py, including against closed forms).

Lorenz's own geometry is checked too: x[i] = F for all i is an exact
equilibrium of the field.

Chaotic systems
---------------
As in the Lorenz-63 file: statistics over a long window and relative claims
between methods, never pointwise values.

Run standalone (writes figures with --figures)::

    python3 tests/test_lorenz_96.py
    python3 tests/test_lorenz_96.py --figures
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _estimation_loader import Harness, load_lib  # noqa: E402


# ==========================================================================
# Lorenz (1996)
# ==========================================================================

N_STATE = 40
FORCING = 8.0
DT = 0.05
OBS_EVERY = 5          # assimilate every 5 model steps -> Dt_obs = 0.25
VAR_R = 1.0            # measurement noise variance, per observed state
VAR_Q = 1.0e-3         # process noise variance per model step, per state
L_ENS = 20             # ensemble size (L << N, the regime that needs a taper)
TAPER_KAPPA = 3.0      # periodic kernel width for the localization taper

SEED = 20261006

# Observe every other state: a sparse network, as in real observing systems.
OBS_INDEX = np.arange(0, N_STATE, 2)
H = np.zeros((len(OBS_INDEX), N_STATE))
for k, i in enumerate(OBS_INDEX):
    H[k, i] = 1.0
R = VAR_R * np.eye(len(OBS_INDEX))


def drift(x, F=FORCING):
    """f(x) -- the Lorenz (1996) field, cyclic indices."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def drift_jacobian(x):
    """df/dx -- four nonzeros per row: on i, i+1, i-1, i-2 (cyclic)."""
    n = x.shape[0]
    J = np.zeros((n, n))
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        J[i, ip1] = x[im1]
        J[i, im2] = -x[im1]
        J[i, im1] = x[ip1] - x[im2]
        J[i, i] = -1.0
    return J


def rk4_step(x, dt=DT):
    k1 = drift(x)
    k2 = drift(x + 0.5 * dt * k1)
    k3 = drift(x + 0.5 * dt * k2)
    k4 = drift(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_tangent(x, dt=DT):
    """Variational Runge-Kutta: the exact Jacobian of the discrete map."""
    I = np.eye(x.shape[0])
    k1 = drift(x);              K1 = drift_jacobian(x)
    a2 = x + 0.5 * dt * k1;     k2 = drift(a2);    K2 = drift_jacobian(a2) @ (I + 0.5 * dt * K1)
    a3 = x + 0.5 * dt * k2;     k3 = drift(a3);    K3 = drift_jacobian(a3) @ (I + 0.5 * dt * K2)
    a4 = x + dt * k3;           k4 = drift(a4);    K4 = drift_jacobian(a4) @ (I + dt * K3)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    F = I + (dt / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
    return x_next, F


def periodic_taper(n=N_STATE, kappa=TAPER_KAPPA):
    """Localization taper on the ring.

    C_ij = exp(-kappa (1 - cos(2 pi (i-j)/n))).  This is positive definite on
    the circle -- it is exp(-kappa) times exp(kappa cos theta), and the latter
    has nonnegative Fourier coefficients (modified Bessel functions) -- so the
    Schur product keeps C o P positive semidefinite.  Correlation decays over
    a few grid points and then flattens at exp(-2 kappa).
    """
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    theta = 2.0 * np.pi * (i - j) / n
    return np.asfortranarray(np.exp(-kappa * (1.0 - np.cos(theta))))


# ==========================================================================
# Realization and the filters
# ==========================================================================

def simulate(n_obs=2000, seed=SEED, x0=None, var_q=VAR_Q):
    """A twin experiment on the ring: truth from the model with process noise
    and observations of every other state."""
    rng = np.random.default_rng(seed)
    if x0 is None:
        x = np.full(N_STATE, FORCING) + rng.normal(scale=0.1, size=N_STATE)
        for _ in range(1000):
            x = rk4_step(x)
        x0 = x.copy()
    Qf = math.sqrt(var_q) * np.eye(N_STATE)

    x_true, y_obs = [], []
    x = x0.copy()
    for k in range(n_obs):
        for _ in range(OBS_EVERY):
            x = rk4_step(x)
            if var_q > 0:
                x = x + Qf @ rng.standard_normal(N_STATE)
        x_true.append(x.copy())
        y_obs.append(H @ x + math.sqrt(VAR_R) * rng.standard_normal(len(OBS_INDEX)))
    return {"x_true": np.array(x_true), "y": [np.asarray(v).ravel() for v in y_obs],
            "Q": np.eye(N_STATE) * var_q}


def ekf_library(run, mu0, P0):
    """The EKF on the ring -- dynamics via u = f(x) - F x, as in Lorenz-63."""
    lib = load_lib()
    kf = lib.KF(N_STATE)
    kf.initialize(mu0, P0)
    x_post = []
    for y_k in run["y"]:
        for _ in range(OBS_EVERY):
            x_prior = kf.state().copy()
            x_next, F = rk4_step_tangent(x_prior)
            u = x_next - F @ x_prior
            kf.time_update(F, run["Q"], u)
        x_pred = kf.state().copy()
        y_eff = y_k - (H @ x_pred) + (H @ x_pred)   # h is linear here
        kf.measurement_update(y_eff, H, R)
        x_post.append(kf.state().copy())
    return {"x_posterior": x_post}


def ensemble_filter(run, mu0, P0, cls="ETKF", L=L_ENS, seed=1,
                    taper=None, inflation=1.0):
    """A Monte Carlo method on the ring.  The forecast is the model run on each
    member (``members`` / ``set_members``); the analysis is the library's."""
    lib = load_lib()
    if cls in ("LETKF", "LEKS"):
        filt = getattr(lib, cls)(N_STATE, L, np.asfortranarray(taper), seed)
    else:
        filt = getattr(lib, cls)(N_STATE, L, seed)
        if taper is not None:
            filt.set_taper(np.asfortranarray(taper))
    if inflation != 1.0:
        filt.set_inflation(inflation)

    rng = np.random.default_rng(seed + 555)
    P0f = np.linalg.cholesky(P0)
    filt.set_members(mu0[:, None] + P0f @ rng.standard_normal((N_STATE, L)))

    x_post, P_post = [], []
    for y_k in run["y"]:
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


def smoothed_of(run, cls="LETKF", L=L_ENS, seed=1, taper=None):
    """Fixed-interval smoothing on the ring (the smoother half of the family)."""
    lib = load_lib()
    n = len(run["y"])
    if cls == "LEKS":
        ks = lib.LEKS(N_STATE, L, np.asfortranarray(taper), seed)
    else:
        ks = lib.EnKS(N_STATE, L, seed)
        if taper is not None:
            ks.set_taper(np.asfortranarray(taper))
    mu0 = np.full(N_STATE, FORCING)
    P0 = 10.0 * np.eye(N_STATE)
    ks.initialize(mu0, P0)
    yv = list(run["y"])
    Hv = [H] * n
    Rv = [R] * n
    Fv = [np.eye(N_STATE)] * n       # unused by EnKS::smooth's own forecast
    Qv = [run["Q"]] * n
    return ks.smooth(yv, Hv, Rv, Fv, Qv)


def rmse(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d ** 2)))


# ==========================================================================
# Figures
# ==========================================================================

def make_figures(outdir, run, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as plt

    os.makedirs(outdir, exist_ok=True)
    xt = run["x_true"]
    n = len(xt)
    written = []
    t = np.arange(n) * OBS_EVERY * DT

    fig, ax = plt.subplots()
    for name, est in results.items():
        err = np.linalg.norm(np.asarray(est["x_posterior"]) - xt, axis=1) / math.sqrt(N_STATE)
        ax.plot(t, err, lw=0.8, label=name)
    ax.set_xlabel(r"time")
    ax.set_ylabel(r"RMSE per state component")
    ax.set_title("Lorenz (1996): estimation error with $L=%d$, $N=%d$" % (L_ENS, N_STATE))
    ax.legend()
    p = os.path.join(outdir, "lorenz96_rmse.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots()
    im = ax.imshow(xt.T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], 0, N_STATE])
    ax.set_xlabel(r"time")
    ax.set_ylabel(r"state index $i$")
    ax.set_title("Lorenz (1996): the truth (the travelling structures)")
    fig.colorbar(im, ax=ax)
    p = os.path.join(outdir, "lorenz96_truth.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots()
    C = periodic_taper()
    ax.plot(C[0], lw=1.0)
    ax.set_xlabel(r"$|i-j|$ on the ring")
    ax.set_ylabel(r"$C_{ij}$")
    ax.set_title(r"The localization taper, $\exp(-\kappa(1-\cos\theta))$")
    p = os.path.join(outdir, "lorenz96_taper.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)
    return written


# ==========================================================================
# Tests
# ==========================================================================

HARNESS = Harness()

# The ensemble must be able to *represent* the tapered analysis.  With a taper
# the gain leaves the ensemble span, so the Joseph target has rank up to
# (L-1) + M rather than L-1 -- see test_taper_raises_the_rank_of_the_analysis.
# L_BIG is chosen so that L-1 comfortably exceeds that rank; L_SMALL is not.
L_BIG = 45
L_SMALL = 20


def _joseph_target(x_prior, P_prior, taper):
    """The covariance the analysis is supposed to have: Joseph form on the
    untapered prior with the gain from the tapered one (thesis eq. 4.26)."""
    Pg = P_prior if taper is None else np.asfortranarray(taper) * P_prior
    S = H @ Pg @ H.T + R
    K = Pg @ H.T @ np.linalg.inv(S)
    IKH = np.eye(P_prior.shape[0]) - K @ H
    return IKH @ P_prior @ IKH.T + K @ R @ K.T


def _run_pair(L_ens, n_obs=400, seed=5):
    """ETKF and ETKF + taper from one shared prior ensemble and one truth."""
    run = simulate(n_obs=n_obs, seed=seed)
    mu0 = np.full(N_STATE, FORCING) + 2.0
    P0 = 10.0 * np.eye(N_STATE)
    C = periodic_taper()
    plain = ensemble_filter(run, mu0, P0, cls="ETKF", L=L_ens, seed=seed)
    loc = ensemble_filter(run, mu0, P0, cls="ETKF", L=L_ens, seed=seed, taper=C)
    return run, C, plain, loc


def test_model_geometry():
    # x[i] = F for all i is an exact equilibrium -- a constant state is
    # advection-free and the -x[i] + F terms cancel.  Analytic, no integration.
    x = np.full(N_STATE, FORCING)
    HARNESS.check(np.linalg.norm(drift(x)) < 1e-10,
                  "x[i] = F is an equilibrium of the Lorenz (1996) field",
                  f"{np.linalg.norm(drift(x)):.2e}")
    J = drift_jacobian(x)
    lam = np.max(np.real(np.linalg.eigvals(J)))
    HARNESS.check(lam > 0, "the symmetric equilibrium is unstable at F = 8",
                  f"{lam:.3f}")


def test_jacobians_match_central_differences():
    rng = np.random.default_rng(4)
    x = np.full(N_STATE, FORCING) + rng.normal(scale=1.0, size=N_STATE)
    eps = 1e-6
    J = drift_jacobian(x)
    Jfd = np.zeros((N_STATE, N_STATE))
    for j in range(N_STATE):
        e = np.zeros(N_STATE); e[j] = eps
        Jfd[:, j] = (drift(x + e) - drift(x - e)) / (2 * eps)
    HARNESS.check(float(np.max(np.abs(J - Jfd))) < 1e-6,
                  "df/dx matches central differences",
                  f"{float(np.max(np.abs(J - Jfd))):.2e}")
    _, F = rk4_step_tangent(x)
    Ffd = np.zeros((N_STATE, N_STATE))
    for j in range(N_STATE):
        e = np.zeros(N_STATE); e[j] = eps
        Ffd[:, j] = (rk4_step(x + e) - rk4_step(x - e)) / (2 * eps)
    HARNESS.check(float(np.max(np.abs(F - Ffd))) < 1e-5,
                  "the RK4 tangent matches central differences of the map",
                  f"{float(np.max(np.abs(F - Ffd))):.2e}")


def test_taper_is_positive_definite_on_the_ring():
    C = periodic_taper()
    w = np.linalg.eigvalsh(C)
    HARNESS.check(w.min() > -1e-10, "the periodic taper is positive definite",
                  f"min eig {w.min():.2e}")
    HARNESS.check(np.allclose(np.diag(C), 1.0), "the taper has unit diagonal")
    HARNESS.check(C[0, N_STATE // 2] < C[0, 1],
                  "the taper decays with distance on the ring")


def test_taper_raises_the_rank_of_the_analysis():
    # The algebraic fact behind the rank condition.  Without a taper the gain
    # lies in the ensemble span and the Joseph target has rank <= L-1.  With
    # one the gain points outside that span and the target can reach (L-1)+M,
    # which no L-member ensemble can represent.
    Lm = L_SMALL
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_STATE, Lm))
    X = X - X.mean(axis=1, keepdims=True)
    P = X @ X.T / (Lm - 1)
    ranks = []
    for taper in (None, periodic_taper()):
        Jt = _joseph_target(None, P, taper)
        w = np.linalg.eigvalsh(Jt)
        ranks.append(int((w > 1e-10 * w.max()).sum()))
    HARNESS.check(ranks[0] <= Lm - 1,
                  "without a taper the Joseph target fits in L-1 dimensions",
                  f"rank {ranks[0]} vs L-1 = {Lm - 1}")
    HARNESS.check(ranks[1] > Lm - 1,
                  "with a taper it does not -- the gain leaves the ensemble span",
                  f"rank {ranks[1]} vs L-1 = {Lm - 1}")


def _target_mismatch(L_ens, taper):
    """How far the analysis covariance is from the tapered Kalman target.

    The ensemble is handed the *exact* prior (mean and sample covariance), so
    the only discrepancy is the truncation the taper forces -- this is the
    |ETKF - LKF| quantity, measured.
    """
    lib = load_lib()
    rng = np.random.default_rng(2)
    mu = np.full(N_STATE, FORCING)
    X = mu[:, None] + rng.normal(scale=1.0, size=(N_STATE, L_ens))
    P = np.cov(X)
    P = 0.5 * (P + P.T)
    target = _joseph_target(mu, P, periodic_taper() if taper else None)

    if taper is not None:
        filt = lib.LETKF(N_STATE, L_ens, np.asfortranarray(periodic_taper()), 3)
    else:
        filt = lib.ETKF(N_STATE, L_ens, 3)
        if taper is not None:
            filt.set_taper(np.asfortranarray(taper))
    filt.set_members(np.asfortranarray(X))
    filt.measurement_update(H @ mu + 0.1, H, R)     # y value is irrelevant here
    got = filt.covariance()
    return float(np.linalg.norm(got - target) / max(np.linalg.norm(target), 1e-300))


def test_localization_helps_when_the_ensemble_can_represent_the_analysis():
    # At L-1 >= rank(Joseph) the ensemble realizes the tapered Kalman analysis
    # exactly, so the estimate improves: the taper removes the spurious
    # long-range correlations that a 45-member sample of a 40-dimensional
    # chaotic system cannot avoid.
    run, C, plain, loc = _run_pair(L_BIG)
    r_plain = rmse(plain["x_posterior"], run["x_true"])
    r_loc = rmse(loc["x_posterior"], run["x_true"])
    print(f"      L={L_BIG}: ETKF rmse {r_plain:.3f}   ETKF + taper rmse {r_loc:.3f}")
    HARNESS.check(r_loc < 0.75 * r_plain,
                  "localization improves the estimate at L = 45, N = 40",
                  f"{r_loc:.3f} vs {r_plain:.3f}")
    err = _target_mismatch(L_BIG, True)
    HARNESS.check(err < 0.05,
                  "at L = 45 the tapered analysis matches the tapered Kalman target",
                  f"relative mismatch {err:.4f}")


def test_truncation_is_visible_when_the_ensemble_cannot():
    # The rank condition is not merely formal: at L = 20 the truncation to
    # L-1 dimensions is a measurable error against the same target.
    err_small = _target_mismatch(L_SMALL, True)
    err_big = _target_mismatch(L_BIG, True)
    print(f"      relative mismatch vs the tapered Kalman target: "
          f"L={L_SMALL} {err_small:.4f}   L={L_BIG} {err_big:.4f}")
    HARNESS.check(err_small > 0.05,
                  "at L = 20 the truncated analysis misses the target",
                  f"relative mismatch {err_small:.4f}")
    HARNESS.check(err_small > 3.0 * err_big,
                  "and the gap is several times the one at L = 45",
                  f"{err_small:.4f} vs {err_big:.4f}")


def test_assimilation_beats_no_assimilation():
    # Compare over a finite window: the free run leaves the attractor and the
    # quadratic nonlinearity blows up, which is the usual fate of an
    # unassimilated Lorenz (1996) forecast.
    run, C, plain, loc = _run_pair(L_BIG, n_obs=300)
    mu0 = np.full(N_STATE, FORCING) + 2.0
    x_free = mu0.copy()
    free = []
    for k in range(len(run["x_true"])):
        for _ in range(OBS_EVERY):
            x_free = rk4_step(x_free)
        free.append(x_free.copy())
        if not np.all(np.isfinite(x_free)):
            break
    n_use = min(len(free), 100, len(run["x_true"]))
    HARNESS.check(rmse(loc["x_posterior"][:n_use], run["x_true"][:n_use]) <
                  0.5 * rmse(free[:n_use], run["x_true"][:n_use]),
                  "assimilating beats a free-running forecast",
                  f"filtered {rmse(loc['x_posterior'][:n_use], run['x_true'][:n_use]):.3f} "
                  f"vs free {rmse(free[:n_use], run['x_true'][:n_use]):.3f}")


def test_figures_are_written():
    import tempfile
    run, C, plain, loc = _run_pair(L_BIG, n_obs=400)
    results = {"ETKF": plain, "ETKF + taper (LETKF)": loc}
    with tempfile.TemporaryDirectory() as d:
        written = make_figures(d, run, results)
        HARNESS.check(len(written) == 3, "all three figures are written")
        HARNESS.check(all(os.path.getsize(p) > 0 for p in written),
                      "every figure is non-empty")


TESTS = [
    test_model_geometry,
    test_jacobians_match_central_differences,
    test_taper_is_positive_definite_on_the_ring,
    test_taper_raises_the_rank_of_the_analysis,
    test_localization_helps_when_the_ensemble_can_represent_the_analysis,
    test_truncation_is_visible_when_the_ensemble_cannot,
    test_assimilation_beats_no_assimilation,
    test_figures_are_written,
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--figures":
        outdir = argv[1] if len(argv) > 1 else os.path.join("figures", "lorenz_96")
        run, C, plain, loc = _run_pair(L_BIG, n_obs=1500)
        results = {"ETKF": plain, "ETKF + taper (LETKF)": loc}
        for p in make_figures(outdir, run, results):
            print("wrote", p)
        return 0

    print(__doc__.splitlines()[0])
    for t in TESTS:
        HARNESS.run(t.__name__, t)
    return HARNESS.summary(len(TESTS))


if __name__ == "__main__":
    sys.exit(main())
