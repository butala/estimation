#!/usr/bin/env python3
"""Reproduction of Kay, *Fundamentals of Statistical Signal Processing,
Volume I: Estimation Theory*, Example 13.4 -- tracking a vehicle with the
extended Kalman filter.

Figures reproduced (they are the content of the example):

    13.22  realization of the vehicle track          -> make_figures()
    13.23  range and bearing of the true track       -> make_figures()
    13.24  true and observed vehicle tracks          -> make_figures()
    13.25  true and extended Kalman filter estimate  -> make_figures()

This is Kay's MIMO case in his own sense: a four-dimensional state observed
through two channels at once (range *and* bearing), nonlinear in the
measurement only.  The dynamics are exactly linear.

Model, as set up in the example
-------------------------------
State (position and velocity in the plane)::

    x[n] = [ r_x[n], r_y[n], v_x[n], v_y[n] ]^T

Dynamics -- constant velocity with time step ``delta`` and process noise on the
velocities only::

    x[n] = A x[n-1] + u[n],     u[n] ~ N(0, Q)
        A = [[1, 0, delta, 0],
             [0, 1, 0,     delta],
             [0, 0, 1,     0    ],
             [0, 0, 0,     1    ]]
        Q = diag(0, 0, var_u, var_u),      var_u = 1e-4

Measurement -- range and bearing from the origin::

    y[n] = h(x[n]) + v[n],      v[n] ~ N(0, R)
        h(x) = [ sqrt(r_x^2 + r_y^2),  atan2(r_y, r_x) ]^T
        R    = diag(var_R, var_beta),      var_R = 0.1,  var_beta = 0.01

Two details of the example that are easy to get wrong and are asserted below:

* The bearing is ``atan2(r_y, r_x)``, not ``arctan(r_y / r_x)``.  On the ideal
  track r_x[n] = 10 - 0.2 n vanishes *exactly* at n = 50, so ``arctan(r_y/r_x)``
  is undefined there and the branch of the angle is lost before and after.
* ``delta = 1`` is folded into A by the construction above; forgetting it
  doubles the velocity scale.

The ideal (noise-free) track is r_x[n] = 10 - 0.2 n, r_y[n] = -5 + 0.2 n, i.e.
constant velocity (-0.2, +0.2), and the true track is one realization of the
process above started from x[0] = [10, -5, -0.2, 0.2].

Time ordering
-------------
Kay's recursion is *time update, then measurement update*.  The library's
``batch`` is measurement, then time.  These coincide at every step provided the
filter is handed the prior at n = 0 rather than the estimate at n = -1, which is
what ``kay_prior`` below does (it applies Kay's initial time update to mu0,
PI0).  With that, the library's posterior at step n is exactly Kay's x-hat(n|n).

Why this lives in the estimation project
----------------------------------------
The extended Kalman filter is *the linear Kalman filter evaluated at the
linearisation*.  Writing H_n = dh/dx at the predicted state x-hat(n|n-1),

    x-hat(n|n) = x-hat(n|n-1) + K_n ( y[n] - h(x-hat(n|n-1)) )
    K_n        = P(n|n-1) H_n^T ( H_n P(n|n-1) H_n^T + R )^{-1}

while ``Filter.measurement_update(y', H', R)`` computes

    x-hat = x-hat^- + K ( y' - H' x-hat^- ).

So with H' = H_n and

    y' = y[n] - h(x-hat(n|n-1)) + H_n x-hat(n|n-1)

we have y' - H' x-hat^- = y[n] - h(x-hat(n|n-1)) exactly, i.e. the library's
linear update *is* Kay's EKF update.  That identity, and an independent
textbook EKF, are what the tests assert.

Run standalone (writes the four figures if matplotlib is available)::

    python3 tests/test_kay_example_13_4.py

or under pytest::

    python3 -m pytest tests/test_kay_example_13_4.py -q

Kay's figures are a single *unseeded* realization, so they cannot be
reproduced pixel for pixel.  What is reproducible -- and what is tested -- is
the model, the recursion, and the statistics of the estimate.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np


# ==========================================================================
# Kay's model -- the constants of Example 13.4
# ==========================================================================

N_STEPS = 101          # n = 0 .. 100
DELTA = 1.0            # time step
VAR_U = 1e-4           # process noise variance on the velocities
VAR_R = 0.1            # range measurement noise variance
VAR_BETA = 1e-2        # bearing measurement noise variance (rad^2)

# Ideal track: r_x = 10 - 0.2 n, r_y = -5 + 0.2 n  (velocity (-0.2, +0.2)).
IDEAL_RX0, IDEAL_RY0 = 10.0, -5.0
IDEAL_VX, IDEAL_VY = -0.2, 0.2

# Default seed for the realization used by --figures and by the docstrings.
# Kay's figures are an unseeded realization, so this is *a* realization of his
# example rather than his exact one.
SEED = 12345

# True initial state and the filter's initial guess.
X0 = np.array([10.0, -5.0, -0.2, 0.2])
MU0 = np.array([5.0, 5.0, 0.0, 0.0])
PI0 = 100.0 * np.eye(4)


def transition(delta: float = DELTA) -> np.ndarray:
    """A -- the constant-velocity transition, x = [r_x, r_y, v_x, v_y].

    Equivalent to ``scipy.linalg.toeplitz([1, 0, 0, 0], r=[1, 0, delta, 0])``,
    which is how Kay's example (and the notebook this was cleaned up from)
    writes it; that equivalence is asserted in the tests.
    """
    return np.array([[1.0, 0.0, delta, 0.0],
                     [0.0, 1.0, 0.0,    delta],
                     [0.0, 0.0, 1.0,    0.0],
                     [0.0, 0.0, 0.0,    1.0]])


def process_noise(var_u: float = VAR_U) -> np.ndarray:
    """Q -- process noise on the velocities only."""
    return np.diag([0.0, 0.0, var_u, var_u])


def measurement_noise(var_R: float = VAR_R, var_beta: float = VAR_BETA) -> np.ndarray:
    """R -- independent range and bearing noise."""
    return np.diag([var_R, var_beta])


def observation(x: np.ndarray) -> np.ndarray:
    """h(x) = [range, bearing].  atan2, not arctan: see the module docstring."""
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def observation_jacobian(x: np.ndarray) -> np.ndarray:
    """dh/dx -- Kay's Jacobian.  Only the position columns are nonzero."""
    r_x, r_y = x[0], x[1]
    rho = np.hypot(r_x, r_y)
    return np.array([[r_x / rho,      r_y / rho,      0.0, 0.0],
                     [-r_y / rho**2,  r_x / rho**2,   0.0, 0.0]])


def ideal_track(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The noise-free track: r_x = 10 - 0.2 n, r_y = -5 + 0.2 n."""
    r_x = IDEAL_RX0 + IDEAL_VX * n
    r_y = IDEAL_RY0 + IDEAL_VY * n
    return r_x, r_y


def kay_prior(mu0: np.ndarray = MU0, PI0: np.ndarray = PI0,
              A: np.ndarray | None = None, Q: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Kay's initial time update.

    ``mu0, PI0`` are the estimate at n = -1; propagating them once produces the
    prior at n = 0, which is what the filter needs so that its first
    measurement update sees y[0] -- matching Kay's time-then-measurement order.
    """
    if A is None:
        A = transition()
    if Q is None:
        Q = process_noise()
    return A @ mu0, A @ PI0 @ A.T + Q


# --------------------------------------------------------------------------
# locate the compiled module (same convention as tests/test_bindings.py)
# --------------------------------------------------------------------------

_KF_CACHE = []


def _load_kf():
    """Return ``estimation.lib.KF`` however the module is available: installed
    as ``estimation.lib``, or built in-tree and pointed at by ``$ESTIMATION_LIB``.

    Memoized: a nanobind module cannot be executed twice in one process (its
    types are registered once), so the lookup must not be repeated.
    """
    if _KF_CACHE:
        return _KF_CACHE[0]
    kf = _load_kf_once()
    _KF_CACHE.append(kf)
    return kf


def _load_kf_once():
    try:
        from estimation.lib import KF
        return KF
    except Exception:  # noqa: BLE001
        pass
    import glob
    import importlib.util
    import pathlib
    candidates = []
    if os.environ.get("ESTIMATION_LIB"):
        candidates.append(os.environ["ESTIMATION_LIB"])
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent
    # Prefer the Python extension module: the glob also matches the C++
    # shared library (build/src/libestimation.dylib), which has no KF.
    found = []
    for pattern in ("lib*.so", "lib*.so.*", "lib*.dylib", "lib*.pyd"):
        found += glob.glob(str(root / "build*" / "src" / pattern))
        found += glob.glob(str(here / pattern))
    found.sort(key=lambda p: ("cpython" not in os.path.basename(p), p))
    candidates += found
    for path in candidates:
        if not os.path.isfile(path):
            continue
        # NB_MODULE(lib, m) exports PyInit_lib, so load as "lib".
        spec = importlib.util.spec_from_file_location("lib", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001
            continue        # not a Python extension (e.g. libestimation.dylib)
        if hasattr(mod, "KF"):
            return mod.KF
    raise SystemExit(
        "cannot find the compiled module (lib*.so); build it and/or set "
        "ESTIMATION_LIB to its path:\n"
        "  cmake -B build-py -DESTATION_BUILD_PYTHON=ON \\\n"
        "        -Dnanobind_DIR=$(python3 -c \"import nanobind; print(nanobind.cmake_dir())\")\n"
        "  cmake --build build-py\n"
        f"searched: {candidates[:6]} ..."
    )


# ==========================================================================
# The two recursions -- the library's linear KF at the linearisation, and an
# independent textbook EKF
# ==========================================================================

def ekf_via_library(y: list[np.ndarray], A: np.ndarray, Q: np.ndarray,
                    R: np.ndarray, mu0: np.ndarray = MU0,
                    PI0: np.ndarray = PI0) -> dict:
    """Kay's EKF, driven by ``estimation.lib.KF``.

    The nonlinear measurement never enters the library: at each step the
    measurement is *re-expressed* as a linear one about the predicted state,
    with the innovation carried in ``y'`` so that ``y' - H' x^-`` is exactly
    ``y[n] - h(x^-)``.  See the module docstring.
    """
    KF = _load_kf()

    kf = KF(A.shape[0])
    mu, PI = kay_prior(mu0, PI0, A, Q)
    kf.initialize(mu, PI)

    x_post, P_post, x_prior, P_prior = [], [], [], []
    for y_n in y:
        x_pred = kf.state().copy()
        P_pred = kf.covariance().copy()
        H_n = observation_jacobian(x_pred)
        h_pred = observation(x_pred)
        # The EKF innovation, written as a linear measurement about x_pred.
        y_eff = y_n - h_pred + H_n @ x_pred

        kf.measurement_update(y_eff, H_n, R)

        x_prior.append(x_pred)
        P_prior.append(P_pred)
        x_post.append(kf.state().copy())
        P_post.append(kf.covariance().copy())

        kf.time_update(A, Q)

    return {"x_posterior": x_post, "P_posterior": P_post,
            "x_prior": x_prior, "P_prior": P_prior}


def ekf_reference(y: list[np.ndarray], A: np.ndarray, Q: np.ndarray,
                  R: np.ndarray, mu0: np.ndarray = MU0,
                  PI0: np.ndarray = PI0) -> dict:
    """The EKF recursion written out as in Kay -- no library involved.

    Included so the test can cross-check the library against a second, literal
    implementation rather than against itself.
    """
    mu, PI = kay_prior(mu0, PI0, A, Q)
    x, P = mu.copy(), PI.copy()
    x_post, P_post, x_prior, P_prior = [], [], [], []
    I = np.eye(A.shape[0])
    for y_n in y:
        H_n = observation_jacobian(x)
        h_pred = observation(x)
        S = H_n @ P @ H_n.T + R
        K = P @ H_n.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        x = x + K @ (y_n - h_pred)                 # x-hat(n|n)
        P = (I - K @ H_n) @ P                      # P(n|n)
        x_prior.append(None)                       # filled below for symmetry
        P_prior.append(None)
        x_post.append(x.copy())
        P_post.append(P.copy())
        x = A @ x                                  # x-hat(n+1|n)
        P = A @ P @ A.T + Q
    return {"x_posterior": x_post, "P_posterior": P_post,
            "x_prior": x_prior, "P_prior": P_prior}


# ==========================================================================
# Realization -- the true track and the noisy range/bearing measurements
# ==========================================================================

def simulate(seed: int = SEED, n_steps: int = N_STEPS,
             A: np.ndarray | None = None, Q: np.ndarray | None = None,
             R: np.ndarray | None = None) -> dict:
    """One realization of the example: true track and measurements.

    ``seed`` fixes the realization.  Kay's figures use an unseeded one, so this
    is *a* realization of his example rather than his exact one.
    """
    if A is None:
        A = transition()
    if Q is None:
        Q = process_noise()
    if R is None:
        R = measurement_noise()
    rng = np.random.default_rng(seed)
    # Q and R are diagonal in the example, but use a general factor so the
    # routine stays correct if they are not.
    Qf = np.linalg.cholesky(Q + 1e-300 * np.eye(Q.shape[0]))
    Rf = np.linalg.cholesky(R)

    x_true = [X0.copy()]
    for _ in range(1, n_steps):
        x_true.append(A @ x_true[-1] + Qf @ rng.standard_normal(A.shape[0]))
    y_true = [observation(x) for x in x_true]
    y = [t + Rf @ rng.standard_normal(len(t)) for t in y_true]

    return {"n": np.arange(n_steps), "x_true": x_true,
            "y_true": y_true, "y": y, "A": A, "Q": Q, "R": R}


def ideal_xy(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return ideal_track(n)


def observed_xy(y: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The measurement converted back to a position -- what a raw
    range/bearing fix would give.  This is Fig 13.24's 'observed track'."""
    r = np.array([y_i[0] for y_i in y])
    b = np.array([y_i[1] for y_i in y])
    return r * np.cos(b), r * np.sin(b)


# ==========================================================================
# Figures 13.22 -- 13.25 (optional: needs matplotlib)
# ==========================================================================

FIG_TITLES = {
    "fig_13_22": "Reproduction of Figure 13.22: Realization of vehicle track",
    "fig_13_23": "Reproduction of Figure 13.23: Range and bearing of true vehicle track",
    "fig_13_24": "Reproduction of Figure 13.24: True and observed vehicle tracks",
    "fig_13_25": "Reproduction of Figure 13.25: True and extended Kalman filter estimate",
}


def make_figures(outdir: str, run: dict, est: dict) -> list[str]:
    """Write Figures 13.22-13.25 to *outdir*.  Returns the file names.

    Uses the Agg backend so this runs headless.  matplotlib is a declared
    dependency (see pyproject.toml).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as plt

    os.makedirs(outdir, exist_ok=True)
    n = run["n"]
    rx_true = np.array([x[0] for x in run["x_true"]])
    ry_true = np.array([x[1] for x in run["x_true"]])
    rx_ideal, ry_ideal = ideal_track(n)
    r_true = np.array([t[0] for t in run["y_true"]])
    beta_true = np.rad2deg([t[1] for t in run["y_true"]])
    rx_obs, ry_obs = observed_xy(run["y"])
    rx_hat = np.array([x[0] for x in est["x_posterior"]])
    ry_hat = np.array([x[1] for x in est["x_posterior"]])

    written = []

    # ---- Figure 13.22: realization of the vehicle track -------------------
    fig, ax = plt.subplots()
    ax.plot(rx_ideal, ry_ideal, ls="--", label="Ideal track")
    ax.plot(rx_true, ry_true, label="True track")
    ax.set_xlabel(r"$r_x[n]$")
    ax.set_ylabel(r"$r_y[n]$")
    ax.set_title(FIG_TITLES["fig_13_22"])
    ax.legend()
    path = os.path.join(outdir, "fig_13_22_track.pdf")
    fig.savefig(path, bbox_inches="tight")   # .pdf -> vector output
    plt.close(fig)
    written.append(path)

    # ---- Figure 13.23: range and bearing of the true track ----------------
    fig, ax = plt.subplots(nrows=2, figsize=(8, 10))
    ax[0].plot(n, r_true)
    ax[0].set_xlabel(r"Sample number, $n$")
    ax[0].set_ylabel(r"$R[n]$")
    ax[0].set_title("Range")
    ax[1].plot(n, beta_true)
    ax[1].set_xlabel(r"Sample number, $n$")
    ax[1].set_ylabel(r"$\beta[n]$")
    ax[1].set_title("Bearing")
    fig.suptitle(FIG_TITLES["fig_13_23"])
    path = os.path.join(outdir, "fig_13_23_range_bearing.pdf")
    fig.savefig(path, bbox_inches="tight")   # .pdf -> vector output
    plt.close(fig)
    written.append(path)

    # ---- Figure 13.24: true and observed tracks ---------------------------
    fig, ax = plt.subplots()
    ax.plot(rx_true, ry_true, label="True track")
    ax.plot(rx_obs, ry_obs, label="Observed track")
    ax.set_xlabel(r"$r_x[n]$")
    ax.set_ylabel(r"$r_y[n]$")
    ax.set_title(FIG_TITLES["fig_13_24"])
    ax.legend()
    path = os.path.join(outdir, "fig_13_24_observed.pdf")
    fig.savefig(path, bbox_inches="tight")   # .pdf -> vector output
    plt.close(fig)
    written.append(path)

    # ---- Figure 13.25: true track and EKF estimate ------------------------
    fig, ax = plt.subplots()
    ax.plot(rx_true, ry_true, label="True track")
    ax.plot(rx_hat, ry_hat, label="Extended Kalman filter estimate")
    ax.set_xlabel(r"$r_x[n]$")
    ax.set_ylabel(r"$r_y[n]$")
    ax.set_title(FIG_TITLES["fig_13_25"])
    ax.legend()
    path = os.path.join(outdir, "fig_13_25_ekf.pdf")
    fig.savefig(path, bbox_inches="tight")   # .pdf -> vector output
    plt.close(fig)
    written.append(path)

    return written


# ==========================================================================
# Tests
# ==========================================================================

_failures = 0
_checks = 0


def check(ok, what, detail=""):
    global _failures, _checks
    _checks += 1
    if ok:
        return
    _failures += 1
    print(f"    FAIL  {what}" + (f"  [{detail}]" if detail else ""))


def run(name, fn):
    print(f"[ {name} ]")
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        global _failures
        _failures += 1
        print(f"    FAIL  {name} raised {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# the model, exactly as Kay sets it up
# --------------------------------------------------------------------------

def test_model_constants():
    A = transition()
    Q = process_noise()
    R = measurement_noise()

    # Constant-velocity structure with delta = 1 folded in.
    check(np.allclose(A, [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
          "A is the constant-velocity transition with delta = 1")
    check(np.allclose(transition(2.0)[0, 2], 2.0), "delta scales the velocity column")

    # Noise structure: process noise on the velocities only, measurement noise
    # diagonal in (range, bearing).
    check(np.allclose(Q, np.diag([0, 0, VAR_U, VAR_U])),
          "Q is process noise on the velocities only")
    check(np.allclose(R, np.diag([VAR_R, VAR_BETA])),
          "R is diag(var_R, var_beta)")

    # Ideal track and the true initial state.
    n = np.arange(N_STEPS)
    rx, ry = ideal_track(n)
    check(np.allclose([rx[0], ry[0]], [10.0, -5.0]), "ideal track starts at (10, -5)")
    check(np.allclose([rx[-1], ry[-1]], [-10.0, 15.0]), "ideal track ends at (-10, 15)")
    check(np.allclose(np.diff(rx), -0.2) and np.allclose(np.diff(ry), 0.2),
          "ideal track has constant velocity (-0.2, +0.2)")
    check(np.allclose([X0[0], X0[1]], [rx[0], ry[0]]), "x[0] lies on the ideal track")
    check(np.allclose([X0[2], X0[3]], [IDEAL_VX, IDEAL_VY]),
          "x[0] carries the ideal velocity")
    check(MU0.shape == (4,) and PI0.shape == (4, 4), "initial guess is 4-dimensional")


def test_transition_matches_kays_toeplitz_construction():
    # Kay (and the original notebook) builds A as
    #     scipy.linalg.toeplitz([1, 0, 0, 0], r = [1, 0, delta, 0])
    # which is an obscure way to say "constant velocity".  transition() spells
    # the matrix out for readability; this pins that the two agree, so the
    # explicit form cannot silently drift from the source's construction.
    from scipy.linalg import toeplitz
    for delta in (0.5, 1.0, 2.0):
        A_ref = toeplitz([1.0, 0.0, 0.0, 0.0], r=[1.0, 0.0, delta, 0.0])
        check(np.allclose(transition(delta), A_ref),
              f"transition(delta={delta}) == toeplitz([1,0,0,0], r=[1,0,delta,0])")


def test_bearing_needs_atan2():
    # r_x[n] = 10 - 0.2 n vanishes exactly at n = 50, so arctan(r_y/r_x) is
    # undefined there and loses the branch either side of it.  This is why the
    # example must use atan2.
    n = np.arange(N_STEPS)
    rx, ry = ideal_track(n)
    check(rx[50] == 0.0, "ideal r_x vanishes exactly at n = 50", f"{rx[50]!r}")

    x50 = np.array([rx[50], ry[50], IDEAL_VX, IDEAL_VY])
    b = observation(x50)[1]
    check(abs(b - math.pi / 2) < 1e-12,
          "atan2 gives +pi/2 at the crossing point", f"{b!r}")

    # On the far side of the crossing r_x < 0 and arctan(r_y/r_x) lands on the
    # wrong branch -- it is off by pi -- while atan2 does not.  (Before the
    # crossing r_x > 0 and the two agree, which is why the naive form can look
    # correct in a short test.)
    for n_i in (51, 60, 100):
        x = np.array([rx[n_i], ry[n_i], 0.0, 0.0])
        naive = np.arctan(ry[n_i] / rx[n_i])
        good = observation(x)[1]
        check(abs(naive - good) > 1.0,
              f"arctan(r_y/r_x) loses the branch at n = {n_i}",
              f"arctan={naive:.3f} atan2={good:.3f}")
    # And it agrees with atan2 while r_x > 0, so the two are only equivalent
    # there -- not in general.
    x49 = np.array([rx[49], ry[49], 0.0, 0.0])
    check(abs(np.arctan(ry[49] / rx[49]) - observation(x49)[1]) < 1e-12,
          "arctan and atan2 agree while r_x > 0 (n = 49)")


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        x = rng.normal(size=4)
        x[0] += 3.0 * np.sign(x[0] if x[0] != 0 else 1.0)   # keep away from the origin
        if np.hypot(x[0], x[1]) < 0.5:
            continue
        H = observation_jacobian(x)
        eps = 1e-6
        Hfd = np.zeros_like(H)
        for j in range(4):
            e = np.zeros(4)
            e[j] = eps
            Hfd[:, j] = (observation(x + e) - observation(x - e)) / (2 * eps)
        worst = max(worst, float(np.max(np.abs(H - Hfd))))
    check(worst < 1e-7, "observation_jacobian matches central differences", f"{worst:.2e}")

    # The measurement depends on position only.
    x = np.array([3.0, -4.0, 0.7, -0.2])
    check(np.allclose(observation_jacobian(x)[:, 2:], 0.0),
          "the Jacobian's velocity columns are zero")


def test_ekf_identity():
    # The library's linear update at the linearisation is Kay's EKF update:
    # the quantity y' - H' x^- that the library sees as its innovation must be
    # exactly the EKF innovation y[n] - h(x^-).  Checked at the priors the
    # filter actually visited.
    run_ = simulate(seed=4242, n_steps=30)
    est = ekf_via_library(run_["y"], run_["A"], run_["Q"], run_["R"])
    worst = 0.0
    scale = 0.0
    for y_n, x_pred in zip(run_["y"], est["x_prior"]):
        H_n = observation_jacobian(x_pred)
        h_pred = observation(x_pred)
        y_eff = y_n - h_pred + H_n @ x_pred          # the linearised measurement
        innovation = y_eff - H_n @ x_pred            # what the library computes
        target = y_n - h_pred                        # what Kay's EKF computes
        worst = max(worst, float(np.max(np.abs(innovation - target))))
        scale = max(scale, float(np.max(np.abs(target))))
    check(worst / max(scale, 1e-300) < 1e-12,
          "y' - H' x^- == y - h(x^-) (the EKF identity)", f"{worst:.2e}")


def test_library_matches_reference_ekf():
    run_ = simulate(seed=99, n_steps=N_STEPS)
    A, Q, R = run_["A"], run_["Q"], run_["R"]
    lib = ekf_via_library(run_["y"], A, Q, R)
    ref = ekf_reference(run_["y"], A, Q, R)

    worst_x = max(float(np.max(np.abs(a - b)))
                  for a, b in zip(lib["x_posterior"], ref["x_posterior"]))
    worst_P = max(float(np.max(np.abs(a - b)))
                  for a, b in zip(lib["P_posterior"], ref["P_posterior"]))
    scale_x = max(float(np.max(np.abs(b))) for b in ref["x_posterior"])
    scale_P = max(float(np.max(np.abs(b))) for b in ref["P_posterior"])
    check(worst_x / scale_x < 1e-10,
          "library EKF matches the textbook EKF, state", f"{worst_x:.2e}")
    check(worst_P / scale_P < 1e-10,
          "library EKF matches the textbook EKF, covariance", f"{worst_P:.2e}")


def test_estimate_tracks_the_truth():
    # Fig 13.25's content: the EKF pulls onto the track from a poor initial
    # guess and then stays there.
    run_ = simulate(seed=2026, n_steps=N_STEPS)
    A, Q, R = run_["A"], run_["Q"], run_["R"]
    est = ekf_via_library(run_["y"], A, Q, R)

    x_true = np.array(run_["x_true"])
    x_hat = np.array(est["x_posterior"])
    err = np.linalg.norm(x_hat[:, :2] - x_true[:, :2], axis=1)

    # The initial guess (5, 5) is far from the track start (10, -5).
    check(err[0] > 3.0, "the EKF starts well away from the track", f"{err[0]:.2f}")
    # After a few steps it has converged onto it.
    tail = err[N_STEPS // 2:]
    check(tail.max() < 3.0, "the EKF stays on the track after convergence",
          f"max tail error {tail.max():.2f}")
    check(float(np.mean(tail)) < 1.5, "mean late-track position error is small",
          f"{np.mean(tail):.2f}")

    # And it beats the raw range/bearing fix (Fig 13.24's other curve).
    rx_obs, ry_obs = observed_xy(run_["y"])
    obs_err = np.hypot(rx_obs - x_true[:, 0], ry_obs - x_true[:, 1])
    check(float(np.mean(err[N_STEPS // 2:])) < float(np.mean(obs_err[N_STEPS // 2:])),
          "the EKF is more accurate than a raw range/bearing fix")


def test_estimate_is_consistent_with_its_covariance():
    # Normalized estimation error squared should be O(1) once converged: the
    # reported P has to be roughly the size of the actual error.
    run_ = simulate(seed=7, n_steps=N_STEPS)
    A, Q, R = run_["A"], run_["Q"], run_["R"]
    est = ekf_via_library(run_["y"], A, Q, R)
    x_true = np.array(run_["x_true"])
    x_hat = np.array(est["x_posterior"])
    nees = []
    for i in range(N_STEPS // 2, N_STEPS):
        d = x_hat[i] - x_true[i]
        P = est["P_posterior"][i]
        nees.append(float(d @ np.linalg.solve(P, d)))
    mean_nees = float(np.mean(nees))
    # Dimension 4: E[NEES] = 4 if consistent. Allow a wide band -- this is one
    # realization of a nonlinear filter.
    check(1.0 < mean_nees < 12.0,
          "mean NEES is O(state dimension = 4)", f"{mean_nees:.2f}")


def test_reproducible_with_a_seed():
    a = simulate(seed=5, n_steps=20)
    b = simulate(seed=5, n_steps=20)
    c = simulate(seed=6, n_steps=20)
    same = all(np.array_equal(u, v) for u, v in zip(a["x_true"], b["x_true"]))
    diff = any(not np.array_equal(u, v) for u, v in zip(a["x_true"], c["x_true"]))
    check(same, "the same seed reproduces the same realization")
    check(diff, "a different seed gives a different realization")


def test_figures_13_22_to_13_25_are_written():
    import tempfile
    run_ = simulate(seed=1, n_steps=N_STEPS)
    est = ekf_via_library(run_["y"], run_["A"], run_["Q"], run_["R"])
    with tempfile.TemporaryDirectory() as d:
        written = make_figures(d, run_, est)
        check(len(written) == 4, "all four figures (13.22-13.25) are written")
        check(sorted(os.path.basename(p) for p in written) ==
              ["fig_13_22_track.pdf", "fig_13_23_range_bearing.pdf",
               "fig_13_24_observed.pdf", "fig_13_25_ekf.pdf"],
              "the four figures carry Kay's figure numbers")
        check(all(os.path.getsize(p) > 0 for p in written),
              "every figure is non-empty")


TESTS = [
    test_model_constants,
    test_transition_matches_kays_toeplitz_construction,
    test_bearing_needs_atan2,
    test_jacobian_matches_finite_differences,
    test_ekf_identity,
    test_library_matches_reference_ekf,
    test_estimate_tracks_the_truth,
    test_estimate_is_consistent_with_its_covariance,
    test_reproducible_with_a_seed,
    test_figures_13_22_to_13_25_are_written,
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # `python3 tests/test_kay_example_13_4.py --figures [dir]`
    # writes Kay's Figures 13.22-13.25 and exits -- the reproducible half of
    # the example, without running the assertions.
    if argv and argv[0] == "--figures":
        outdir = argv[1] if len(argv) > 1 else os.path.join("figures", "kay_13_4")
        run_ = simulate(seed=SEED, n_steps=N_STEPS)
        est = ekf_via_library(run_["y"], run_["A"], run_["Q"], run_["R"])
        for path in make_figures(outdir, run_, est):
            print("wrote", path)
        return 0

    print(__doc__.splitlines()[0])
    for t in TESTS:
        run(t.__name__, t)
    print(f"\n{_checks} checks passed, {_failures} failed ({len(TESTS)} cases)")
    print("SOME CHECKS FAILED" if _failures else "ALL CHECKS PASSED")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
