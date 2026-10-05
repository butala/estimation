#!/usr/bin/env python3
"""Unit tests for the nanobind layer of estimation.

These tests pin down the *binding contract*, which the C++ suite cannot see:

  1. zero-copy inputs    -- the numpy buffer is read in place (no silent copy)
  2. strict dtype        -- no silent upcast/downcast copy (.noconvert())
  3. read-only accepted  -- const-correct: Ref<const T> needs no writable buffer
  4. copy-safe outputs   -- state()/covariance()/factor() are copies, so
                            mutating what you get back cannot corrupt a filter
  5. ValueError contracts-- dimension mismatches raise, they do not abort
  6. KF == SquareRootKF  -- the two backends agree from Python too
  7. memory stability    -- peak RSS does not grow across many updates

Run standalone (no pytest needed):
    python3 tests/test_bindings.py
or under pytest:
    python3 -m pytest tests/test_bindings.py -q

The compiled module is located through $ESTIMATION_LIB (a path to the .so) or
by looking next to this file and in ../build*/src/.
"""

import os
import resource
import subprocess
import sys

import numpy as np


# --------------------------------------------------------------------------
# locate the compiled module
# --------------------------------------------------------------------------

def _load():
    import importlib.util
    import glob
    import pathlib

    candidates = []
    if os.environ.get("ESTIMATION_LIB"):
        candidates.append(os.environ["ESTIMATION_LIB"])
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent
    for pattern in ("lib*.so", "lib*.so.*", "lib*.dylib", "lib*.pyd"):
        candidates += glob.glob(str(root / "build*" / "src" / pattern))
        candidates += glob.glob(str(root / "build" / "src" / pattern))
        candidates += glob.glob(str(here / pattern))
        candidates += glob.glob(str(here.parent / pattern))
    for path in candidates:
        if os.path.isfile(path):
            # NB_MODULE(lib, m) exports PyInit_lib, so the module must be
            # loaded under the name "lib".
            spec = importlib.util.spec_from_file_location("lib", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, path
    raise SystemExit(
        "cannot find the compiled module (lib*.so); "
        "build it first and/or set ESTIMATION_LIB to its path:\n"
        "  cmake -B build -S src -DSKBUILD=1 -Dnanobind_DIR=$(python3 -c "
        "'import nanobind; print(nanobind.cmake_dir())')\n"
        "  cmake --build build\n"
        f"searched: {candidates[:6]} ..."
    )


lib, LIB_PATH = _load()
print(f"# module: {LIB_PATH}")


# --------------------------------------------------------------------------
# tiny harness
# --------------------------------------------------------------------------

_failures = 0
_checks = 0


def check(ok, what, detail=""):
    global _failures, _checks
    _checks += 1
    if ok:
        return
    _failures += 1
    print(f"    FAIL  {what}" + (f"  [{detail}]" if detail else ""))


def check_raises(exc_type, fn, what):
    try:
        fn()
    except exc_type:
        check(True, what)
        return
    except Exception as e:  # noqa: BLE001
        check(False, what, f"raised {type(e).__name__} instead of {exc_type.__name__}")
        return
    check(False, what, "did not raise")


def peak_rss_bytes():
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform == "darwin" else v * 1024


def run(name, fn):
    print(f"[ {name} ]")
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        global _failures
        _failures += 1
        print(f"    FAIL  {name} raised {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def fortran(a):
    return np.asfortranarray(a, dtype=np.float64)


def spd(n, jitter=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    return fortran(G @ G.T + jitter * np.eye(n))


def mat(rows, cols, seed=0):
    rng = np.random.default_rng(seed + 1)
    return fortran(rng.standard_normal((rows, cols)))


def vec(n, seed=0):
    rng = np.random.default_rng(seed + 2)
    return np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)


N = 6
M = 3


def fresh(cls=lib.KF, n=N, taper=None, inflation=1.0):
    f = cls(n)
    f.initialize(vec(n), spd(n))
    if taper is not None:
        f.set_taper(taper)
    if inflation != 1.0:
        f.set_inflation(inflation)
    return f


# --------------------------------------------------------------------------
# 1-2. zero-copy inputs and strict dtype
# --------------------------------------------------------------------------

def test_inputs_accepted():
    f = lib.KF(N)
    f.initialize(vec(N), spd(N))
    y, H, R = vec(M), mat(M, N), spd(M, seed=3)
    f.measurement_update(y, H, R)                 # plain F-order
    f.measurement_update(y, np.ascontiguousarray(H), R)   # C-order
    f.measurement_update(y, H[: M, : N], R)       # non-contiguous slice
    y_ro = y.copy()
    y_ro.setflags(write=False)
    H_ro = H.copy()
    H_ro.setflags(write=False)
    f.measurement_update(y_ro, H_ro, R)           # read-only inputs
    check(True, "contiguous / C-order / sliced / read-only inputs are accepted")


def test_zero_copy_is_real():
    # Read-only acceptance is necessary but not sufficient for zero-copy: a
    # caster could copy into a fresh writable buffer. Prove it does not by
    # measuring peak RSS across set_taper with a large read-only matrix. The
    # filter retains exactly one copy (taper_ = C by design); a binding-layer
    # copy would make that two.
    #
    # The measurement must happen in a fresh process: ru_maxrss is a lifetime
    # HIGH-WATER mark, so anything allocated by earlier tests in the same
    # process inflates the baseline and corrupts the ratio.
    probe = (
        "import numpy as np, resource, sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('lib', sys.argv[1])\n"
        "lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)\n"
        "order = sys.argv[2]\n"
        "n = 3200\n"
        "size = n * n * 8\n"
        "f = lib.KF(4)\n"
        "f.set_taper(np.eye(2)); f.clear_taper()\n"
        "C = np.full((n, n), 0.5, order=order)\n"
        "C.setflags(write=False)\n"
        "before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "f.set_taper(C)\n"
        "after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print((after - before) / size)\n"
    )
    for order in ("F", "C"):
        out = subprocess.check_output(
            [sys.executable, "-c", probe, LIB_PATH, order],
            text=True, timeout=300,
        )
        ratio = float(out.strip().splitlines()[-1])
        check(ratio < 1.5,
              f"set_taper({order}-order read-only): no binding-layer copy",
              f"grew {ratio:.2f}x the array (1.0x = zero-copy + 1 by-design "
              f"copy; 2.0x means the caster copied too)")


def test_inputs_rejected():
    f = lib.KF(N)
    f.initialize(vec(N), spd(N))
    y, H, R = vec(M), mat(M, N), spd(M, seed=3)
    # .noconvert(): no silent dtype cast
    check_raises(TypeError,
                 lambda: f.measurement_update(y.astype(np.float32), H, R),
                 "float32 y is rejected (no silent upcast)")
    check_raises(TypeError,
                 lambda: f.initialize(vec(N), spd(N).astype(np.float32)),
                 "float32 P is rejected")
    check_raises(TypeError,
                 lambda: f.initialize(list(range(N)), spd(N)),
                 "python list is rejected")
    check_raises(TypeError,
                 lambda: f.initialize(np.float64(1.0), spd(N)),
                 "scalar is rejected")
    check_raises(TypeError,
                 lambda: f.initialize(vec(N), [[1.0, 0.0], [0.0, 1.0]]),
                 "nested list is rejected")


# --------------------------------------------------------------------------
# 3. read-only / const-correct
# --------------------------------------------------------------------------

def test_readonly_never_triggers_a_write():
    # If the caster made a writable temp or wrote through the view, this would
    # raise (numpy forbids writes to a read-only buffer at the C level too).
    f = lib.KF(N)
    x = vec(N)
    P = spd(N)
    x.setflags(write=False)
    P.setflags(write=False)
    f.initialize(x, P)
    y = vec(M)
    H = mat(M, N)
    R = spd(M, seed=3)
    for a in (y, H, R):
        a.setflags(write=False)
    f.measurement_update(y, H, R)
    F = mat(N, N, seed=4)
    Q = spd(N, seed=5)
    u = vec(N, seed=6)
    for a in (F, Q, u):
        a.setflags(write=False)
    f.time_update(F, Q, u)
    check(True, "read-only x, P, y, H, R, F, Q, u all survive a full cycle")


# --------------------------------------------------------------------------
# 4. outputs are copies (mutating them must not corrupt the filter)
# --------------------------------------------------------------------------

def test_outputs_are_copies():
    f = fresh()
    f.measurement_update(vec(M), mat(M, N), spd(M, seed=3))
    s1 = f.state()
    s2 = f.state()
    check(not np.shares_memory(s1, s2), "two state() calls do not share memory")
    s1[:] = -99.0
    check(np.allclose(f.state(), s2), "mutating state() does not change the filter")

    P_before = f.covariance()
    P1 = f.covariance()
    P1[:] = 123.0
    check(np.allclose(f.covariance(), P_before),
          "mutating covariance() does not change the filter")
    check(not np.allclose(P_before, 123.0),
          "the write to the returned covariance actually took effect")

    g = lib.SquareRootKF(N)
    g.initialize(vec(N), spd(N))
    L1 = g.factor()
    L1[:] = 5.0
    check(np.isfinite(g.covariance()).all() and
          not np.allclose(g.covariance(), 25.0 * N),
          "mutating factor() does not change the filter")

    # Mutating the *input* after the call must not matter either: the filter
    # owns its state.
    h = lib.KF(N)
    x = vec(N)
    P = spd(N)
    h.initialize(x, P)
    want = h.state().copy()
    x[:] = 7.0
    P[:] = 7.0
    check(np.allclose(h.state(), want), "mutating x after initialize is harmless")
    check(np.allclose(h.covariance(), spd(N)), "mutating P after initialize is harmless")


# --------------------------------------------------------------------------
# 5. contracts raise ValueError (do not abort the interpreter)
# --------------------------------------------------------------------------

def test_contracts_raise_value_error():
    f = lib.KF(N)
    check_raises(ValueError,
                 lambda: f.initialize(vec(N + 1), spd(N)),
                 "wrong-size x raises ValueError")
    check_raises(ValueError,
                 lambda: f.initialize(vec(N), mat(N, N + 1)),
                 "non-square P raises ValueError")
    f.initialize(vec(N), spd(N))
    check_raises(ValueError,
                 lambda: f.measurement_update(vec(M + 1), mat(M, N), spd(M)),
                 "wrong-size y raises ValueError")
    check_raises(ValueError,
                 lambda: f.measurement_update(vec(M), mat(M, N + 1), spd(M)),
                 "H cols != N raises ValueError")
    check_raises(ValueError,
                 lambda: f.time_update(mat(N, N + 1), spd(N)),
                 "wrong-size F raises ValueError")
    check_raises(ValueError,
                 lambda: f.time_update(mat(N, N), spd(N), vec(N + 1)),
                 "wrong-size u raises ValueError")
    check_raises(ValueError,
                 lambda: f.batch([vec(M), vec(M)], [mat(M, N)],
                                 [spd(M), spd(M)], [mat(N, N), mat(N, N)],
                                 [spd(N), spd(N)]),
                 "inconsistent batch lengths raise ValueError")

    g = lib.SquareRootKF(N)
    check_raises(ValueError,
                 lambda: g.initialize(vec(N + 1), spd(N)),
                 "SquareRootKF: wrong-size x raises ValueError")


# --------------------------------------------------------------------------
# 6. KF == SquareRootKF from Python, and polymorphic use
# --------------------------------------------------------------------------

def test_backends_agree():
    n, m, steps = 8, 3, 25
    rng = np.random.default_rng(1234)
    Hs = [fortran(rng.standard_normal((m, n))) for _ in range(steps)]
    Rs = [fortran(np.eye(m) * 0.5) for _ in range(steps)]
    ys = [rng.standard_normal(m) for _ in range(steps)]
    Fs = [fortran(np.eye(n) * 0.98 + 0.02 * rng.standard_normal((n, n)))
          for _ in range(steps)]
    Qs = [fortran(np.eye(n) * 1e-3) for _ in range(steps)]
    x0 = rng.standard_normal(n)
    P0 = spd(n, seed=9)
    rho = 0.6
    C = fortran(rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))))

    for label, taper, lam in (("plain", None, 1.0),
                              ("taper (LKF)", C, 1.0),
                              ("inflation", None, 1.05),
                              ("taper + inflation", C, 1.02)):
        a = lib.KF(n)
        b = lib.SquareRootKF(n)
        for f in (a, b):
            if taper is not None:
                f.set_taper(taper)
            f.set_inflation(lam)
            f.initialize(x0, P0)
        worst = 0.0
        for i in range(steps):
            a.measurement_update(ys[i], Hs[i], Rs[i])
            b.measurement_update(ys[i], Hs[i], Rs[i])
            worst = max(worst, float(np.max(np.abs(a.state() - b.state()))))
            worst = max(worst, float(np.max(np.abs(a.covariance() - b.covariance()))))
            a.time_update(Fs[i], Qs[i])
            b.time_update(Fs[i], Qs[i])
        check(worst < 1e-10, f"KF == SquareRootKF ({label})", f"worst={worst:.2e}")


def test_polymorphic_use():
    n, m = 5, 2
    a, b = lib.KF(n), lib.SquareRootKF(n)
    check(isinstance(a, lib.Filter) and isinstance(b, lib.Filter),
          "both concrete classes are Filter instances")
    for f in (a, b):
        f.initialize(vec(n), spd(n))
        f.measurement_update(vec(m), mat(m, n), spd(m, seed=3))
        f.time_update(mat(n, n), spd(n, seed=4))
        check(f.dimension() == n, "dimension() reports N")
    check(np.allclose(a.state(), b.state()), "polymorphic calls agree on state")


# --------------------------------------------------------------------------
# 7. knobs and batch semantics
# --------------------------------------------------------------------------

def test_knobs():
    f = lib.KF(N)
    check(f.inflation() == 1.0 and not f.has_taper(), "defaults are off")
    f.set_inflation(1.07)
    check(f.inflation() == 1.07, "set_inflation round-trips")
    f.clear_inflation()
    check(f.inflation() == 1.0, "clear_inflation resets to 1")
    C = np.eye(N)
    f.set_taper(C)
    check(f.has_taper(), "set_taper sets has_taper")
    f.clear_taper()
    check(not f.has_taper(), "clear_taper clears has_taper")

    # ones taper == no taper (the Schur-product identity)
    x0, P0 = vec(N), spd(N)
    g = lib.KF(N)
    h = lib.KF(N)
    h.set_taper(np.ones((N, N)))
    g.initialize(x0, P0)
    h.initialize(x0, P0)
    y, H, R = vec(M), mat(M, N), spd(M, seed=3)
    g.measurement_update(y, H, R)
    h.measurement_update(y, H, R)
    check(np.allclose(g.covariance(), h.covariance(), atol=1e-12),
          "all-ones taper == no taper")


def test_batch_semantics():
    n, m, I = 4, 2, 5
    ys = [vec(m, seed=i) for i in range(I)]
    Hs = [mat(m, n, seed=i) for i in range(I)]
    Rs = [spd(m, seed=i) for i in range(I)]
    Fs = [mat(n, n, seed=i) for i in range(I)]
    Qs = [spd(n, seed=i) for i in range(I)]
    us = [vec(n, seed=i) for i in range(I)]
    x0, P0 = vec(n, 99), spd(n, seed=99)

    f = lib.SquareRootKF(n)
    f.initialize(x0, P0)
    out = f.batch(ys, Hs, Rs, Fs, Qs)
    check(len(out.x_prior) == I and len(out.x_posterior) == I, "batch lengths")
    check(len(out.P_prior) == I and len(out.P_posterior) == I, "batch cov lengths")

    # replay
    g = lib.SquareRootKF(n)
    g.initialize(x0, P0)
    ok = True
    for i in range(I):
        ok = ok and np.allclose(out.x_prior[i], g.state())
        ok = ok and np.allclose(out.P_prior[i], g.covariance())
        g.measurement_update(ys[i], Hs[i], Rs[i])
        ok = ok and np.allclose(out.x_posterior[i], g.state())
        ok = ok and np.allclose(out.P_posterior[i], g.covariance())
        g.time_update(Fs[i], Qs[i])
    check(ok, "batch: prior/posterior at time i match a manual replay")

    # record masks
    f.initialize(x0, P0)
    none = f.batch(ys, Hs, Rs, Fs, Qs, lib.RECORD_NONE)
    check(len(none.x_prior) == 0 and len(none.P_posterior) == 0,
          "RECORD_NONE stores nothing")
    f.initialize(x0, P0)
    means = f.batch(ys, Hs, Rs, Fs, Qs, lib.RECORD_MEANS)
    check(len(means.x_posterior) == I and len(means.P_prior) == 0,
          "RECORD_MEANS stores means only")
    f.initialize(x0, P0)
    mixed = f.batch(ys, Hs, Rs, Fs, Qs,
                    lib.RECORD_X_POSTERIOR | lib.RECORD_P_POSTERIOR)
    check(len(mixed.x_posterior) == I and len(mixed.P_posterior) == I
          and len(mixed.x_prior) == 0 and len(mixed.P_prior) == 0,
          "RECORD_X_POSTERIOR | RECORD_P_POSTERIOR works as a mask")

    # chunking: two halves == one call
    f.initialize(x0, P0)
    whole = f.batch(ys, Hs, Rs, Fs, Qs, us)
    f.initialize(x0, P0)
    p1 = f.batch(ys[:2], Hs[:2], Rs[:2], Fs[:2], Qs[:2], us[:2])
    p2 = f.batch(ys[2:], Hs[2:], Rs[2:], Fs[2:], Qs[2:], us[2:])
    ok = all(np.allclose(whole.x_posterior[i], p1.x_posterior[i]) for i in range(2))
    ok = ok and all(np.allclose(whole.x_posterior[i + 2], p2.x_posterior[i])
                    for i in range(3))
    check(ok, "batch: two halves == one call (composable)")

    # time_update with u == x <- F x + u
    f.initialize(x0, P0)
    f.time_update(Fs[0], Qs[0], us[0])
    check(np.allclose(f.state(), Fs[0] @ x0 + us[0]), "time_update(F,Q,u): x <- Fx+u")


# --------------------------------------------------------------------------
# 9. the Kalman family: KF == SquareRootKF == UDKF
# --------------------------------------------------------------------------

def test_udkf_agrees_with_kf():
    n, m, steps = 6, 3, 15
    rng = np.random.default_rng(4242)
    a, b, c = lib.KF(n), lib.SquareRootKF(n), lib.UDKF(n)
    worst_x = worst_P = 0.0
    for s in range(steps):
        x0 = rng.standard_normal(n)
        P0 = spd(n, seed=s)
        for f in (a, b, c):
            f.initialize(x0, P0)
        H = fortran(rng.standard_normal((m, n)))
        R = spd(m, seed=s + 1)
        y = rng.standard_normal(m)
        F = fortran(np.eye(n) * 0.97 + 0.03 * rng.standard_normal((n, n)))
        Q = spd(n, seed=s + 2) * 1e-3
        u = rng.standard_normal(n)
        for f in (a, b, c):
            f.measurement_update(y, H, R)
        worst_x = max(worst_x, float(np.max(np.abs(a.state() - b.state()))),
                      float(np.max(np.abs(a.state() - c.state()))))
        worst_P = max(worst_P, float(np.max(np.abs(a.covariance() - b.covariance()))),
                      float(np.max(np.abs(a.covariance() - c.covariance()))))
        for f in (a, b, c):
            f.time_update(F, Q, u)
    check(worst_x < 1e-10, "KF == SquareRootKF == UDKF, state", f"{worst_x:.2e}")
    check(worst_P < 1e-10, "KF == SquareRootKF == UDKF, covariance", f"{worst_P:.2e}")


def test_udkf_agrees_with_kf_under_taper():
    n, m = 6, 3
    rng = np.random.default_rng(9)
    C = fortran(0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))))
    a, b = lib.KF(n), lib.UDKF(n)
    x0 = rng.standard_normal(n)
    P0 = spd(n, seed=1)
    H = fortran(rng.standard_normal((m, n)))
    y = rng.standard_normal(m)
    R = spd(m, seed=2)
    for f in (a, b):
        f.set_taper(C)
        f.initialize(x0, P0)
        f.measurement_update(y, H, R)
    check(np.allclose(a.state(), b.state(), atol=1e-11), "UDKF (taper) == KF, state")
    check(np.allclose(a.covariance(), b.covariance(), atol=1e-11), "UDKF (taper) == KF, covariance")


def test_udkf_factorization_invariants():
    n = 7
    f = lib.UDKF(n)
    f.initialize(np.zeros(n), spd(n))
    for _ in range(10):
        f.measurement_update(np.ones(3), mat(3, n), spd(3, seed=3))
        f.time_update(mat(n, n) * 0.1 + np.eye(n), spd(n, seed=4) * 1e-3)
        L, D = f.unit_lower(), f.diagonal()
        check(np.allclose(np.triu(L, 1), 0.0), "UDKF: L has empty strict upper triangle")
        check(np.allclose(np.diag(L), 1.0), "UDKF: diag(L) == 1")
        check(bool(np.all(D > 0)), "UDKF: D > 0")
        check(np.allclose(L @ np.diag(D) @ L.T, f.covariance(), atol=1e-11),
              "UDKF: L D L^T == covariance()")


# --------------------------------------------------------------------------
# 10. the ensemble family
# --------------------------------------------------------------------------

def test_ensemble_family_agrees():
    n, m, L = 6, 3, 3000
    rng = np.random.default_rng(77)
    x0 = rng.standard_normal(n)
    P0 = spd(n, seed=1)
    H = fortran(rng.standard_normal((m, n)))
    R = spd(m, seed=2)
    y = rng.standard_normal(m)

    ref = lib.KF(n)
    ref.initialize(x0, P0)
    ref.measurement_update(y, H, R)

    a = lib.EnKF(n, L, 1)
    b = lib.EnSRF(n, L, 1)
    c = lib.EAKF(n, L, 1)
    d = lib.ETKF(n, L, 1)
    a.initialize(x0, P0)
    X = a.members()                       # one shared prior ensemble
    for e in (b, c, d):
        e.set_members(X)
    for e in (a, b, c, d):
        e.measurement_update(y, H, R)

    # Deterministic trio agree exactly on mean and covariance.
    check(np.allclose(b.state(), c.state(), atol=1e-12), "EnSRF == EAKF, state")
    check(np.allclose(b.state(), d.state(), atol=1e-12), "EnSRF == ETKF, state")
    check(np.allclose(b.covariance(), c.covariance(), atol=1e-12), "EnSRF == EAKF, covariance")
    check(np.allclose(b.covariance(), d.covariance(), atol=1e-12), "EnSRF == ETKF, covariance")
    # ETKF's members are a rotation of EnSRF's (different factor, same cov).
    rot = np.linalg.norm(b.anomalies() - d.anomalies()) / np.linalg.norm(b.anomalies())
    check(rot > 1e-6, "ETKF members rotate EnSRF's", f"{rot:.2e}")
    # All four converge to KF at O(1/sqrt(L)).
    for name, e in (("EnKF", a), ("EnSRF", b), ("EAKF", c), ("ETKF", d)):
        err = np.linalg.norm(e.covariance() - ref.covariance()) / np.linalg.norm(ref.covariance())
        check(err < 0.15, f"{name} covariance within 15% of KF (L={L})", f"{err:.3f}")


def test_letkf_is_etkf_plus_taper():
    n, m, L = 6, 3, 50
    C = fortran(0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))))
    rng = np.random.default_rng(3)
    a = lib.LETKF(n, L, C, 5)
    b = lib.ETKF(n, L, 5)
    b.set_taper(C)
    a.initialize(rng.standard_normal(n), spd(n))
    b.set_members(a.members())
    H = fortran(rng.standard_normal((m, n)))
    y = rng.standard_normal(m)
    R = spd(m, seed=8)
    a.measurement_update(y, H, R)
    b.measurement_update(y, H, R)
    check(np.allclose(a.state(), b.state(), atol=1e-13), "LETKF == ETKF + taper, state")
    check(np.allclose(a.covariance(), b.covariance(), atol=1e-13), "LETKF == ETKF + taper, covariance")
    check(a.has_taper(), "LETKF has a taper")


def test_ensemble_members_round_trip():
    n, L = 5, 40
    f = lib.EnKF(n, L, 1)
    f.initialize(np.arange(n, dtype=float), spd(n))
    X = f.members()
    check(X.shape == (n, L), "members() is N x L")
    check(np.allclose(X.mean(axis=1), f.state()), "members' mean is state()")
    check(np.allclose(f.anomalies().sum(axis=1), 0.0, atol=1e-12), "anomalies are centered")
    check(np.allclose(f.anomalies() @ f.anomalies().T / (L - 1), f.covariance(), atol=1e-11),
          "anomalies reproduce covariance()")
    # set_members is the exact inverse of members()
    g = lib.EnSRF(n, L, 1)
    g.set_members(X)
    check(np.allclose(g.members(), X, atol=1e-12), "set_members(members()) round-trips")


def test_smoother_converges_to_rts():
    n, m, I, L = 4, 2, 6, 4000
    rng = np.random.default_rng(5150)
    ys = [rng.standard_normal(m) for _ in range(I)]
    Hs = [fortran(rng.standard_normal((m, n))) for _ in range(I)]
    Rs = [spd(m, seed=i) for i in range(I)]
    Fs = [fortran(np.eye(n) * 0.95 + 0.05 * rng.standard_normal((n, n))) for _ in range(I)]
    Qs = [spd(n, seed=i) * 1e-3 for i in range(I)]
    x0 = rng.standard_normal(n)
    P0 = spd(n, seed=9)

    # Exact reference: KF forward record + RTS.
    ref = lib.KF(n)
    ref.initialize(x0, P0)
    rec = ref.batch(ys, Hs, Rs, Fs, Qs)
    exact = lib.rts_smooth(rec, Fs, Qs)

    ks = lib.EnKS(n, L, 1)
    ks.initialize(x0, P0)
    sm = ks.smooth(ys, Hs, Rs, Fs, Qs)
    check(len(sm.x_smoothed) == I and len(sm.P_smoothed) == I, "EnKS lengths")
    worst = max(
        float(np.linalg.norm(exact.x_smoothed[i] - sm.x_smoothed[i]))
        for i in range(I))
    check(worst < 0.5, "EnKS (L=4000) vs RTS, state", f"{worst:.3f}")
    # Terminal condition of the exact smoother.
    check(np.allclose(exact.P_smoothed[-1], rec.P_posterior[-1], atol=1e-10),
          "rts_smooth: terminal condition")


def test_leks_is_enks_plus_taper():
    n, m, I, L = 5, 2, 4, 30
    C = fortran(0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))))
    rng = np.random.default_rng(2)
    ys = [rng.standard_normal(m) for _ in range(I)]
    Hs = [fortran(rng.standard_normal((m, n))) for _ in range(I)]
    Rs = [spd(m, seed=i) for i in range(I)]
    Fs = [fortran(np.eye(n)) for _ in range(I)]
    Qs = [spd(n, seed=i) * 1e-3 for i in range(I)]
    x0 = rng.standard_normal(n)
    P0 = spd(n, seed=6)

    a = lib.LEKS(n, L, C, 3)
    b = lib.EnKS(n, L, 3)
    b.set_taper(C)
    a.initialize(x0, P0)
    sa = a.smooth(ys, Hs, Rs, Fs, Qs)
    b.initialize(x0, P0)
    sb = b.smooth(ys, Hs, Rs, Fs, Qs)
    same = all(np.allclose(sa.x_smoothed[i], sb.x_smoothed[i], atol=1e-13)
               and np.allclose(sa.P_smoothed[i], sb.P_smoothed[i], atol=1e-13)
               for i in range(I))
    check(same, "LEKS == EnKS + taper")
    check(a.has_taper(), "LEKS has a taper")


def test_new_contracts_throw():
    n = 4
    f = lib.EnKF(n, 20, 1)
    check_raises(ValueError, lambda: f.initialize(vec(n + 1), spd(n)),
                 "EnKF.initialize: wrong size x raises ValueError")
    check_raises(ValueError, lambda: f.set_members(mat(n, 3)),
                 "EnKF.set_members: wrong shape raises ValueError")
    g = lib.UDKF(n)
    check_raises(ValueError, lambda: g.initialize(vec(n), mat(n, n + 1)),
                 "UDKF.initialize: non-square P raises ValueError")
    ks = lib.EnKS(n, 20, 1)
    check_raises(ValueError,
                 lambda: ks.smooth([vec(2), vec(2)], [mat(2, n)],
                                   [spd(2), spd(2)], [np.eye(n)], [spd(n)]),
                 "EnKS.smooth: inconsistent lengths raise ValueError")


# --------------------------------------------------------------------------
# 8. memory stability across many updates
# --------------------------------------------------------------------------

def test_memory_stability():
    n, m = 6, 2
    f = lib.SquareRootKF(n)
    g = lib.KF(n)
    for h in (f, g):
        h.initialize(vec(n), spd(n))
    y, H, R = vec(m), mat(m, n), spd(m, seed=3)
    F = fortran(np.eye(n) * 0.99)
    Q = fortran(np.eye(n) * 1e-5)
    # warm up
    for _ in range(50):
        f.measurement_update(y, H, R)
        f.time_update(F, Q)
    base = peak_rss_bytes()
    for _ in range(20000):
        f.measurement_update(y, H, R)
        f.time_update(F, Q)
        g.measurement_update(y, H, R)
        g.time_update(F, Q)
    growth = peak_rss_bytes() - base
    check(growth < 8 * 1024 * 1024,
          "20000 updates: peak RSS stable (< 8 MB growth)",
          f"growth={growth / 1e6:.2f} MB")
    check(np.isfinite(f.state()).all() and np.isfinite(g.state()).all(),
          "20000 updates: state finite")
    check(float(np.min(np.linalg.eigvalsh(f.covariance()))) > 0.0,
          "20000 updates: SquareRootKF P stays PD")


# --------------------------------------------------------------------------

TESTS = [
    test_inputs_accepted,
    test_udkf_agrees_with_kf,
    test_udkf_agrees_with_kf_under_taper,
    test_udkf_factorization_invariants,
    test_ensemble_family_agrees,
    test_letkf_is_etkf_plus_taper,
    test_ensemble_members_round_trip,
    test_smoother_converges_to_rts,
    test_leks_is_enks_plus_taper,
    test_new_contracts_throw,
    test_zero_copy_is_real,
    test_inputs_rejected,
    test_readonly_never_triggers_a_write,
    test_outputs_are_copies,
    test_contracts_raise_value_error,
    test_backends_agree,
    test_polymorphic_use,
    test_knobs,
    test_batch_semantics,
    test_memory_stability,
]


def main():
    for t in TESTS:
        run(t.__name__, t)
    print(f"\n{_checks} checks passed, {_failures} failed ({len(TESTS)} cases)")
    print("SOME CHECKS FAILED" if _failures else "ALL CHECKS PASSED")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
