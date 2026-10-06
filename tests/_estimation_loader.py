"""How the Python tests find the compiled ``estimation.lib`` module.

Prefer an installed package (``pip install -e .``), then fall back to an
in-tree extension pointed at by ``$ESTIMATION_LIB`` or found under ``build*/``.

``load_lib()`` is memoized: a nanobind module cannot be executed twice in one
process (its types are registered once), so the lookup must not be repeated.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import pathlib
import sys

_CACHE = {}


def load_lib():
    """Return the compiled module (``estimation.lib``)."""
    if "lib" in _CACHE:
        return _CACHE["lib"]
    mod = _load_lib_once()
    _CACHE["lib"] = mod
    return mod


def _load_lib_once():
    try:
        import estimation.lib as mod
        return mod
    except Exception:  # noqa: BLE001 -- fall through to the in-tree lookup
        pass

    here = pathlib.Path(__file__).resolve().parent
    root = here.parent
    candidates = []
    if os.environ.get("ESTIMATION_LIB"):
        candidates.append(os.environ["ESTIMATION_LIB"])
    # Prefer the Python extension: the glob also matches the C++ shared
    # library (build/src/libestimation.dylib), which has no KF.
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
        except Exception:  # noqa: BLE001 -- e.g. the C++ shared library
            continue
        if hasattr(mod, "KF"):
            return mod

    sys.exit(
        "cannot find the compiled module (lib*); install the package or set "
        "ESTIMATION_LIB to its path:\n"
        "  python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -e .\n"
        f"searched: {candidates[:6]} ..."
    )


class Harness:
    """The check/run/count harness used by the example tests."""

    def __init__(self):
        self.checks = 0
        self.failures = 0

    def check(self, ok, what, detail=""):
        self.checks += 1
        if ok:
            return
        self.failures += 1
        print(f"    FAIL  {what}" + (f"  [{detail}]" if detail else ""))

    def check_close(self, a, b, tol, what):
        err = abs(a - b)
        self.check(err <= tol, what, f"err={err:.3e} tol={tol:.1e}")

    def run(self, name, fn):
        print(f"[ {name} ]")
        sys.stdout.flush()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            print(f"    FAIL  {name} raised {type(e).__name__}: {e}")

    def summary(self, n_cases):
        print(f"\n{self.checks} checks passed, {self.failures} failed ({n_cases} cases)")
        print("SOME CHECKS FAILED" if self.failures else "ALL CHECKS PASSED")
        return 1 if self.failures else 0
