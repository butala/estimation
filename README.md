# estimation

Kalman-family state estimation in C++ (Eigen) with Python bindings (nanobind).

## Filters

**Exact Kalman filter** — three uncertainty representations, one map:

| Class | Stores | Notes |
|---|---|---|
| `KF` | `P` | covariance form, the reference / "oracle" |
| `SquareRootKF` | `L`, `P = L Lᵀ` | Cholesky factor form; `P` never formed |
| `UDKF` | `L`, `D`, `P = L D Lᵀ` | Bierman's factored form; `O(N²)` per scalar row |

**Monte Carlo Kalman methods** — one shared LKF mean, four anomaly algebras:

| Class | Anomaly update | Reference |
|---|---|---|
| `EnKF` | `(I−KH)X + KE`, recentered `E ~ N(0,R)` | Evensen 1994; Burgers et al. 1998 |
| `EnSRF` | sequential `(I−αkh)X` | Whitaker & Hamill 2002 |
| `EAKF` | sequential observation-space regression | Anderson 2001 |
| `ETKF` | block `X C^{−1/2}` | Bishop, Etherton & Hodyss 2001 |

**Localization is a knob, not an algorithm:** `LETKF` is `ETKF` plus a *required*
taper, `LEKS` is `EnKS` plus a required taper — exactly as `LKF = KF + set_taper`.

Every filter is `template <typename T = double>` and is explicitly instantiated
for `double` and `float`; `Scalar = T` is the coefficient type the bodies use so
the same source is precision-agnostic. Numeric literals go through
`Scalar(...)` to keep arithmetic in `T`, and floors use
`std::numeric_limits<Scalar>::min()` rather than a decimal literal (which
underflows to zero in `float`).

**Smoothers** (fixed interval):

| Class | What it is |
|---|---|
| `rts_smooth(record, F, Q)` | exact Rauch–Tung–Striebel on a `BatchOutput` — the reference |
| `EnKS` | Evensen's ensemble smoother: EnKF forward + lag-one cross-covariances |
| `LEKS` | `EnKS` + a required taper (the taper acts on the smoother regression) |

### Equivalences the tests enforce

- `KF == SquareRootKF == UDKF` — exactly (all three are the same filter), with
  and without a taper
- `EnSRF == EAKF == ETKF` — exactly the same mean and covariance from one
  shared prior ensemble (they coincide for sequential scalar rows; ETKF's
  members differ by a rotation, per Sakov & Oke 2008)
- all four ensemble methods, and `LETKF`/`LEKS`, converge to `KF`/the `LKF` as
  `L → ∞` (statistical)
- `LETKF == ETKF + set_taper`, `LEKS == EnKS + set_taper` — exact
- `EnKS → rts_smooth` as `L → ∞`

## API

One interface (`estimation::Filter<T>`, bound as `estimation.lib.Filter`), four
verbs and two knobs, identical in C++ and Python:

```python
f = KF(N)                     # or SquareRootKF(N)

f.initialize(x, P)            # "assume this prior"
f.measurement_update(y, H, R) # data, model, noise
f.time_update(F, Q)           # model, noise
f.time_update(F, Q, u)        # x <- F x + u   (u = known input contribution)

f.set_taper(C)                # localization: gain from C o P   -> LKF
f.set_inflation(lambda)       # multiplicative, on the prior covariance

f.state()                     # the state estimate (a copy)
f.covariance()                # P (a copy; the sample covariance for ensembles)

f.batch(y, H, R, F, Q, u=None, record=RECORD_EVERYTHING)
```

Two conventions worth knowing, because they are the theory:

* **The taper is a covariance-space operation.** The gain is formed from the
  tapered covariance `C o P` while the covariance update is the Joseph form on
  the **untapered** `P` (Butala et al., IEEE TIP 2009, eq. (4.26)). With a
  non-trivial taper the filter is the tapered / localized Kalman filter (LKF) —
  the `L -> infinity` limit of the localized EnKF. So
  `‖EnKF − KF‖ <= ‖EnKF − LKF‖ + ‖LKF − KF‖` is expressible as "the same class
  with `set_taper` on and off". Note the Schur-product identity element is the
  **all-ones** matrix (`Ones(n,n)`), not `Identity(n,n)`: `I o P = diag(P)`.
* **Inflation scales the prior covariance at the start of every
  `measurement_update`** (`P <- lambda P`), matching standard practice.
  `set_inflation(1)` between analyses if you assimilate several independent
  observation vectors at one epoch.

`batch` records `*_prior[i]` and `*_posterior[i]` **at time i** — exactly the
sequence an RTS smoother needs — and is composable (two halves == one call).
`record` is a bitmask (`RECORD_MEANS`, `RECORD_COVARIANCES`, `RECORD_NONE`, …)
so the `O(N²)` sequences are opt-in.

### Verified binding contract

Established by `tests/test_bindings.py` (see below):

| | Behaviour |
|---|---|
| float64 inputs, **any layout** (F-order, C-order/numpy default, slices) | **zero-copy** strided views |
| read-only numpy arrays | accepted (const-correct; no writable temp is made) |
| `float32`, lists, scalars | **rejected** — `.noconvert()`, no silent cast |
| `state()`, `covariance()`, `factor()` | **copies** — mutating them cannot corrupt a filter |
| dimension / length mismatches | `ValueError` (always on, not `assert`) |

## Build

Needs CMake ≥ 3.20, a C++20 compiler, and Eigen (3.4+ or 5.x).

```bash
cmake -B build
cmake --build build -j
```

This produces `build/src/libestimation.dylib` and the test binary
`build/src/test`.

### Python extension module

The simplest route builds and installs the extension together with its
dependencies (`numpy`, `scipy`, `matplotlib` -- see `pyproject.toml`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

After that `import estimation.lib` works and no `ESTIMATION_LIB` variable is
needed. Alternatively, build the extension in-tree (needs `nanobind` and Python
development headers):

```bash
cmake -B build-py -DESTIMATION_BUILD_PYTHON=ON \
      -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())")
cmake --build build-py -j
export ESTIMATION_LIB=$(ls "$PWD"/build-py/src/lib.cpython-*.so)
```

Both routes are supported by the Python tests; they prefer the installed
package and fall back to `$ESTIMATION_LIB`.

## Test

### C++ unit tests — 46 cases / 230 checks

```bash
./build/src/test
```

Exits 0 on success. Covers: scalar closed forms; the Kalman gain against an
explicit inverse; an **independent Joseph / gain reference written from the
paper**; `KF == SquareRootKF == UDKF` in every configuration (plain, taper,
inflation, both) and through `Filter<T>&`; the Bierman downdate against a dense
factorization; UD factorization invariants; taper = LKF semantics; inflation;
the `u` overload; edge cases (`N=1`, `Q=0`, rank-deficient `Q`, `F=0`, tiny
`R`); 500-step drift-free stability; `batch`/`Record` masks and chunking;
`EnSRF == EAKF == ETKF` and the ETKF rotation; ensemble convergence to
`KF`/`LETKF`; `rts_smooth` monotonicity; `EnKS → RTS`; `LEKS == EnKS + taper`;
contract violations; 20000-update memory stability; and four smoke cases for
the explicitly instantiated `float` build (`KF == SquareRootKF == UDKF`, with
and without a taper, the ensemble trio with a finiteness check on the ETKF
eigenvalue clamp, and `EnKS`/`rts_smooth` finiteness).

### Worked example from the literature — `tests/test_kay_example_13_4.py`

A complete reproduction of Kay, *Fundamentals of Statistical Signal
Processing, Vol. I*, **Example 13.4** (vehicle tracking with the extended
Kalman filter; Figures 13.22–13.25). It is Kay's MIMO case — a 4-dimensional
state observed through two channels at once, range and bearing — and the
measurement is nonlinear while the dynamics are exactly linear.

The reason it lives in this project: **the EKF is this library's linear `KF`
evaluated at the linearisation.** With `H_n = dh/dx` at the predicted state,
`y' = y - h(x^-) + H_n x^-` and `H' = H_n` give `y' - H' x^- == y - h(x^-)`
exactly, so `measurement_update(y', H', R)` *is* Kay's EKF update. That identity
and an independent textbook EKF are asserted, alongside:

- the model constants and the ideal track, as Kay sets them up
- `observation_jacobian` against central differences
- `atan2` is **required**, not a detail: `r_x[n] = 10 - 0.2 n` vanishes exactly
  at `n = 50`, so `arctan(r_y/r_x)` is undefined there and is off by `pi` for
  `r_x < 0`. The test pins both the crossing and the branch loss.
- the estimate converges onto the track and beats a raw range/bearing fix
- mean NEES is O(state dimension) — the reported `P` is the right size
- seeded reproducibility

`scipy` is used only to verify that `transition(delta)` agrees with the
`toeplitz([1, 0, 0, 0], r=[1, 0, delta, 0])` construction in the source, which
`transition()` spells out as an explicit matrix for readability.

**Run it** (34 checks / 10 cases):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                       # numpy, scipy, matplotlib + the extension
python3 tests/test_kay_example_13_4.py # assertions
```

and to write Kay's Figures 13.22–13.25 to `figures/kay_13_4/`:

```bash
python3 tests/test_kay_example_13_4.py --figures
```

### Python binding tests — 20 cases / 119 checks

```bash
python3 tests/test_bindings.py
```
(after `pip install -e .`; or set `ESTIMATION_LIB` to an in-tree `lib*.so`)

(Or `python3 -m pytest tests/test_bindings.py -q`.) Covers the zero-copy
contract, strict dtype, read-only inputs, output copy-safety, `ValueError`
contracts, `KF == SquareRootKF` from Python, knobs, `batch` semantics, and
memory stability.

### Memory checks

ASan + UBSan (bounds, use-after-free, UB):

```bash
cmake -B build-asan -DESTIMATION_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-asan -j
ASAN_OPTIONS=detect_leaks=0 ./build-asan/src/test
```

Leak counts (LSan is not supported on macOS/arm64 — use `leaks`):

```bash
MallocStackLogging=1 leaks --atExit -- ./build/src/test
```

Expected: `0 leaks for 0 total leaked bytes`.

### Everything

```bash
cmake -B build && cmake --build build -j && ./build/src/test
cmake -B build-asan -DESTIMATION_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  && cmake --build build-asan -j && ASAN_OPTIONS=detect_leaks=0 ./build-asan/src/test
MallocStackLogging=1 leaks --atExit -- ./build/src/test
cmake -B build-py -DESTIMATION_BUILD_PYTHON=ON \
  -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())") \
  && cmake --build build-py -j
export ESTIMATION_LIB=$(ls "$PWD"/build-py/src/lib.cpython-*.so)
python3 tests/test_bindings.py
```

## Layout

```
CMakeLists.txt              project entry point
src/
  filter.hpp                Filter<T> interface, Types, Record, BatchOutput, taper/inflation
  kalman_filter.hpp/.cpp    KF<T>            (covariance form)
  square_root_kf.hpp/.cpp   SquareRootKF<T>  (Cholesky factor form)
  ud_filter.hpp/.cpp        UDKF<T>          (Bierman factored form)
  ensemble_filter.hpp/.cpp  EnsembleFilter<T>, EnKF, EnSRF, EAKF, ETKF, LETKF
  smoother.hpp/.cpp         rts_smooth, EnKS<T>, LEKS<T>
  module.cpp                nanobind bindings
  testing.hpp               minimal dependency-free test harness
  test.cpp                  C++ unit tests
tests/
  test_bindings.py          Python binding tests (binding contract + the family)
  test_kay_example_13_4.py  Kay, Estimation Theory Ex 13.4 (EKF worked example)
```

Build flags: `-O3`, `-mtune=native` always; `-march=native` only for local
builds (never for wheels). **No `-ffast-math`** — it flushes NaN/Inf and
reassociates, which is unsafe for a filter (NaN is the divergence alarm) and
makes results compiler-dependent.

## Related documents

* `ENKF_CONVERGENCE_ERRATUM.{md,pdf}` — the gap in the proof of EnKF
  convergence noted by Mandel–Cobb–Beezley, and the corrected proof.
* `RESEARCH_AGENDA.md` — state of the art in ensemble Kalman convergence theory,
  and the prioritized next steps.
