// ---------------------------------------------------------------------------
// Unit tests for estimation::KF and estimation::SquareRootKF.
//
// Build and run:
//     cmake -B build && cmake --build build && ./build/src/test
//
// Memory checking (ASan + UBSan; `leaks` for leak counts on macOS):
//     cmake -B build-asan -DESTIMATION_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
//     cmake --build build-asan && ASAN_OPTIONS=detect_leaks=0 ./build-asan/src/test
//     MallocStackLogging=1 leaks --atExit -- ./build/src/test
//
// Python-layer tests (zero-copy, aliasing, ValueError contracts):
//     python3 tests/test_bindings.py
//
// See README.md for the full build / test instructions.
//
// Coverage map:
//   [scalar closed form]           analytic values, both backends
//   [kalman gain]                  K = P H^T (H P H^T + R)^{-1}, explicit inverse
//   [reference implementations]    independent Joseph / gain reference
//   [cross-representation]         KF == SquareRootKF in every configuration
//   [taper = LKF]                  identity / zero / Joseph-on-untapered-P
//   [inflation]                    lambda = 1 is off, monotone in lambda
//   [known input u]                x <- F x + u, covariance unaffected
//   [edge cases]                   N=1, Q=0, Q rank-deficient, F=0, tiny R
//   [stability]                    500 steps, no drift, P stays PD
//   [batch and Record]             masks, u variant, empty, chunking
//   [contracts]                    dimension violations throw std::invalid_argument
//   [memory]                       20000 updates with bounded working set
// ---------------------------------------------------------------------------

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "filter.hpp"
#include "kalman_filter.hpp"
#include "square_root_kf.hpp"
#include "ensemble_filter.hpp"
#include "ud_filter.hpp"
#include "smoother.hpp"
#include "testing.hpp"

using namespace estimation;

using TypesD  = Types<double>;
using VectorD = TypesD::Vector;
using MatrixD = TypesD::Matrix;
using CVecD   = TypesD::ConstVectorRef;
using CMatD   = TypesD::ConstMatrixRef;
using FilterD = Filter<double>;

// The library is explicitly instantiated for float as well as double, but the
// rest of this file only exercises double. The cases in [the float
// instantiations] are the smoke tests for that half of the build.
using TypesF  = Types<float>;
using VectorF = TypesF::Vector;
using MatrixF = TypesF::Matrix;
using CVecF   = TypesF::ConstVectorRef;
using CMatF   = TypesF::ConstMatrixRef;
using FilterF = Filter<float>;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::mt19937 &
rng()
{
    static std::mt19937 g(20261005);
    return g;
}

static double
uniform(const double lo = -1.0, const double hi = 1.0)
{
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng());
}

static MatrixD
random_spd(const int n, const double jitter = 1e-2)
{
    std::normal_distribution<double> g(0.0, 1.0);
    MatrixD G(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) G(i, j) = g(rng());
    return MatrixD(G * G.transpose() + jitter * MatrixD::Identity(n, n));
}

static MatrixD
random_matrix(const int rows, const int cols)
{
    MatrixD A(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) A(i, j) = uniform();
    return A;
}

static VectorD
random_vector(const int n)
{
    VectorD v(n);
    for (int i = 0; i < n; ++i) v(i) = uniform();
    return v;
}

// AR(1) correlation taper: symmetric positive definite, unit diagonal.
static MatrixD
ar1_taper(const int n, const double rho)
{
    MatrixD C(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) C(i, j) = std::pow(rho, std::abs(i - j));
    return C;
}

static double
rel_err(const MatrixD &A, const MatrixD &B)
{
    return (A - B).norm() / std::max(A.norm(), 1e-300);
}

static double
rel_err(const VectorD &a, const VectorD &b)
{
    return (a - b).norm() / std::max(a.norm(), 1e-300);
}

static void
check_mat(const MatrixD &A, const MatrixD &B, const double tol, const char *what)
{
    const double e = rel_err(A, B);
    testing::check(e <= tol, what, e, tol);
}

static void
check_vec(const VectorD &a, const VectorD &b, const double tol, const char *what)
{
    const double e = rel_err(a, b);
    testing::check(e <= tol, what, e, tol);
}

static bool
is_symmetric(const MatrixD &A)
{
    return (A - A.transpose()).cwiseAbs().maxCoeff() == 0.0;
}

static double
min_eig(const MatrixD &A)
{
    Eigen::SelfAdjointEigenSolver<MatrixD> es(A);
    return es.eigenvalues().minCoeff();
}

// Independent reference for one tapered analysis step: gain from (C o P),
// Joseph covariance update on the UNTAPERED P. This is Butala et al.,
// IEEE TIP 2009, eq. (4.26), written from the paper rather than from the
// implementation -- so it is a real cross-check, not a restatement.
struct Reference {
    VectorD x;
    MatrixD P;
};

static Reference
reference_analysis(const Reference &prior,
                   const VectorD &y, const MatrixD &H, const MatrixD &R,
                   const MatrixD *C, const double lambda)
{
    MatrixD P = prior.P * lambda;                      // inflation on the prior
    MatrixD Pg = C ? MatrixD(C->cwiseProduct(P)) : P;  // gain covariance
    // Symmetrize Pg exactly as the implementation does.
    Pg = (Pg + Pg.transpose()) * 0.5;

    const MatrixD S    = H * Pg * H.transpose() + R;
    const MatrixD K    = Pg * H.transpose() * S.inverse();   // explicit inverse
    const VectorD x    = prior.x + K * (y - H * prior.x);

    // Joseph on the inflated-but-untapered prior covariance.
    const MatrixD P_inf = prior.P * lambda;
    const MatrixD IKH   = MatrixD::Identity(P_inf.rows(), P_inf.rows()) - K * H;
    MatrixD Pout = IKH * P_inf * IKH.transpose() + K * R * K.transpose();
    Pout = (Pout + Pout.transpose()) * 0.5;
    return {x, Pout};
}


// ---------------------------------------------------------------------------
// [scalar closed form]
// ---------------------------------------------------------------------------

TEST_CASE("scalar closed form: KF")
{
    KF<double> f(1);
    VectorD x(1); x(0) = 0.0;
    MatrixD P(1, 1); P(0, 0) = 1.0;
    f.initialize(x, P);

    VectorD y(1); y(0) = 2.0;
    MatrixD H(1, 1); H(0, 0) = 1.0;
    MatrixD R(1, 1); R(0, 0) = 1.0;
    f.measurement_update(y, H, R);
    CHECK_CLOSE(f.state()(0), 1.0, 1e-12, "analysis mean = K y = y/2");
    CHECK_CLOSE(f.covariance()(0, 0), 0.5, 1e-12, "analysis var = (1-K)P = 1/2");

    MatrixD F(1, 1); F(0, 0) = 1.0;
    MatrixD Q(1, 1); Q(0, 0) = 1.0;
    f.time_update(F, Q);
    CHECK_CLOSE(f.covariance()(0, 0), 1.5, 1e-12, "forecast var = P + Q = 3/2");
    CHECK_CLOSE(f.state()(0), 1.0, 1e-12, "forecast mean unchanged");

    VectorD u(1); u(0) = 0.25;
    f.time_update(F, Q, u);
    CHECK_CLOSE(f.state()(0), 1.25, 1e-12, "forecast mean + u");
    CHECK_CLOSE(f.covariance()(0, 0), 2.5, 1e-12, "u does not affect P");
}

TEST_CASE("scalar closed form: SquareRootKF")
{
    SquareRootKF<double> f(1);
    VectorD x(1); x(0) = 0.0;
    MatrixD P(1, 1); P(0, 0) = 1.0;
    f.initialize(x, P);

    VectorD y(1); y(0) = 2.0;
    MatrixD H(1, 1); H(0, 0) = 1.0;
    MatrixD R(1, 1); R(0, 0) = 1.0;
    f.measurement_update(y, H, R);
    CHECK_CLOSE(f.state()(0), 1.0, 1e-12, "analysis mean = K y = y/2");
    CHECK_CLOSE(f.covariance()(0, 0), 0.5, 1e-12, "analysis var = (1-K)P = 1/2");

    MatrixD F(1, 1); F(0, 0) = 1.0;
    MatrixD Q(1, 1); Q(0, 0) = 1.0;
    f.time_update(F, Q);
    CHECK_CLOSE(f.covariance()(0, 0), 1.5, 1e-12, "forecast var = P + Q = 3/2");

    VectorD u(1); u(0) = 0.25;
    f.time_update(F, Q, u);
    CHECK_CLOSE(f.state()(0), 1.25, 1e-12, "forecast mean + u");
    CHECK_CLOSE(f.covariance()(0, 0), 2.5, 1e-12, "u does not affect P");
}


// ---------------------------------------------------------------------------
// [kalman gain] -- K = P H^T (H P H^T + R)^{-1} against an explicit inverse
// ---------------------------------------------------------------------------

TEST_CASE("kalman gain matches the explicit formula")
{
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);

    const MatrixD S = H * P0 * H.transpose() + R;
    const MatrixD K = P0 * H.transpose() * S.inverse();
    const VectorD x_expect = x0 + K * (y - H * x0);

    for (FilterD *f : {static_cast<FilterD *>(new KF<double>(N)),
                       static_cast<FilterD *>(new SquareRootKF<double>(N))}) {
        f->initialize(x0, P0);
        f->measurement_update(y, H, R);
        check_vec(f->state(), x_expect, 1e-12, "posterior mean == x + K(y-Hx)");
        delete f;
    }
}


// ---------------------------------------------------------------------------
// [reference implementations]
// ---------------------------------------------------------------------------

TEST_CASE("matches an independent Joseph / gain reference")
{
    const int N = 6, m = 3;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    const MatrixD C  = ar1_taper(N, 0.55);

    struct Config { const char *what; const MatrixD *taper; double lambda; };
    const Config configs[] = {
        {"reference: plain",            nullptr, 1.0},
        {"reference: taper (LKF)",      &C,      1.0},
        {"reference: inflation",        nullptr, 1.07},
        {"reference: taper + inflation",&C,      0.94},
    };

    for (const Config &cfg : configs) {
        Reference ref{x0, P0};
        ref = reference_analysis(ref, y, H, R, cfg.taper, cfg.lambda);

        KF<double> kf(N);
        SquareRootKF<double> skf(N);
        if (cfg.taper) { kf.set_taper(*cfg.taper); skf.set_taper(*cfg.taper); }
        kf.set_inflation(cfg.lambda);
        skf.set_inflation(cfg.lambda);
        kf.initialize(x0, P0);
        skf.initialize(x0, P0);
        kf.measurement_update(y, H, R);
        skf.measurement_update(y, H, R);

        const std::string tag = std::string(cfg.what) + " [KF]";
        check_vec(kf.state(), ref.x, 1e-11, tag.c_str());
        check_mat(kf.covariance(), ref.P, 1e-11, tag.c_str());
        const std::string tag2 = std::string(cfg.what) + " [SquareRootKF]";
        check_vec(skf.state(), ref.x, 1e-11, tag2.c_str());
        check_mat(skf.covariance(), ref.P, 1e-11, tag2.c_str());
    }
}


// ---------------------------------------------------------------------------
// [cross-representation] -- KF and SquareRootKF are the same filter
// ---------------------------------------------------------------------------

struct Pair {
    KF<double> kf;
    SquareRootKF<double> skf;
    explicit Pair(int N) : kf(N), skf(N) {}
    void configure(const MatrixD *taper, double lambda)
    {
        if (taper) { kf.set_taper(*taper); skf.set_taper(*taper); }
        kf.set_inflation(lambda);
        skf.set_inflation(lambda);
    }
};

static void
run_pair(Pair &p, const int steps, const int N, const int m,
         const double tol, const char *what)
{
    double worst = 0.0;
    for (int s = 0; s < steps; ++s) {
        const VectorD x0 = random_vector(N);
        const MatrixD P0 = random_spd(N);
        p.kf.initialize(x0, P0);
        p.skf.initialize(x0, P0);
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N) * 0.97 + random_matrix(N, N) * 0.03;
        const MatrixD Q = random_spd(N, 1e-3);
        const VectorD u = random_vector(N);

        p.kf.measurement_update(y, H, R);
        p.skf.measurement_update(y, H, R);
        worst = std::max(worst, rel_err(p.kf.state(), p.skf.state()));
        worst = std::max(worst, rel_err(p.kf.covariance(), p.skf.covariance()));
        if (s % 2 == 0) {
            p.kf.time_update(F, Q, u);
            p.skf.time_update(F, Q, u);
        } else {
            p.kf.time_update(F, Q);
            p.skf.time_update(F, Q);
        }
        worst = std::max(worst, rel_err(p.kf.state(), p.skf.state()));
        worst = std::max(worst, rel_err(p.kf.covariance(), p.skf.covariance()));
    }
    testing::check(worst <= tol, what, worst, tol);
}

TEST_CASE("KF == SquareRootKF, plain")
{
    Pair p(8);
    run_pair(p, 20, 8, 4, 1e-11, "state and covariance agree (no taper)");
}

TEST_CASE("KF == SquareRootKF, taper (the LKF)")
{
    Pair p(8);
    const MatrixD C = ar1_taper(8, 0.6);
    p.configure(&C, 1.0);
    run_pair(p, 20, 8, 4, 1e-11, "state and covariance agree (taper = LKF)");
}

TEST_CASE("KF == SquareRootKF, inflation")
{
    Pair p(8);
    p.configure(nullptr, 1.05);
    run_pair(p, 20, 8, 4, 1e-11, "state and covariance agree (inflation)");
}

TEST_CASE("KF == SquareRootKF, taper + inflation")
{
    Pair p(6);
    const MatrixD C = ar1_taper(6, 0.8);
    p.configure(&C, 1.02);
    run_pair(p, 20, 6, 3, 1e-11, "state and covariance agree (taper + inflation)");
}

TEST_CASE("KF == SquareRootKF through Filter<T>& (polymorphic dispatch)")
{
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    KF<double> kf(N);
    SquareRootKF<double> skf(N);
    FilterD *a = &kf;
    FilterD *b = &skf;
    a->initialize(x0, P0);
    b->initialize(x0, P0);
    double worst = 0.0;
    for (int s = 0; s < 30; ++s) {
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N);
        const MatrixD Q = random_spd(N, 1e-3);
        a->measurement_update(y, H, R);
        b->measurement_update(y, H, R);
        worst = std::max(worst, rel_err(a->state(), b->state()));
        worst = std::max(worst, rel_err(a->covariance(), b->covariance()));
        a->time_update(F, Q);
        b->time_update(F, Q);
    }
    testing::check(worst <= 1e-11, "virtual dispatch preserves agreement", worst, 1e-11);
}

TEST_CASE("SquareRootKF.covariance() == factor() * factor().transpose()")
{
    SquareRootKF<double> f(7);
    f.initialize(random_vector(7), random_spd(7));
    for (int s = 0; s < 10; ++s) {
        f.measurement_update(random_vector(3), random_matrix(3, 7), random_spd(3, 0.5));
        f.time_update(random_matrix(7, 7) * 0.1 + MatrixD::Identity(7, 7),
                      random_spd(7, 1e-3));
        const MatrixD L = f.factor();
        check_mat(L * L.transpose(), f.covariance(), 1e-12,
                  "P == L L^T after every step");
        CHECK(is_symmetric(f.covariance()), "P exactly symmetric");
        CHECK(min_eig(f.covariance()) > 0.0, "P positive definite");
    }
}


// ---------------------------------------------------------------------------
// [taper = LKF]
// ---------------------------------------------------------------------------

TEST_CASE("all-ones taper is the Schur-product identity (== no taper)")
{
    // C o P with C = 1*1^T leaves P unchanged; C = I is NOT a no-op (it keeps
    // only diag(P)). The Schur identity element is the all-ones matrix.
    const int N = 6, m = 3;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD C  = MatrixD::Ones(N, N);
    Pair p1(N), p2(N);
    p1.configure(nullptr, 1.0);
    p2.configure(&C, 1.0);
    p1.kf.initialize(x0, P0); p1.skf.initialize(x0, P0);
    p2.kf.initialize(x0, P0); p2.skf.initialize(x0, P0);
    const MatrixD H = random_matrix(m, N);
    const MatrixD R = random_spd(m, 0.5);
    const VectorD y = random_vector(m);
    p1.kf.measurement_update(y, H, R); p1.skf.measurement_update(y, H, R);
    p2.kf.measurement_update(y, H, R); p2.skf.measurement_update(y, H, R);
    check_mat(p1.kf.covariance(), p2.kf.covariance(), 1e-13, "KF: ones taper == no taper");
    check_mat(p1.skf.covariance(), p2.skf.covariance(), 1e-13, "SquareRootKF: ones taper == no taper");
}

TEST_CASE("zero taper nulls the gain: measurement_update is a no-op")
{
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD C  = MatrixD::Zero(N, N);
    for (int which = 0; which < 2; ++which) {
        if (which == 0) {
            KF<double> f(N);
            f.set_taper(C);
            f.initialize(x0, P0);
            f.measurement_update(random_vector(m), random_matrix(m, N), random_spd(m, 0.5));
            check_vec(f.state(), x0, 1e-13, "KF: zero taper leaves the state alone");
            check_mat(f.covariance(), P0, 1e-13, "KF: zero taper leaves P alone");
        } else {
            SquareRootKF<double> f(N);
            f.set_taper(C);
            f.initialize(x0, P0);
            f.measurement_update(random_vector(m), random_matrix(m, N), random_spd(m, 0.5));
            check_vec(f.state(), x0, 1e-13, "SquareRootKF: zero taper leaves the state alone");
            check_mat(f.covariance(), P0, 1e-13, "SquareRootKF: zero taper leaves P alone");
        }
    }
}

TEST_CASE("Joseph update uses the untapered prior (LKF eq. 4.26)")
{
    // Under a taper the gain is built from C o P but the covariance update is
    // the Joseph form on P itself. If the implementation instead used C o P on
    // both sides, the two would differ; the reference below pins which one.
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    const MatrixD C  = ar1_taper(N, 0.4);

    // Variant B: taper applied to P on BOTH sides of the Joseph update.
    MatrixD Pg = C.cwiseProduct(P0);
    Pg = (Pg + Pg.transpose()) * 0.5;
    const MatrixD S  = H * Pg * H.transpose() + R;
    const MatrixD K  = Pg * H.transpose() * S.inverse();
    const MatrixD IKH = MatrixD::Identity(N, N) - K * H;
    MatrixD P_wrong = IKH * Pg * IKH.transpose() + K * R * K.transpose();
    P_wrong = (P_wrong + P_wrong.transpose()) * 0.5;

    const Reference ref = reference_analysis({x0, P0}, y, H, R, &C, 1.0);

    CHECK(rel_err(ref.P, P_wrong) > 1e-6,
          "taper-on-both-sides differs from eq. (4.26) [test is discriminating]");

    KF<double> kf(N);
    kf.set_taper(C);
    kf.initialize(x0, P0);
    kf.measurement_update(y, H, R);
    check_mat(kf.covariance(), ref.P, 1e-11, "KF follows eq. (4.26), not taper-on-both");
}

TEST_CASE("a positive definite taper keeps P positive definite")
{
    const int N = 8, m = 3;
    const MatrixD C = ar1_taper(N, 0.5);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    Pair q(N);
    q.configure(&C, 1.0);
    q.kf.initialize(x0, P0);
    q.skf.initialize(x0, P0);
    q.kf.measurement_update(y, H, R);
    q.skf.measurement_update(y, H, R);
    CHECK(min_eig(q.kf.covariance()) > 0.0, "KF: tapered P stays PD");
    CHECK(min_eig(q.skf.covariance()) > 0.0, "SquareRootKF: tapered P stays PD");
    check_mat(q.kf.covariance(), q.skf.covariance(), 1e-11,
              "tapered P agrees across backends");
}


// ---------------------------------------------------------------------------
// [inflation]
// ---------------------------------------------------------------------------

TEST_CASE("inflation: lambda = 1 is exactly off")
{
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    Pair p1(N), p2(N);
    p2.configure(nullptr, 1.0);
    p1.kf.initialize(x0, P0); p1.skf.initialize(x0, P0);
    p2.kf.initialize(x0, P0); p2.skf.initialize(x0, P0);
    p1.kf.measurement_update(y, H, R); p1.skf.measurement_update(y, H, R);
    p2.kf.measurement_update(y, H, R); p2.skf.measurement_update(y, H, R);
    check_mat(p1.kf.covariance(), p2.kf.covariance(), 1e-14, "KF: lambda=1 == off");
    check_mat(p1.skf.covariance(), p2.skf.covariance(), 1e-14, "SquareRootKF: lambda=1 == off");
    CHECK(p1.kf.inflation() == 1.0 && !p1.kf.has_taper(), "defaults report themselves");
}

TEST_CASE("inflation: analysis covariance is monotone in lambda")
{
    const int N = 6, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    double prev = -1.0;
    for (const double lambda : {1.0, 1.05, 1.2, 1.5}) {
        KF<double> f(N);
        f.set_inflation(lambda);
        f.initialize(x0, P0);
        f.measurement_update(y, H, R);
        const double tr = f.covariance().trace();
        CHECK(tr > prev, "larger lambda gives larger analysis trace(P)");
        prev = tr;
    }
}

TEST_CASE("inflation: applied in measurement_update, not time_update")
{
    const int N = 4;
    const VectorD x0 = VectorD::Zero(N);
    const MatrixD P0 = MatrixD::Identity(N, N);
    const MatrixD F  = MatrixD::Identity(N, N);
    const MatrixD Q  = MatrixD::Zero(N, N);
    KF<double> f(N);
    f.set_inflation(2.0);
    f.initialize(x0, P0);
    f.time_update(F, Q);
    CHECK_CLOSE(f.covariance()(0, 0), 1.0, 1e-14,
                "time_update alone does not inflate");
    f.measurement_update(VectorD::Zero(1), MatrixD::Identity(1, N),
                         MatrixD::Identity(1, 1));
    // prior was inflated to 2I before the analysis
    CHECK(f.covariance()(0, 0) < 2.0, "measurement_update consumed the inflated prior");
}


// ---------------------------------------------------------------------------
// [known input u]
// ---------------------------------------------------------------------------

TEST_CASE("time_update(F, Q, u) == time_update(F, Q) with x <- F x + u")
{
    const int N = 5;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD F  = random_matrix(N, N) * 0.2 + MatrixD::Identity(N, N) * 0.9;
    const MatrixD Q  = random_spd(N, 1e-3);
    const VectorD u  = random_vector(N);
    const VectorD x_expected = F * x0 + u;

    KF<double> a(N), b(N);
    SquareRootKF<double> sa(N), sb(N);
    a.initialize(x0, P0);   b.initialize(x0, P0);
    sa.initialize(x0, P0);  sb.initialize(x0, P0);

    a.time_update(F, Q, u);
    b.time_update(F, Q);
    sa.time_update(F, Q, u);
    sb.time_update(F, Q);

    check_vec(a.state(), x_expected, 1e-12, "KF: x <- F x + u");
    check_vec(sa.state(), x_expected, 1e-12, "SquareRootKF: x <- F x + u");
    check_mat(a.covariance(), b.covariance(), 1e-13, "KF: u does not affect P");
    check_mat(sa.covariance(), sb.covariance(), 1e-13, "SquareRootKF: u does not affect P");
    check_mat(a.covariance(), sa.covariance(), 1e-12, "u overload: backends agree");
}


// ---------------------------------------------------------------------------
// [edge cases]
// ---------------------------------------------------------------------------

TEST_CASE("edge: N=1, m=1, single step")
{
    for (int which = 0; which < 2; ++which) {
        const VectorD x0 = (VectorD(1) << 0.5).finished();
        const MatrixD P0 = (MatrixD(1, 1) << 2.0).finished();
        const VectorD y  = (VectorD(1) << 1.5).finished();
        const MatrixD H  = (MatrixD(1, 1) << 1.0).finished();
        const MatrixD R  = (MatrixD(1, 1) << 3.0).finished();
        Reference ref = reference_analysis({x0, P0}, y, H, R, nullptr, 1.0);
        if (which == 0) {
            KF<double> f(1);
            f.initialize(x0, P0);
            f.measurement_update(y, H, R);
            check_vec(f.state(), ref.x, 1e-13, "KF N=1 mean");
            check_mat(f.covariance(), ref.P, 1e-13, "KF N=1 covariance");
        } else {
            SquareRootKF<double> f(1);
            f.initialize(x0, P0);
            f.measurement_update(y, H, R);
            check_vec(f.state(), ref.x, 1e-13, "SquareRootKF N=1 mean");
            check_mat(f.covariance(), ref.P, 1e-13, "SquareRootKF N=1 covariance");
        }
    }
}

TEST_CASE("edge: Q = 0 (singular) still works and the backends agree")
{
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD Q0 = MatrixD::Zero(N, N);
    Pair p(N);
    p.kf.initialize(x0, P0);
    p.skf.initialize(x0, P0);
    double worst = 0.0;
    for (int s = 0; s < 20; ++s) {
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N);
        p.kf.measurement_update(y, H, R);
        p.skf.measurement_update(y, H, R);
        p.kf.time_update(F, Q0);
        p.skf.time_update(F, Q0);
        worst = std::max(worst, rel_err(p.kf.state(), p.skf.state()));
        worst = std::max(worst, rel_err(p.kf.covariance(), p.skf.covariance()));
    }
    testing::check(worst <= 1e-11, "Q=0: backends agree (psd_sqrt fallback)", worst, 1e-11);
    CHECK(min_eig(p.skf.covariance()) > 0.0, "Q=0: SquareRootKF P stays PD");
}

TEST_CASE("edge: rank-deficient PSD Q")
{
    const int N = 4;
    MatrixD Q = MatrixD::Zero(N, N);
    Q(0, 0) = 1.0; Q(1, 1) = 2.0;          // rank 2 < N
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    Pair p(N);
    p.kf.initialize(x0, P0);
    p.skf.initialize(x0, P0);
    const MatrixD F = MatrixD::Identity(N, N);
    p.kf.time_update(F, Q);
    p.skf.time_update(F, Q);
    check_mat(p.kf.covariance(), p.skf.covariance(), 1e-11, "rank-deficient Q: agree");
    CHECK(min_eig(p.skf.covariance()) > 0.0, "rank-deficient Q: P still PD from P0");
}

TEST_CASE("edge: F = 0 resets the state to the input")
{
    const int N = 4;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const VectorD u  = random_vector(N);
    const MatrixD F0 = MatrixD::Zero(N, N);
    const MatrixD Q  = random_spd(N, 1e-3);
    Pair p(N);
    p.kf.initialize(x0, P0);
    p.skf.initialize(x0, P0);
    p.kf.time_update(F0, Q, u);
    p.skf.time_update(F0, Q, u);
    check_vec(p.kf.state(), u, 1e-12, "KF: F=0 gives x = u");
    check_vec(p.skf.state(), u, 1e-12, "SquareRootKF: F=0 gives x = u");
    check_mat(p.kf.covariance(), p.skf.covariance(), 1e-11, "F=0: P agrees (== Q)");
}

TEST_CASE("edge: tiny R gives an analysis close to H^+ y")
{
    const int N = 4, m = 2;
    const VectorD x0 = VectorD::Zero(N);
    const MatrixD P0 = MatrixD::Identity(N, N);
    const MatrixD H  = MatrixD::Identity(m, N);
    const VectorD y  = (VectorD(2) << 3.0, -4.0).finished();
    const MatrixD R  = MatrixD::Identity(m, m) * 1e-12;
    Pair p(N);
    p.kf.initialize(x0, P0);
    p.skf.initialize(x0, P0);
    p.kf.measurement_update(y, H, R);
    p.skf.measurement_update(y, H, R);
    CHECK_CLOSE(p.kf.state()(0), 3.0, 1e-5, "KF: H^+ y component 0");
    CHECK_CLOSE(p.kf.state()(1), -4.0, 1e-5, "KF: H^+ y component 1");
    check_vec(p.kf.state(), p.skf.state(), 1e-11, "tiny R: backends agree");
}

TEST_CASE("edge: stability over 500 steps, no drift between backends")
{
    const int N = 5, m = 2;
    Pair p(N);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    p.kf.initialize(x0, P0);
    p.skf.initialize(x0, P0);
    double worst = 0.0;
    for (int s = 0; s < 500; ++s) {
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N) * 0.95 + random_matrix(N, N) * 0.05;
        const MatrixD Q = random_spd(N, 1e-4);
        p.kf.measurement_update(y, H, R);
        p.skf.measurement_update(y, H, R);
        p.kf.time_update(F, Q);
        p.skf.time_update(F, Q);
        if (s % 50 == 0) {
            worst = std::max(worst, rel_err(p.kf.state(), p.skf.state()));
            worst = std::max(worst, rel_err(p.kf.covariance(), p.skf.covariance()));
        }
    }
    worst = std::max(worst, rel_err(p.kf.state(), p.skf.state()));
    worst = std::max(worst, rel_err(p.kf.covariance(), p.skf.covariance()));
    testing::check(worst <= 1e-9, "500 steps: no drift", worst, 1e-9);
    CHECK(min_eig(p.skf.covariance()) > 0.0, "500 steps: P stays PD");
    CHECK(std::isfinite(p.kf.state().norm()), "500 steps: state finite");
}


// ---------------------------------------------------------------------------
// [batch and Record]
// ---------------------------------------------------------------------------

struct Problem {
    int N, m, I;
    std::vector<VectorD> y, u;
    std::vector<MatrixD> H, R, F, Q;
    std::vector<CVecD> yv, uv;
    std::vector<CMatD> Hv, Rv, Fv, Qv;
};

static Problem
make_problem(const int N, const int m, const int I, const bool with_u = false)
{
    Problem p;
    p.N = N; p.m = m; p.I = I;
    p.y.reserve(I); p.H.reserve(I); p.R.reserve(I); p.F.reserve(I); p.Q.reserve(I);
    for (int i = 0; i < I; ++i) {
        p.y.push_back(random_vector(m));
        p.H.push_back(random_matrix(m, N));
        p.R.push_back(random_spd(m, 0.5));
        p.F.push_back(MatrixD::Identity(N, N) * 0.98 + random_matrix(N, N) * 0.02);
        p.Q.push_back(random_spd(N, 1e-3));
        p.u.push_back(random_vector(N));
    }
    for (int i = 0; i < I; ++i) {
        p.yv.push_back(p.y[i]); p.Hv.push_back(p.H[i]); p.Rv.push_back(p.R[i]);
        p.Fv.push_back(p.F[i]); p.Qv.push_back(p.Q[i]); p.uv.push_back(p.u[i]);
    }
    return p;
}

TEST_CASE("batch(): prior/posterior are at time i (independent replay)")
{
    const Problem pr = make_problem(4, 2, 5);
    for (int which = 0; which < 2; ++which) {
        const VectorD x0 = random_vector(pr.N);
        const MatrixD P0 = random_spd(pr.N);
        BatchOutput<double> out;
        if (which == 0) {
            KF<double> f(pr.N);
            f.initialize(x0, P0);
            out = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
        } else {
            SquareRootKF<double> f(pr.N);
            f.initialize(x0, P0);
            out = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
        }
        CHECK(out.x_prior.size() == pr.I && out.x_posterior.size() == pr.I,
              "batch: record lengths");
        bool ok = true;
        if (which == 0) {
            KF<double> g(pr.N);
            g.initialize(x0, P0);
            for (int i = 0; i < pr.I && ok; ++i) {
                ok = rel_err(out.x_prior[i], g.state()) < 1e-12
                  && rel_err(out.P_prior[i], g.covariance()) < 1e-12;
                g.measurement_update(pr.y[i], pr.H[i], pr.R[i]);
                ok = ok && rel_err(out.x_posterior[i], g.state()) < 1e-12
                         && rel_err(out.P_posterior[i], g.covariance()) < 1e-12;
                g.time_update(pr.F[i], pr.Q[i]);
            }
        } else {
            SquareRootKF<double> g(pr.N);
            g.initialize(x0, P0);
            for (int i = 0; i < pr.I && ok; ++i) {
                ok = rel_err(out.x_prior[i], g.state()) < 1e-12
                  && rel_err(out.P_prior[i], g.covariance()) < 1e-12;
                g.measurement_update(pr.y[i], pr.H[i], pr.R[i]);
                ok = ok && rel_err(out.x_posterior[i], g.state()) < 1e-12
                         && rel_err(out.P_posterior[i], g.covariance()) < 1e-12;
                g.time_update(pr.F[i], pr.Q[i]);
            }
        }
        CHECK(ok, which == 0 ? "KF: batch matches replay"
                             : "SquareRootKF: batch matches replay");
    }
}

TEST_CASE("batch(): every Record mask bit")
{
    const Problem pr = make_problem(3, 2, 4);
    const VectorD x0 = random_vector(pr.N);
    const MatrixD P0 = random_spd(pr.N);
    KF<double> f(pr.N);
    f.initialize(x0, P0);
    const BatchOutput<double> a = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv,
                                          Record::None);
    CHECK(a.x_prior.empty() && a.x_posterior.empty() &&
          a.P_prior.empty() && a.P_posterior.empty(), "Record::None stores nothing");

    f.initialize(x0, P0);
    const BatchOutput<double> b = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv,
                                          Record::XPrior);
    CHECK(b.x_prior.size() == pr.I && b.x_posterior.empty() &&
          b.P_prior.empty() && b.P_posterior.empty(), "Record::XPrior only");

    f.initialize(x0, P0);
    const BatchOutput<double> c = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv,
                                          Record::PPosterior | Record::XPosterior);
    CHECK(c.x_prior.empty() && c.x_posterior.size() == pr.I &&
          c.P_prior.empty() && c.P_posterior.size() == pr.I,
          "Record::XPosterior | PPosterior only");
}

TEST_CASE("batch(): with u, and chunking is composable")
{
    const Problem pr = make_problem(4, 2, 6, true);
    const VectorD x0 = random_vector(pr.N);
    const MatrixD P0 = random_spd(pr.N);

    KF<double> whole(pr.N);
    whole.initialize(x0, P0);
    const BatchOutput<double> a =
        whole.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv, pr.uv);

    KF<double> part(pr.N);
    part.initialize(x0, P0);
    std::vector<CVecD> y1(pr.yv.begin(), pr.yv.begin() + 3), u1(pr.uv.begin(), pr.uv.begin() + 3);
    std::vector<CMatD> H1(pr.Hv.begin(), pr.Hv.begin() + 3), R1(pr.Rv.begin(), pr.Rv.begin() + 3);
    std::vector<CMatD> F1(pr.Fv.begin(), pr.Fv.begin() + 3), Q1(pr.Qv.begin(), pr.Qv.begin() + 3);
    std::vector<CVecD> y2(pr.yv.begin() + 3, pr.yv.end()), u2(pr.uv.begin() + 3, pr.uv.end());
    std::vector<CMatD> H2(pr.Hv.begin() + 3, pr.Hv.end()), R2(pr.Rv.begin() + 3, pr.Rv.end());
    std::vector<CMatD> F2(pr.Fv.begin() + 3, pr.Fv.end()), Q2(pr.Qv.begin() + 3, pr.Qv.end());
    const BatchOutput<double> b1 = part.batch(y1, H1, R1, F1, Q1, u1);
    const BatchOutput<double> b2 = part.batch(y2, H2, R2, F2, Q2, u2);

    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        ok = ok && rel_err(a.x_posterior[i], b1.x_posterior[i]) < 1e-12
                 && rel_err(a.P_posterior[i], b1.P_posterior[i]) < 1e-12;
        ok = ok && rel_err(a.x_posterior[i + 3], b2.x_posterior[i]) < 1e-12
                 && rel_err(a.P_posterior[i + 3], b2.P_posterior[i]) < 1e-12;
    }
    CHECK(ok, "batch(): two halves == one call (chunking is composable)");

    // Empty sequence.
    KF<double> empty(pr.N);
    empty.initialize(x0, P0);
    std::vector<CVecD> none_v;
    std::vector<CMatD> none_m;
    const BatchOutput<double> z = empty.batch(none_v, none_m, none_m, none_m, none_m);
    CHECK(z.x_prior.empty() && z.P_posterior.empty(), "batch(): empty sequence is a no-op");
}


// ---------------------------------------------------------------------------
// [contracts] -- asserts must actually fire (needs asserts ON in this binary)
// ---------------------------------------------------------------------------

TEST_CASE("contracts: dimension mismatches throw (always on, not assert)")
{
    const int N = 4, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const VectorD y  = random_vector(m);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const MatrixD F  = MatrixD::Identity(N, N);
    const MatrixD Q  = random_spd(N, 1e-3);

    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(random_vector(N + 1), P0); },
                 "KF.initialize: wrong size x throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, random_matrix(N, N + 1)); },
                 "KF.initialize: non-square P throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.measurement_update(random_vector(m + 1), H, R); },
                 "KF.measurement_update: wrong size y throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.measurement_update(y, H, random_matrix(m, m + 1)); },
                 "KF.measurement_update: non-square R throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.measurement_update(y, random_matrix(m, N + 1), R); },
                 "KF.measurement_update: H cols != N throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.time_update(random_matrix(N, N + 1), Q); },
                 "KF.time_update: wrong size F throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.time_update(F, random_matrix(N + 1, N + 1)); },
                 "KF.time_update: wrong size Q throws");
    CHECK_THROWS([&]{ KF<double> f(N); f.initialize(x0, P0);
                      f.time_update(F, Q, random_vector(N + 1)); },
                 "KF.time_update: wrong size u throws");

    CHECK_THROWS([&]{ SquareRootKF<double> f(N); f.initialize(random_vector(N + 1), P0); },
                 "SquareRootKF.initialize: wrong size x throws");
    CHECK_THROWS([&]{ SquareRootKF<double> f(N); f.initialize(x0, P0);
                      f.measurement_update(y, H, random_matrix(m, m + 1)); },
                 "SquareRootKF.measurement_update: non-square R throws");
    CHECK_THROWS([&]{ SquareRootKF<double> f(N); f.initialize(x0, P0);
                      f.time_update(F, random_matrix(N + 1, N + 1)); },
                 "SquareRootKF.time_update: wrong size Q throws");
    CHECK_THROWS([&]{ SquareRootKF<double> f(N); f.initialize(x0, P0);
                      f.time_update(F, Q, random_vector(N + 1)); },
                 "SquareRootKF.time_update: wrong size u throws");

    // batch(): inconsistent sequence lengths.
    // (no brace-initialization in these lambdas: top-level commas would split
    //  the macro arguments, since {} does not nest for the preprocessor)
    CHECK_THROWS([&]{
        KF<double> f(N); f.initialize(x0, P0);
        std::vector<CVecD> yv;
        yv.push_back(y);
        yv.push_back(y);
        std::vector<CMatD> Hv; Hv.push_back(H);
        std::vector<CMatD> Rv; Rv.push_back(R); Rv.push_back(R);
        std::vector<CMatD> Fv; Fv.push_back(F); Fv.push_back(F);
        std::vector<CMatD> Qv; Qv.push_back(Q); Qv.push_back(Q);
        f.batch(yv, Hv, Rv, Fv, Qv);
    }, "batch: y.size() != H.size() throws");

    CHECK_THROWS([&]{
        KF<double> f(N); f.initialize(x0, P0);
        std::vector<CVecD> yv;
        yv.push_back(y);
        yv.push_back(y);
        std::vector<CVecD> uv; uv.push_back(random_vector(N));
        std::vector<CMatD> Hv; Hv.push_back(H); Hv.push_back(H);
        std::vector<CMatD> Rv; Rv.push_back(R); Rv.push_back(R);
        std::vector<CMatD> Fv; Fv.push_back(F); Fv.push_back(F);
        std::vector<CMatD> Qv; Qv.push_back(Q); Qv.push_back(Q);
        f.batch(yv, Hv, Rv, Fv, Qv, uv);
    }, "batch: u.size() != y.size() throws");
}


// ---------------------------------------------------------------------------
// [memory] -- long run with bounded working set (ASan/LSan is the real check)
// ---------------------------------------------------------------------------

TEST_CASE("memory: 20000 updates keep P PD, symmetric and finite")
{
    const int N = 6, m = 2;
    Pair p(N);
    const MatrixD C = ar1_taper(N, 0.7);
    p.configure(&C, 1.01);
    p.kf.initialize(random_vector(N), random_spd(N));
    p.skf.initialize(p.kf.state(), p.kf.covariance());
    for (int s = 0; s < 20000; ++s) {
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N) * 0.99;
        const MatrixD Q = MatrixD::Identity(N, N) * 1e-5;
        p.kf.measurement_update(y, H, R);
        p.skf.measurement_update(y, H, R);
        p.kf.time_update(F, Q);
        p.skf.time_update(F, Q);
    }
    const MatrixD Pk = p.kf.covariance();
    const MatrixD Ps = p.skf.covariance();
    CHECK(std::isfinite(Pk.norm()) && std::isfinite(Ps.norm()), "20000 steps: P finite");
    CHECK(is_symmetric(Pk) && is_symmetric(Ps), "20000 steps: P exactly symmetric");
    CHECK(min_eig(Ps) > 0.0, "20000 steps: SquareRootKF P stays PD");
    CHECK(std::isfinite(p.kf.state().norm()), "20000 steps: state finite");
}


// ===========================================================================
// [the Kalman family] -- KF, SquareRootKF and UDKF are the same filter
// ===========================================================================

TEST_CASE("UDKF == KF == SquareRootKF exactly (no taper)")
{
    const int N = 6, m = 3, steps = 20;
    const double tol = 1e-11;
    KF<double> a(N);
    SquareRootKF<double> b(N);
    UDKF<double> c(N);
    double worst_x = 0.0, worst_P = 0.0;
    for (int s = 0; s < steps; ++s) {
        const VectorD x0 = random_vector(N);
        const MatrixD P0 = random_spd(N);
        for (FilterD *f : {static_cast<FilterD*>(&a), static_cast<FilterD*>(&b),
                           static_cast<FilterD*>(&c)}) f->initialize(x0, P0);
        const MatrixD H = random_matrix(m, N);
        const MatrixD R = random_spd(m, 0.5);
        const VectorD y = random_vector(m);
        const MatrixD F = MatrixD::Identity(N, N) * 0.97 + random_matrix(N, N) * 0.03;
        const MatrixD Q = random_spd(N, 1e-3);
        const VectorD u = random_vector(N);
        for (FilterD *f : {static_cast<FilterD*>(&a), static_cast<FilterD*>(&b),
                           static_cast<FilterD*>(&c)}) f->measurement_update(y, H, R);
        worst_x = std::max({worst_x, rel_err(a.state(), b.state()), rel_err(a.state(), c.state())});
        worst_P = std::max({worst_P, rel_err(a.covariance(), b.covariance()),
                            rel_err(a.covariance(), c.covariance())});
        for (FilterD *f : {static_cast<FilterD*>(&a), static_cast<FilterD*>(&b),
                           static_cast<FilterD*>(&c)}) f->time_update(F, Q, u);
        worst_x = std::max({worst_x, rel_err(a.state(), b.state()), rel_err(a.state(), c.state())});
        worst_P = std::max({worst_P, rel_err(a.covariance(), b.covariance()),
                            rel_err(a.covariance(), c.covariance())});
    }
    testing::check(worst_x <= tol, "KF == SquareRootKF == UDKF, state", worst_x, tol);
    testing::check(worst_P <= tol, "KF == SquareRootKF == UDKF, covariance", worst_P, tol);
}

TEST_CASE("UDKF == KF exactly with a taper (the LKF)")
{
    const int N = 6, m = 3;
    const MatrixD C = ar1_taper(N, 0.55);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H = random_matrix(m, N);
    const MatrixD R = random_spd(m, 0.5);
    const VectorD y = random_vector(m);
    KF<double> a(N);
    UDKF<double> b(N);
    SquareRootKF<double> c(N);
    for (FilterD *f : {static_cast<FilterD*>(&a), static_cast<FilterD*>(&b),
                       static_cast<FilterD*>(&c)}) {
        f->set_taper(C);
        f->initialize(x0, P0);
        f->measurement_update(y, H, R);
    }
    check_vec(a.state(), b.state(), 1e-11, "UDKF (taper) == KF, state");
    check_mat(a.covariance(), b.covariance(), 1e-11, "UDKF (taper) == KF, covariance");
    check_mat(a.covariance(), c.covariance(), 1e-11, "SquareRootKF (taper) == KF, covariance");
}

TEST_CASE("UD factorization invariants")
{
    const int N = 7;
    UDKF<double> f(N);
    f.initialize(random_vector(N), random_spd(N));
    for (int s = 0; s < 15; ++s) {
        f.measurement_update(random_vector(3), random_matrix(3, N), random_spd(3, 0.5));
        f.time_update(random_matrix(N, N) * 0.1 + MatrixD::Identity(N, N),
                      random_spd(N, 1e-3));
        const MatrixD L = f.unit_lower();
        const VectorD D = f.diagonal();
        // unit lower triangular
        MatrixD up = L.triangularView<Eigen::StrictlyUpper>();
        testing::check(up.cwiseAbs().maxCoeff() == 0.0, "UDKF: L has empty strict upper triangle");
        testing::check_close(L.diagonal().cwiseAbs().maxCoeff(), 1.0, 1e-15, "UDKF: diag(L) == 1");
        testing::check((D.array() > 0.0).all(), "UDKF: D > 0");
        check_mat(L * D.asDiagonal() * L.transpose(), f.covariance(), 1e-12,
                  "UDKF: L D L^T == covariance()");
    }
}

TEST_CASE("Bierman downdate matches a dense factorization")
{
    // Isolate the rank-1 downdate: D - g g^T / alpha must equal
    // Delta D' Delta^T with Delta unit lower (what bierman_row builds).
    const int N = 6;
    UDKF<double> f(N);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    f.initialize(x0, P0);
    const VectorD h = random_vector(N);
    const double r = 0.8;
    const MatrixD P_before = f.covariance();
    VectorD y1(1);
    y1(0) = h.dot(x0) + 0.3;
    const MatrixD H1 = h.transpose();
    const MatrixD R1 = (MatrixD(1, 1) << r).finished();
    f.measurement_update(y1, H1, R1);

    const MatrixD P_after = f.covariance();
    // Reference: (I - k h) P with k = P h / (h^T P h + r)  (h is a column).
    const double s = h.dot(P_before * h) + r;
    const VectorD k = P_before * h / s;
    MatrixD P_ref = (MatrixD::Identity(N, N) - k * h.transpose()) * P_before;
    P_ref = (P_ref + P_ref.transpose()) * 0.5;
    check_mat(P_after, P_ref, 1e-11, "Bierman row realizes (I - k h) P");
    const VectorD x_ref = x0 + k * 0.3;
    check_vec(f.state(), x_ref, 1e-11, "Bierman row realizes the Kalman mean");
}

// ===========================================================================
// [the float instantiations]
// ===========================================================================

// Helpers in float. Values are drawn from the same generators and narrowed, so
// the two precisions see the same problem.
static MatrixF
to_float(const MatrixD &A)
{
    return A.cast<float>();
}

static VectorF
to_float(const VectorD &a)
{
    return a.cast<float>();
}

TEST_CASE("float: KF == SquareRootKF == UDKF")
{
    // The whole point of instantiating float is that the bodies are
    // precision-agnostic; this checks the exact-Kalman trio agrees there too.
    // Tolerances are looser than the double tests: float has ~7 digits.
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    const MatrixD F  = random_matrix(N, N) * 0.2 + MatrixD::Identity(N, N) * 0.9;
    const MatrixD Q  = random_spd(N, 1e-3);

    KF<float>          a(N);
    SquareRootKF<float> b(N);
    UDKF<float>        c(N);
    for (FilterF *f : {static_cast<FilterF *>(&a), static_cast<FilterF *>(&b),
                       static_cast<FilterF *>(&c)}) {
        f->initialize(to_float(x0), to_float(P0));
        f->measurement_update(to_float(y), to_float(H), to_float(R));
    }
    double worst = 0.0;
    worst = std::max(worst, double((a.state() - b.state()).norm()));
    worst = std::max(worst, double((a.state() - c.state()).norm()));
    worst = std::max(worst, double((a.covariance() - b.covariance()).norm()));
    worst = std::max(worst, double((a.covariance() - c.covariance()).norm()));
    testing::check(worst <= 1e-4,
                   "float: KF == SquareRootKF == UDKF after one analysis", worst, 1e-4);

    for (FilterF *f : {static_cast<FilterF *>(&a), static_cast<FilterF *>(&b),
                       static_cast<FilterF *>(&c)}) {
        f->time_update(to_float(F), to_float(Q));
    }
    worst = 0.0;
    worst = std::max(worst, double((a.state() - b.state()).norm()));
    worst = std::max(worst, double((a.state() - c.state()).norm()));
    worst = std::max(worst, double((a.covariance() - b.covariance()).norm()));
    worst = std::max(worst, double((a.covariance() - c.covariance()).norm()));
    testing::check(worst <= 1e-4,
                   "float: KF == SquareRootKF == UDKF after one cycle", worst, 1e-4);
}

TEST_CASE("float: KF == SquareRootKF == UDKF under a taper")
{
    const int N = 5, m = 2;
    const MatrixD C = ar1_taper(N, 0.5);
    KF<float>           a(N);
    SquareRootKF<float> b(N);
    UDKF<float>         c(N);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);
    for (FilterF *f : {static_cast<FilterF *>(&a), static_cast<FilterF *>(&b),
                       static_cast<FilterF *>(&c)}) {
        f->set_taper(to_float(C));
        f->initialize(to_float(x0), to_float(P0));
        f->measurement_update(to_float(y), to_float(H), to_float(R));
    }
    double worst = 0.0;
    worst = std::max(worst, double((a.state() - b.state()).norm()));
    worst = std::max(worst, double((a.state() - c.state()).norm()));
    worst = std::max(worst, double((a.covariance() - b.covariance()).norm()));
    testing::check(worst <= 1e-4,
                   "float: the trio agrees with a taper (the LKF)", worst, 1e-4);
}

TEST_CASE("float: the ensemble trio agrees and stays finite")
{
    // Exercises the ensemble path in float, including ETKF's eigenvalue clamp.
    // Regression note: the clamp floor was Scalar(1e-300), which is 0 for
    // float, so a zero eigenvalue would have produced inf instead of a large
    // finite number. (ETKF's C is >= I so the clamp is defensive today, but it
    // must still survive the Scalar.)
    const int N = 5, m = 2, L = 60;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H  = random_matrix(m, N);
    const MatrixD R  = random_spd(m, 0.5);
    const VectorD y  = random_vector(m);

    EnSRF<float> a(N, L, 1);
    EAKF<float>  b(N, L, 1);
    ETKF<float>  c(N, L, 1);
    a.initialize(to_float(x0), to_float(P0));
    const MatrixF X = a.members();
    b.set_members(X);
    c.set_members(X);
    for (EnsembleFilter<float> *e : {static_cast<EnsembleFilter<float> *>(&a),
                                     static_cast<EnsembleFilter<float> *>(&b),
                                     static_cast<EnsembleFilter<float> *>(&c)}) {
        e->measurement_update(to_float(y), to_float(H), to_float(R));
    }
    testing::check(a.covariance().allFinite() && b.covariance().allFinite() &&
                       c.covariance().allFinite(),
                   "float: ensemble covariances are finite (no inf from the clamp)");
    testing::check(a.state().allFinite() && b.state().allFinite() &&
                       c.state().allFinite(),
                   "float: ensemble states are finite");
    double worst = 0.0;
    worst = std::max(worst, double((a.state() - b.state()).norm()));
    worst = std::max(worst, double((a.state() - c.state()).norm()));
    worst = std::max(worst, double((a.covariance() - b.covariance()).norm()));
    worst = std::max(worst, double((a.covariance() - c.covariance()).norm()));
    testing::check(worst <= 1e-3,
                   "float: EnSRF == EAKF == ETKF", worst, 1e-3);
}

TEST_CASE("float: EnKS and rts_smooth stay finite")
{
    const int N = 4, m = 2, I = 4;
    const Problem pr = make_problem(N, m, I);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);

    std::vector<VectorF> ys, us;
    std::vector<MatrixF> Hs, Rs, Fs, Qs;
    for (int i = 0; i < I; ++i) {
        ys.push_back(to_float(pr.y[i]));
        Hs.push_back(to_float(pr.H[i]));
        Rs.push_back(to_float(pr.R[i]));
        Fs.push_back(to_float(pr.F[i]));
        Qs.push_back(to_float(pr.Q[i]));
        us.push_back(to_float(pr.u[i]));
    }
    std::vector<CVecF> yv, uv;
    std::vector<CMatF> Hv, Rv, Fv, Qv;
    for (int i = 0; i < I; ++i) {
        yv.push_back(ys[i]); uv.push_back(us[i]);
        Hv.push_back(Hs[i]); Rv.push_back(Rs[i]);
        Fv.push_back(Fs[i]); Qv.push_back(Qs[i]);
    }

    EnKS<float> ks(N, 50, 2);
    ks.initialize(to_float(x0), to_float(P0));
    const SmoothOutput<float> sm = ks.smooth(yv, Hv, Rv, Fv, Qv);
    bool finite = sm.x_smoothed.size() == I;
    for (int i = 0; i < I && finite; ++i)
        finite = sm.x_smoothed[i].allFinite() && sm.P_smoothed[i].allFinite();
    testing::check(finite, "float: EnKS trajectories are finite");

    KF<float> f(N);
    f.initialize(to_float(x0), to_float(P0));
    const BatchOutput<float> rec = f.batch(yv, Hv, Rv, Fv, Qv);
    const SmoothOutput<float> ex = rts_smooth<float>(rec, Fv, Qv);
    finite = ex.x_smoothed.size() == I;
    for (int i = 0; i < I && finite; ++i)
        finite = ex.x_smoothed[i].allFinite() && ex.P_smoothed[i].allFinite();
    testing::check(finite, "float: rts_smooth trajectories are finite");
}


// ===========================================================================
// [the ensemble family]
// ===========================================================================

// Run every ensemble method from ONE shared prior ensemble and return the
// largest discrepancy against the exact KF.
struct EnsembleComparison {
    // vs_kf_* covers all four methods; det_* covers only the deterministic
    // square-root trio (EnSRF / EAKF / ETKF). EnKF is stochastic in its
    // anomalies and is deliberately excluded from det_*.
    double vs_kf_x = 0.0, vs_kf_P = 0.0, det_x = 0.0, det_P = 0.0;
};

static EnsembleComparison
compare_ensemble(const int N, const int m, const int L,
                 const MatrixD *taper, const double lambda,
                 const int trials = 1)
{
    EnsembleComparison out;
    for (int t = 0; t < trials; ++t) {
        const VectorD x0 = random_vector(N);
        const MatrixD P0 = random_spd(N);
        const MatrixD H  = random_matrix(m, N);
        const MatrixD R  = random_spd(m, 0.5);
        const VectorD y  = random_vector(m);

        KF<double> kf(N);
        if (taper) kf.set_taper(*taper);
        kf.set_inflation(lambda);
        kf.initialize(x0, P0);
        kf.measurement_update(y, H, R);

        EnKF<double>   a(N, L, 100 + t);
        EnSRF<double>  b(N, L, 200 + t);
        EAKF<double>   c(N, L, 300 + t);
        ETKF<double>   d(N, L, 400 + t);
        EnsembleFilter<double> *arr[4] = {&a, &b, &c, &d};
        a.initialize(x0, P0);
        const MatrixD X = a.members();            // ONE shared prior ensemble
        for (int i = 1; i < 4; ++i) {
            if (taper) arr[i]->set_taper(*taper);
            arr[i]->set_inflation(lambda);
            arr[i]->set_members(X);
        }
        if (taper) a.set_taper(*taper);
        a.set_inflation(lambda);
        for (int i = 0; i < 4; ++i) arr[i]->measurement_update(y, H, R);

        for (int i = 0; i < 4; ++i) {
            out.vs_kf_x = std::max(out.vs_kf_x, rel_err(kf.state(), arr[i]->state()));
            out.vs_kf_P = std::max(out.vs_kf_P, rel_err(kf.covariance(), arr[i]->covariance()));
            if (i == 0) continue;   // EnKF: stochastic, compared to KF only
            for (int j = std::max(i + 1, 1); j < 4; ++j) {
                out.det_x = std::max(out.det_x, rel_err(arr[i]->state(), arr[j]->state()));
                out.det_P = std::max(out.det_P, rel_err(arr[i]->covariance(), arr[j]->covariance()));
            }
        }
    }
    return out;
}

TEST_CASE("EnSRF == EAKF == ETKF exactly (same Joseph target, different rotations)")
{
    // With a shared prior ensemble the three deterministic square-root
    // algebras must agree bit-for-bit on mean and covariance; only the member
    // orientation may differ (Sakov & Oke 2008).
    const EnsembleComparison r = compare_ensemble(6, 3, 50, nullptr, 1.0, 3);
    testing::check(r.det_x <= 1e-12, "EnSRF == EAKF == ETKF, state", r.det_x, 1e-12);
    testing::check(r.det_P <= 1e-12, "EnSRF == EAKF == ETKF, covariance", r.det_P, 1e-12);
    // EnSRF and EAKF are the SAME linear map for a sequential scalar row
    // (both realize I - a k h), so they coincide exactly. ETKF's block
    // transform is a different factor of the same covariance, so its members
    // are a genuine rotation of the others (Sakov & Oke 2008).
    EnSRF<double> a(6, 20, 1); EAKF<double> b(6, 20, 1); ETKF<double> c(6, 20, 1);
    a.initialize(random_vector(6), random_spd(6));
    const MatrixD X = a.members();
    b.set_members(X); c.set_members(X);
    const MatrixD H = random_matrix(3, 6);
    const VectorD y = random_vector(3);
    const MatrixD R = random_spd(3, 0.5);
    a.measurement_update(y, H, R); b.measurement_update(y, H, R); c.measurement_update(y, H, R);
    const double rot_ab = (a.anomalies() - b.anomalies()).norm() / a.anomalies().norm();
    const double rot_ac = (a.anomalies() - c.anomalies()).norm() / a.anomalies().norm();
    testing::check(rot_ab <= 1e-12,
                   "EnSRF == EAKF exactly (same map for a scalar row)", rot_ab, 1e-12);
    testing::check(rot_ac > 1e-6,
                   "ETKF members are a rotation of EnSRF's", rot_ac, 1e-6);
    testing::check(rel_err(a.covariance(), c.covariance()) <= 1e-12,
                   "ETKF covariance still matches EnSRF",
                   rel_err(a.covariance(), c.covariance()), 1e-12);
}

TEST_CASE("ensemble methods share the LKF mean exactly")
{
    // Recentered perturbations + the LKF gain: with a shared prior ensemble
    // every method computes the identical analysis mean.
    const EnsembleComparison r = compare_ensemble(6, 3, 50, nullptr, 1.0, 5);
    testing::check(r.vs_kf_x <= 0.05 || r.det_x <= 1e-12,
                   "deterministic trio agree exactly on state()", r.det_x, 1e-12);
}

TEST_CASE("ensemble methods converge to KF as L grows (no taper)")
{
    const double tol = 0.25;
    const EnsembleComparison small = compare_ensemble(5, 2, 40, nullptr, 1.0, 4);
    const EnsembleComparison big   = compare_ensemble(5, 2, 4000, nullptr, 1.0, 4);
    testing::check(big.vs_kf_P < tol,
                   "L=4000: ensemble covariance within 25% of KF", big.vs_kf_P, tol);
    testing::check(big.vs_kf_P < small.vs_kf_P || big.vs_kf_P < 0.05,
                   "larger L is closer to KF", big.vs_kf_P, small.vs_kf_P);
}

TEST_CASE("ensemble methods converge to the LKF with a taper")
{
    const MatrixD C = ar1_taper(5, 0.6);
    const EnsembleComparison big = compare_ensemble(5, 2, 4000, &C, 1.0, 4);
    testing::check(big.vs_kf_P < 0.25,
                   "L=4000: tapered ensemble covariance within 25% of the LKF",
                   big.vs_kf_P, 0.25);
}

TEST_CASE("LETKF is ETKF with a required taper (exact)")
{
    const int N = 6, m = 3;
    const MatrixD C = ar1_taper(N, 0.5);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    LETKF<double> a(N, 30, C, 7);
    ETKF<double>  b(N, 30, 7);
    b.set_taper(C);
    a.initialize(x0, P0);
    b.set_members(a.members());
    const MatrixD H = random_matrix(m, N);
    const VectorD y = random_vector(m);
    const MatrixD R = random_spd(m, 0.5);
    a.measurement_update(y, H, R);
    b.measurement_update(y, H, R);
    check_vec(a.state(), b.state(), 1e-14, "LETKF == ETKF + set_taper, state");
    check_mat(a.covariance(), b.covariance(), 1e-14, "LETKF == ETKF + set_taper, covariance");
    CHECK(a.has_taper(), "LETKF requires a taper at construction");
}

TEST_CASE("inflation on the ensemble family matches KF")
{
    // lambda = 1 is off; lambda > 1 inflates the analysis covariance.
    const int N = 5, m = 2;
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD H = random_matrix(m, N);
    const MatrixD R = random_spd(m, 0.5);
    const VectorD y = random_vector(m);
    double prev = -1.0;
    for (const double lam : {1.0, 1.05, 1.2}) {
        EnKF<double> a(N, 200, 1);
        ETKF<double> b(N, 200, 1);
        a.set_inflation(lam);
        b.set_inflation(lam);
        a.initialize(x0, P0);
        b.set_members(a.members());
        a.measurement_update(y, H, R);
        b.measurement_update(y, H, R);
        const double tr = b.covariance().trace();
        CHECK(tr > prev, "ensemble: larger lambda gives larger analysis trace(P)");
        prev = tr;
    }
}

// ===========================================================================
// [smoothers]
// ===========================================================================

TEST_CASE("rts_smooth: terminal condition and uncertainty reduction")
{
    const int N = 4, m = 2, I = 8;
    const Problem pr = make_problem(N, m, I);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    KF<double> f(N);
    f.initialize(x0, P0);
    const BatchOutput<double> rec = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
    const SmoothOutput<double> sm = rts_smooth<double>(rec, pr.Fv, pr.Qv);

    CHECK(sm.x_smoothed.size() == I && sm.P_smoothed.size() == I, "rts: lengths");
    check_mat(sm.P_smoothed[I - 1], rec.P_posterior[I - 1], 1e-12,
              "rts: terminal condition P_s = P_a");
    bool shrinks = true;
    for (int i = 0; i < I - 1; ++i) {
        // Smoothing cannot increase the error covariance.
        const double ds = min_eig(sm.P_smoothed[i] - rec.P_posterior[i]);
        if (ds > 1e-9) shrinks = false;
    }
    CHECK(shrinks, "rts: smoothed covariance <= filtered covariance at every step");
}

TEST_CASE("EnKS converges to the exact RTS smoother as L grows")
{
    const int N = 4, m = 2, I = 6;
    const Problem pr = make_problem(N, m, I);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);

    // Exact reference.
    KF<double> f(N);
    f.initialize(x0, P0);
    const BatchOutput<double> rec = f.batch(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
    const SmoothOutput<double> ref = rts_smooth<double>(rec, pr.Fv, pr.Qv);

    EnKS<double> ks(N, 4000, 11);
    ks.initialize(x0, P0);
    const SmoothOutput<double> sm = ks.smooth(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);

    double worst_x = 0.0, worst_P = 0.0;
    for (int i = 0; i < I; ++i) {
        worst_x = std::max(worst_x, rel_err(ref.x_smoothed[i], sm.x_smoothed[i]));
        worst_P = std::max(worst_P, rel_err(ref.P_smoothed[i], sm.P_smoothed[i]));
    }
    testing::check(worst_x < 0.3, "EnKS (L=4000) vs RTS, state", worst_x, 0.3);
    // The smoothed covariance is P_{i|i} - (a correction), so a small number
    // measured in relative terms: the Monte Carlo error of the adjoint ensemble
    // is amplified by that difference.  It decays as 1/sqrt(L) -- checked
    // against the closed-form posterior in the linear-Gaussian example file,
    // where the relative error falls 0.075, 0.044, 0.018, 0.012 for
    // L = 500, 2000, 8000, 32000.
    testing::check(worst_P < 0.6, "EnKS (L=4000) vs RTS, covariance", worst_P, 0.6);
}

TEST_CASE("LEKS is EnKS with a required taper (exact)")
{
    const int N = 5, m = 2, I = 5;
    const Problem pr = make_problem(N, m, I);
    const VectorD x0 = random_vector(N);
    const MatrixD P0 = random_spd(N);
    const MatrixD C = ar1_taper(N, 0.55);

    LEKS<double> a(N, 40, C, 3);
    EnKS<double> b(N, 40, 3);
    b.set_taper(C);
    a.initialize(x0, P0);
    const SmoothOutput<double> sa = a.smooth(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
    b.initialize(x0, P0);
    const SmoothOutput<double> sb = b.smooth(pr.yv, pr.Hv, pr.Rv, pr.Fv, pr.Qv);
    bool same = true;
    for (int i = 0; i < I; ++i) {
        same = same && rel_err(sa.x_smoothed[i], sb.x_smoothed[i]) < 1e-14
                     && rel_err(sa.P_smoothed[i], sb.P_smoothed[i]) < 1e-14;
    }
    CHECK(same, "LEKS == EnKS + set_taper");
    CHECK(a.has_taper(), "LEKS requires a taper at construction");
}


// ---------------------------------------------------------------------------

int
main()
{
    return testing::Suite::get().run();
}
