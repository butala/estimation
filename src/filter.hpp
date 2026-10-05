#pragma once

// ---------------------------------------------------------------------------
// estimation::Filter -- the common interface for Kalman-family methods.
//
// Two conceptual families share this interface:
//
//   Exact Kalman filter, different uncertainty representations:
//       KF           -- stores P            (covariance form)
//       SquareRootKF -- stores L, P = L L^T (Cholesky / square-root form)
//       (UDKF later) -- stores U, D, P = U D U^T
//
//   Monte Carlo Kalman methods, different update algebras:
//       EnKF, EnSRF, EAKF, ETKF, LETKF, EnKS (future)
//
// The same four verbs and two knobs on every backend:
//
//     f.initialize(x, P);              // "assume this prior"
//     f.measurement_update(y, H, R);   // data, model, noise
//     f.time_update(F, Q);             // model, noise
//     f.time_update(F, Q, u);          // x <- F x + u
//     f.set_taper(C);                  // localization   -> LKF / LETKF
//     f.set_inflation(lambda);         // multiplicative inflation
//
// Two conventions worth stating once, because they are the theory:
//
//  * The taper is a covariance-space operation. The gain is formed from the
//    tapered covariance C o P, while the covariance update uses the UNTAPERED
//    P in Joseph form (Butala et al., IEEE TIP 2009, eq. (4.26)). With a
//    non-trivial taper the filter is the tapered / localized Kalman filter
//    (LKF), which is the L -> infinity limit of the localized EnKF. Hence
//    ||EnKF - KF|| <= ||EnKF - LKF|| + ||LKF - KF|| is expressible as "the
//    same class with set_taper on and off". Square-root backends materialize
//    the covariance in the gain when a taper is set.
//
//  * Multiplicative inflation scales the PRIOR covariance at the start of
//    every measurement_update (P <- lambda P), matching standard practice.
//    If several independent observation vectors are assimilated at one epoch,
//    set_inflation(1) between them to avoid inflating more than once.
// ---------------------------------------------------------------------------

#ifdef PYTHON_MODULE
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#endif

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>


namespace estimation {

    // Contract violation (bad dimensions, inconsistent sequence lengths).
    //
    // Thrown rather than asserted, on purpose:
    //   * assert() is compiled out under NDEBUG, so a release build / wheel
    //     would otherwise have no dimension checks at all and mismatched
    //     arguments would go straight into Eigen as undefined behaviour.
    //   * nanobind maps std::invalid_argument to Python ValueError, so a bad
    //     call from Python raises instead of aborting the interpreter.
    inline void
    require(const bool ok, const char *what)
    {
        if (!ok) throw std::invalid_argument(what);
    }

    // Shared type vocabulary. Const*Ref is a read-only view: the const is on
    // the data, not on the view. That is the idiomatic Eigen spelling for
    // input parameters, and it accepts const objects and temporaries.
    template <typename T>
    struct Types {
        using Scalar = T;
        using Vector = Eigen::Vector<T, Eigen::Dynamic>;
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        // Input views accept ANY strided layout -- row-major (numpy's
        // default), column-major, or a slice -- as a zero-copy view.
        //
        // A plain Ref<const Matrix> only matches column-major outer-stride-1,
        // so a row-major numpy array would have to be canonicalized with a
        // full copy on every call. That copy is invisible from Python and
        // dominates the cost of a small update, so the dynamic-stride Ref is
        // what makes the zero-copy contract uniform rather than layout-
        // dependent. See tests/test_bindings.py::test_zero_copy_is_real.
        using ConstVectorRef =
            Eigen::Ref<const Vector, 0, Eigen::InnerStride<Eigen::Dynamic>>;
        using ConstMatrixRef =
            Eigen::Ref<const Matrix, 0,
                       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

        // Mutable views (unused so far; kept for symmetry and for future
        // zero-copy output policies).
        using VectorRef = Eigen::Ref<Vector>;
        using MatrixRef = Eigen::Ref<Matrix>;
    };


    // Which sequences batch() records. Covariances are O(N^2) per step, so
    // recording is opt-in per kind rather than all-or-nothing.
    struct Record {
        static constexpr unsigned None        = 0;
        static constexpr unsigned XPrior      = 1u << 0;
        static constexpr unsigned XPosterior  = 1u << 1;
        static constexpr unsigned PPrior      = 1u << 2;
        static constexpr unsigned PPosterior  = 1u << 3;
        static constexpr unsigned Means       = XPrior | XPosterior;
        static constexpr unsigned Covariances = PPrior | PPosterior;
        static constexpr unsigned Everything  = Means | Covariances;
    };


    // The forward record. Index i is time i: *_prior[i] is the state /
    // covariance immediately before measurement_update at step i, and
    // *_posterior[i] immediately after. This is exactly the (prior, posterior)
    // sequence an RTS smoother needs.
    template <typename T>
    struct BatchOutput {
        using Vector = typename Types<T>::Vector;
        using Matrix = typename Types<T>::Matrix;

        std::vector<Vector> x_prior;
        std::vector<Vector> x_posterior;
        std::vector<Matrix> P_prior;
        std::vector<Matrix> P_posterior;
    };


    template <typename T>
    class Filter {
    public:
        using Scalar         = typename Types<T>::Scalar;
        using Vector         = typename Types<T>::Vector;
        using Matrix         = typename Types<T>::Matrix;
        using ConstVectorRef = typename Types<T>::ConstVectorRef;
        using ConstMatrixRef = typename Types<T>::ConstMatrixRef;
        using VectorRef      = typename Types<T>::VectorRef;
        using MatrixRef      = typename Types<T>::MatrixRef;

        explicit Filter(std::size_t N)
            : N_(N), taper_(), inflation_(Scalar(1)) {}

        virtual ~Filter() = default;

        Filter(const Filter &) = delete;
        Filter &operator=(const Filter &) = delete;

        std::size_t dimension() const { return N_; }

        // ---- state and uncertainty --------------------------------------------
        virtual const Vector &state() const = 0;

        // Materializes P. For sample-based backends (EnKF, ETKF, ...) this is
        // the sample covariance and costs O(N^2 L); it is not a cheap getter.
        virtual Matrix covariance() const = 0;

        // ---- the two updates ---------------------------------------------------
        virtual void initialize(ConstVectorRef x, ConstMatrixRef P) = 0;

        virtual void measurement_update(ConstVectorRef y,
                                        ConstMatrixRef H,
                                        ConstMatrixRef R) = 0;

        virtual void time_update(ConstMatrixRef F, ConstMatrixRef Q) = 0;

        // x <- F x + u. Pass u = B u for the general form x <- F x + B u.
        virtual void time_update(ConstMatrixRef F, ConstMatrixRef Q,
                                 ConstVectorRef u) = 0;

        // ---- knobs (defaults reproduce the plain Kalman filter) ----------------
        void set_taper(ConstMatrixRef C)     { taper_ = C; }
        void clear_taper()                   { taper_.resize(0, 0); }
        bool has_taper() const               { return taper_.size() > 0; }

        void set_inflation(Scalar lambda)    { inflation_ = lambda; }
        void clear_inflation()               { inflation_ = Scalar(1); }
        Scalar inflation() const             { return inflation_; }

        // ---- forward pass -------------------------------------------------------
        // Runs measurement_update / time_update over a sequence and records the
        // requested quantities. Call initialize() first -- batch() starts from
        // wherever the filter currently is, which makes it composable across
        // chunks (and leaves the final forecast in the filter).
        BatchOutput<T>
        batch(const std::vector<ConstVectorRef> &y,
              const std::vector<ConstMatrixRef> &H,
              const std::vector<ConstMatrixRef> &R,
              const std::vector<ConstMatrixRef> &F,
              const std::vector<ConstMatrixRef> &Q,
              unsigned record = Record::Everything)
        {
            return batch(y, H, R, F, Q, std::vector<ConstVectorRef>(), record);
        }

        BatchOutput<T>
        batch(const std::vector<ConstVectorRef> &y,
              const std::vector<ConstMatrixRef> &H,
              const std::vector<ConstMatrixRef> &R,
              const std::vector<ConstMatrixRef> &F,
              const std::vector<ConstMatrixRef> &Q,
              const std::vector<ConstVectorRef> &u,
              unsigned record = Record::Everything)
        {
            require(y.size() == H.size() && H.size() == R.size() &&
                        R.size() == F.size() && F.size() == Q.size(),
                    "batch: y, H, R, F, Q must all have the same length");
            require(u.empty() || u.size() == y.size(),
                    "batch: u must be empty or have the same length as y");

            const std::size_t I = y.size();
            const bool want_x_prior     = record & Record::XPrior;
            const bool want_x_posterior = record & Record::XPosterior;
            const bool want_P_prior     = record & Record::PPrior;
            const bool want_P_posterior = record & Record::PPosterior;

            BatchOutput<T> out;
            out.x_prior.reserve(want_x_prior ? I : 0);
            out.x_posterior.reserve(want_x_posterior ? I : 0);
            out.P_prior.reserve(want_P_prior ? I : 0);
            out.P_posterior.reserve(want_P_posterior ? I : 0);

            for (std::size_t i = 0; i < I; ++i) {
                if (want_x_prior) out.x_prior.push_back(state());
                if (want_P_prior) out.P_prior.push_back(covariance());

                measurement_update(y[i], H[i], R[i]);

                if (want_x_posterior) out.x_posterior.push_back(state());
                if (want_P_posterior) out.P_posterior.push_back(covariance());

                if (!u.empty()) time_update(F[i], Q[i], u[i]);
                else            time_update(F[i], Q[i]);
            }
            return out;
        }

    protected:
        std::size_t N_;
        Matrix      taper_;       // empty means identity
        Scalar      inflation_;   // 1 means off

        // P <- (P + P^T)/2 in place, no temporary. Every covariance this
        // library produces is exactly symmetric -- that is what lets LLT read a
        // single triangle and trust it.
        static void symmetrize_in_place(Matrix &P)
        {
            const Eigen::Index n = P.rows();
            for (Eigen::Index j = 0; j < n; ++j) {
                for (Eigen::Index i = 0; i <= j; ++i) {
                    const Scalar v = Scalar(0.5) * (P(i, j) + P(j, i));
                    P(i, j) = v;
                    P(j, i) = v;
                }
            }
        }

        // The covariance the gain is formed from: C o P with a taper, else P.
        // A taper is a symmetric correlation-type matrix (Schur product
        // theorem then keeps C o P positive semidefinite); the product is
        // symmetrized so that a mildly asymmetric input C cannot make the gain
        // covariance asymmetric.
        Matrix gain_covariance(const Matrix &P) const
        {
            if (!has_taper()) return P;
            Matrix Pg = taper_.cwiseProduct(P);
            symmetrize_in_place(Pg);
            return Pg;
        }

        // P <- lambda P applied to a covariance matrix.
        void apply_inflation(Matrix &P) const
        {
            if (inflation_ != Scalar(1)) P *= inflation_;
        }

        // L <- sqrt(lambda) L applied to a square-root factor (P = L L^T).
        void apply_inflation_sqrt(Matrix &L) const
        {
            if (inflation_ != Scalar(1)) L *= std::sqrt(inflation_);
        }
    };

} // namespace estimation
