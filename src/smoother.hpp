#pragma once

// ---------------------------------------------------------------------------
// Smoothers for the fixed-interval problem.
//
//   rts_smooth(record, F, Q)   exact Rauch-Tung-Striebel, operating on a
//                              forward Filter::BatchOutput -- the reference
//                              that EnKS is validated against
//   EnKS<T>                    Butala, Fathpour & Bhatt's localized ensemble
//                              Kalman smoother -- a Monte Carlo approximation
//                              to the BRYSON-FRAZIER smoother (IEEE, 2012).
//                              An EnKF forward pass, then a backward recursion
//                              on an ensemble adjoint variable, then the
//                              smoothed estimates.  It inverts only the
//                              INNOVATION covariance R_e,i (M x M, and >= R > 0),
//                              never the forecast covariance: that is the whole
//                              point of using the BF rather than the RTS form,
//                              whose P_{i|i-1}^{-1} is both undesirable and
//                              singular for L < N.
//   LEKS<T>                    = EnKS + a REQUIRED taper (localized EnKS)
//
// Same rule as the filters: localization is a knob, not a separate algorithm.
// The taper enters the EnKS exactly where it enters the EnKF -- in the gain and
// in the innovation covariance -- and additionally in the third stage, where
// C'_i and C''_i of the paper are both taken to be the filter's taper. With no
// taper LEKS is EnKS; with no taper rts_smooth is the exact Kalman smoother.
//
// rts_smooth is included as the exact reference: the two agree on the smoothed
// mean in the linear-Gaussian case, which is how the tests check the BF form
// against something independent of it.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <random>
#include <vector>

#include "filter.hpp"
#include "ensemble_filter.hpp"


namespace estimation {

    template <typename T>
    struct SmoothOutput {
        using Vector = typename Types<T>::Vector;
        using Matrix = typename Types<T>::Matrix;

        std::vector<Vector> x_smoothed;   // index i is time i
        std::vector<Matrix> P_smoothed;
    };


    // Exact RTS on the record produced by Filter::batch. Requires the record
    // to contain means and covariances (Record::Means | Record::Covariances).
    template <typename T = double>
    SmoothOutput<T>
    rts_smooth(const BatchOutput<T> &record,
               const std::vector<typename Types<T>::ConstMatrixRef> &F,
               const std::vector<typename Types<T>::ConstMatrixRef> &Q,
               const std::vector<typename Types<T>::ConstVectorRef> &u =
                   std::vector<typename Types<T>::ConstVectorRef>());


    template <typename T = double>
    class EnKS {
    public:
        using Scalar         = typename Types<T>::Scalar;
        using Vector         = typename Types<T>::Vector;
        using Matrix         = typename Types<T>::Matrix;
        using ConstVectorRef = typename Types<T>::ConstVectorRef;
        using ConstMatrixRef = typename Types<T>::ConstMatrixRef;

        EnKS(std::size_t N, std::size_t L, std::uint64_t seed = 0);

        void set_taper(ConstMatrixRef C)  { fwd_.set_taper(C); }
        void clear_taper()                { fwd_.clear_taper(); }
        bool has_taper() const            { return fwd_.has_taper(); }
        void set_inflation(Scalar lambda) { fwd_.set_inflation(lambda); }
        void clear_inflation()            { fwd_.clear_inflation(); }
        Scalar inflation() const          { return fwd_.inflation(); }
        std::size_t ensemble_size() const { return fwd_.ensemble_size(); }

        // Set the initial prior (forwarded to the forward EnKF).
        void initialize(ConstVectorRef x, ConstMatrixRef P) { fwd_.initialize(x, P); }

        // Fixed-interval smoothing over the sequence, starting from the
        // current prior. Leaves the filter at its final forecast and returns
        // the smoothed trajectories.
        SmoothOutput<T>
        smooth(const std::vector<ConstVectorRef> &y,
               const std::vector<ConstMatrixRef> &H,
               const std::vector<ConstMatrixRef> &R,
               const std::vector<ConstMatrixRef> &F,
               const std::vector<ConstMatrixRef> &Q,
               const std::vector<ConstVectorRef> &u =
                   std::vector<ConstVectorRef>());

    protected:
        EnKF<T> fwd_;        // the forward pass (the EnKF of stage 1)
        std::mt19937_64 rng_;   // for the Z_i of the adjoint ensemble, eq. (13)
    };


    // Localized ensemble Kalman smoother = EnKS + a mandatory taper.
    template <typename T = double>
    class LEKS : public EnKS<T> {
    public:
        using ConstMatrixRef = typename Types<T>::ConstMatrixRef;
        LEKS(std::size_t N, std::size_t L, ConstMatrixRef C,
             std::uint64_t seed = 0)
            : EnKS<T>(N, L, seed)
        {
            this->set_taper(C);
        }
    };

} // namespace estimation
