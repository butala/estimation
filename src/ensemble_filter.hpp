#pragma once

// ---------------------------------------------------------------------------
// Ensemble Kalman methods, all built on estimation::Filter<T>.
//
//     EnsembleFilter<T>   shared ensemble state, RNG, and the LKF mean update
//       |- EnKF<T>        perturbed observations (recentered)      [Evensen 1994,
//       |                                                        Burgers et al. 1998]
//       |- EnSRF<T>       Whitaker & Hamill (2002) sequential alpha
//       |- EAKF<T>        Anderson (2001) sequential regression adjustment
//       |- ETKF<T>        Bishop, Etherton & Hodyss (2001) block transform
//       `- LETKF<T>       = ETKF + a REQUIRED taper                [Hunt et al.]
//
// Design rule that makes the whole family comparable: every method uses the
// SAME analysis mean -- the LKF mean, i.e. x <- x + K (y - H x) with K built
// from the gain covariance (C o P with a taper, else P) -- and the SAME
// covariance target -- the Joseph form on the untapered P (Butala et al.,
// IEEE TIP 2009, eq. (4.26)). They differ only in how the analysis ANOMALIES
// are realized:
//
//   EnKF   random:      X^a = (I - K H) X + K E,     E = recentered N(0, R) draws
//   EnSRF  sequential:  X^a = (I - a k h) X          (Whitaker & Hamill alpha)
//   EAKF   sequential:  X^a = X + K_x (Y^a - Y)      (Anderson regression)
//   ETKF   block:       X^a = X C^{-1/2}             (Bishop transform)
//
// In the no-taper case all four realize exactly the same analysis covariance
// (proved in the Whitaker-Hamill / Anderson / Bishop papers and re-checked by
// the unit tests); they differ only by a rotation of the ensemble. With a
// taper the gain leaves the ensemble span, so the Joseph target has rank up to
// L-1+M and the square-root methods fall back to a common Joseph factor
// (joseph_anomalies) which is exact whenever L >= N+1.
//
// Perturbations are RECENTERED, so the ensemble mean is exactly the LKF mean
// at every step for every method. The randomness lives entirely in the
// anomalies -- which is what makes "all methods agree on state()" an exact,
// testable statement rather than a statistical one.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <random>

#include "filter.hpp"


namespace estimation {

    template <typename T = double>
    class EnsembleFilter : public Filter<T> {
    public:
        using Scalar         = typename Filter<T>::Scalar;
        using Vector         = typename Filter<T>::Vector;
        using Matrix         = typename Filter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;

        EnsembleFilter(std::size_t N, std::size_t L, std::uint64_t seed = 0);

        std::size_t ensemble_size() const { return L_; }

        const Vector &state() const override { return x_; }
        Matrix covariance() const override;            // 1/(L-1) X~ X~^T

        Matrix members() const;                     // N x L, columns = members
        Matrix anomalies() const;                      // N x L, centered, sum 0
        void set_members(ConstMatrixRef X);

        void initialize(ConstVectorRef x, ConstMatrixRef P) override;
        void time_update(ConstMatrixRef F, ConstMatrixRef Q) override;
        void time_update(ConstMatrixRef F, ConstMatrixRef Q,
                         ConstVectorRef u) override;

        // Each algebra implements the whole measurement update; the base
        // supplies the shared linear algebra below. EnKF is block/stochastic,
        // EnSRF and EAKF are inherently sequential (one scalar row at a time,
        // after whitening R to I), ETKF is a block transform.
        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override = 0;

    protected:
        std::size_t   L_;
        Vector        x_;
        Matrix        X_;                              // N x L, columns sum to 0
        std::mt19937_64 rng_;

        // ---- shared linear algebra ------------------------------------------
        Matrix sample_covariance() const;              // symmetrized 1/(L-1) X~ X~^T
        void   apply_inflation_to_anomalies();         // X~ <- sqrt(lambda) X~

        // Whiten the observation: y' = L_R^{-1} y, H' = L_R^{-1} H, R' = I.
        // The KF with (y', H', I) equals the KF with (y, H, R), and a whitened
        // R is what the sequential updates (EnSRF / EAKF / UDKF) need.
        static Matrix lower_factor(ConstMatrixRef A);
        void whiten(ConstVectorRef y, ConstMatrixRef H, ConstMatrixRef R,
                    Vector &y_w, Matrix &H_w) const;

        // K = P_g H^T (H P_g H^T + R)^{-1} with P_g = gain_covariance(P).
        Matrix kalman_gain(const Matrix &P, ConstMatrixRef H,
                           ConstMatrixRef R) const;

        // Analysis anomalies realizing the Joseph target
        //     P^a = (I - K H) P (I - K H)^T + K R K^T
        // as 1/(L-1) X~^a X~^a^T.  Used by EnSRF / EAKF / ETKF whenever a taper
        // is set (the tapered gain leaves the ensemble span); exact whenever
        // the target rank is <= L-1.
        Matrix joseph_anomalies(const Matrix &Xs, const Matrix &K,
                                ConstMatrixRef H, ConstMatrixRef L_R) const;

        // Draw and recenter an N x L matrix of iid N(0, Sigma) columns.
        Matrix draw_recentered(const Matrix &Sigma_chol);

        // Fallback for the tapered path of the sequential methods: replace the
        // anomalies wholesale by the Joseph factor.
        void update_anomalies_joseph(const Matrix &K, ConstMatrixRef H,
                                     ConstMatrixRef R);
    };


    // ---- the four update algebras ------------------------------------------

    // Perturbed observations. Each member sees y + e^l; the perturbations are
    // recentered so the mean is exactly the LKF mean and the randomness lives
    // in the anomalies. Faithful to Butala et al. (4.21) up to that recentering.
    template <typename T = double>
    class EnKF : public EnsembleFilter<T> {
    public:
        using Scalar         = typename EnsembleFilter<T>::Scalar;
        using Vector         = typename EnsembleFilter<T>::Vector;
        using Matrix         = typename EnsembleFilter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;
        using EnsembleFilter<T>::EnsembleFilter;
        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override;
    };


    // Sequential scalar updates with the Whitaker & Hamill alpha factor,
    // which makes (I - a k h) P (I - a k h)^T = (I - k h) P exactly.
    template <typename T = double>
    class EnSRF : public EnsembleFilter<T> {
    public:
        using Scalar         = typename EnsembleFilter<T>::Scalar;
        using Vector         = typename EnsembleFilter<T>::Vector;
        using Matrix         = typename EnsembleFilter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;
        using EnsembleFilter<T>::EnsembleFilter;
        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override;
    };


    // Anderson's sequential regression ("adjustment") in observation space.
    // Covariantly identical to EnSRF; the ensemble rotation differs.
    template <typename T = double>
    class EAKF : public EnsembleFilter<T> {
    public:
        using Scalar         = typename EnsembleFilter<T>::Scalar;
        using Vector         = typename EnsembleFilter<T>::Vector;
        using Matrix         = typename EnsembleFilter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;
        using EnsembleFilter<T>::EnsembleFilter;
        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override;
    };


    // Bishop's ensemble transform: X^a = X C^{-1/2}, C = I + Y^T R^{-1} Y/(L-1).
    // Single block update over the whole measurement vector.
    template <typename T = double>
    class ETKF : public EnsembleFilter<T> {
    public:
        using Scalar         = typename EnsembleFilter<T>::Scalar;
        using Vector         = typename EnsembleFilter<T>::Vector;
        using Matrix         = typename EnsembleFilter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;
        using EnsembleFilter<T>::EnsembleFilter;
        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override;
    };


    // Local ETKF. NOT a separate algorithm: it is ETKF with a localization
    // taper, which is why the taper is mandatory at construction. That keeps
    // "localization is a knob" true across the library -- LKF = KF + taper and
    // LETKF = ETKF + taper are then literally the same statement.
    template <typename T = double>
    class LETKF : public ETKF<T> {
    public:
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;
        LETKF(std::size_t N, std::size_t L, ConstMatrixRef C,
              std::uint64_t seed = 0)
            : ETKF<T>(N, L, seed)
        {
            this->set_taper(C);
        }
    };

} // namespace estimation
