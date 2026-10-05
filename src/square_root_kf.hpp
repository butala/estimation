#pragma once

// ---------------------------------------------------------------------------
// estimation::SquareRootKF -- the Kalman filter in Cholesky / square-root form.
//
// State is stored as (x, L) with P = L L^T and L lower triangular. Every step
// updates the factor directly and re-triangularizes with QR, so P is never
// formed and is never allowed to lose symmetry or positive definiteness. This
// is the numerically stable cousin of KF, and the deterministic relative of
// the ensemble square-root filters (ETKF/EnSRF): where the ETKF transforms an
// anomaly matrix X^a = X^f T, this filter transforms a Cholesky factor.
//
// Same map as KF, including the tapered (LKF) variant: the gain comes from the
// tapered covariance C o P while the covariance update is the Joseph form on
// the UNTAPERED P (Butala et al., IEEE TIP 2009, eq. (4.26)), realized here as
// a QR of the stacked Joseph factors -- never as a subtraction of covariances.
// The cross-check in test.cpp verifies KF and SquareRootKF agree to roundoff
// with and without a taper.
//
// Cost is O(N^3) per update (the QR re-triangularization). For large M with
// diagonal R the sequential scalar form is O(N^2) per scalar measurement --
// that is the UDKF (Bierman/Thornton) backend, not this one.
// ---------------------------------------------------------------------------

#include "filter.hpp"


namespace estimation {

    template <typename T = double>
    class SquareRootKF : public Filter<T> {
    public:
        using Scalar         = typename Filter<T>::Scalar;
        using Vector         = typename Filter<T>::Vector;
        using Matrix         = typename Filter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;

        explicit SquareRootKF(std::size_t N);

        const Vector &state() const override { return x_; }
        Matrix covariance() const override { return L_ * L_.transpose(); }

        // The Cholesky factor itself (P = L L^T). Cheap; no materialization.
        const Matrix &factor() const { return L_; }

        void initialize(ConstVectorRef x, ConstMatrixRef P) override;

        void measurement_update(ConstVectorRef y,
                                ConstMatrixRef H,
                                ConstMatrixRef R) override;

        void time_update(ConstMatrixRef F, ConstMatrixRef Q) override;
        void time_update(ConstMatrixRef F, ConstMatrixRef Q,
                         ConstVectorRef u) override;

    private:
        Vector x_;
        Matrix L_;   // lower triangular, P = L_ L_^T

        // Lower triangular L with L L^T = M M^T, positive diagonal.
        static Matrix retriangulate(const Matrix &M);

        // Any matrix G with G G^T = A for symmetric positive semidefinite A.
        // Robust to (numerically) singular A, which LLT is not.
        static Matrix psd_sqrt(const Matrix &A);
    };

} // namespace estimation
