#pragma once

// ---------------------------------------------------------------------------
// estimation::KF -- the Kalman filter in covariance form.
//
// Stores (x, P) directly. Same map as SquareRootKF (which stores P = L L^T),
// including the tapered (LKF) variant -- see filter.hpp for the conventions.
// The covariance update is the Joseph form on the untapered P, which keeps P
// symmetric positive definite and is what makes the two backends agree when a
// taper is set. For P- = F P F^T + Q the product is evaluated before being
// assigned to P, and the result is re-symmetrized: F P F^T is symmetric only
// in exact arithmetic.
//
// O(N^3) per update, dominated by the Cholesky of the m x m innovation
// covariance and the two O(N^2 m) products. Fine for moderate N; this is the
// reference ("oracle") implementation for the Monte Carlo methods.
// ---------------------------------------------------------------------------

#include "filter.hpp"


namespace estimation {

    template <typename T = double>
    class KF : public Filter<T> {
    public:
        using Scalar         = typename Filter<T>::Scalar;
        using Vector         = typename Filter<T>::Vector;
        using Matrix         = typename Filter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;

        explicit KF(std::size_t N);

        const Vector &state() const override { return x_; }
        Matrix covariance() const override { return P_; }

        void initialize(ConstVectorRef x, ConstMatrixRef P) override;

        void measurement_update(ConstVectorRef y,
                                ConstMatrixRef H,
                                ConstMatrixRef R) override;

        void time_update(ConstMatrixRef F, ConstMatrixRef Q) override;
        void time_update(ConstMatrixRef F, ConstMatrixRef Q,
                         ConstVectorRef u) override;

    private:
        Vector x_;
        Matrix P_;
    };

} // namespace estimation
