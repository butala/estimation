#pragma once

// ---------------------------------------------------------------------------
// estimation::UDKF -- the Kalman filter in Bierman's factored (LD^T) form.
//
// Stores P = L D L^T with L unit lower triangular and D positive diagonal
// (Bierman's "UD" with U = L^T). Symmetry and positive definiteness are
// structural invariants -- D > 0 is maintained entrywise -- so unlike the
// covariance form there is nothing to symmetrize and no way for round-off to
// make P indefinite.
//
// Measurement updates use Bierman's sequential scalar downdate, O(N^2) per
// scalar row after whitening R to I -- the reason the factored form exists.
// The downdate factorizes the rank-1 modification D - g g^T / alpha in O(N^2)
// via the bordered (Schur) recursion
//
//     alpha_{j+1} = alpha_j - g_j^2 / d_j
//     d'_j        = d_j - g_j^2 / alpha_j
//     Delta[i,j]  = -g_j g_i / (d_j alpha_{j+1})        i > j
//
// giving D - g g^T/alpha = Delta D' Delta^T with Delta unit lower, hence
// L^+ = L Delta (still unit lower) and D^+ = D'.  Verified against a dense
// LDL^T of the same rank-1 modification in the unit tests.
//
// With a taper the gain is no longer Bierman's "natural" gain and the update
// falls back to the Joseph form of the LKF (thesis (4.26)) with refactorization
// -- which is what keeps UDKF == KF == SquareRootKF for every taper setting.
//
// The time update currently forms F P F^T + Q and refactorizes it; Thornton's
// O(N^2) UD time update for Q = G D_Q G^T is the natural next optimization.
// ---------------------------------------------------------------------------

#include "filter.hpp"


namespace estimation {

    template <typename T = double>
    class UDKF : public Filter<T> {
    public:
        using Scalar         = typename Filter<T>::Scalar;
        using Vector         = typename Filter<T>::Vector;
        using Matrix         = typename Filter<T>::Matrix;
        using ConstVectorRef = typename Filter<T>::ConstVectorRef;
        using ConstMatrixRef = typename Filter<T>::ConstMatrixRef;

        explicit UDKF(std::size_t N);

        const Vector &state() const override { return x_; }
        Matrix covariance() const override;

        // The factorization itself: P = L_ diag(D_) L_^T.
        const Matrix &unit_lower() const { return L_; }
        const Vector &diagonal() const  { return D_; }

        void initialize(ConstVectorRef x, ConstMatrixRef P) override;

        void measurement_update(ConstVectorRef y, ConstMatrixRef H,
                                ConstMatrixRef R) override;

        void time_update(ConstMatrixRef F, ConstMatrixRef Q) override;
        void time_update(ConstMatrixRef F, ConstMatrixRef Q,
                         ConstVectorRef u) override;

    private:
        Vector x_;
        Matrix L_;   // unit lower triangular
        Vector D_;   // positive diagonal

        void refactor(ConstMatrixRef A);          // SPD A -> (L_, D_) via LDL^T
        void bierman_row(const Vector &h, const Scalar z_minus_hx, const Scalar r);
    };

} // namespace estimation
