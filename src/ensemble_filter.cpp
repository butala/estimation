#include <cassert>
#include <limits>

#include "ensemble_filter.hpp"


namespace estimation {

    // =========================================================================
    // EnsembleFilter: shared ensemble state and linear algebra
    // =========================================================================

    template <typename T>
    EnsembleFilter<T>::EnsembleFilter(const std::size_t N, const std::size_t L,
                                      const std::uint64_t seed)
        : Filter<T>(N), L_(L), x_(Vector::Zero(N)),
          X_(Matrix::Zero(N, static_cast<Eigen::Index>(L))),
          rng_(seed ? seed : 0x9e3779b97f4a7c15ULL)
    {
        require(L >= 2, "EnsembleFilter: need at least 2 members");
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::covariance() const
    {
        Matrix P = (X_ * X_.transpose()) / Scalar(L_ - 1);
        this->symmetrize_in_place(P);
        return P;
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::anomalies() const
    {
        return X_;
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::members() const
    {
        return (X_.colwise() + x_);
    }


    template <typename T>
    void
    EnsembleFilter<T>::set_members(const ConstMatrixRef X)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Eigen::Index L = static_cast<Eigen::Index>(L_);
        require(X.rows() == N && X.cols() == L,
                "set_members: X must be N x L");
        x_ = X.rowwise().mean();
        X_ = (X.colwise() - x_);
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::sample_covariance() const
    {
        Matrix P = (X_ * X_.transpose()) / Scalar(L_ - 1);
        this->symmetrize_in_place(P);
        return P;
    }


    template <typename T>
    void
    EnsembleFilter<T>::apply_inflation_to_anomalies()
    {
        if (this->inflation_ != Scalar(1))
            X_ *= std::sqrt(this->inflation_);
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::lower_factor(const ConstMatrixRef A)
    {
        Matrix S = A;
        // LLT reads a single triangle; make that the truth.
        S = ((S + S.transpose()).eval() * Scalar(0.5));
        Eigen::LLT<Matrix> llt(S);
        if (llt.info() == Eigen::Success) return Matrix(llt.matrixL());
        // Positive semidefinite (or round-off): fall back to an eigen factor.
        Eigen::SelfAdjointEigenSolver<Matrix> es(S);
        require(es.info() == Eigen::Success, "lower_factor: eigen decomposition failed");
        const Vector s = es.eigenvalues().cwiseMax(Scalar(0)).cwiseSqrt();
        return es.eigenvectors() * s.asDiagonal();
    }


    template <typename T>
    void
    EnsembleFilter<T>::whiten(const ConstVectorRef y, const ConstMatrixRef H,
                              const ConstMatrixRef R,
                              Vector &y_w, Matrix &H_w) const
    {
        // Whiten by the (invertible) factor: L_R y' = y, L_R H' = H.
        const Matrix L_R = lower_factor(R);
        const Eigen::FullPivLU<Matrix> lu(L_R);
        require(lu.isInvertible(), "whiten: R factor is singular");
        y_w = lu.solve(y);
        H_w = lu.solve(H);
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::kalman_gain(const Matrix &P, const ConstMatrixRef H,
                                   const ConstMatrixRef R) const
    {
        const Matrix Pg   = this->gain_covariance(P);
        const Matrix P_HT = Pg * H.transpose();
        const Matrix S    = H * P_HT + R;
        Eigen::LLT<Matrix> lltS((S + S.transpose()).eval() * Scalar(0.5));
        require(lltS.info() == Eigen::Success,
                "kalman_gain: innovation covariance is not positive definite");
        return lltS.solve(P_HT.transpose()).transpose();
    }


    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::draw_recentered(const Matrix &Sigma_chol)
    {
        const Eigen::Index k = Sigma_chol.rows();
        const Eigen::Index L = static_cast<Eigen::Index>(L_);
        std::normal_distribution<double> gauss(0.0, 1.0);
        Matrix G(k, L);
        for (Eigen::Index j = 0; j < L; ++j)
            for (Eigen::Index i = 0; i < k; ++i)
                G(i, j) = static_cast<Scalar>(gauss(rng_));
        // Recenter so the perturbations carry no mean offset.
        G.colwise() -= G.rowwise().mean();
        return Sigma_chol * G;
    }


    // Analysis anomalies realizing
    //     P^a = (I - K H) P (I - K H)^T + K R K^T  (= 1/(L-1) W W^T)
    // with W = [ (I - K H) X~ , sqrt(L-1) K L_R ].  We return an N x L factor
    // whose columns sum to zero:
    //     X~^a = U_r sqrt(Lambda_r) Q,   Q = (first r columns of 1^perp)^T
    // which is exact whenever rank(W) <= L-1 (in particular whenever L >= N+1)
    // and otherwise the best rank-(L-1) approximation in Frobenius norm.
    template <typename T>
    typename EnsembleFilter<T>::Matrix
    EnsembleFilter<T>::joseph_anomalies(const Matrix &Xs, const Matrix &K,
                                        const ConstMatrixRef H,
                                        const ConstMatrixRef L_R) const
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Eigen::Index L = static_cast<Eigen::Index>(L_);
        const Eigen::Index m = H.rows();

        Matrix W(N, L + m);
        W.leftCols(L)  = Xs - K * (H * Xs);                    // (I - K H) X~
        W.rightCols(m) = K * L_R * std::sqrt(Scalar(L - 1));   // sqrt(L-1) K L_R

        Matrix A = (W * W.transpose()).eval();
        this->symmetrize_in_place(A);
        Eigen::SelfAdjointEigenSolver<Matrix> es(A);
        require(es.info() == Eigen::Success, "joseph_anomalies: eigen failed");
        // Eigenvalues ASCEND, so the dominant subspace is at the END: take the
        // r largest, not the first r.  Counting from the front makes r collapse
        // to 0 the moment the smallest eigenvalue is a non-positive round-off,
        // which silently zeroes the analysis covariance.
        const Eigen::Index n = A.rows();
        const Vector lam = es.eigenvalues().cwiseMax(Scalar(0));
        // Rank by a RELATIVE tolerance: an exact "> 0" test lets round-off in
        // the smallest eigenvalue flip the truncation between runs, which makes
        // the analysis nondeterministic.
        const Scalar tol =
            lam(n - 1) * std::numeric_limits<Scalar>::epsilon() * Scalar(n);
        Eigen::Index r = 0;
        while (r < n && lam(n - 1 - r) > tol) ++r;
        if (r > L - 1) r = L - 1;   // keep the columns summing to zero possible

        // Orthonormal basis of 1^perp in R^L, as the trailing columns of a
        // Householder reflector built from the all-ones vector (whose leading
        // column is therefore parallel to 1).
        Eigen::HouseholderQR<Matrix> qr(Vector::Ones(L));
        const Matrix Qm = qr.householderQ() * Matrix::Identity(L, L);
        const Matrix B  = Qm.rightCols(L - 1);            // L x (L-1)
        const Matrix Qr = B.leftCols(r).transpose();      // r x L, Qr 1 = 0

        const Matrix Ur = es.eigenvectors().rightCols(r);  // the r largest
        // Zero-initialized: Eigen leaves Matrix(n, n) uninitialized, so setting
        // only the diagonal leaves garbage off-diagonal entries -- and U Sr Qr
        // then carries that garbage into the anomalies (a scale error and a
        // source of run-to-run nondeterminism).
        Matrix Sr = Matrix::Zero(r, r);
        for (Eigen::Index i = 0; i < r; ++i) Sr(i, i) = std::sqrt(lam(n - r + i));
        return (Ur * Sr * Qr).eval();
    }


    template <typename T>
    void
    EnsembleFilter<T>::update_anomalies_joseph(const Matrix &K,
                                               const ConstMatrixRef H,
                                               const ConstMatrixRef R)
    {
        const Matrix L_R = lower_factor(R);
        X_ = joseph_anomalies(X_, K, H, L_R);
    }


    template <typename T>
    void
    EnsembleFilter<T>::initialize(const ConstVectorRef x, const ConstMatrixRef P)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Eigen::Index L = static_cast<Eigen::Index>(L_);
        require(x.size() == N, "initialize: x must have length N");
        require(P.rows() == N && P.cols() == N, "initialize: P must be N x N");

        const Matrix L_P = lower_factor(P);
        // Draw N(0, I) columns and recenter: then E[X~ X~^T/(L-1)] = L_P L_P^T
        // exactly and the sample mean is exactly x.
        std::normal_distribution<double> gauss(0.0, 1.0);
        Matrix G(N, L);
        for (Eigen::Index j = 0; j < L; ++j)
            for (Eigen::Index i = 0; i < N; ++i)
                G(i, j) = static_cast<Scalar>(gauss(rng_));
        G.colwise() -= G.rowwise().mean();
        x_ = x;
        X_ = L_P * G;
    }


    template <typename T>
    void
    EnsembleFilter<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        require(F.rows() == N && F.cols() == N, "time_update: F must be N x N");
        require(Q.rows() == N && Q.cols() == N, "time_update: Q must be N x N");

        // x <- F x, X~ <- F X~ + U~ with U~ recentered iid N(0, Q). Recentering
        // keeps the sample mean exactly F x (matching KF/SquareRootKF) and
        // keeps E[U~ U~^T/(L-1)] = Q.
        X_ = (F * X_).eval() + draw_recentered(lower_factor(Q));
        x_ = (F * x_).eval();
    }


    template <typename T>
    void
    EnsembleFilter<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q,
                                   const ConstVectorRef u)
    {
        require(u.size() == static_cast<Eigen::Index>(this->N_),
                "time_update: u must have length N");
        time_update(F, Q);
        x_ += u;
    }


    // =========================================================================
    // EnKF: perturbed observations (recentered)
    // =========================================================================

    template <typename T>
    void
    EnKF<T>::measurement_update(const ConstVectorRef y, const ConstMatrixRef H,
                                const ConstMatrixRef R)
    {
        this->apply_inflation_to_anomalies();
        const auto P = this->sample_covariance();
        const auto K = this->kalman_gain(P, H, R);
        const auto e = (y - H * this->x_).eval();
        this->x_ += K * e;

        // X~^a = (I - K H) X~ + K E,  E = recentered N(0, R) draws (M x L).
        // Then 1/(L-1) X~^a X~^a^T -> Joseph(P, K) as L -> infinity, and the
        // mean is untouched because E's columns sum to zero.
        this->X_ = (this->X_ - K * (H * this->X_)).eval()
                 + K * this->draw_recentered(this->lower_factor(R));
    }


    // =========================================================================
    // EnSRF: Whitaker & Hamill (2002) sequential alpha
    // =========================================================================

    // For a scalar row (h, 1) and k = P h^T / (h P h^T + 1), the alpha factor
    //     a = 1 / (1 + sqrt(1 / (h P h^T + 1)))
    // makes (I - a k h) P (I - a k h)^T = (I - k h) P exactly -- i.e. one
    // sequential step realizes the scalar Joseph target with no extra noise.
    template <typename T>
    void
    EnSRF<T>::measurement_update(const ConstVectorRef y, const ConstMatrixRef H,
                                 const ConstMatrixRef R)
    {
        this->apply_inflation_to_anomalies();
        Vector y_w;
        Matrix H_w;
        this->whiten(y, H, R, y_w, H_w);
        const Eigen::Index m = H_w.rows();

        if (this->has_taper()) {
            // The tapered gain leaves the ensemble span: use the Joseph target
            // of the LKF in one shot (identical for EnSRF / EAKF / ETKF).
            const Matrix P = this->sample_covariance();
            const Matrix K = this->kalman_gain(P, H, R);
            const Vector e = (y - H * this->x_).eval();
            this->x_ += K * e;
            this->update_anomalies_joseph(K, H, R);
            return;
        }

        Matrix Xs = this->X_;
        Vector x  = this->x_;
        for (Eigen::Index j = 0; j < m; ++j) {
            const Matrix h = H_w.row(j);                   // 1 x N
            const Matrix Y = (h * Xs).eval();              // 1 x L
            const Scalar v = (Y.squaredNorm() / Scalar(this->L_ - 1));   // h P h^T
            const Scalar s = v + Scalar(1);                // whitened: +1
            const Matrix k = ((Xs * Y.transpose()) / (Scalar(this->L_ - 1) * s)).eval();
            const Scalar a = Scalar(1) / (Scalar(1) + std::sqrt(Scalar(1) / s));

            x  += k * (y_w(j) - (h * x)(0, 0));
            Xs -= ((a * k) * (h * Xs)).eval();
        }
        this->x_ = x;
        this->X_ = Xs;
    }


    // =========================================================================
    // EAKF: Anderson (2001) sequential regression adjustment
    // =========================================================================

    // For a scalar row, the observed-space spread is scaled by 1/sqrt(h P h^T+1)
    // and the state anomalies are dragged along by regression:
    //     X~^a = X~ + Cov(x, y) Var(y)^{-1} (Y^a - Y),   Y^a = Y / sqrt(s).
    // This realizes (I - k h) P exactly (see the derivation in the tests), and
    // is covariantly identical to EnSRF with a different rotation of members.
    template <typename T>
    void
    EAKF<T>::measurement_update(const ConstVectorRef y, const ConstMatrixRef H,
                                const ConstMatrixRef R)
    {
        this->apply_inflation_to_anomalies();
        Vector y_w;
        Matrix H_w;
        this->whiten(y, H, R, y_w, H_w);
        const Eigen::Index m = H_w.rows();

        if (this->has_taper()) {
            const Matrix P = this->sample_covariance();
            const Matrix K = this->kalman_gain(P, H, R);
            const Vector e = (y - H * this->x_).eval();
            this->x_ += K * e;
            this->update_anomalies_joseph(K, H, R);
            return;
        }

        Matrix Xs = this->X_;
        Vector x  = this->x_;
        for (Eigen::Index j = 0; j < m; ++j) {
            const Matrix h = H_w.row(j);                   // 1 x N
            const Matrix Y = (h * Xs).eval();              // 1 x L
            const Scalar v = Y.squaredNorm() / Scalar(this->L_ - 1);     // Var(y^b)
            const Scalar s = v + Scalar(1);
            const Scalar inv_sqrt_s = Scalar(1) / std::sqrt(s);

            // Mean: the usual Kalman update with the whitened scalar row.
            const Matrix k = ((Xs * Y.transpose()) / (Scalar(this->L_ - 1) * s)).eval();
            x += k * (y_w(j) - (h * x)(0, 0));

            // Anomalies: regression adjustment of the spread in observation
            // space, mapped back through Cov(x, y) Var(y)^{-1}.
            if (v > Scalar(0)) {
                const Matrix cov_xy = ((Xs * Y.transpose()) / Scalar(this->L_ - 1)).eval(); // N x 1
                Xs += (cov_xy / v) * (Y * (inv_sqrt_s - Scalar(1)));
            }
        }
        this->x_ = x;
        this->X_ = Xs;
    }


    // =========================================================================
    // ETKF: Bishop et al. (2001) block transform X^a = X C^{-1/2}
    // =========================================================================

    template <typename T>
    void
    ETKF<T>::measurement_update(const ConstVectorRef y, const ConstMatrixRef H,
                                const ConstMatrixRef R)
    {
        this->apply_inflation_to_anomalies();
        const Matrix P = this->sample_covariance();
        const Matrix K = this->kalman_gain(P, H, R);
        const Vector e = (y - H * this->x_).eval();
        this->x_ += K * e;

        if (this->has_taper()) {
            this->update_anomalies_joseph(K, H, R);
            return;
        }

        // C = I + Y^T R^{-1} Y / (L-1) with Y = H X~ (M x L). The transform
        // T = C^{-1/2} gives X~^a = X~ T and 1/(L-1) X~^a X~^a^T = (I - K H) P
        // (Woodbury), i.e. exactly the untapered Joseph target.
        const Matrix Y  = (H * this->X_).eval();                 // M x L
        Eigen::LLT<Matrix> lltR(R);
        require(lltR.info() == Eigen::Success,
                "ETKF: R must be positive definite");
        const Matrix Ri_Y = lltR.solve(Y);                       // R^{-1} Y
        Matrix C = (Y.transpose() * Ri_Y) / Scalar(this->L_ - 1);
        C.diagonal().array() += Scalar(1);
        this->symmetrize_in_place(C);

        Eigen::SelfAdjointEigenSolver<Matrix> es(C);
        require(es.info() == Eigen::Success, "ETKF: eigen decomposition failed");
        // Eigenvalues of C are >= 1 (C = I + ...), so this clamp is defensive;
        // it must still be a value that survives the Scalar. numeric_limits::min()
        // is the smallest positive normal -- Scalar(1e-300) is 0 for float.
        const Vector lam =
            es.eigenvalues().cwiseMax(std::numeric_limits<Scalar>::min());
        const Vector inv_sqrt = lam.cwiseSqrt().cwiseInverse();
        // T = C^{-1/2} = U diag(lambda^{-1/2}) U^T (symmetric).
        const Matrix Tmat = (es.eigenvectors() * inv_sqrt.asDiagonal()
                             * es.eigenvectors().transpose()).eval();
        this->X_ = (this->X_ * Tmat).eval();
    }


    template class EnsembleFilter<double>;
    template class EnsembleFilter<float>;
    template class EnKF<double>;
    template class EnKF<float>;
    template class EnSRF<double>;
    template class EnSRF<float>;
    template class EAKF<double>;
    template class EAKF<float>;
    template class ETKF<double>;
    template class ETKF<float>;
    template class LETKF<double>;
    template class LETKF<float>;

} // namespace estimation
