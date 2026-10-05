#include "square_root_kf.hpp"


namespace estimation {

    template <typename T>
    SquareRootKF<T>::SquareRootKF(const std::size_t N)
        : Filter<T>(N), x_(Vector::Zero(N)), L_(Matrix::Zero(N, N))
    {
    }


    // Lower triangular L with L L^T = M M^T and positive diagonal.
    //
    // Thin QR of M^T (K x N, K >= N) gives M^T = Q R with R upper triangular,
    // hence M = R^T Q^T and M M^T = R^T R. Sign-flipping the rows of R leaves
    // R^T R unchanged and lets us fix the sign convention.
    template <typename T>
    typename SquareRootKF<T>::Matrix
    SquareRootKF<T>::retriangulate(const Matrix &M)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(M.rows());
        require(M.cols() >= M.rows(),
                "retriangulate: need at least as many columns as rows");

        Eigen::HouseholderQR<Matrix> qr(M.transpose());
        Matrix R = qr.matrixQR().topLeftCorner(N, N);
        R.template triangularView<Eigen::StrictlyLower>().setZero();

        for (Eigen::Index i = 0; i < N; ++i) {
            if (R(i, i) < Scalar(0)) R.row(i) *= Scalar(-1);
        }
        return R.transpose();
    }


    // Any G with G G^T = A for symmetric positive semidefinite A. LLT first
    // (fast, triangular); fall back to an eigen factorization with the
    // negative eigenvalues of round-off clipped away, so that singular A (e.g.
    // Q = 0 in a deterministic model) is handled rather than rejected.
    template <typename T>
    typename SquareRootKF<T>::Matrix
    SquareRootKF<T>::psd_sqrt(const Matrix &A)
    {
        Eigen::LLT<Matrix> llt(A);
        if (llt.info() == Eigen::Success) return Matrix(llt.matrixL());

        Eigen::SelfAdjointEigenSolver<Matrix> es(A);
        require(es.info() == Eigen::Success,
                "psd_sqrt: eigen decomposition failed");
        const Vector s = es.eigenvalues().cwiseMax(Scalar(0)).cwiseSqrt();
        return es.eigenvectors() * s.asDiagonal();
    }


    template <typename T>
    void
    SquareRootKF<T>::initialize(const ConstVectorRef x, const ConstMatrixRef P)
    {
        require(x.size() == static_cast<Eigen::Index>(this->N_),
                "initialize: x must have length N");
        require(P.rows() == static_cast<Eigen::Index>(this->N_) &&
                P.cols() == static_cast<Eigen::Index>(this->N_),
                "initialize: P must be N x N");
        x_ = x;
        Matrix P_sym = P;
        this->symmetrize_in_place(P_sym);
        L_ = retriangulate(psd_sqrt(P_sym));
    }


    template <typename T>
    void
    SquareRootKF<T>::measurement_update(const ConstVectorRef y,
                                        const ConstMatrixRef H,
                                        const ConstMatrixRef R)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Eigen::Index m = H.rows();

        require(y.size() == m,
                "measurement_update: y must have length H.rows()");
        require(H.cols() == N,
                "measurement_update: H must have N columns");
        require(R.rows() == m && R.cols() == m,
                "measurement_update: R must be H.rows() x H.rows()");

        // Multiplicative inflation on the prior covariance: P <- lambda P.
        this->apply_inflation_sqrt(L_);

        // Innovation covariance S = H P_g H^T + R and cross covariance P_g H^T,
        // where P_g is the gain covariance (C o P with a taper, else P). The
        // untapered factor L_ is needed again below for the Joseph update, so
        // only the gain path ever materializes P.
        const Matrix HL = H * L_;                 // H L   (m x N)
        Matrix P_HT;                              // P_g H^T (N x m)
        Matrix S;                                 // (m x m)
        if (this->has_taper()) {
            Matrix P = L_ * L_.transpose();
            this->symmetrize_in_place(P);
            const Matrix Pg = this->gain_covariance(P);
            P_HT = Pg * H.transpose();
            S    = H * P_HT + R;
        }
        else {
            P_HT = L_ * HL.transpose();           // L L^T H^T
            S    = HL * HL.transpose() + R;
        }

        Eigen::LLT<Matrix> lltS(S);
        require(lltS.info() == Eigen::Success,
                "measurement_update: innovation covariance is not positive definite");
        const Matrix K = lltS.solve(P_HT.transpose()).transpose();  // P_g H^T S^{-1}

        const Vector e = (y - H * x_).eval();
        x_ += K * e;

        // Joseph form on the UNTAPERED P (Butala et al., IEEE TIP 2009 (4.26)):
        //     P+ = (I - K H) P (I - K H)^T + K R K^T = M M^T
        // with M = [ (I - K H) L , K L_R ] and L_R L_R^T = R. QR, not a
        // subtraction of covariances, so P+ stays symmetric positive definite.
        const Matrix L_R = psd_sqrt(Matrix(R));
        Matrix M(N, N + m);
        M.leftCols(N)  = L_ - K * HL;             // (I - K H) L
        M.rightCols(m) = K * L_R;
        L_ = retriangulate(M);
    }


    template <typename T>
    void
    SquareRootKF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        require(F.rows() == N && F.cols() == N,
                "time_update: F must be N x N");
        require(Q.rows() == N && Q.cols() == N,
                "time_update: Q must be N x N");

        // P- = F P+ F^T + Q = (F L)(F L)^T + L_Q L_Q^T = M M^T
        // with M = [ F L , L_Q ]. Inflation is applied in measurement_update.
        Matrix Q_sym = Q;
        this->symmetrize_in_place(Q_sym);
        const Matrix L_Q = psd_sqrt(Q_sym);
        Matrix M(N, 2 * N);
        M.leftCols(N)  = F * L_;
        M.rightCols(N) = L_Q;
        L_ = retriangulate(M);

        x_ = (F * x_).eval();
    }


    template <typename T>
    void
    SquareRootKF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q,
                                 const ConstVectorRef u)
    {
        require(u.size() == static_cast<Eigen::Index>(this->N_),
                "time_update: u must have length N");
        time_update(F, Q);
        x_ += u;
    }


    template class SquareRootKF<double>;
    template class SquareRootKF<float>;

} // namespace estimation
