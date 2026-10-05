#include "kalman_filter.hpp"


namespace estimation {

    template <typename T>
    KF<T>::KF(const std::size_t N)
        : Filter<T>(N), x_(Vector::Zero(N)), P_(Matrix::Zero(N, N))
    {
    }


    template <typename T>
    void
    KF<T>::initialize(const ConstVectorRef x, const ConstMatrixRef P)
    {
        require(x.size() == static_cast<Eigen::Index>(this->N_),
                "initialize: x must have length N");
        require(P.rows() == static_cast<Eigen::Index>(this->N_) &&
                P.cols() == static_cast<Eigen::Index>(this->N_),
                "initialize: P must be N x N");
        x_ = x;
        P_ = P;
        // Enforce the invariant "P_ is exactly symmetric".
        this->symmetrize_in_place(P_);
    }


    template <typename T>
    void
    KF<T>::measurement_update(const ConstVectorRef y,
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
        this->apply_inflation(P_);

        // Gain from the GAIN covariance (C o P with a taper, else P).
        const Matrix Pg   = this->gain_covariance(P_);
        const Matrix P_HT = Pg * H.transpose();          // P_g H^T (N x m)
        const Matrix S    = H * P_HT + R;                // (m x m)

        Eigen::LLT<Matrix> lltS(S);
        require(lltS.info() == Eigen::Success,
                "measurement_update: innovation covariance is not positive definite");
        const Matrix K = lltS.solve(P_HT.transpose()).transpose();  // P_g H^T S^{-1}

        const Vector e = (y - H * x_).eval();
        x_ += K * e;

        // Joseph form on the UNTAPERED prior P (Butala et al., IEEE TIP 2009
        // (4.26)). Keeping the snapshot of the prior also breaks the aliasing
        // of P_ on both sides of the assignment.
        const Matrix P_prior = P_;
        const Matrix IKH     = Matrix::Identity(N, N) - K * H;
        P_ = (IKH * P_prior * IKH.transpose() + K * R * K.transpose()).eval();
        this->symmetrize_in_place(P_);
    }


    template <typename T>
    void
    KF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        require(F.rows() == N && F.cols() == N,
                "time_update: F must be N x N");
        require(Q.rows() == N && Q.cols() == N,
                "time_update: Q must be N x N");

        x_ = (F * x_).eval();

        // Evaluate F P F^T before assigning to P: the destination P also occurs
        // in the product, and Eigen does not guard against that aliasing.
        // Inflation is applied in measurement_update.
        const Matrix FP = (F * P_).eval();
        P_ = (FP * F.transpose() + Q).eval();
        this->symmetrize_in_place(P_);
    }


    template <typename T>
    void
    KF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q,
                       const ConstVectorRef u)
    {
        require(u.size() == static_cast<Eigen::Index>(this->N_),
                "time_update: u must have length N");
        time_update(F, Q);
        x_ += u;
    }


    template class KF<double>;
    template class KF<float>;

} // namespace estimation
