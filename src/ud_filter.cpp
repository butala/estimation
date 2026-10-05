#include <cassert>

#include "ud_filter.hpp"


namespace estimation {

    template <typename T>
    UDKF<T>::UDKF(const std::size_t N)
        : Filter<T>(N), x_(Vector::Zero(N)),
          L_(Matrix::Identity(N, N)), D_(Vector::Ones(N))
    {
    }


    template <typename T>
    typename UDKF<T>::Matrix
    UDKF<T>::covariance() const
    {
        Matrix P = (L_ * D_.asDiagonal()) * L_.transpose();
        this->symmetrize_in_place(P);
        return P;
    }


    // SPD A -> L_ D_ L_^T with L_ unit lower.
    //
    // Written out rather than calling Eigen::LDLT: Eigen pivots and returns
    // P^T L D L^T P, whose `matrixL()` is NOT a factor of A -- reconstructing
    // L D L^T from it is off by the permutation (this was a real bug). The
    // unpivoted recursion is what Bierman's downdate assumes.
    template <typename T>
    void
    UDKF<T>::refactor(const ConstMatrixRef A)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        Matrix S = A;
        this->symmetrize_in_place(S);
        L_ = Matrix::Identity(N, N);
        D_ = Vector::Zero(N);
        for (Eigen::Index j = 0; j < N; ++j) {
            Scalar s = S(j, j);
            for (Eigen::Index k = 0; k < j; ++k) s -= L_(j, k) * L_(j, k) * D_(k);
            require(s > Scalar(0), "UDKF::refactor: matrix is not positive definite");
            D_(j) = s;
            for (Eigen::Index i = j + 1; i < N; ++i) {
                Scalar t = S(i, j);
                for (Eigen::Index k = 0; k < j; ++k) t -= L_(i, k) * L_(j, k) * D_(k);
                L_(i, j) = t / D_(j);
            }
        }
    }


    template <typename T>
    void
    UDKF<T>::initialize(const ConstVectorRef x, const ConstMatrixRef P)
    {
        require(x.size() == static_cast<Eigen::Index>(this->N_),
                "initialize: x must have length N");
        require(P.rows() == static_cast<Eigen::Index>(this->N_) &&
                P.cols() == static_cast<Eigen::Index>(this->N_),
                "initialize: P must be N x N");
        x_ = x;
        refactor(P);
    }


    // Bierman's sequential scalar downdate for the whitened row (h, r):
    //     g = D L^T h^T,  alpha = r + h P h^T = r + f^T g   (f = L^T h^T)
    //     k = L g / alpha,   x += k (z - h x)
    //     P^+ = L (D - g g^T / alpha) L^T = (L Delta) D' (L Delta)^T
    // with Delta D' Delta^T = D - g g^T / alpha from the bordered recursion.
    template <typename T>
    void
    UDKF<T>::bierman_row(const Vector &h, const Scalar z_minus_hx,
                         const Scalar r)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Vector f = L_.transpose() * h;             // L^T h^T
        Vector g(N);
        for (Eigen::Index i = 0; i < N; ++i) g(i) = D_(i) * f(i);

        Scalar alpha = r + f.dot(g);                     // = h P h^T + r
        require(alpha > Scalar(0), "bierman_row: non-positive innovation variance");

        x_ += ((L_ * g) / alpha) * z_minus_hx;

        // D - g g^T / alpha = Delta D' Delta^T (Delta unit lower), O(N^2).
        Vector Dnew(N);
        Matrix Delta = Matrix::Identity(N, N);
        Scalar a = alpha;
        for (Eigen::Index j = 0; j < N; ++j) {
            Dnew(j) = D_(j) - g(j) * g(j) / a;
            const Scalar a_next = a - g(j) * g(j) / D_(j);
            for (Eigen::Index i = j + 1; i < N; ++i)
                Delta(i, j) = -g(j) * g(i) / (D_(j) * a_next);
            a = a_next;
        }
        // L^+ = L Delta (unit lower stays unit lower).
        L_ = (L_ * Delta).eval();
        D_ = Dnew;
        for (Eigen::Index i = 0; i < N; ++i) {
            L_(i, i) = Scalar(1);
            if (D_(i) < Scalar(0)) D_(i) = Scalar(0);
        }
    }


    template <typename T>
    void
    UDKF<T>::measurement_update(const ConstVectorRef y, const ConstMatrixRef H,
                                const ConstMatrixRef R)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        const Eigen::Index m = H.rows();
        require(y.size() == m,
                "measurement_update: y must have length H.rows()");
        require(H.cols() == N, "measurement_update: H must have N columns");
        require(R.rows() == m && R.cols() == m,
                "measurement_update: R must be H.rows() x H.rows()");

        Matrix P = covariance();
        this->apply_inflation(P);
        refactor(P);

        if (this->has_taper()) {
            // The tapered gain is not Bierman's natural gain: do the Joseph
            // form of the LKF and refactor, exactly as KF/SquareRootKF do.
            const Matrix Pg   = this->gain_covariance(P);
            const Matrix P_HT = Pg * H.transpose();
            const Matrix S    = H * P_HT + R;
            Eigen::LLT<Matrix> lltS(((S + S.transpose()).eval() * Scalar(0.5)));
            require(lltS.info() == Eigen::Success,
                    "measurement_update: innovation covariance is not PD");
            const Matrix K = lltS.solve(P_HT.transpose()).transpose();
            x_ += K * ((y - H * x_).eval());
            const Matrix IKH = Matrix::Identity(N, N) - K * H;
            const Matrix Pout =
                (IKH * P * IKH.transpose() + K * R * K.transpose()).eval();
            refactor(Pout);
            return;
        }

        // Whitened sequential Bierman updates (R -> I). For a whitened system
        // these compose to exactly the block analysis.
        Matrix Rm = R;
        Rm = ((Rm + Rm.transpose()).eval() * Scalar(0.5));
        Eigen::LLT<Matrix> lltR(Rm);
        require(lltR.info() == Eigen::Success, "measurement_update: R must be PD");
        const Matrix L_R(lltR.matrixL());
        const Eigen::FullPivLU<Matrix> lu(L_R);
        require(lu.isInvertible(), "measurement_update: R factor is singular");
        const Vector y_w = lu.solve(y);
        const Matrix H_w = lu.solve(H);

        for (Eigen::Index j = 0; j < m; ++j) {
            const Vector h = H_w.row(j).transpose();
            const Scalar innov = y_w(j) - (H_w.row(j) * x_)(0, 0);
            bierman_row(h, innov, Scalar(1));
        }
    }


    template <typename T>
    void
    UDKF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q)
    {
        const Eigen::Index N = static_cast<Eigen::Index>(this->N_);
        require(F.rows() == N && F.cols() == N, "time_update: F must be N x N");
        require(Q.rows() == N && Q.cols() == N, "time_update: Q must be N x N");

        Matrix A = (F * ((L_ * D_.asDiagonal()) * L_.transpose()) * F.transpose()
                    + Q).eval();
        refactor(A);
        x_ = (F * x_).eval();
    }


    template <typename T>
    void
    UDKF<T>::time_update(const ConstMatrixRef F, const ConstMatrixRef Q,
                         const ConstVectorRef u)
    {
        require(u.size() == static_cast<Eigen::Index>(this->N_),
                "time_update: u must have length N");
        time_update(F, Q);
        x_ += u;
    }


    template class UDKF<double>;
    template class UDKF<float>;

} // namespace estimation
