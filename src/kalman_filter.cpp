#include <cassert>

#include "kalman_filter.hpp"

#include <iostream>


namespace estimation {

    template <typename T>
    estimation::KF<T>::KF(const size_t N) :
        N{N}
    {
        x_hat.resize(N);
        P.resize(N, N);
    }


    template <typename T>
    KF<T>::Vector
    KF<T>::get_x_hat() const
    { return x_hat; }


    template <typename T>
    KF<T>::Matrix
    KF<T>::get_P() const
    { return P; }


    template <typename T>
    void
    KF<T>::initialize(
        const KF<T>::VectorRef &x_hat,
        const KF<T>::MatrixRef &P)
    {
        assert(this->x_hat.size() == N);
        assert(this->P.rows() == N && this->P.cols() == N);

        this->x_hat = x_hat;
        this->P = P;
    }


    template <typename T>
    KF<T>::Vector
    KF<T>::measurement_update(
        const KF<T>::VectorRef &y,
        const KF<T>::MatrixRef &H,
        const KF<T>::MatrixRef &R)
    {
        const size_t M = H.rows();
        const size_t N = H.cols();

        assert(y.size() == M);
        assert(R.rows() == R.cols() == M);

        A.resize(M, M);
        B.resize(M, N);
        b.resize(M);
        C.resize(N, M);

        A = H * (P * H.transpose());
        A += R;

        Eigen::LLT<KF<T>::MatrixRef> llt(A);

        B = llt.solve(H * P);
        b = llt.solve(y - H * x_hat);

        C = P * H.transpose();

        x_hat += C * b;
        P -= C * B;

        return x_hat;
    }


    template <typename T>
    KF<T>::Vector
    KF<T>::time_update(
        const KF<T>::MatrixRef &F,
        const KF<T>::MatrixRef &Q)
    {
        assert(F.rows() == F.cols() == N);
        assert(Q.rows() == Q.cols() == N);

        x_hat = F * x_hat;
        P = F * P * F.transpose() + Q;

        return x_hat;
    }

    template <typename T>
    KF<T>::Vector
    KF<T>::time_update(
        const KF<T>::MatrixRef &F,
        const KF<T>::MatrixRef &Q,
        const KF<T>::VectorRef &z)
    {
        assert(z.size() == N);

        time_update(F, Q);
        x_hat += z;

        return x_hat;
    }


    template <typename T>
    KF<T>::BatchOutput
    KF<T>::batch(const BatchOutputConfig &config,
                 const KF<T>::VectorRef &mu,
                 const KF<T>::MatrixRef &PI,
                 const std::vector<const KF<T>::VectorRef> &y,
                 const std::vector<const KF<T>::MatrixRef> &H,
                 const std::vector<const KF<T>::MatrixRef> &R,
                 const std::vector<const KF<T>::MatrixRef> &F,
                 const std::vector<const KF<T>::MatrixRef> &Q,
                 const std::vector<const KF<T>::VectorRef> &z) {
        BatchOutput output;

        assert(mu.size() == N);
        assert(PI.rows() == PI.cols() == N);

        assert(y.size() == H.size() == R.size() == F.size() == Q.size());
        size_t I = y.size();

        if (z.size() > 0) {
            assert(z.size() == I);
        }

        initialize(mu, PI);
        for (size_t i = 0; i < I; i++) {
            // measurement update
            measurement_update(y[i], H[i], R[i]);
            if (config.save_x_hat_posterior) {
                output.x_hat_posterior.push_back(x_hat);
            }
            if (config.save_P_posterior) {
                output.P_posterior.push_back(P);
            }
            // time update
            if (z.size() > 0) {
                time_update(F[i], Q[i], z[i]);
            }
            else {
                time_update(F[i], Q[i]);
            }
            if (config.save_x_hat_prior) {
                output.x_hat_prior.push_back(x_hat);
            }
            if (config.save_P_prior) {
                output.P_prior.push_back(P);
            }
        }

        return output;
    }


    template <typename T>
    KF<T>::BatchOutput
    KF<T>::batch(const BatchOutputConfig &config,
                 const KF<T>::VectorRef &mu,
                 const KF<T>::MatrixRef &PI,
                 const std::vector<const KF<T>::VectorRef> &y,
                 const std::vector<const KF<T>::MatrixRef> &H,
                 const std::vector<const KF<T>::MatrixRef> &R,
                 const std::vector<const KF<T>::MatrixRef> &F,
                 const std::vector<const KF<T>::MatrixRef> &Q) {
        std::vector<const KF<T>::VectorRef> z = std::vector<const KF<T>::VectorRef>();
        return batch(config, mu, PI, y, H, R, F, Q, z);
    }


    template class KF<double>;
    template class KF<float>;

}
