#include <cassert>

// #include <Eigen/Dense>

#include "kalman_filter.hpp"

#include <iostream>


namespace estimation {

    template <class T>
    estimation::KF<T>::KF(const size_t N) :
        N{N}
    {
        x_hat.resize(N);
        P.resize(N, N);
    }


    template <class T>
    KF<T>::Vector
    KF<T>::get_x_hat() const
    { return x_hat; }


    template <class T>
    KF<T>::Matrix
    KF<T>::get_P() const
    { return P; }


    template <class T>
    void
    KF<T>::initialize(
        const Eigen::Ref<KF<T>::Vector> &x_hat,
        const Eigen::Ref<KF<T>::Matrix> &P)
    {
        assert(this->x_hat.size() == N);
        assert(this->P.rows() == N && this->P.cols() == N);

        this->x_hat = x_hat;
        this->P = P;
    }


    template <class T>
    KF<T>::Vector
    KF<T>::measurement_update(
        const Eigen::Ref<KF<T>::Vector> &y,
        const Eigen::Ref<KF<T>::Matrix> &H,
        const Eigen::Ref<KF<T>::Matrix> &R)
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

        Eigen::LLT<Eigen::Ref<KF<T>::Matrix>> llt(A);

        B = llt.solve(H * P);
        b = llt.solve(y - H * x_hat);

        C = P * H.transpose();

        x_hat += C * b;
        P -= C * B;

        return x_hat;
    }


    template <class T>
    KF<T>::Vector
    KF<T>::time_update(
        const Eigen::Ref<KF<T>::Matrix> &F,
        const Eigen::Ref<KF<T>::Matrix> &Q,
        const Eigen::Ref<KF<T>::Vector> &z)
    {
        assert(F.rows() == F.cols() == N);
        assert(Q.rows() == Q.cols() == N);
        if (z.size() > 0) {
            assert(z.size() == N);
        }

        x_hat = F * x_hat;
        P = F * P * F.transpose() + Q;

        if (z.size() > 0) {
            x_hat += z;
        }

        return x_hat;
    }


    template class KF<double>;
    template class KF<float>;

}
