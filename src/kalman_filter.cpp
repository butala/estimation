#include <cassert>

// #include <Eigen/Dense>

#include "kalman_filter.hpp"

#include <iostream>



template <class T>
estimation::KF<T>::KF(const size_t N) :
    N{N}
{
    x_hat.resize(N);
    P.resize(N, N);
}


template <class T>
Eigen::Vector<T, Eigen::Dynamic>
estimation::KF<T>::get_x_hat() const
{ return x_hat; }


template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
estimation::KF<T>::get_P() const
{ return P; }


template <class T>
void
estimation::KF<T>::initialize(
    const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &x_hat,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &P)
{
    assert(this->x_hat.size() == N);
    assert(this->P.rows() == N && this->P.cols() == N);

    this->x_hat = x_hat;
    this->P = P;
}


template <class T>
Eigen::Vector<T, Eigen::Dynamic>
estimation::KF<T>::measurement_update(
    const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &y,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &H,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &R)
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

    Eigen::LLT<Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> llt(A);

    B = llt.solve(H * P);
    b = llt.solve(y - H * x_hat);

    C = P * H.transpose();

    x_hat += C * b;
    P -= C * B;

    return x_hat;
}


template <class T>
Eigen::Vector<T, Eigen::Dynamic>
estimation::KF<T>::time_update(
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &F,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Q,
    const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &z)
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


template class estimation::KF<double>;
// template class estimation::KF<float>;



void measurement_update(
    const Eigen::Map<Eigen::VectorXd> &x_hat) {
    // Eigen::Ref<Eigen::MatrixXd> &P,
    // const Eigen::Ref<Eigen::VectorXd> &y,
    // const Eigen::Ref<Eigen::MatrixXd> &H,
    // const Eigen::Ref<Eigen::MatrixXd> &R) {
    // Kalman filter measurement update

    // std::cout << x_hat[0] << '\n';
    //x_hat[0] = 10.;
    // std::cout << x_hat[0] << '\n';

    //Eigen::VectorXd x_hat = x_hat_ref;

    //x_hat[0] = 10;
    std::cout << x_hat << '\n';

    // PASS THESE AS OPTIONAL ARGUMENTS
    // Eigen::MatrixXd A;
    // Eigen::MatrixXd B;
    // Eigen::VectorXd b;
    // Eigen::MatrixXd C;

    // const size_t M = H.rows();
    // const size_t N = H.cols();

    // assert(x_hat.size() == N);
    // assert(P.rows() == P.cols() == N);
    // assert(y.size() == M);
    // assert(R.rows() == R.cols() == M);

    // A.resize(M, M);
    // B.resize(M, N);
    // b.resize(M);
    // C.resize(N, M);

    // A = H * (P * H.transpose());
    // A += R;

    // Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(A);

    // B = llt.solve(H * P);
    // b = llt.solve(y - H * x_hat);

    // C = P * H.transpose();

    // std::cout << x_hat[0] << '\n';
    // x_hat += C * b;
    // std::cout << x_hat[0] << '\n';

    // P -= C * B;
}
