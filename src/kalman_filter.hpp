#pragma once

#ifdef PYTHON_MODULE
#include <nanobind/eigen/dense.h>
#endif

#include <Eigen/Dense>


namespace estimation {

    template <class T = double>
    class KF {
    public:
        KF(const size_t N);

        Eigen::Vector<T, Eigen::Dynamic>
        get_x_hat() const;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        get_P() const;

        void
        initialize(const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &x_hat,
                   const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &P);

        Eigen::Vector<T, Eigen::Dynamic>
        measurement_update(const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &y,
                           const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &H,
                           const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &R);


        Eigen::Vector<T, Eigen::Dynamic>
        time_update(const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &F,
                    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Q,
                    const Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic>> &z = Eigen::Vector<T, 0>());

    private:
        size_t N;
        Eigen::Vector<T, Eigen::Dynamic> x_hat;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> P;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B;
        Eigen::Vector<T, Eigen::Dynamic> b;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C;
    };

}
