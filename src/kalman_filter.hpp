#pragma once

#ifdef PYTHON_MODULE
#include <nanobind/eigen/dense.h>
#endif

#include <Eigen/Dense>


namespace estimation {

    template <typename T = double>
    class KF {
    public:
        KF(const size_t N);

        typedef typename Eigen::Vector<T, Eigen::Dynamic> Vector;
        typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

        Vector
        get_x_hat() const;

        Matrix
        get_P() const;

        void
        initialize(const Eigen::Ref<Vector> &x_hat,
                   const Eigen::Ref<Matrix> &P);

        Eigen::Vector<T, Eigen::Dynamic>
        measurement_update(const Eigen::Ref<Vector> &y,
                           const Eigen::Ref<Matrix> &H,
                           const Eigen::Ref<Matrix> &R);


        Eigen::Vector<T, Eigen::Dynamic>
        time_update(const Eigen::Ref<Matrix> &F,
                    const Eigen::Ref<Matrix> &Q,
                    const Eigen::Ref<Vector> &z = Eigen::Vector<T, 0>());

    private:
        size_t N;
        Vector x_hat;
        Matrix P;
        Matrix A;
        Matrix B;
        Vector b;
        Matrix C;
    };

}
