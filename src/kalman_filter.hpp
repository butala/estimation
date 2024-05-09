#pragma once

#ifdef PYTHON_MODULE
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#endif

#include <string>
#include <vector>
#include <format>
#include <Eigen/Dense>


namespace estimation {

    template <typename T = double>
    class KF {
    public:
        KF(const size_t N);

        typedef typename Eigen::Vector<T, Eigen::Dynamic> Vector;
        typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

        typedef typename Eigen::Ref<Vector> VectorRef;
        typedef typename Eigen::Ref<Matrix> MatrixRef;


        Vector
        get_x_hat() const;

        Matrix
        get_P() const;

        void
        initialize(const VectorRef &x_hat,
                   const MatrixRef &P);

        Eigen::Vector<T, Eigen::Dynamic>
        measurement_update(const VectorRef &y,
                           const MatrixRef &H,
                           const MatrixRef &R);


        Eigen::Vector<T, Eigen::Dynamic>
        time_update(const MatrixRef &F,
                    const MatrixRef &Q);

        Eigen::Vector<T, Eigen::Dynamic>
        time_update(const MatrixRef &F,
                    const MatrixRef &Q,
                    const VectorRef &z);


        struct BatchOutputConfig {
            BatchOutputConfig() :
                path(""),
                x_hat_posterior_template(""),
                x_hat_prior_template(""),
                P_posterior_template(""),
                P_prior_template(""),
                save_x_hat_posterior(false),
                save_x_hat_prior(false),
                save_P_posterior(false),
                save_P_prior(false) {;}

            std::string path;

            std::string x_hat_posterior_template;
            std::string x_hat_prior_template;

            std::string P_posterior_template;
            std::string P_prior_template;

            bool save_x_hat_posterior;
            bool save_x_hat_prior;

            bool save_P_posterior;
            bool save_P_prior;
        };


        class BatchOutput {
        public:
            BatchOutput() :
                x_hat_posterior(),
                x_hat_prior(),
                P_posterior(),
                P_prior() {;}

            std::vector<Vector> x_hat_posterior;
            std::vector<Vector> x_hat_prior;

            std::vector<Matrix> P_posterior;
            std::vector<Matrix> P_prior;
        };


        BatchOutput
        batch(const BatchOutputConfig &config,
              const VectorRef &mu,
              const MatrixRef &PI,
              const std::vector<const VectorRef> &y,
              const std::vector<const MatrixRef> &H,
              const std::vector<const MatrixRef> &R,
              const std::vector<const MatrixRef> &F,
              const std::vector<const MatrixRef> &Q,
              const std::vector<const VectorRef> &z);

        BatchOutput
        batch(const BatchOutputConfig &config,
              const VectorRef &mu,
              const MatrixRef &PI,
              const std::vector<const VectorRef> &y,
              const std::vector<const MatrixRef> &H,
              const std::vector<const MatrixRef> &R,
              const std::vector<const MatrixRef> &F,
              const std::vector<const MatrixRef> &Q);


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
