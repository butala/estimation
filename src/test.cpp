// #include "estimation.hpp"
#include "kalman_filter.hpp"

int main(int argc, char** argv) {
    // Eigen::Matrix2d A;
    // A << 1, 2, 3, 4;

    // Eigen::Map<Eigen::MatrixXd> A_mat(A.data(), A.rows(), A.cols());

    // print_matrix(A_mat);

    estimation::KF kf(10);

    return 0;
}
