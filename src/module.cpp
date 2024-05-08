#include <nanobind/nanobind.h>

#include "kalman_filter.hpp"

namespace nb = nanobind;

using namespace nb::literals;

namespace estimation {

    NB_MODULE(lib, m) {
        nb::class_<KF<double>>(m, "KF")
            .def(nb::init<const size_t>())
            .def("x_hat", &KF<double>::get_x_hat)
            .def("P", &KF<double>::get_P)
            .def("initialize", &KF<double>::initialize,
                 "x_hat"_a.noconvert(),
                 "P"_a.noconvert())
            .def("measurement_update", &KF<double>::measurement_update,
                 "y"_a.noconvert(),
                 "H"_a.noconvert(),
                 "R"_a.noconvert())
            .def("time_update", &KF<double>::time_update,
                 "F"_a.noconvert(),
                 "Q"_a.noconvert(),
                 "z"_a.noconvert() = Eigen::Vector<double, 0>());
    }

}
