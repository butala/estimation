#include <nanobind/nanobind.h>

#include "kalman_filter.hpp"

namespace nb = nanobind;

using namespace estimation;


NB_MODULE(lib, m) {
    nb::class_<KF<double>>(m, "KF")
        .def(nb::init<const size_t>())
        .def("x_hat", &KF<double>::get_x_hat)
        .def("P", &KF<double>::get_P)
        .def("initialize", &KF<double>::initialize,
             nb::arg("x_hat").noconvert(),
             nb::arg("P").noconvert())
        .def("measurement_update", &KF<double>::measurement_update,
             nb::arg("y").noconvert(),
             nb::arg("H").noconvert(),
             nb::arg("R").noconvert());
}
