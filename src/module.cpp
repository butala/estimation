#include <nanobind/nanobind.h>

#include "kalman_filter.hpp"

namespace nb = nanobind;

using namespace nb::literals;

namespace estimation {

    NB_MODULE(lib, m) {
        nb::class_<KF<double>> KF_(m, "KF");

        KF_
            .def(nb::init<const size_t>())
            .def("x_hat", &KF<double>::get_x_hat)
            .def("P", &KF<double>::get_P)
            .def("initialize",
                 &KF<double>::initialize,
                 "x_hat"_a.noconvert(),
                 "P"_a.noconvert())
            .def("measurement_update",
                 &KF<double>::measurement_update,
                 "y"_a.noconvert(),
                 "H"_a.noconvert(),
                 "R"_a.noconvert())
            .def("time_update",
                 nb::overload_cast<const KF<double>::MatrixRef &,
                 const KF<double>::MatrixRef &>
                 (&KF<double>::time_update),
                 "F"_a.noconvert(),
                 "Q"_a.noconvert())
            .def("time_update_with_input",
                 nb::overload_cast<const KF<double>::MatrixRef &,
                 const KF<double>::MatrixRef &,
                 const KF<double>::VectorRef &>
                 (&KF<double>::time_update),
                 "F"_a.noconvert(),
                 "Q"_a.noconvert(),
                 "z"_a.noconvert())
            .def("batch",
                 nb::overload_cast<
                 const KF<double>::BatchOutputConfig &,
                 const KF<double>::VectorRef &,
                 const KF<double>::MatrixRef &,
                 const std::vector<const KF<double>::VectorRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &>
                 (&KF<double>::batch),
                 "config"_a,
                 "mu"_a.noconvert(),
                 "PI"_a.noconvert(),
                 "y"_a,
                 "H"_a,
                 "R"_a,
                 "F"_a,
                 "Q"_a)
            .def("batch_with_input",
                 nb::overload_cast<
                 const KF<double>::BatchOutputConfig &,
                 const KF<double>::VectorRef &,
                 const KF<double>::MatrixRef &,
                 const std::vector<const KF<double>::VectorRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::MatrixRef> &,
                 const std::vector<const KF<double>::VectorRef> &>
                 (&KF<double>::batch),
                 "config"_a,
                 "mu"_a.noconvert(),
                 "PI"_a.noconvert(),
                 "y"_a,
                 "H"_a,
                 "R"_a,
                 "F"_a,
                 "Q"_a,
                 "z"_a);


        nb::class_<KF<double>::BatchOutputConfig>(KF_, "BatchOutputConfig")
            .def(nb::init<>())
            .def_rw("path", &KF<double>::BatchOutputConfig::path)
            .def_rw("x_hat_posterior_template", &KF<double>::BatchOutputConfig::x_hat_posterior_template)
            .def_rw("x_hat_prior_template", &KF<double>::BatchOutputConfig::x_hat_prior_template)
            .def_rw("P_posterior_template", &KF<double>::BatchOutputConfig::P_posterior_template)
            .def_rw("P_prior_template", &KF<double>::BatchOutputConfig::P_prior_template)
            .def_rw("save_x_hat_posterior", &KF<double>::BatchOutputConfig::save_x_hat_posterior)
            .def_rw("save_x_hat_prior", &KF<double>::BatchOutputConfig::save_x_hat_prior)
            .def_rw("save_P_posterior", &KF<double>::BatchOutputConfig::save_P_posterior)
            .def_rw("save_P_prior", &KF<double>::BatchOutputConfig::save_P_posterior);


        nb::class_<KF<double>::BatchOutput>(KF_, "BatchOutput")
            .def_rw("x_hat_posterior", &KF<double>::BatchOutput::x_hat_posterior)
            .def_rw("x_hat_prior", &KF<double>::BatchOutput::x_hat_prior)
            .def_rw("P_posterior", &KF<double>::BatchOutput::P_posterior)
            .def_rw("P_prior", &KF<double>::BatchOutput::P_prior);
    }

}
