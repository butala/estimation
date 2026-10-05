#include <nanobind/nanobind.h>

#include "filter.hpp"
#include "kalman_filter.hpp"
#include "square_root_kf.hpp"

namespace nb = nanobind;

using namespace nb::literals;

namespace estimation {

    NB_MODULE(lib, m) {
        using FilterT  = Filter<double>;
        using ConstVec = FilterT::ConstVectorRef;
        using ConstMat = FilterT::ConstMatrixRef;
        using VecList  = std::vector<ConstVec>;
        using MatList  = std::vector<ConstMat>;

        // ---- shared interface ------------------------------------------------
        // Every method in the family has exactly these names and these
        // argument orders, in C++ and in Python.
        nb::class_<FilterT>(m, "Filter")
            .def("dimension", &FilterT::dimension)
            .def("state", &FilterT::state)
            .def("covariance", &FilterT::covariance)
            .def("initialize", &FilterT::initialize,
                 "x"_a.noconvert(), "P"_a.noconvert())
            .def("measurement_update", &FilterT::measurement_update,
                 "y"_a.noconvert(), "H"_a.noconvert(), "R"_a.noconvert())
            .def("time_update",
                 [](FilterT &self, const ConstMat &F, const ConstMat &Q) {
                     self.time_update(F, Q);
                 },
                 "F"_a.noconvert(), "Q"_a.noconvert())
            .def("time_update",
                 [](FilterT &self, const ConstMat &F, const ConstMat &Q,
                    const ConstVec &u) {
                     self.time_update(F, Q, u);
                 },
                 "F"_a.noconvert(), "Q"_a.noconvert(), "u"_a.noconvert())
            .def("set_taper", &FilterT::set_taper, "C"_a.noconvert())
            .def("clear_taper", &FilterT::clear_taper)
            .def("has_taper", &FilterT::has_taper)
            .def("set_inflation", &FilterT::set_inflation, "lambda"_a)
            .def("clear_inflation", &FilterT::clear_inflation)
            .def("inflation", &FilterT::inflation)
            .def("batch",
                 [](FilterT &self, const VecList &y, const MatList &H,
                    const MatList &R, const MatList &F, const MatList &Q,
                    unsigned record) {
                     return self.batch(y, H, R, F, Q, record);
                 },
                 "y"_a, "H"_a, "R"_a, "F"_a, "Q"_a,
                 "record"_a = Record::Everything)
            .def("batch",
                 [](FilterT &self, const VecList &y, const MatList &H,
                    const MatList &R, const MatList &F, const MatList &Q,
                    const VecList &u, unsigned record) {
                     return self.batch(y, H, R, F, Q, u, record);
                 },
                 "y"_a, "H"_a, "R"_a, "F"_a, "Q"_a, "u"_a,
                 "record"_a = Record::Everything);

        nb::class_<BatchOutput<double>>(m, "BatchOutput")
            .def(nb::init<>())
            .def_rw("x_prior",     &BatchOutput<double>::x_prior)
            .def_rw("x_posterior", &BatchOutput<double>::x_posterior)
            .def_rw("P_prior",     &BatchOutput<double>::P_prior)
            .def_rw("P_posterior", &BatchOutput<double>::P_posterior);

        // Record mask constants.
        m.attr("RECORD_NONE")        = nb::int_(Record::None);
        m.attr("RECORD_X_PRIOR")     = nb::int_(Record::XPrior);
        m.attr("RECORD_X_POSTERIOR") = nb::int_(Record::XPosterior);
        m.attr("RECORD_P_PRIOR")     = nb::int_(Record::PPrior);
        m.attr("RECORD_P_POSTERIOR") = nb::int_(Record::PPosterior);
        m.attr("RECORD_MEANS")       = nb::int_(Record::Means);
        m.attr("RECORD_COVARIANCES") = nb::int_(Record::Covariances);
        m.attr("RECORD_EVERYTHING")  = nb::int_(Record::Everything);

        // ---- exact Kalman filter, two uncertainty representations -----------
        nb::class_<KF<double>, FilterT>(m, "KF")
            .def(nb::init<std::size_t>(), "N"_a);

        nb::class_<SquareRootKF<double>, FilterT>(m, "SquareRootKF")
            .def(nb::init<std::size_t>(), "N"_a)
            .def("factor", &SquareRootKF<double>::factor);
    }

} // namespace estimation
