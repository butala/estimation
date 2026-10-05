#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "filter.hpp"
#include "kalman_filter.hpp"
#include "square_root_kf.hpp"
#include "ensemble_filter.hpp"
#include "ud_filter.hpp"
#include "smoother.hpp"

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
        // argument orders, in C++ and in Python. LETKF and LEKS are their
        // unlocalized counterparts plus a required taper -- localization is a
        // knob, not a separate algorithm.
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
            .def("tapered", &FilterT::tapered, "A"_a.noconvert())
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

        nb::class_<SmoothOutput<double>>(m, "SmoothOutput")
            .def(nb::init<>())
            .def_rw("x_smoothed", &SmoothOutput<double>::x_smoothed)
            .def_rw("P_smoothed", &SmoothOutput<double>::P_smoothed);

        m.attr("RECORD_NONE")        = nb::int_(Record::None);
        m.attr("RECORD_X_PRIOR")     = nb::int_(Record::XPrior);
        m.attr("RECORD_X_POSTERIOR") = nb::int_(Record::XPosterior);
        m.attr("RECORD_P_PRIOR")     = nb::int_(Record::PPrior);
        m.attr("RECORD_P_POSTERIOR") = nb::int_(Record::PPosterior);
        m.attr("RECORD_MEANS")       = nb::int_(Record::Means);
        m.attr("RECORD_COVARIANCES") = nb::int_(Record::Covariances);
        m.attr("RECORD_EVERYTHING")  = nb::int_(Record::Everything);

        // ---- ensemble common -------------------------------------------------
        // EnKF / EnSRF / EAKF / ETKF share the ensemble representation.
        nb::class_<EnsembleFilter<double>, FilterT>(m, "EnsembleFilter")
            .def("ensemble_size", &EnsembleFilter<double>::ensemble_size)
            .def("members", &EnsembleFilter<double>::members)
            .def("anomalies", &EnsembleFilter<double>::anomalies)
            .def("set_members", &EnsembleFilter<double>::set_members,
                 "X"_a.noconvert());

        // ---- exact Kalman filter, three uncertainty representations ----------
        nb::class_<KF<double>, FilterT>(m, "KF")
            .def(nb::init<std::size_t>(), "N"_a);

        nb::class_<SquareRootKF<double>, FilterT>(m, "SquareRootKF")
            .def(nb::init<std::size_t>(), "N"_a)
            .def("factor", &SquareRootKF<double>::factor);

        nb::class_<UDKF<double>, FilterT>(m, "UDKF")
            .def(nb::init<std::size_t>(), "N"_a)
            .def("unit_lower", &UDKF<double>::unit_lower)
            .def("diagonal", &UDKF<double>::diagonal);

        // ---- Monte Carlo Kalman methods --------------------------------------
        nb::class_<EnKF<double>, EnsembleFilter<double>>(m, "EnKF")
            .def(nb::init<std::size_t, std::size_t, std::uint64_t>(),
                 "N"_a, "L"_a, "seed"_a = 0);

        nb::class_<EnSRF<double>, EnsembleFilter<double>>(m, "EnSRF")
            .def(nb::init<std::size_t, std::size_t, std::uint64_t>(),
                 "N"_a, "L"_a, "seed"_a = 0);

        nb::class_<EAKF<double>, EnsembleFilter<double>>(m, "EAKF")
            .def(nb::init<std::size_t, std::size_t, std::uint64_t>(),
                 "N"_a, "L"_a, "seed"_a = 0);

        nb::class_<ETKF<double>, EnsembleFilter<double>>(m, "ETKF")
            .def(nb::init<std::size_t, std::size_t, std::uint64_t>(),
                 "N"_a, "L"_a, "seed"_a = 0);

        nb::class_<LETKF<double>, ETKF<double>>(m, "LETKF")
            .def(nb::init<std::size_t, std::size_t, ConstMat, std::uint64_t>(),
                 "N"_a, "L"_a, "C"_a.noconvert(), "seed"_a = 0);

        // ---- smoothers -------------------------------------------------------
        nb::class_<EnKS<double>>(m, "EnKS")
            .def(nb::init<std::size_t, std::size_t, std::uint64_t>(),
                 "N"_a, "L"_a, "seed"_a = 0)
            .def("ensemble_size", &EnKS<double>::ensemble_size)
            .def("initialize", &EnKS<double>::initialize,
                 "x"_a.noconvert(), "P"_a.noconvert())
            .def("set_taper", &EnKS<double>::set_taper, "C"_a.noconvert())
            .def("clear_taper", &EnKS<double>::clear_taper)
            .def("has_taper", &EnKS<double>::has_taper)
            .def("set_inflation", &EnKS<double>::set_inflation, "lambda"_a)
            .def("clear_inflation", &EnKS<double>::clear_inflation)
            .def("inflation", &EnKS<double>::inflation)
            .def("smooth",
                 [](EnKS<double> &self, const VecList &y, const MatList &H,
                    const MatList &R, const MatList &F, const MatList &Q,
                    const VecList &u) {
                     return self.smooth(y, H, R, F, Q, u);
                 },
                 "y"_a, "H"_a, "R"_a, "F"_a, "Q"_a, "u"_a)
            .def("smooth",
                 [](EnKS<double> &self, const VecList &y, const MatList &H,
                    const MatList &R, const MatList &F, const MatList &Q) {
                     return self.smooth(y, H, R, F, Q);
                 },
                 "y"_a, "H"_a, "R"_a, "F"_a, "Q"_a);

        nb::class_<LEKS<double>, EnKS<double>>(m, "LEKS")
            .def(nb::init<std::size_t, std::size_t, ConstMat, std::uint64_t>(),
                 "N"_a, "L"_a, "C"_a.noconvert(), "seed"_a = 0);

        // Exact RTS smoother on a forward record (the reference EnKS is
        // validated against).
        m.def("rts_smooth",
              [](const BatchOutput<double> &rec, const MatList &F,
                 const MatList &Q, const VecList &u) {
                  return rts_smooth<double>(rec, F, Q, u);
              },
              "record"_a, "F"_a, "Q"_a, "u"_a);
        m.def("rts_smooth",
              [](const BatchOutput<double> &rec, const MatList &F,
                 const MatList &Q) {
                  return rts_smooth<double>(rec, F, Q);
              },
              "record"_a, "F"_a, "Q"_a);
    }

} // namespace estimation
