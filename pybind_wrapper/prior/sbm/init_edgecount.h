#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGECOUNT_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGECOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/python/edgecount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initEdgeCountPrior(py::module& m){
    py::class_<EdgeCountPrior, Prior<size_t>, PyEdgeCountPrior<>>(m, "EdgeCountPrior")
        .def(py::init<>())
        .def("get_log_likelihood_ratio_from_graphmove", &EdgeCountPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &EdgeCountPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &EdgeCountPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &EdgeCountPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &EdgeCountPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &EdgeCountPrior::applyBlockMove,
            py::arg("move"));

    py::class_<EdgeCountDeltaPrior, EdgeCountPrior>(m, "EdgeCountDeltaPrior")
        .def(py::init<size_t>(), py::arg("edge_count"));

    py::class_<EdgeCountPoissonPrior, EdgeCountPrior>(m, "EdgeCountPoissonPrior")
        .def(py::init<double>(), py::arg("mean"));
}

}

#endif
