#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_UTIL_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_UTIL_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace FastMIDyNet{

template <typename SBMPrior, typename... PriorBaseClasses>
py::class_<SBMPrior, PriorBaseClasses...> declareSBMPrior(py::module& m, std::string pyName){
    return py::class_<SBMPrior, PriorBaseClasses...>(m, pyName.c_str())
        .def(py::init<>())
        .def("get_log_likelihood_ratio_from_graphmove", &SBMPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &SBMPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_graphmove", &SBMPrior::getLogPriorRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_blockmove", &SBMPrior::getLogPriorRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &SBMPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &SBMPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graph_move", &SBMPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_block_move", &SBMPrior::applyBlockMove,
            py::arg("move"));
}

}

#endif
