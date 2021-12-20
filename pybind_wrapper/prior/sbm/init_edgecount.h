#ifndef FAST_MIDYNET_PYWRAPPER_INIT_EDGECOUNT_SBMPRIOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_EDGECOUNT_SBMPRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/sbm/edge_count.h"

class PyEdgeCountPrior: public FastMIDyNet::EdgeCountPrior {
    public:
        virtual double getLogLikelihoodFromState(const size_t& state) const {
            PYBIND11_OVERLOAD_PURE(double, FastMIDyNet::EdgeCountPrior, getLogLikelihoodFromState, state);
        };
};


template <typename EdgeCountPriorType>
pybind11::class_<EdgeCountPriorType> defineEdgeCountPrior(pybind11::module& m, std::string pyName){
    return pybind11::class_<EdgeCountPriorType>(m, pyName.c_str())
        .def("get_state", &EdgeCountPriorType::getState)
        .def("set_state", &EdgeCountPriorType::setState, pybind11::arg("state"))
        .def("sample", &EdgeCountPriorType::sample)
        .def("sample_state", &EdgeCountPriorType::sampleState)
        .def("sample_priors", &EdgeCountPriorType::samplePriors)
        .def("get_logLikelihood", &EdgeCountPriorType::getLogLikelihood)
        .def("get_logPrior", &EdgeCountPriorType::getLogPrior)
        .def("get_logJoint", &EdgeCountPriorType::getLogJoint)
        .def("get_logLikelihood_ratio_from_graphmove", &EdgeCountPriorType::getLogLikelihoodRatioFromGraphMove,
            pybind11::arg("move"))
        .def("get_logJoint_ratio_from_graphmove", &EdgeCountPriorType::getLogJointRatioFromGraphMove,
            pybind11::arg("move"))
        .def("get_logJoint_ratio_from_blockmove", &EdgeCountPriorType::getLogJointRatioFromBlockMove,
            pybind11::arg("move"))
        .def("apply_graph_move", &EdgeCountPriorType::applyGraphMove,
            pybind11::arg("move"))
        .def("apply_block_move", &EdgeCountPriorType::applyBlockMove,
            pybind11::arg("move"))
        ;
}

void initEdgeCountPrior(pybind11::module& m){
    defineEdgeCountPrior<PyEdgeCountPrior>(m, "EdgeCountPrior");

    defineEdgeCountPrior<FastMIDyNet::EdgeCountDeltaPrior>(m, "EdgeCountDeltaPrior")
        .def(pybind11::init<size_t>(), pybind11::arg("edge_count"));

    defineEdgeCountPrior<FastMIDyNet::EdgeCountPoissonPrior>(m, "EdgeCountPoissonPrior")
        .def(pybind11::init<double>(), pybind11::arg("mean"));
}

#endif
