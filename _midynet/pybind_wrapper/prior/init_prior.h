#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PRIOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "sbm/init.h"


namespace py = pybind11;
namespace FastMIDyNet{


template <typename StateType>
py::class_<Prior<StateType>, NestedRandomVariable, PyPrior<StateType>> declarePriorBaseClass(py::module& m, std::string pyName){
    return py::class_<Prior<StateType>, NestedRandomVariable, PyPrior<StateType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("get_state", &Prior<StateType>::getState)
        .def("set_state", &Prior<StateType>::setState, py::arg("state"))
        .def("sample_state", &Prior<StateType>::sampleState)
        .def("sample_priors", &Prior<StateType>::samplePriors)
        .def("sample", &Prior<StateType>::sample)
        .def("get_log_likelihood", &Prior<StateType>::getLogLikelihood)
        .def("get_log_prior", &Prior<StateType>::getLogPrior)
        .def("get_log_joint", &Prior<StateType>::getLogJoint)
        .def("get_log_joint_ratio_from_graph_move", &Prior<StateType>::getLogJointRatioFromGraphMove, py::arg("move"))
        .def("get_log_joint_ratio_from_block_move", &Prior<StateType>::getLogJointRatioFromBlockMove, py::arg("move"))
        .def("apply_graph_move", &Prior<StateType>::applyGraphMove, py::arg("move"))
        .def("apply_block_move", &Prior<StateType>::applyBlockMove, py::arg("move"))
        ;
}

void initPrior(pybind11::module& m){
    declarePriorBaseClass<size_t>(m, "UnIntPrior");
    declarePriorBaseClass<std::vector<size_t>>(m, "UnIntVectorPrior");
    declarePriorBaseClass<MultiGraph>(m, "MultigraphPrior");
    pybind11::module sbm = m.def_submodule("sbm");
    initSBMPriors(sbm);
}

}

#endif
