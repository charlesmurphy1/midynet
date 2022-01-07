#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/random_graph/python/randomgraph.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initRandomGraphBaseClass(py::module& m){
    py::class_<RandomGraph, PyRandomGraph<>>(m, "RandomGraph")
        .def(py::init<size_t>(), py::arg("size"))
        .def("get_state", &RandomGraph::getState)
        .def("set_state", &RandomGraph::setState, py::arg("state"))
        .def("get_size", &RandomGraph::getSize)
        .def("get_log_likehood", &RandomGraph::getLogLikelihood)
        .def("get_log_prior", &RandomGraph::getLogPrior)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likehood_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogLikelihoodRatio), py::arg("move"))
        .def("get_log_prior_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogPriorRatio), py::arg("move"))
        .def("get_log_joint_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogJointRatio), py::arg("move"))
        .def("apply_move", py::overload_cast<const GraphMove&>(&RandomGraph::applyMove), py::arg("move"))
        .def("sample", &RandomGraph::sample)
        .def("check_self_consistency", &RandomGraph::checkSelfConsistency)
        ;
}

}
#endif
