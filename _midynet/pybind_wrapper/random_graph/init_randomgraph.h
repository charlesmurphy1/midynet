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
        .def(py::init<size_t>(), py::arg("size")=0)
        .def("get_state", &RandomGraph::getState)
        .def("set_state", &RandomGraph::setState, py::arg("state"))
        .def("get_size", &RandomGraph::getSize)
        .def("get_blocks", &RandomGraph::getBlocks)
        .def("get_block_of_idx", &RandomGraph::getBlockOfIdx, py::arg("vertex_idx"))
        .def("get_block_count", &RandomGraph::getBlockCount)
        .def("get_vertex_counts", &RandomGraph::getVertexCountsInBlocks)
        .def("get_edge_matrix", &RandomGraph::getEdgeMatrix)
        .def("get_edge_counts", &RandomGraph::getEdgeCountsInBlocks)
        .def("get_edge_count", &RandomGraph::getEdgeCount)
        .def("get_degrees", &RandomGraph::getDegrees)
        .def("get_degree_of_idx", &RandomGraph::getDegreeOfIdx, py::arg("vertex_idx"))
        .def("get_degree_counts", &RandomGraph::getDegreeCountsInBlocks)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likelihood", &RandomGraph::getLogLikelihood)
        .def("get_log_prior", &RandomGraph::getLogPrior)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likehood_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogLikelihoodRatio), py::arg("move"))
        .def("get_log_prior_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogPriorRatio), py::arg("move"))
        .def("get_log_joint_ratio", py::overload_cast<const GraphMove&>(&RandomGraph::getLogJointRatio), py::arg("move"))
        .def("apply_move", py::overload_cast<const GraphMove&>(&RandomGraph::applyMove), py::arg("move"))
        .def("sample", &RandomGraph::sample)
        .def("check_self_consistency", &RandomGraph::checkSelfConsistency)
        .def("check_safety", &RandomGraph::checkSafety)
        ;
}

}
#endif
