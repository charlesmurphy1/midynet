#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/random_graph/python/randomgraph.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initRandomGraphBaseClass(py::module& m){
    py::class_<RandomGraph, NestedRandomVariable, PyRandomGraph<>>(m, "RandomGraph")
        .def(py::init<size_t>(), py::arg("size")=0)
        .def("get_graph", &RandomGraph::getGraph)
        .def("set_graph", &RandomGraph::setGraph, py::arg("graph"))
        .def("get_size", &RandomGraph::getSize)
        .def("get_blocks", &RandomGraph::getBlocks)
        .def("get_block_of_idx", &RandomGraph::getBlockOfIdx, py::arg("vertex_idx"))
        .def("get_block_count", &RandomGraph::getBlockCount)
        .def("get_vertex_counts", &RandomGraph::getVertexCountsInBlocks)
        .def("get_edge_matrix", &RandomGraph::getEdgeMatrix)
        .def("get_edge_counts", &RandomGraph::getEdgeCountsInBlocks)
        .def("get_edge_count", &RandomGraph::getEdgeCount)
        .def("get_average_degree", &RandomGraph::getAverageDegree)
        .def("get_degrees", &RandomGraph::getDegrees)
        .def("get_degree_of_idx", &RandomGraph::getDegreeOfIdx, py::arg("vertex_idx"))
        .def("get_degree_counts", &RandomGraph::getDegreeCountsInBlocks)
        .def("compute_block_count", &RandomGraph::computeBlockCount)
        .def("compute_vertex_counts", &RandomGraph::computeVertexCountsInBlocks)
        .def("compute_edge_matrix", &RandomGraph::computeEdgeMatrix)
        .def("compute_edge_counts", &RandomGraph::computeEdgeCountsInBlocks)
        .def("compute_degree_counts", &RandomGraph::computeDegreeCountsInBlocks)
        .def("sample", &RandomGraph::sample)
        .def("sample_graph", &RandomGraph::sampleGraph)
        .def("get_log_likelihood", &RandomGraph::getLogLikelihood)
        .def("get_log_prior", &RandomGraph::getLogPrior)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likelihood_ratio_from_graph_move", &RandomGraph::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
        .def("get_log_likelihood_ratio_from_block_move", &RandomGraph::getLogLikelihoodRatioFromBlockMove, py::arg("move"))
        .def("get_log_prior_ratio_from_graph_move", &RandomGraph::getLogPriorRatioFromGraphMove, py::arg("move"))
        .def("get_log_prior_ratio_from_block_move", &RandomGraph::getLogPriorRatioFromBlockMove, py::arg("move"))
        .def("get_log_joint_ratio_from_graph_move", &RandomGraph::getLogJointRatioFromGraphMove, py::arg("move"))
        .def("get_log_joint_ratio_from_block_move", &RandomGraph::getLogJointRatioFromBlockMove, py::arg("move"))
        .def("apply_graph_move", &RandomGraph::applyGraphMove, py::arg("move"))
        .def("apply_block_move", &RandomGraph::applyBlockMove, py::arg("move"))
        .def("is_compatible", &RandomGraph::isCompatible)
        ;
}

}
#endif
