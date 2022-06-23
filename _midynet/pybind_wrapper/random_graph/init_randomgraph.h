#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/python/randomgraph.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label>
py::class_<VertexLabeledRandomGraph<Label>, RandomGraph, PyVertexLabeledRandomGraph<Label>> declareVertexLabeledRandomGraph(py::module& m, std::string pyName){
    return py::class_<VertexLabeledRandomGraph<Label>, RandomGraph, PyVertexLabeledRandomGraph<Label>>(m, pyName.c_str())
        .def(py::init<size_t>(), py::arg("size")=0)
        .def("sample_labels", &VertexLabeledRandomGraph<Label>::sampleLabels)
        .def("set_labels", &VertexLabeledRandomGraph<Label>::setLabels, py::arg("labels"))
        .def("get_vertex_labels", &VertexLabeledRandomGraph<Label>::getVertexLabels)
        .def("get_label_counts", &VertexLabeledRandomGraph<Label>::getLabelCounts)
        .def("get_edge_label_counts", &VertexLabeledRandomGraph<Label>::getEdgeLabelCounts)
        .def("get_label_graph", &VertexLabeledRandomGraph<Label>::getLabelGraph)
        .def("get_label_of_idx", &VertexLabeledRandomGraph<Label>::getLabelOfIdx, py::arg("vertex"))
        .def("get_log_likelihood_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogLikelihoodRatioFromLabelMove, py::arg("move"))
        .def("get_log_prior_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogPriorRatioFromLabelMove, py::arg("move"))
        .def("get_log_joint_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogJointRatioFromLabelMove, py::arg("move"))
        .def("apply_label_move", &VertexLabeledRandomGraph<Label>::applyLabelMove, py::arg("move"))
        ;
}

void initRandomGraphBaseClass(py::module& m){
    py::class_<RandomGraph, NestedRandomVariable, PyRandomGraph<>>(m, "RandomGraph")
        .def(py::init<size_t>(), py::arg("size")=0)
        .def("get_graph", &RandomGraph::getGraph)
        .def("set_graph", &RandomGraph::setGraph, py::arg("graph"))
        .def("get_size", &RandomGraph::getSize)
        .def("set_size", &RandomGraph::setSize)
        .def("get_edge_count", &RandomGraph::getEdgeCount)
        .def("get_average_degree", &RandomGraph::getAverageDegree)
        .def("sample", &RandomGraph::sample)
        .def("get_log_likelihood", &RandomGraph::getLogLikelihood)
        .def("get_log_prior", &RandomGraph::getLogPrior)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likelihood_ratio_from_graph_move", &RandomGraph::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
        .def("get_log_prior_ratio_from_graph_move", &RandomGraph::getLogPriorRatioFromGraphMove, py::arg("move"))
        .def("get_log_joint_ratio_from_graph_move", &RandomGraph::getLogJointRatioFromGraphMove, py::arg("move"))
        .def("apply_graph_move", &RandomGraph::applyGraphMove, py::arg("move"))
        .def("is_compatible", &RandomGraph::isCompatible)
        ;
    declareVertexLabeledRandomGraph<BlockIndex>(m, "BlockLabeledRandomGraph");
}

}
#endif
