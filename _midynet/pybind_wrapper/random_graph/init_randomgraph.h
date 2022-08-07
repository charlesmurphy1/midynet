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
        .def(py::init<size_t>(), py::arg("size"))
        .def("get_labels", &VertexLabeledRandomGraph<Label>::getLabels)
        .def("get_label_count", &VertexLabeledRandomGraph<Label>::getLabelCount)
        .def("get_vertex_counts", &VertexLabeledRandomGraph<Label>::getVertexCounts)
        .def("get_edge_label_counts", &VertexLabeledRandomGraph<Label>::getEdgeLabelCounts)
        .def("get_label_graph", &VertexLabeledRandomGraph<Label>::getLabelGraph)
        .def("get_label_of_idx", [](const VertexLabeledRandomGraph<Label>& self, BaseGraph::VertexIndex vertex) { return self.getLabelOfIdx(vertex); }, py::arg("vertex"))
        .def("set_labels", &VertexLabeledRandomGraph<Label>::setLabels, py::arg("labels"))
        .def("sample_labels", &VertexLabeledRandomGraph<Label>::sampleLabels)
        .def("get_log_likelihood_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogLikelihoodRatioFromLabelMove, py::arg("move"))
        .def("get_log_prior_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogPriorRatioFromLabelMove, py::arg("move"))
        .def("get_log_joint_ratio_from_label_move", &VertexLabeledRandomGraph<Label>::getLogJointRatioFromLabelMove, py::arg("move"))
        .def("apply_label_move", &VertexLabeledRandomGraph<Label>::applyLabelMove, py::arg("move"))
        .def("is_valid_label_move", &VertexLabeledRandomGraph<Label>::isValidLabelMove, py::arg("move"))
        ;
}

template<typename Label>
py::class_<NestedVertexLabeledRandomGraph<Label>, VertexLabeledRandomGraph<Label>, PyNestedVertexLabeledRandomGraph<Label>> declareNestedVertexLabeledRandomGraph(py::module& m, std::string pyName){
    return py::class_<NestedVertexLabeledRandomGraph<Label>, VertexLabeledRandomGraph<Label>, PyNestedVertexLabeledRandomGraph<Label>>(m, pyName.c_str())
        .def(py::init<size_t>(), py::arg("size"))
        .def("set_nested_labels", &NestedVertexLabeledRandomGraph<Label>::setNestedLabels, py::arg("nested_labels"))
        .def("get_depth", &NestedVertexLabeledRandomGraph<Label>::getDepth)
        .def("get_label_of_idx", [](const NestedVertexLabeledRandomGraph<Label>& self, BaseGraph::VertexIndex vertex, Level level) { return self.getLabelOfIdx(vertex, level); }, py::arg("vertex"), py::arg("level"))
        .def("get_nested_label_of_idx", &NestedVertexLabeledRandomGraph<Label>::getNestedLabelOfIdx, py::arg("vertex"), py::arg("level"))
        .def("get_nested_labels", [](const NestedVertexLabeledRandomGraph<Label>& self){ return self.getNestedLabels(); })
        .def("get_nested_labels", [](const NestedVertexLabeledRandomGraph<Label>& self, Level level){ return self.getNestedLabels(level); })
        .def("get_nested_label_count", [](const NestedVertexLabeledRandomGraph<Label>& self){ return self.getNestedLabelCount(); })
        .def("get_nested_label_count", [](const NestedVertexLabeledRandomGraph<Label>& self, Level level){ return self.getNestedLabelCount(level); })
        .def("get_nested_vertex_counts", [](const NestedVertexLabeledRandomGraph<Label>& self){ return self.getNestedVertexCounts(); })
        .def("get_nested_vertex_counts", [](const NestedVertexLabeledRandomGraph<Label>& self, Level level){ return self.getNestedVertexCounts(level); })
        .def("get_nested_edge_label_counts", [](const NestedVertexLabeledRandomGraph<Label>& self){ return self.getNestedEdgeLabelCounts(); })
        .def("get_nested_edge_label_counts", [](const NestedVertexLabeledRandomGraph<Label>& self, Level level){ return self.getNestedEdgeLabelCounts(level); })
        .def("get_nested_label_graph", [](const NestedVertexLabeledRandomGraph<Label>& self){ return self.getNestedLabelGraph(); })
        .def("get_nested_label_graph", [](const NestedVertexLabeledRandomGraph<Label>& self, Level level){ return self.getNestedLabelGraph(level); })
        ;
}

void initRandomGraphBaseClass(py::module& m){
    py::class_<RandomGraph, NestedRandomVariable, PyRandomGraph<>>(m, "RandomGraph")
        .def(py::init<size_t>(), py::arg("size"))
        .def("get_state", &RandomGraph::getState)
        .def("set_state", &RandomGraph::setState, py::arg("state"))
        .def("get_size", &RandomGraph::getSize)
        .def("set_size", &RandomGraph::setSize)
        .def("get_edge_count", &RandomGraph::getEdgeCount)
        .def("get_average_degree", &RandomGraph::getAverageDegree)
        .def("sample", &RandomGraph::sample)
        .def("sample_state", &RandomGraph::sampleState)
        .def("sample_prior", &RandomGraph::samplePrior)
        .def("get_log_likelihood", &RandomGraph::getLogLikelihood)
        .def("get_log_prior", &RandomGraph::getLogPrior)
        .def("get_log_joint", &RandomGraph::getLogJoint)
        .def("get_log_likelihood_ratio_from_graph_move", &RandomGraph::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
        .def("get_log_prior_ratio_from_graph_move", &RandomGraph::getLogPriorRatioFromGraphMove, py::arg("move"))
        .def("get_log_joint_ratio_from_graph_move", &RandomGraph::getLogJointRatioFromGraphMove, py::arg("move"))
        .def("apply_graph_move", &RandomGraph::applyGraphMove, py::arg("move"))
        .def("is_compatible", &RandomGraph::isCompatible)
        .def("is_valid_graph_move", &RandomGraph::isValidGraphMove, py::arg("move"))
        ;
    declareVertexLabeledRandomGraph<BlockIndex>(m, "BlockLabeledRandomGraph");
    declareNestedVertexLabeledRandomGraph<BlockIndex>(m, "NestedBlockLabeledRandomGraph");
}

}
#endif
