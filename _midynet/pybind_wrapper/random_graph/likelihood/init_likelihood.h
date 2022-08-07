#ifndef FAST_MIDYNET_PYWRAPPER_INIT_LIKELIHOOD_H
#define FAST_MIDYNET_PYWRAPPER_INIT_LIKELIHOOD_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/random_graph/likelihood/python/likelihood.hpp"

namespace py = pybind11;
namespace FastMIDyNet{


// py::class_<GraphLikelihoodModel, NestedRandomVariable, PyGraphLikelihoodModel<>> declareGraphLikelihoodBaseClass(py::module& m, std::string pyName){
//     return py::class_<GraphLikelihoodModel, NestedRandomVariable, PyGraphLikelihoodModel<>>(m, pyName.c_str())
//         .def("get_log_likelihood", &GraphLikelihoodModel::getLogLikelihood)
//         .def("get_log_likelihood_ratio_from_graph_move", &GraphLikelihoodModel::setState, py::arg("move"))
//         .def("sample", &GraphLikelihoodModel::sample)
//         ;
// }

template<typename Label>
py::class_<VertexLabeledGraphLikelihoodModel<Label>, GraphLikelihoodModel, PyVertexLabeledGraphLikelihoodModel<Label>> declareVertexLabeledGraphLikelihoodBaseClass(py::module& m, std::string pyName){
    return py::class_<VertexLabeledGraphLikelihoodModel<Label>, GraphLikelihoodModel, PyVertexLabeledGraphLikelihoodModel<Label>>(m, pyName.c_str())
        .def("get_log_likelihood_ratio_from_label_move", &VertexLabeledGraphLikelihoodModel<Label>::getLogLikelihoodRatioFromLabelMove, py::arg("move"))
        ;
}

void initGraphLikelihoods(py::module& m){
    py::class_<GraphLikelihoodModel, NestedRandomVariable, PyGraphLikelihoodModel<>>(m, "GraphLikelihoodModel")
        .def("get_log_likelihood", &GraphLikelihoodModel::getLogLikelihood)
        .def("get_log_likelihood_ratio_from_graph_move", &GraphLikelihoodModel::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
        .def("sample", &GraphLikelihoodModel::sample)
        ;

    declareVertexLabeledGraphLikelihoodBaseClass<BlockIndex>(m, "BlockLabeledGraphLikelihoodBaseClass");

}

}


#endif
