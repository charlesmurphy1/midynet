#ifndef FAST_MIDYNET_PYWRAPPER_DATA_INIT_H
#define FAST_MIDYNET_PYWRAPPER_DATA_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/data/python/data_model.hpp"
#include "FastMIDyNet/data/data_model.hpp"
#include "FastMIDyNet/data/dynamics/dynamics.hpp"
#include "init_dynamics.h"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename GraphPriorType>
py::class_<DataModel<GraphPriorType>, NestedRandomVariable, PyDataModel<GraphPriorType>> declareDataModel(py::module& m, std::string pyName){
    return py::class_<DataModel<GraphPriorType>, NestedRandomVariable, PyDataModel<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<>())
        .def(py::init<GraphPriorType&>(), py::arg("graph_prior"))
        .def("get_size", &DataModel<GraphPriorType>::getSize)
        .def("get_graph", &DataModel<GraphPriorType>::getGraph)
        .def("set_graph", &DataModel<GraphPriorType>::setGraph, py::arg("graph"))
        .def("set_graph_prior", &DataModel<GraphPriorType>::setGraphPrior)
        .def("get_graph_prior", &DataModel<GraphPriorType>::getGraphPrior)
        .def("sample", &DataModel<GraphPriorType>::sample)
        .def("sample_state", &DataModel<GraphPriorType>::sampleState)
        .def("sample_prior", &DataModel<GraphPriorType>::samplePrior)
        .def("get_log_likelihood", &DataModel<GraphPriorType>::getLogLikelihood)
        .def("get_log_prior", &DataModel<GraphPriorType>::getLogPrior)
        .def("get_log_joint", &DataModel<GraphPriorType>::getLogJoint)
        .def("get_log_likelihood_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogPriorRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("apply_graph_move", &DataModel<GraphPriorType>::applyGraphMove,
            py::arg("move"))
        ;
}

template<typename GraphPriorType>
py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>> declareDynamics(py::module& m, std::string pyName){
    return py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<GraphPriorType&, size_t, size_t, bool, bool>())
        ;
}



void initDataModels(py::module& m){
    declareDataModel<RandomGraph>(m, "DataModel");
    declareDataModel<BlockLabeledRandomGraph>(m, "BlockLabeledDataModel");
    declareDataModel<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDataModel");

    py::module dynamics = m.def_submodule("dynamics");
    initDynamics(dynamics);
}


}

#endif
