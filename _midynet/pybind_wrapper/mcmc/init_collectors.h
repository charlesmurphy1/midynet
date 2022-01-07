#ifndef FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_COLLECTORS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/mcmc/callbacks/collector.h"
#include "FastMIDyNet/mcmc/python/callback.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initCollectors(py::module& m){
    /* Collect base classes */
    py::class_<Collector, CallBack, PyCollector<>>(m, "Collector")
        .def(py::init<>())
        .def("collect", &Collector::collect)
        .def("clear", &Collector::clear);

    py::class_<SweepCollector, Collector, PyCollector<SweepCollector>>(m, "SweepCollector")
        .def(py::init<>());

    py::class_<StepCollector, Collector, PyCollector<StepCollector>>(m, "StepCollector")
        .def(py::init<>());

    /* Graph collector classes */
    py::class_<CollectGraphOnSweep, Collector>(m, "CollectGraphOnSweep")
        .def(py::init<>())
        .def("get_graphs", &CollectGraphOnSweep::getGraphs);

    py::class_<CollectEdgeMultiplicityOnSweep, Collector>(m, "CollectEdgeMultiplicityOnSweep")
        .def(py::init<>())
        .def("get_edge_multiplicity", &CollectEdgeMultiplicityOnSweep::getEdgeMultiplicity);

    py::class_<WriteGraphToFileOnSweep, Collector>(m, "WriteGraphToFileOnSweep")
        .def(py::init<std::string, std::string>(), py::arg("filename"), py::arg("ext")=".b");

    /* Metrics collector classes */
    py::class_<CollectLikelihoodOnSweep, Collector>(m, "CollectLikelihoodOnSweep")
        .def(py::init<>())
        .def("get_log_likelihoods", &CollectLikelihoodOnSweep::getLogLikelihoods);

    py::class_<CollectPriorOnSweep, Collector>(m, "CollectPriorOnSweep")
        .def(py::init<>())
        .def("get_log_priors", &CollectPriorOnSweep::getLogPriors);

    py::class_<CollectJointOnSweep, Collector>(m, "CollectJointOnSweep")
        .def(py::init<>())
        .def("get_log_joints", &CollectJointOnSweep::getLogJoints);
}

}

#endif
