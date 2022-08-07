#ifndef FAST_MIDYNET_PYWRAPPER_INIT_CONFIGURATION_H
#define FAST_MIDYNET_PYWRAPPER_INIT_CONFIGURATION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/configuration.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initConfiguration(py::module& m){
    py::class_<ConfigurationModelBase, RandomGraph>(m, "ConfigurationModelBase")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, DegreePrior&>(), py::arg("size"), py::arg("degree_prior"))
        .def("get_degree_prior", &ConfigurationModelBase::getDegreePrior)
        .def("set_degree_prior", &ConfigurationModelBase::setDegreePrior, py::arg("prior"))
        ;

    py::class_<ConfigurationModel, ConfigurationModelBase>(m, "ConfigurationModel")
        .def( py::init<std::vector<size_t>>(), py::arg("degrees") )
        ;

    py::class_<ConfigurationModelFamily, ConfigurationModelBase>(m, "ConfigurationModelFamily")
        .def(
            py::init<size_t, double, bool, bool>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("hyperprior"),
            py::arg("canonical")
        )
        ;
}

}

#endif
