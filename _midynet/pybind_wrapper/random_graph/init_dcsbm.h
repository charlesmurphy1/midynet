#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_DCSBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_DCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/random_graph/configuration.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDegreeCorrectedStochasticBlockModelFamily(py::module& m){
    py::class_<DegreeCorrectedStochasticBlockModelFamily, StochasticBlockModelFamily>(m, "DegreeCorrectedStochasticBlockModelFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, BlockPrior&, EdgeMatrixPrior&, DegreePrior&>(),
            py::arg("size"), py::arg("block_prior"), py::arg("edge_matrix_prior"), py::arg("degree_prior"))
        .def("get_degree_prior", &DegreeCorrectedStochasticBlockModelFamily::getDegreePrior)
        .def("set_degree_prior", &DegreeCorrectedStochasticBlockModelFamily::setDegreePrior)
        ;
}

void initConfigurationModelFamily(py::module& m){
    py::class_<ConfigurationModelFamily, DegreeCorrectedStochasticBlockModelFamily>(m, "ConfigurationModelFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&, DegreePrior&>(),
            py::arg("size"), py::arg("edge_count_prior"), py::arg("degree_prior"))
        .def("get_edge_count_prior", &ConfigurationModelFamily::getEdgeCountPrior)
        .def("set_edge_count_prior", &ConfigurationModelFamily::setEdgeCountPrior)
    ;
}

}

#endif
