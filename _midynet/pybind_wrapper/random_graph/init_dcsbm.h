#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_DCSBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_DCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/random_graph/python/sbm.hpp"
#include "FastMIDyNet/random_graph/configuration.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDegreeCorrectedStochasticBlockModelFamily(py::module& m){
    py::class_<DegreeCorrectedStochasticBlockModelFamily, StochasticBlockModelFamily>(m, "DegreeCorrectedStochasticBlockModelFamily")
        .def(py::init<BlockPrior&, EdgeMatrixPrior&, DegreePrior&>(), py::arg("block_prior"), py::arg("edge_matrix_prior"), py::arg("degree_prior"))
        .def("get_degree_of_idx", &DegreeCorrectedStochasticBlockModelFamily::getDegreeOfIdx,
            py::arg("idx"))
        .def("get_degrees", &DegreeCorrectedStochasticBlockModelFamily::getDegrees)
        .def("get_block_count", &DegreeCorrectedStochasticBlockModelFamily::getBlockCount)
        .def("get_degree_count_in_blocks", &DegreeCorrectedStochasticBlockModelFamily::getDegreeCountsInBlocks);
}

void initConfigurationModelFamily(py::module& m){
    py::class_<ConfigurationModelFamily, DegreeCorrectedStochasticBlockModelFamily>(m, "ConfigurationModelFamily")
        .def(py::init<DegreePrior&>(), py::arg("degree_prior"))
    ;
}

}

#endif
