#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_SBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_SBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/random_graph/python/sbm.hpp"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/random_graph/sbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initStochasticBlockModelFamily(py::module& m){
    py::class_<StochasticBlockModelFamily, RandomGraph, PyStochasticBlockModelFamily<>>(m, "StochasticBlockModelFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, BlockPrior&, EdgeMatrixPrior&>(), py::arg("size"), py::arg("blocks"), py::arg("edge_matrix"))
        .def("sample_blocks", &StochasticBlockModelFamily::sampleBlocks)
        .def("set_blocks", &StochasticBlockModelFamily::setBlocks, py::arg("blocks"))
        .def("get_block_of_idx", &StochasticBlockModelFamily::getBlockOfIdx,
            py::arg("idx"))
        .def("get_block_prior", &StochasticBlockModelFamily::getBlockPrior)
        .def("set_block_prior", &StochasticBlockModelFamily::setBlockPrior)
        .def("get_edge_matrix_prior", &StochasticBlockModelFamily::getEdgeMatrixPrior)
        .def("set_edge_matrix_prior", &StochasticBlockModelFamily::setEdgeMatrixPrior)
        ;
}

void initErdosRenyiFamily(py::module& m){
    py::class_<ErdosRenyiFamily, StochasticBlockModelFamily>(m, "ErdosRenyiFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_cout_prior"))
        .def("get_edge_count_prior", &ErdosRenyiFamily::getEdgeCountPrior)
        .def("set_edge_count_prior", &ErdosRenyiFamily::setEdgeCountPrior)
    ;

    py::class_<SimpleErdosRenyiFamily, RandomGraph>(m, "SimpleErdosRenyiFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_cout_prior"))
        .def("get_edge_count_prior", &SimpleErdosRenyiFamily::getEdgeCountPrior)
        .def("set_edge_count_prior", &SimpleErdosRenyiFamily::setEdgeCountPrior)
    ;
}

}

#endif
