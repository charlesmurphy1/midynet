#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGEMATRIX_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGEMATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "declare.h"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/python/edgematrix.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initEdgeMatrixPrior(py::module& m){
    declareSBMPrior<EdgeMatrixPrior, Prior<std::vector<std::vector<size_t>>>, PyEdgeMatrixPrior<>>(m, "EdgeMatrixPrior")
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"))
        .def("get_edge_count", &EdgeMatrixPrior::getEdgeCount)
        .def("get_edge_counts_in_blocks", &EdgeMatrixPrior::getEdgeCountsInBlocks)
        .def("get_graph", &EdgeMatrixPrior::getGraph)
        .def("get_edge_count_prior", &EdgeMatrixPrior::getEdgeCountPrior)
        .def("set_edge_count_prior", &EdgeMatrixPrior::setEdgeCountPrior)
        .def("get_block_prior", &EdgeMatrixPrior::getBlockPrior)
        .def("set_block_prior", &EdgeMatrixPrior::setBlockPrior)
        ;


    py::class_<EdgeMatrixUniformPrior, EdgeMatrixPrior>(m, "EdgeMatrixUniformPrior")
        .def(py::init<>())
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"));

}

}

#endif
