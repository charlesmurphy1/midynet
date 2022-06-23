#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGEMATRIX_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGEMATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/python/edgematrix.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initEdgeMatrixPrior(py::module& m){
    py::class_<EdgeMatrixPrior, BlockLabeledPrior<MultiGraph>, PyEdgeMatrixPrior<>>(m, "EdgeMatrixPrior")
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"))
        .def("get_edge_count", &EdgeMatrixPrior::getEdgeCount)
        .def("get_edge_counts", &EdgeMatrixPrior::getEdgeCounts)
        .def("get_graph", &EdgeMatrixPrior::getGraph)
        .def("set_partition", &EdgeMatrixPrior::setPartition)
        .def("get_edge_count_prior", &EdgeMatrixPrior::getEdgeCountPrior)
        .def("set_edge_count_prior", &EdgeMatrixPrior::setEdgeCountPrior)
        .def("get_block_prior", &EdgeMatrixPrior::getBlockPrior)
        .def("set_block_prior", &EdgeMatrixPrior::setBlockPrior)
        ;


    py::class_<EdgeMatrixDeltaPrior, EdgeMatrixPrior>(m, "EdgeMatrixDeltaPrior")
        .def(py::init<>())
        .def(py::init<MultiGraph>(), py::arg("edge_matrix"))
        .def(py::init<MultiGraph, EdgeCountPrior&, BlockPrior&>(),
            py::arg("edge_matrix"), py::arg("edge_count_prior"), py::arg("block_prior"));

    py::class_<EdgeMatrixUniformPrior, EdgeMatrixPrior>(m, "EdgeMatrixUniformPrior")
        .def(py::init<>())
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"));

}

}

#endif
