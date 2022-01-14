#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_DEGREE_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "declare.h"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/prior/sbm/python/degree.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initDegreePrior(py::module& m){
    declareSBMPrior<DegreePrior, Prior<std::vector<size_t>>, PyDegreePrior<>>(m, "DegreePrior")
        .def(py::init<BlockPrior&, EdgeMatrixPrior&>(), py::arg("block_prior"), py::arg("edge_matrix"))
        .def("get_degree_of_idx", &DegreePrior::getDegreeOfIdx)
        .def("get_degree_count_in_blocks", &DegreePrior::getDegreeCountsInBlocks)
        .def("get_graph", &DegreePrior::getGraph)
        .def("get_block_prior", &DegreePrior::getBlockPrior)
        .def("set_block_prior", &DegreePrior::setBlockPrior, py::arg("block_prior"))
        .def("get_edge_matrix_prior", &DegreePrior::getEdgeMatrixPrior)
        .def("set_edge_matrix_prior", &DegreePrior::setEdgeMatrixPrior, py::arg("edge_matrix_prior"))
        ;

    py::class_<DegreeDeltaPrior, DegreePrior>(m, "DegreeDeltaPrior")
        .def(py::init<>())
        .def(py::init<const DegreeSequence&, BlockPrior&, EdgeMatrixPrior&>(),
            py::arg("degrees"), py::arg("block_prior"), py::arg("edge_matrix"))
        .def("set_degrees", &DegreeDeltaPrior::setDegrees)
        ;

    py::class_<DegreeUniformPrior, DegreePrior>(m, "DegreeUniformPrior")
        .def(py::init<>())
        .def(py::init<BlockPrior&, EdgeMatrixPrior&>(), py::arg("block_prior"), py::arg("edge_matrix"));


}

}

#endif
