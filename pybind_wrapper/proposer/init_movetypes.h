#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initMoveTypes(py::module& m){
    py::class_<GraphMove>(m, "GraphMove")
        .def(py::init<std::vector<BaseGraph::Edge>, std::vector<BaseGraph::Edge>>(),
            py::arg("removed_edges"), py::arg("added_edges"))
        .def_readonly("removed_edges", &GraphMove::removedEdges)
        .def_readonly("added_edges", &GraphMove::addedEdges);

    py::class_<BlockMove>(m, "BlockMove")
        .def(py::init<BaseGraph::VertexIndex, BlockIndex, BlockIndex, int>(),
            py::arg("vertex_id"), py::arg("prev_block_id"), py::arg("next_block_id"), py::arg("added_blocks")=0)
        .def_readonly("vertex_id", &BlockMove::vertexIdx)
        .def_readonly("prev_block_id", &BlockMove::prevBlockIdx)
        .def_readonly("next_block_id", &BlockMove::nextBlockIdx)
        .def_readonly("added_blocks", &BlockMove::addedBlocks);

}

}

#endif
