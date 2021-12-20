#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"


void initMoveTypes(pybind11::module& m){
    pybind11::class_<FastMIDyNet::GraphMove>(m, "GraphMove")
        .def(pybind11::init<std::vector<BaseGraph::Edge>, std::vector<BaseGraph::Edge>>(), pybind11::arg("removed_edges"), pybind11::arg("added_edges"))
        .def_readonly("removed_edges", &FastMIDyNet::GraphMove::removedEdges)
        .def_readonly("added_edges", &FastMIDyNet::GraphMove::addedEdges);

    pybind11::class_<FastMIDyNet::BlockMove>(m, "BlockMove")
        .def(pybind11::init<BaseGraph::VertexIndex, FastMIDyNet::BlockIndex, FastMIDyNet::BlockIndex, int>(), pybind11::arg("vertex_id"), pybind11::arg("prev_block_id"), pybind11::arg("next_block_id"), pybind11::arg("added_blocks")=0)
        .def_readonly("vertex_id", &FastMIDyNet::BlockMove::vertexIdx)
        .def_readonly("prev_block_id", &FastMIDyNet::BlockMove::prevBlockIdx)
        .def_readonly("next_block_id", &FastMIDyNet::BlockMove::nextBlockIdx)
        .def_readonly("added_blocks", &FastMIDyNet::BlockMove::addedBlocks);

}

#endif
