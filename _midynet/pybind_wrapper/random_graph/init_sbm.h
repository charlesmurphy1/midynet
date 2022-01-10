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
        .def("get_block_of_idx", &StochasticBlockModelFamily::getBlockOfIdx,
            py::arg("idx"))
        .def("get_blocks", &StochasticBlockModelFamily::getBlocks)
        .def("get_block_count", &StochasticBlockModelFamily::getBlockCount)
        .def("get_vertex_count_in_blocks", &StochasticBlockModelFamily::getVertexCountsInBlocks)
        .def("get_edge_count", &StochasticBlockModelFamily::getEdgeCount)
        .def("get_edge_count_in_blocks", &StochasticBlockModelFamily::getEdgeCountsInBlocks)
        .def("get_edge_matrix", &StochasticBlockModelFamily::getEdgeMatrix)
        .def("get_log_likehood_ratio", py::overload_cast<const BlockMove&>(&StochasticBlockModelFamily::getLogLikelihoodRatio),
            py::arg("move"))
        .def("get_log_prior_ratio", py::overload_cast<const BlockMove&>(&StochasticBlockModelFamily::getLogPriorRatio),
            py::arg("move"))
        .def("get_log_joint_ratio", py::overload_cast<const BlockMove&>(&StochasticBlockModelFamily::getLogJointRatio),
            py::arg("move"))
        .def("apply_move", py::overload_cast<const BlockMove&>(&StochasticBlockModelFamily::applyMove),
            py::arg("move"))
        ;
}

void initErdosRenyiFamily(py::module& m){
    py::class_<ErdosRenyiFamily, StochasticBlockModelFamily>(m, "ErdosRenyiFamily")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_cout_prior"))
    ;
}

}

#endif
