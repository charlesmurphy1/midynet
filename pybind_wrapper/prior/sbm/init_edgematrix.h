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
    py::class_<EdgeMatrixPrior, Prior<std::vector<std::vector<size_t>>>, PyEdgeMatrixPrior<>>(m, "EdgeMatrixPrior")
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"))
        .def("get_log_likelihood_ratio_from_graphmove", &EdgeMatrixPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &EdgeMatrixPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_graphmove", &EdgeMatrixPrior::getLogPriorRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_blockmove", &EdgeMatrixPrior::getLogPriorRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &EdgeMatrixPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &EdgeMatrixPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &EdgeMatrixPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &EdgeMatrixPrior::applyBlockMove,
            py::arg("move"))
        .def("get_block_count", &EdgeMatrixPrior::getBlockCount)
        .def("get_edge_count", &EdgeMatrixPrior::getEdgeCount)
        .def("get_edge_counts_in_blocks", &EdgeMatrixPrior::getEdgeCountsInBlocks)
        .def("get_blocks", &EdgeMatrixPrior::getBlocks)
        .def("get_graph", &EdgeMatrixPrior::getGraph);
        // .def("set_graph", &EdgeMatrixPrior::setGraph)


    py::class_<EdgeMatrixUniformPrior, EdgeMatrixPrior>(m, "EdgeMatrixUniformPrior")
        .def(py::init<EdgeCountPrior&, BlockPrior&>(), py::arg("edge_count_prior"), py::arg("block_prior"));

}

}

#endif
