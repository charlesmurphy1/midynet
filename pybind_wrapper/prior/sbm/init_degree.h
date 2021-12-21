#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_DEGREE_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/prior/sbm/python/degree.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initDegreePrior(py::module& m){
    py::class_<DegreePrior, Prior<std::vector<size_t>>, PyDegreePrior<>>(m, "DegreePrior")
        .def(py::init<BlockPrior&, EdgeMatrixPrior&>(), py::arg("block_prior"), py::arg("edge_matrix"))
        .def("get_log_likelihood_ratio_from_graphmove", &DegreePrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &DegreePrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_graphmove", &DegreePrior::getLogPriorRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_blockmove", &DegreePrior::getLogPriorRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &DegreePrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &DegreePrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &DegreePrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &DegreePrior::applyBlockMove,
            py::arg("move"))
        .def("get_size", &DegreePrior::getSize)
        .def("get_block_of_idx", &DegreePrior::getBlockOfIdx)
        .def("get_blocks", &DegreePrior::getBlocks)
        .def("get_degree_of_idx", &DegreePrior::getDegreeOfIdx)
        .def("get_block_count", &DegreePrior::getBlockCount)
        .def("get_edge_count", &DegreePrior::getEdgeCount)
        .def("get_edge_count_in_blocks", &DegreePrior::getEdgeCountsInBlocks)
        .def("get_vertex_count_in_blocks", &DegreePrior::getVertexCountsInBlocks)
        .def("get_degree_count_in_blocks", &DegreePrior::getDegreeCountsInBlocks)
        .def("get_graph", &DegreePrior::getGraph)
        ;

    py::class_<DegreeUniformPrior, DegreePrior>(m, "DegreeUniformPrior")
        .def(py::init<BlockPrior&, EdgeMatrixPrior&>(), py::arg("block_prior"), py::arg("edge_matrix"));


}

}

#endif
