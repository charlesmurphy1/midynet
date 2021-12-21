#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/python/block.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initBlockPrior(py::module& m){
    py::class_<BlockPrior, Prior<std::vector<size_t>>, PyBlockPrior<>>(m, "BlockPrior")
        .def(py::init<size_t>(), py::arg("size"))
        .def("get_log_likelihood_ratio_from_graphmove", &BlockPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &BlockPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_graphmove", &BlockPrior::getLogPriorRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_prior_ratio_from_blockmove", &BlockPrior::getLogPriorRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &BlockPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &BlockPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &BlockPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &BlockPrior::applyBlockMove,
            py::arg("move"))
        .def("get_size", &BlockPrior::getSize)
        .def("get_block_count", &BlockPrior::getBlockCount)
        .def("get_vertex_count", &BlockPrior::getVertexCountsInBlocks)
        .def("get_block_of_idx", &BlockPrior::getBlockOfIdx);



    py::class_<BlockDeltaPrior, BlockPrior>(m, "BlockDeltaPrior")
        .def(py::init<const std::vector<size_t>&>(), py::arg("blocks"));

    py::class_<BlockUniformPrior, BlockPrior>(m, "BlockUniformPrior")
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"));

    py::class_<BlockHyperPrior, BlockPrior, PyBlockPrior<BlockHyperPrior>>(m, "BlockHyperPrior")
        .def(py::init<VertexCountPrior&>(), py::arg("vertex_count_prior"));

    py::class_<BlockUniformHyperPrior, BlockHyperPrior>(m, "BlockUniformHyperPrior")
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"));

}

}

#endif
