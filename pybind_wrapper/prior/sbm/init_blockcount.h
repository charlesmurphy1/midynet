#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCKCOUNT_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/python/blockcount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initBlockCountPrior(py::module& m){
    py::class_<BlockCountPrior, Prior<size_t>, PyBlockCountPrior<>>(m, "BlockCountPrior")
        .def(py::init<>())
        .def("get_log_likelihood_ratio_from_graphmove", &BlockCountPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &BlockCountPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &BlockCountPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &BlockCountPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &BlockCountPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &BlockCountPrior::applyBlockMove,
            py::arg("move"));

    py::class_<BlockCountDeltaPrior, BlockCountPrior>(m, "BlockCountDeltaPrior")
        .def(py::init<size_t>(), py::arg("block_count"));

    py::class_<BlockCountPoissonPrior, BlockCountPrior>(m, "BlockCountPoissonPrior")
        .def(py::init<double>(), py::arg("mean"));
}

}

#endif
