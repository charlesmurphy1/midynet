#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCKCOUNT_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "declare.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/python/blockcount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{


void initBlockCountPrior(py::module& m){
    declareSBMPrior<BlockCountPrior, Prior<size_t>, PyBlockCountPrior<>>(m, "BlockCountPrior");

    py::class_<BlockCountDeltaPrior, BlockCountPrior>(m, "BlockCountDeltaPrior")
        .def(py::init<size_t>(), py::arg("block_count"));

    py::class_<BlockCountPoissonPrior, BlockCountPrior>(m, "BlockCountPoissonPrior")
        .def(py::init<double>(), py::arg("mean"))
        .def("get_mean", &BlockCountPoissonPrior::getMean)
        .def("set_mean", &BlockCountPoissonPrior::setMean, py::arg("mean"))
        ;
}

}

#endif
