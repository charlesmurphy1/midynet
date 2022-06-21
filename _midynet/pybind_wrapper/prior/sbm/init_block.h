#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/python/block.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initBlockPrior(py::module& m){
    py::class_<BlockPrior, BlockLabeledPrior<std::vector<size_t>>, PyBlockPrior<>>(m, "BlockPrior")
        .def(py::init<>())
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        .def("get_size", &BlockPrior::getSize)
        .def("set_size", &BlockPrior::setSize)
        .def("get_block_count", &BlockPrior::getBlockCount)
        .def("get_effective_block_count", &BlockPrior::getEffectiveBlockCount)
        .def("get_block_count_prior", &BlockPrior::getBlockCountPrior)
        .def("get_block_count_prior_ref", &BlockPrior::getBlockCountPriorRef)
        .def("set_block_count_prior", &BlockPrior::setBlockCountPrior)
        .def("get_vertex_counts", &BlockPrior::getVertexCounts)
        .def("get_block_of_idx", &BlockPrior::getBlockOfIdx)
        ;

    py::class_<BlockDeltaPrior, BlockPrior>(m, "BlockDeltaPrior")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>(), py::arg("blocks"));

    py::class_<BlockUniformPrior, BlockPrior>(m, "BlockUniformPrior")
        .def(py::init<>())
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        ;

    py::class_<BlockUniformHyperPrior, BlockPrior>(m, "BlockUniformHyperPrior")
        .def(py::init<>())
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        ;

}

}

#endif
