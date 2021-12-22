#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "declare.h"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/python/block.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

void initBlockPrior(py::module& m){
    declareSBMPrior<BlockPrior, Prior<std::vector<size_t>>, PyBlockPrior<>>(m, "BlockPrior")
        .def(py::init<size_t>(), py::arg("size"))
        .def("get_size", &BlockPrior::getSize)
        .def("set_size", &BlockPrior::setSize)
        .def("get_block_count", &BlockPrior::getBlockCount)
        .def("get_vertex_count", &BlockPrior::getVertexCountsInBlocks)
        .def("get_block_of_idx", &BlockPrior::getBlockOfIdx)
        ;

    py::class_<BlockDeltaPrior, BlockPrior>(m, "BlockDeltaPrior")
        .def(py::init<const std::vector<size_t>&>(), py::arg("blocks"));

    py::class_<BlockUniformPrior, BlockPrior>(m, "BlockUniformPrior")
        .def(py::init<>())
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        .def("get_block_count_prior", &BlockUniformPrior::getBlockCountPrior)
        .def("set_block_count_prior", &BlockUniformPrior::setBlockCountPrior)
        ;

    py::class_<BlockHyperPrior, BlockPrior, PyBlockPrior<BlockHyperPrior>>(m, "BlockHyperPrior")
        .def(py::init<VertexCountPrior&>(), py::arg("vertex_count_prior"))
        .def("get_vertex_count_prior", &BlockHyperPrior::getVertexCountPrior)
        .def("set_vertex_count_prior", &BlockHyperPrior::setVertexCountPrior)
        ;

    py::class_<BlockUniformHyperPrior, BlockHyperPrior>(m, "BlockUniformHyperPrior")
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        .def("get_block_count_prior", &BlockUniformHyperPrior::getBlockCountPrior)
        .def("set_block_count_prior", &BlockUniformHyperPrior::setBlockCountPrior)
        ;

}

}

#endif
