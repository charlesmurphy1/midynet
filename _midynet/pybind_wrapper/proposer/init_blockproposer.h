#ifndef FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initBlockProposer(py::module& m){
    py::class_<BlockProposer, Proposer<BlockMove>, PyBlockProposer<>>(m, "BlockProposer")
        .def(py::init<>())
        .def("set_up", &BlockProposer::setUp, py::arg("random_graph"));

    py::class_<UniformBlockProposer, BlockProposer>(m, "UniformBlockProposer")
        .def(py::init<double>(), py::arg("create_new_block")=0.1) ;

    py::class_<PeixotoBlockProposer, BlockProposer>(m, "PeixotoBlockProposer")
        .def(py::init<double,double>(), py::arg("create_new_block")=0.1, py::arg("shift")=1) ;

}

}

#endif
