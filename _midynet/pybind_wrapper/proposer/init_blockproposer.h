#ifndef FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/block_proposer/generic.h"
#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initBlockProposer(py::module& m){
    py::class_<BlockProposer, Proposer<BlockMove>, PyBlockProposer<>>(m, "BlockProposer")
        .def(py::init<>())
        .def("set_up", &BlockProposer::setUp, py::arg("random_graph"))
        .def("get_log_proposal_prob_ratio", &BlockProposer::getLogProposalProbRatio, py::arg("move"))
        .def("apply_graph_move", &BlockProposer::applyGraphMove, py::arg("move"))
        .def("apply_block_move", &BlockProposer::applyBlockMove, py::arg("move"));

    py::class_<BlockGenericProposer, BlockProposer>(m, "BlockGenericProposer")
        .def(py::init<>());

    py::class_<BlockUniformProposer, BlockProposer>(m, "BlockUniformProposer")
        .def(py::init<double>(), py::arg("create_new_block")=0.1) ;

    py::class_<BlockPeixotoProposer, BlockProposer>(m, "BlockPeixotoProposer")
        .def(py::init<double,double>(), py::arg("create_new_block")=0.1, py::arg("shift")=1) ;

}

}

#endif
