#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/python/proposer.hpp"

namespace py = pybind11;
namespace FastMIDyNet{


template<typename MoveType>
py::class_<Proposer<MoveType>, PyProposer<MoveType>> declareProposerBaseClass(py::module& m, std::string pyName){
    return py::class_<Proposer<MoveType>, PyProposer<MoveType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("propose_move", &Proposer<MoveType>::proposeMove)
        .def("get_log_proposal_ratio", &Proposer<MoveType>::getLogProposalProbRatio, py::arg("move"))
        .def("update", &Proposer<MoveType>::updateProbabilities, py::arg("move")) ;
}

void initProposerBaseClass(py::module& m){
    declareProposerBaseClass<GraphMove>(m, "EdgeProposerBase");
    declareProposerBaseClass<BlockMove>(m, "BlockProposerBase");
}

}

#endif
