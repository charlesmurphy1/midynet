#ifndef FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/python/label_proposer.hpp"

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/proposer/label/peixoto.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label>
py::class_<LabelProposer<Label>, Proposer<LabelMove<Label>>, PyLabelProposer<Label>> declareLabelProposer(py::module&m, std::string pyName){
    return py::class_<LabelProposer<Label>, Proposer<LabelMove<Label>>, PyLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<>())
        .def("set_up", &LabelProposer<Label>::setUp, py::arg("random_graph"))
        .def("get_log_proposal_prob_ratio", &LabelProposer<Label>::getLogProposalProbRatio, py::arg("move"))
        .def("apply_graph_move", &LabelProposer<Label>::applyGraphMove, py::arg("move"))
        .def("apply_labe_move", &LabelProposer<Label>::applyLabelMove, py::arg("move"))
        ;
}

void initLabelProposer(py::module& m){
    declareLabelProposer<BlockIndex>(m, "BlockProposer");

    py::class_<LabelUniformProposer<BlockIndex>, LabelProposer<BlockIndex>>(m, "BlockUniformProposer")
        .def(py::init<double>(), py::arg("label_creation_prob")=0.1)
        ;

    py::class_<LabelPeixotoProposer<BlockIndex>, LabelProposer<BlockIndex>>(m, "BlockPeixotoProposer")
        .def(py::init<double,double>(), py::arg("label_creation_prob")=0.1, py::arg("shift")=1)
        ;
}

}

#endif
