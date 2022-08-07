#ifndef FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_BLOCKPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/python/label.hpp"
#include "FastMIDyNet/proposer/python/nested_label.hpp"

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/label/uniform.hpp"
#include "FastMIDyNet/proposer/label/mixed.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/uniform.hpp"
#include "FastMIDyNet/proposer/nested_label/mixed.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label>
py::class_<LabelProposer<Label>, Proposer<LabelMove<Label>>, PyLabelProposer<Label>> declareLabelProposer(py::module&m, std::string pyName){
    return py::class_<LabelProposer<Label>, Proposer<LabelMove<Label>>, PyLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
        .def("set_up_with_prior", &LabelProposer<Label>::setUpWithPrior, py::arg("graph_prior"))
        .def("get_log_proposal_prob_ratio", &LabelProposer<Label>::getLogProposalProbRatio, py::arg("move"))
        .def("get_log_proposal_prob", &LabelProposer<Label>::getLogProposalProb, py::arg("move"), py::arg("reverse")=false)
        .def("apply_label_move", &LabelProposer<Label>::applyLabelMove, py::arg("move"))
        .def("propose_label_move", &LabelProposer<Label>::proposeLabelMove, py::arg("vertex"))
        .def("propose_new_label_move", &LabelProposer<Label>::proposeNewLabelMove, py::arg("vertex"))
        ;
}

template<typename Label>
py::class_<NestedLabelProposer<Label>, LabelProposer<Label>, PyNestedLabelProposer<Label>> declareNestedLabelProposer(py::module&m, std::string pyName){
    return py::class_<NestedLabelProposer<Label>, LabelProposer<Label>, PyNestedLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
        .def("set_up_with_nested_prior", &NestedLabelProposer<Label>::setUpWithNestedPrior, py::arg("nested_graph_prior"))
        .def("sample_level", &NestedLabelProposer<Label>::sampleLevel)
        ;
}

template<typename Label>
py::class_<GibbsLabelProposer<Label>, LabelProposer<Label>, PyGibbsLabelProposer<Label>> declareGibbsLabelProposer(py::module&m, std::string pyName){
    return py::class_<GibbsLabelProposer<Label>, LabelProposer<Label>, PyGibbsLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1)
        ;
}

template<typename Label>
py::class_<GibbsNestedLabelProposer<Label>, NestedLabelProposer<Label>, PyGibbsNestedLabelProposer<Label>> declareGibbsNestedLabelProposer(py::module&m, std::string pyName){
    return py::class_<GibbsNestedLabelProposer<Label>, NestedLabelProposer<Label>, PyGibbsNestedLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1)
        ;
}

template<typename Label>
py::class_<RestrictedLabelProposer<Label>, LabelProposer<Label>, PyRestrictedLabelProposer<Label>> declareRestrictedLabelProposer(py::module&m, std::string pyName){
    return py::class_<RestrictedLabelProposer<Label>, LabelProposer<Label>, PyRestrictedLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
        ;
}

template<typename Label>
py::class_<RestrictedNestedLabelProposer<Label>, NestedLabelProposer<Label>, PyRestrictedNestedLabelProposer<Label>> declareRestrictedNestedLabelProposer(py::module&m, std::string pyName){
    return py::class_<RestrictedNestedLabelProposer<Label>, NestedLabelProposer<Label>, PyRestrictedNestedLabelProposer<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
        ;
}

template<typename Label>
py::class_<MixedSampler<Label>, PyMixedSampler<Label>> declareMixedSampler(py::module& m, std::string pyName){
    return py::class_<MixedSampler<Label>, PyMixedSampler<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("shift")=1)
        .def("get_shift", &MixedSampler<Label>::getShift)
        ;
}

template<typename Label>
py::class_<MixedNestedSampler<Label>, PyMixedNestedSampler<Label>> declareMixedNestedSampler(py::module& m, std::string pyName){
    return py::class_<MixedNestedSampler<Label>, PyMixedNestedSampler<Label>>(m, pyName.c_str())
        .def(py::init<double>(), py::arg("shift")=1)
        .def("get_shift", &MixedNestedSampler<Label>::getShift)
        ;
}



void initLabelProposer(py::module& m){
    declareLabelProposer<BlockIndex>(m, "BlockProposer");
    declareGibbsLabelProposer<BlockIndex>(m, "GibbsBlockProposer");
    declareRestrictedLabelProposer<BlockIndex>(m, "RestrictedBlockProposer");
    declareMixedSampler<BlockIndex>(m, "MixedBlockSampler");

    declareNestedLabelProposer<BlockIndex>(m, "NestedBlockProposer");
    declareGibbsNestedLabelProposer<BlockIndex>(m, "GibbsNestedBlockProposer");
    declareRestrictedNestedLabelProposer<BlockIndex>(m, "RestrictedNestedBlockProposer");
    declareMixedNestedSampler<BlockIndex>(m, "MixedNestedBlockSampler");

    py::class_<GibbsUniformLabelProposer<BlockIndex>, GibbsLabelProposer<BlockIndex>>(m, "GibbsUniformBlockProposer")
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1)
        ;
    py::class_<GibbsMixedLabelProposer<BlockIndex>, GibbsLabelProposer<BlockIndex>, MixedSampler<BlockIndex>>(m, "GibbsMixedBlockProposer")
        .def(py::init<double, double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1, py::arg("shift")=1)
        ;
    py::class_<GibbsUniformNestedLabelProposer<BlockIndex>, GibbsNestedLabelProposer<BlockIndex>>(m, "GibbsUniformNestedBlockProposer")
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1)
        ;
    py::class_<GibbsMixedNestedLabelProposer<BlockIndex>, GibbsNestedLabelProposer<BlockIndex>, MixedNestedSampler<BlockIndex>>(m, "GibbsMixedNestedBlockProposer")
        .def(py::init<double, double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("label_creation_prob")=0.1, py::arg("shift")=1)
        ;
    py::class_<RestrictedUniformLabelProposer<BlockIndex>, RestrictedLabelProposer<BlockIndex>>(m, "RestrictedUniformBlockProposer")
    .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
    ;
    py::class_<RestrictedMixedLabelProposer<BlockIndex>, RestrictedLabelProposer<BlockIndex>, MixedSampler<BlockIndex>>(m, "RestrictedMixedBlockProposer")
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("shift")=1)
        ;
    py::class_<RestrictedUniformNestedLabelProposer<BlockIndex>, RestrictedNestedLabelProposer<BlockIndex>>(m, "RestrictedUniformNestedBlockProposer")
    .def(py::init<double>(), py::arg("sample_label_count_prob")=0.1)
    ;
    py::class_<RestrictedMixedNestedLabelProposer<BlockIndex>, RestrictedNestedLabelProposer<BlockIndex>, MixedNestedSampler<BlockIndex>>(m, "RestrictedMixedNestedBlockProposer")
        .def(py::init<double, double>(), py::arg("sample_label_count_prob")=0.1, py::arg("shift")=1)
        ;
}

}

#endif
