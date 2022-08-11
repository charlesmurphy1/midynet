#ifndef FAST_MIDYNET_PYTHON_LABEL_HPP
#define FAST_MIDYNET_PYTHON_LABEL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/label/base.hpp"
#include "FastMIDyNet/proposer/label/mixed.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label, typename BaseClass = LabelProposer<Label>>
class PyLabelProposer: public PyProposer<LabelMove<Label>, BaseClass>{
protected:
    const double getLogProposalProbForReverseMove(const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbForReverseMove, move); }
    const double getLogProposalProbForMove(const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbForMove, move); }
    bool isCreatingLabelMove(const LabelMove<Label>& move, bool reverse) const override { PYBIND11_OVERRIDE_PURE(bool, BaseClass, isCreatingLabelMove, move, reverse); }
public:
    using PyProposer<LabelMove<Label>, BaseClass>::PyProposer;

    /* Pure abstract methods */
    const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex& vertex) const override { PYBIND11_OVERRIDE_PURE(const LabelMove<Label>, BaseClass, proposeLabelMove, vertex); }
    const LabelMove<Label> proposeNewLabelMove(const BaseGraph::VertexIndex& vertex) const override { PYBIND11_OVERRIDE_PURE(const LabelMove<Label>, BaseClass, proposeNewLabelMove, vertex); }

    /* Abstract & overloaded methods */
    const double getLogProposalProb(const LabelMove<Label>& move, bool reverse) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogProposalProb, move, reverse); }
    void applyLabelMove(const LabelMove<Label>& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyLabelMove, move); }
    void setUpWithPrior(const VertexLabeledRandomGraph<Label>& graphPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setUpWithPrior, graphPrior); }
};

template<typename Label, typename BaseClass = GibbsLabelProposer<Label>>
class PyGibbsLabelProposer: public PyLabelProposer<Label, BaseClass>{
public:
    using PyLabelProposer<Label, BaseClass>::PyLabelProposer;
};

template<typename Label, typename BaseClass = RestrictedLabelProposer<Label>>
class PyRestrictedLabelProposer: public PyLabelProposer<Label, BaseClass>{
public:
    using PyLabelProposer<Label, BaseClass>::PyLabelProposer;
};

template<typename Label, typename BaseClass = MixedSampler<Label>>
class PyMixedSampler: public BaseClass{
protected:
    const Label sampleLabelUniformly() const override { PYBIND11_OVERRIDE_PURE(const Label, BaseClass, sampleLabelUniformly, ); }
    const size_t getAvailableLabelCount() const override { PYBIND11_OVERRIDE_PURE(const size_t, BaseClass, getAvailableLabelCount, ); }
public:
    using BaseClass::BaseClass;
};

}

#endif
