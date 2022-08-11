#ifndef FAST_MIDYNET_PYTHON_NESTED_LABEL_HPP
#define FAST_MIDYNET_PYTHON_NESTED_LABEL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/python/label.hpp"
#include "FastMIDyNet/proposer/nested_label/base.hpp"
#include "FastMIDyNet/proposer/nested_label/uniform.hpp"
#include "FastMIDyNet/proposer/nested_label/mixed.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label, typename BaseClass = NestedLabelProposer<Label>>
class PyNestedLabelProposer: public PyLabelProposer<Label, BaseClass>{
protected:
    using PyLabelProposer<Label, BaseClass>::getLogProposalProbForMove;
    using PyLabelProposer<Label, BaseClass>::getLogProposalProbForReverseMove;
public:
    using PyLabelProposer<Label, BaseClass>::PyLabelProposer;

    /* Pure abstract methods */
    using PyLabelProposer<Label, BaseClass>::proposeLabelMove;

    /* Abstract & overloaded methods */
    void setUpWithNestedPrior(const NestedVertexLabeledRandomGraph<Label>& graphPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setUpWithNestedPrior, graphPrior); }
};

template<typename Label, typename BaseClass = GibbsNestedLabelProposer<Label>>
class PyGibbsNestedLabelProposer: public PyNestedLabelProposer<Label, BaseClass>{
public:
    using PyNestedLabelProposer<Label, BaseClass>::PyNestedLabelProposer;
};

template<typename Label, typename BaseClass = RestrictedNestedLabelProposer<Label>>
class PyRestrictedNestedLabelProposer: public PyNestedLabelProposer<Label, BaseClass>{
public:
    using PyNestedLabelProposer<Label, BaseClass>::PyNestedLabelProposer;
};

template<typename Label, typename BaseClass = MixedNestedSampler<Label>>
class PyMixedNestedSampler: public BaseClass{
protected:
    const Label sampleLabelUniformlyAtLevel(Level level) const override { PYBIND11_OVERRIDE_PURE(const Label, BaseClass, sampleLabelUniformlyAtLevel, level); }
    const size_t getAvailableLabelCountAtLevel(Level level) const override { PYBIND11_OVERRIDE_PURE(const size_t, BaseClass, getAvailableLabelCountAtLevel, level); }
public:
    using BaseClass::BaseClass;
};

}

#endif
