#ifndef FAST_MIDYNET_PYTHON_BLOCK_PROPOSER_HPP
#define FAST_MIDYNET_PYTHON_BLOCK_PROPOSER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename Label, typename BaseClass = LabelProposer<Label>>
class PyLabelProposer: public PyProposer<LabelMove<Label>, BaseClass>{
public:
    using PyProposer<LabelMove<Label>, BaseClass>::PyProposer;

    /* Pure abstract methods */
    const double getLogProposalProbRatio(const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbRatio, move); }
    const LabelMove<Label> proposeMove(const BaseGraph::VertexIndex& id) const override { PYBIND11_OVERRIDE_PURE(const LabelMove<Label>, BaseClass, proposeMove, id); }

    /* Abstract & overloaded methods */
    void applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void applyLabelMove(const LabelMove<Label>& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyLabelMove, move); }
};


}

#endif
