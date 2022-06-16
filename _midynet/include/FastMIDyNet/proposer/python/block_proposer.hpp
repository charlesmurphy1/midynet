#ifndef FAST_MIDYNET_PYTHON_BLOCK_PROPOSER_HPP
#define FAST_MIDYNET_PYTHON_BLOCK_PROPOSER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename BaseClass = BlockProposer>
class PyBlockProposer: public PyProposer<BlockMove, BaseClass>{
protected:
    // bool creatingNewBlock(const BlockMove& move) const { PYBIND11_OVERRIDE_PURE(bool, BaseClass, creatingNewBlock, move); }
    // bool destroyingBlock(const BlockMove& move) const { PYBIND11_OVERRIDE_PURE(bool, BaseClass, destroyingBlock, move); }
public:
    using PyProposer<BlockMove, BaseClass>::PyProposer;

    /* Pure abstract methods */
    void setUp(const RandomGraph& randomGraph) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, setUp, randomGraph); }
    const double getLogProposalProbRatio(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogProposalProbRatio, move); }
    const BlockMove proposeMove(const BaseGraph::VertexIndex& id) const override { PYBIND11_OVERRIDE_PURE(const BlockMove, BaseClass, proposeMove, id); }

    /* Abstract & overloaded methods */
    void applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void applyBlockMove(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyBlockMove, move); }
};


}

#endif
