#ifndef FAST_MIDYNET_GENERIC_PROPOSER_H
#define FAST_MIDYNET_GENERIC_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockGenericProposer: public BlockProposer {
protected:
    bool creatingNewBlock(const BlockMove&) const { return false; }
    bool destroyingBlock(const BlockMove&) const { return false; }
public:
    BlockGenericProposer(){}
    const BlockMove proposeMove(const BaseGraph::VertexIndex& vertex) const override{
        return {vertex, (*m_blocksPtr)[vertex], (*m_blocksPtr)[vertex], 0};
    }
    const double getLogProposalProbRatio(const BlockMove&) const override { return 0;};
    void checkSelfSafety() const override { }
};

} // namespace FastMIDyNet


#endif
