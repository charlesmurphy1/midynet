#ifndef FAST_MIDYNET_GENERIC_PROPOSER_H
#define FAST_MIDYNET_GENERIC_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockGenericProposer: public BlockProposer {
    const std::vector<BlockIndex>* m_blocksPtr;
    public:
        BlockGenericProposer(){}
        BlockMove proposeMove() const override{
            return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], 0};
        }
        void setUp(const RandomGraph& randomGraph) override { m_blocksPtr = &randomGraph.getBlocks(); }
        const double getLogProposalProbRatio(const BlockMove&) const override { return 0;};
        void checkSelfSafety() const override { }
};

} // namespace FastMIDyNet


#endif
