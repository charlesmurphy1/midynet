#ifndef FAST_MIDYNET_GENERIC_PROPOSER_H
#define FAST_MIDYNET_GENERIC_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockGenericProposer: public BlockProposer {
    public:
        BlockGenericProposer(){}
        BlockMove proposeMove() const override{
            return {0, 0, 0, 0};
        }
        void setUp(const RandomGraph& randomGraph) override {};
        double getLogProposalProbRatio(const BlockMove&) const override { return 0;};
        void checkSafety() const override { }
};

} // namespace FastMIDyNet


#endif
