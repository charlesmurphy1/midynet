#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockProposer: public Proposer<BlockMove> {
public:
    virtual void setUp(const RandomGraph& randomGraph) = 0;
    virtual const double getLogProposalProbRatio(const BlockMove& move) const = 0;
    virtual void updateProbabilities(const GraphMove& move) {};
    virtual void updateProbabilities(const BlockMove& move) {};

};

} // namespace FastMIDyNet


#endif
