#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockProposer: public Proposer<BlockMove> {
protected:
    virtual bool creatingNewBlock(const BlockMove&) const = 0;
    virtual bool destroyingBlock(const BlockMove&) const = 0;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;
    virtual const BlockMove proposeMove(const BaseGraph::VertexIndex) const = 0;
public:
    virtual void setUp(const RandomGraph& randomGraph) = 0;
    virtual const double getLogProposalProbRatio(const BlockMove& move) const = 0;

    const BlockMove proposeMove() const override{
        auto vertexIdx = m_vertexDistribution(rng);
        auto move = proposeMove(vertexIdx);
        if (creatingNewBlock(move))
            ++move.addedBlocks;
        else if (destroyingBlock(move))
            --move.addedBlocks;
        return move;
    }
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyBlockMove(const BlockMove& move) {};

};

} // namespace FastMIDyNet


#endif
