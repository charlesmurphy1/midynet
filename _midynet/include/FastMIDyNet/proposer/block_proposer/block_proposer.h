#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {

class BlockProposer: public Proposer<BlockMove> {
protected:
    bool creatingNewBlock(const BlockMove& move) const {
        return m_vertexCountsPtr->get(move.nextBlockIdx) == 0;
    };
    bool destroyingBlock(const BlockMove& move) const {
        return move.prevBlockIdx != move.nextBlockIdx and m_vertexCountsPtr->get(move.prevBlockIdx) == 1 ;
    }
    const BlockSequence* m_blocksPtr = nullptr;
    const CounterMap<BlockIndex>* m_vertexCountsPtr = nullptr;
    const int getAddedBlocks(const BlockMove& move) const {
        return (int) creatingNewBlock(move) - (int) destroyingBlock(move);
    }
public:
    virtual void setUp(const RandomGraph& randomGraph) {
        m_blocksPtr = &randomGraph.getBlocks();
        m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
    };
    virtual const double getLogProposalProbRatio(const BlockMove& move) const = 0;
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyBlockMove(const BlockMove& move) {};

    virtual const BlockMove proposeRawMove() const = 0;
    const BlockMove proposeMove() const override {
        BlockMove move = proposeRawMove();
        move.addedBlocks = getAddedBlocks(move);
        return move;
    }


};

} // namespace FastMIDyNet


#endif
