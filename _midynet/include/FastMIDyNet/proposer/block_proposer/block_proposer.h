#ifndef FAST_MIDYNET_BLOCKPROPOSER_H
#define FAST_MIDYNET_BLOCKPROPOSER_H


#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/rng.h"


namespace FastMIDyNet {

class BlockProposer: public Proposer<BlockMove> {
protected:
    const BlockSequence* m_blocksPtr = nullptr;
    const CounterMap<BlockIndex>* m_vertexCountsPtr = nullptr;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;


    bool creatingNewBlock(const BlockMove& move) const {
        return m_vertexCountsPtr->get(move.nextBlockIdx) == 0;
    };
    bool destroyingBlock(const BlockMove& move) const {
        return move.prevBlockIdx != move.nextBlockIdx and m_vertexCountsPtr->get(move.prevBlockIdx) == 1 ;
    }
    const int getAddedBlocks(const BlockMove& move) const {
        return (int) creatingNewBlock(move) - (int) destroyingBlock(move);
    }
public:
    virtual void setUp(const RandomGraph& randomGraph) {
        m_blocksPtr = &randomGraph.getBlocks();
        m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
        m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
    };
    virtual const double getLogProposalProbRatio(const BlockMove& move) const = 0;
    virtual void applyGraphMove(const GraphMove& move) {};
    virtual void applyBlockMove(const BlockMove& move) {};
    virtual const BlockMove proposeMove(const BaseGraph::VertexIndex&) const = 0;
    const BlockMove proposeMove() const override{
        auto vertexIdx = m_vertexDistribution(rng);
        return proposeMove(vertexIdx);
    }
};

} // namespace FastMIDyNet


#endif
