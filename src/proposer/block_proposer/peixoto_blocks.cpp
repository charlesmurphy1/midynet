#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto_proposer.h"
#include <random>


namespace FastMIDyNet {


PeixotoBlockProposer::PeixotoBlockProposer(double createNewBlockProbability, double shift):
    m_createNewBlockDistribution(createNewBlockProbability),
    m_blockCreationProbability(createNewBlockProbability),
    m_shift(shift) {
    assertValidProbability(createNewBlockProbability);
}

BlockMove PeixotoBlockProposer::proposeMove(BaseGraph::VertexIndex movedVertex) {
    BlockIndex prevBlockIdx = (*m_blocksPtr)[movedVertex];
    if (m_createNewBlockDistribution(rng) == 1){
        return {movedVertex, prevBlockIdx, *m_blockCountPtr, 1};
    } else if ( m_graphPtr->getDegreeOfIdx(movedVertex) == 0 ){
        std::uniform_int_distribution<size_t> dist(0, *m_blockCountPtr-1);
        BlockIndex nextBlockIdx = dist(rng);
        int addedBlocks = 0;
        if ( destroyingBlock(prevBlockIdx, nextBlockIdx) ){
            --addedBlocks;
        }
        return {movedVertex, prevBlockIdx, nextBlockIdx, addedBlocks};

    } else {

        auto neighbors = m_graphPtr->getNeighboursOfIdx(movedVertex);
        const BaseGraph::VertexIndex& randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
        BlockIndex t = (*m_blocksPtr)[randomNeighbor];
        double probUniformSampling = m_shift * (*m_blockCountPtr) / ((*m_edgeCountsPtr)[t] + m_shift * (*m_blockCountPtr));

        BlockIndex nextBlockIdx;
        int addedBlocks = 0;
        if ( m_uniform01(rng) < probUniformSampling){
            std::uniform_int_distribution<size_t> dist(0, *m_blockCountPtr-1);
            nextBlockIdx = dist(rng);
        } else {
            nextBlockIdx = generateCategorical<size_t, size_t>( (*m_edgeMatrixPtr)[t] );
        }

        if ( destroyingBlock(prevBlockIdx, nextBlockIdx) ){
            --addedBlocks;
        }

        return {movedVertex, prevBlockIdx, nextBlockIdx, addedBlocks};
    }
}

void PeixotoBlockProposer::setUp(const StochasticBlockModelFamily& sbmGraph) {
    m_blockCountPtr = &sbmGraph.getBlockCount();
    m_blocksPtr = &sbmGraph.getBlocks();
    m_vertexCountsPtr = &sbmGraph.getVertexCountsInBlocks();
    m_edgeMatrixPtr = &sbmGraph.getEdgeMatrix();
    m_edgeCountsPtr = &sbmGraph.getEdgeCountsInBlocks();
    m_graphPtr = &sbmGraph.getState();
}

double PeixotoBlockProposer::getLogProposalProb(const BlockMove& move) const {
    if ( move.addedBlocks == 1){
         return log(m_blockCreationProbability);
    } else if (m_graphPtr->getDegreeOfIdx(move.vertexIdx) == 0){
        return log(1 - m_blockCreationProbability) - log(*m_blockCountPtr);
    } else {
        size_t degree = 0;
        double weight = 0;
        auto r = move.prevBlockIdx, s = move.nextBlockIdx;
        for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
            auto t = (*m_blocksPtr) [ neighbor.vertexIndex];
            degree += neighbor.label;
            weight += neighbor.label * ( (*m_edgeMatrixPtr)[t][s] + m_shift ) / ((*m_edgeCountsPtr)[t] + m_shift * (*m_blockCountPtr)) ;
        }
        double logProposal = log(1 - m_blockCreationProbability) + log(weight) - log(degree);

        return logProposal;
    }
}


IntMap<std::pair<BlockIndex, BlockIndex>> PeixotoBlockProposer::getEdgeMatrixDiff(const BlockMove& move) const {
    IntMap<std::pair<BlockIndex, BlockIndex>> edgeMatDiff;
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        BlockIndex t = (*m_blocksPtr)[neighbor.vertexIndex];
        edgeMatDiff.decrement(getOrderedEdge({r, t}), neighbor.label);
        edgeMatDiff.increment(getOrderedEdge({s, t}), neighbor.label);
    }
     return edgeMatDiff;
}

IntMap<BlockIndex> PeixotoBlockProposer::getEdgeCountsDiff(const BlockMove& move) const {
    IntMap<BlockIndex> edgeCountsDiff;
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        BlockIndex t = (*m_blocksPtr)[neighbor.vertexIndex];
        edgeCountsDiff.decrement(r, neighbor.label);
        edgeCountsDiff.increment(s, neighbor.label);
    }
     return edgeCountsDiff;
}

double PeixotoBlockProposer::getReverseLogProposalProb(const BlockMove& move) const {
    if ( move.addedBlocks == -1){
         return log(m_blockCreationProbability);
    } else if (m_graphPtr->getDegreeOfIdx(move.vertexIdx) == 0){
        return -log(*m_blockCountPtr);
    } else {
        auto edgeMatDiff = getEdgeMatrixDiff(move);
        auto edgeCountsDiff = getEdgeCountsDiff(move);
        size_t degree = 0;
        double weight = 0;
        auto r = move.prevBlockIdx, s = move.nextBlockIdx;
        for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
            BlockIndex t;
            if (neighbor.vertexIndex == move.vertexIdx)
                t = r;
            else
                t = (*m_blocksPtr) [ neighbor.vertexIndex ];
            size_t nextErt = (*m_edgeMatrixPtr)[t][r] + edgeMatDiff.get(getOrderedEdge({r, t}));
            size_t nextEt = (*m_edgeCountsPtr)[t] + edgeCountsDiff.get(t);
            degree += neighbor.label;
            weight += neighbor.label * ( nextErt + m_shift ) / (nextEt + m_shift * (*m_blockCountPtr + move.addedBlocks)) ;
        }
        return log(1 - m_blockCreationProbability) + log ( weight ) - log( degree );
    }
}

} // namespace FastMIDyNet
