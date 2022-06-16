#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"
#include <random>


namespace FastMIDyNet {

const BlockMove BlockPeixotoProposer::proposeMove(const BaseGraph::VertexIndex& movedVertex) const {

    BlockIndex prevBlockIdx = (*m_blocksPtr)[movedVertex], nextBlockIdx;
    size_t B = m_vertexCountsPtr->size();
    if (m_createNewBlockDistribution(rng) == 1)
        return {movedVertex, prevBlockIdx, B};
    if ( (*m_degreesPtr)[movedVertex] == 0 ){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        BlockIndex nextBlockIdx = dist(rng);
        BlockMove move = {movedVertex, prevBlockIdx, nextBlockIdx};
        return move;
    }

    auto neighbors = m_graphPtr->getNeighboursOfIdx(movedVertex);
    BaseGraph::VertexIndex randomNeighbor = movedVertex;
    while(randomNeighbor == movedVertex)
        randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
    BlockIndex t = (*m_blocksPtr)[randomNeighbor];

    double probUniformSampling = m_shift * B / (m_edgeCountsPtr->get(t) + m_shift * B);
    if ( m_uniform01(rng) < probUniformSampling){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        nextBlockIdx = dist(rng);
    } else {
        std::uniform_int_distribution<int> dist(0, (*m_edgeCountsPtr)[t] - 1);
        int mult = dist(rng);
        for (auto s : m_edgeMatrixPtr->getNeighboursOfIdx(t)){
            mult -= ((t == s.vertexIndex) ? 2 : 1) * s.label;
            nextBlockIdx = s.vertexIndex;
            if (mult < 0) break;
        }
    }

    BlockMove move = {movedVertex, prevBlockIdx, nextBlockIdx};
    return move;
}

void BlockPeixotoProposer::setUp(const RandomGraph& randomGraph) {
    BlockProposer::setUp(randomGraph);
    m_edgeMatrixPtr = &randomGraph.getEdgeMatrix();
    m_edgeCountsPtr = &randomGraph.getEdgeCountsInBlocks();
    m_degreesPtr = &randomGraph.getDegrees();
    m_graphPtr = &randomGraph.getGraph();
}

const double BlockPeixotoProposer::getLogProposalProb(const BlockMove& move) const {
    size_t B = m_vertexCountsPtr->size();
    if ( creatingNewBlock(move) )
         return log(m_blockCreationProbability);
    double weight = 0, degree = 0;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        if (move.vertexIdx == neighbor.vertexIndex)
            continue;
        auto t = (*m_blocksPtr) [ neighbor.vertexIndex ];
        size_t Est = ((t == move.nextBlockIdx) ? 2 : 1) * m_edgeMatrixPtr->getEdgeMultiplicityIdx(t, move.nextBlockIdx);
        size_t Et = (*m_edgeCountsPtr)[t];

        degree += neighbor.label;
        weight += neighbor.label * ( Est + m_shift ) / (Et + m_shift * B) ;
    }

    if (degree == 0)
       return log(1 - m_blockCreationProbability) - log(B);
    double logProposal = log(1 - m_blockCreationProbability) + log(weight) - log(degree);
    return logProposal;
}


IntMap<std::pair<BlockIndex, BlockIndex>> BlockPeixotoProposer::getEdgeMatrixDiff(const BlockMove& move) const {
    IntMap<std::pair<BlockIndex, BlockIndex>> edgeMatDiff;
    BlockIndex r = move.prevBlockIdx, s = move.nextBlockIdx;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        BlockIndex t = (*m_blocksPtr)[neighbor.vertexIndex];
        if (move.vertexIdx == neighbor.vertexIndex)
            t = move.prevBlockIdx;
        edgeMatDiff.decrement(getOrderedEdge({r, t}), neighbor.label);
        if (move.vertexIdx == neighbor.vertexIndex)
            t = move.nextBlockIdx;
        edgeMatDiff.increment(getOrderedEdge({s, t}), neighbor.label);
    }
     return edgeMatDiff;
}

IntMap<BlockIndex> BlockPeixotoProposer::getEdgeCountsDiff(const BlockMove& move) const {
    IntMap<BlockIndex> edgeCountsDiff;
    size_t degree = (*m_degreesPtr)[move.vertexIdx];
    edgeCountsDiff.decrement(move.prevBlockIdx, degree);
    edgeCountsDiff.increment(move.nextBlockIdx, degree);
     return edgeCountsDiff;
}

const double BlockPeixotoProposer::getReverseLogProposalProb(const BlockMove& move) const {
    int addedBlocks = getAddedBlocks(move) ;
    size_t B = m_vertexCountsPtr->size() + addedBlocks;
    if ( destroyingBlock(move) )
         return log(m_blockCreationProbability);

    auto edgeMatDiff = getEdgeMatrixDiff(move);
    auto edgeCountsDiff = getEdgeCountsDiff(move);


    double weight = 0, degree = 0;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        if (move.vertexIdx == neighbor.vertexIndex)
            continue;
        auto t = (*m_blocksPtr) [ neighbor.vertexIndex ];
        size_t Ert = ((t == move.prevBlockIdx) ? 2 : 1) * (m_edgeMatrixPtr->getEdgeMultiplicityIdx(t, move.prevBlockIdx) + edgeMatDiff.get({t, move.prevBlockIdx}));
        size_t Et = (*m_edgeCountsPtr)[t] + edgeCountsDiff.get(t);
        degree += neighbor.label;
        weight += neighbor.label * ( Ert + m_shift ) / (Et + m_shift * B) ;
    }

    if (degree == 0)
       return log(1 - m_blockCreationProbability) - log(B);
    double logReverseProposal = log(1 - m_blockCreationProbability) + log ( weight ) - log( degree );
    return logReverseProposal;
}

} // namespace FastMIDyNet
