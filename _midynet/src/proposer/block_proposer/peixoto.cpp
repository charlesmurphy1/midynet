#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"
#include <random>


namespace FastMIDyNet {

const BlockMove BlockPeixotoProposer::proposeRawMove(BaseGraph::VertexIndex movedVertex) const {

    BlockIndex prevBlockIdx = (*m_blocksPtr)[movedVertex];
    size_t B = m_vertexCountsPtr->size();
    if (m_createNewBlockDistribution(rng) == 1)
        return {movedVertex, prevBlockIdx, B};
    if ( m_graphPtr->getDegreeOfIdx(movedVertex) == 0 ){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        BlockIndex nextBlockIdx = dist(rng);
        BlockMove move = {movedVertex, prevBlockIdx, nextBlockIdx};
        return move;

    }

    auto neighbors = m_graphPtr->getNeighboursOfIdx(movedVertex);
    BaseGraph::VertexIndex randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
    while(randomNeighbor == movedVertex)
        randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
    BlockIndex t = (*m_blocksPtr)[randomNeighbor];
    double probUniformSampling = m_shift * (B) / ((*m_edgeCountsPtr)[t] + m_shift * B);

    BlockIndex nextBlockIdx;
    if ( m_uniform01(rng) < probUniformSampling){
        std::uniform_int_distribution<size_t> dist(0, B-1);
        nextBlockIdx = dist(rng);
    } else {
        nextBlockIdx = generateCategorical<size_t, size_t>( (*m_edgeMatrixPtr)[t] );
    }

    BlockMove move = {movedVertex, prevBlockIdx, nextBlockIdx};
    return move;
}

void BlockPeixotoProposer::setUp(const RandomGraph& randomGraph) {
    BlockProposer::setUp(randomGraph);
    m_edgeMatrixPtr = &randomGraph.getEdgeMatrix();
    m_edgeCountsPtr = &randomGraph.getEdgeCountsInBlocks();
    m_graphPtr = &randomGraph.getGraph();
    m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
}

const double BlockPeixotoProposer::getLogProposalProb(const BlockMove& move) const {
    size_t B = m_vertexCountsPtr->size();
    if ( creatingNewBlock(move) )
         return log(m_blockCreationProbability);
    double weight = 0, degree = 0;
    auto r = move.prevBlockIdx, s = move.nextBlockIdx;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        if (move.vertexIdx == neighbor.vertexIndex)
            continue;
        auto t = (*m_blocksPtr) [ neighbor.vertexIndex ];
        size_t Est = (*m_edgeMatrixPtr)[t][move.nextBlockIdx];
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
        edgeMatDiff.decrement({r, t}, neighbor.label);
        edgeMatDiff.decrement({t, r}, neighbor.label);
        if (move.vertexIdx == neighbor.vertexIndex)
            t = move.nextBlockIdx;
        edgeMatDiff.increment({s, t}, neighbor.label);
        edgeMatDiff.increment({t, s}, neighbor.label);
    }
     return edgeMatDiff;
}

IntMap<BlockIndex> BlockPeixotoProposer::getEdgeCountsDiff(const BlockMove& move) const {
    IntMap<BlockIndex> edgeCountsDiff;
    size_t degree = m_graphPtr->getDegreeOfIdx(move.vertexIdx);
    edgeCountsDiff.decrement(move.prevBlockIdx, degree);
    edgeCountsDiff.increment(move.nextBlockIdx, degree);
     return edgeCountsDiff;
}

const double BlockPeixotoProposer::getReverseLogProposalProb(const BlockMove& move) const {
    int addedBlocks = getAddedBlocks(move) ;
    size_t B = m_vertexCountsPtr->size() + addedBlocks;
    if ( addedBlocks == -1)
         return log(m_blockCreationProbability);

    auto edgeMatDiff = getEdgeMatrixDiff(move);
    auto edgeCountsDiff = getEdgeCountsDiff(move);


    double weight = 0, degree = 0;
    for (auto neighbor : m_graphPtr->getNeighboursOfIdx(move.vertexIdx)){
        if (move.vertexIdx == neighbor.vertexIndex)
            continue;
        auto t = (*m_blocksPtr) [ neighbor.vertexIndex ];
        size_t Ert = (*m_edgeMatrixPtr)[t][move.prevBlockIdx] + edgeMatDiff.get({t, move.prevBlockIdx});
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
