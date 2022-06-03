#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/proposer/block_proposer/peixoto.h"
#include <random>


namespace FastMIDyNet {

BlockMove BlockPeixotoProposer::proposeMove(BaseGraph::VertexIndex movedVertex) const {
    BlockIndex prevBlockIdx = (*m_blocksPtr)[movedVertex];
    if (m_createNewBlockDistribution(rng) == 1)
        return {movedVertex, prevBlockIdx, *m_blockCountPtr, 1};
    if ( m_graphPtr->getDegreeOfIdx(movedVertex) == 0 ){
        std::uniform_int_distribution<size_t> dist(0, *m_blockCountPtr-1);
        BlockIndex nextBlockIdx = dist(rng);
        int addedBlocks = 0;
        if ( destroyingBlock(prevBlockIdx, nextBlockIdx) ){
            --addedBlocks;
        }
        return {movedVertex, prevBlockIdx, nextBlockIdx, addedBlocks};

    }

    auto neighbors = m_graphPtr->getNeighboursOfIdx(movedVertex);
    BaseGraph::VertexIndex randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
    while(randomNeighbor == movedVertex)
        randomNeighbor = sampleUniformlyFrom(neighbors.begin(), neighbors.end())->vertexIndex;
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

void BlockPeixotoProposer::setUp(const RandomGraph& randomGraph) {
    m_blockCountPtr = &randomGraph.getBlockCount();
    m_blocksPtr = &randomGraph.getBlocks();
    m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
    m_edgeMatrixPtr = &randomGraph.getEdgeMatrix();
    m_edgeCountsPtr = &randomGraph.getEdgeCountsInBlocks();
    m_graphPtr = &randomGraph.getGraph();
    m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
}

const double BlockPeixotoProposer::getLogProposalProb(const BlockMove& move) const {
    size_t B = *m_blockCountPtr;
    if ( move.addedBlocks == 1)
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
    // std::cout << weight << ", " << degree << std::endl;
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
    size_t B = *m_blockCountPtr + move.addedBlocks;
    if ( move.addedBlocks == -1)
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
