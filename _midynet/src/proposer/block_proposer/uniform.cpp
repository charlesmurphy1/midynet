#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/utility/functions.h"
#include <random>


namespace FastMIDyNet {

const BlockMove BlockUniformProposer::proposeMove(BaseGraph::VertexIndex movedVertex) const {
    size_t B = m_vertexCountsPtr->size();
    if (B == 1 && m_blockCreationProbability == 0)
        return {movedVertex, (*m_blocksPtr)[movedVertex], (*m_blocksPtr)[movedVertex], 0};


    const BlockIndex& currentBlock = (*m_blocksPtr)[movedVertex];

    BlockIndex newBlock;
    if (m_createNewBlockDistribution(rng)){
        newBlock = B;
    }
    else if (B > 1) {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, B - 1)(rng);
    } else {
        return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], 0};
    }
    BlockMove move = {movedVertex, currentBlock, newBlock, 0};
    return move;
}

void BlockUniformProposer::setUp(const RandomGraph& randomGraph) {
    m_blocksPtr = &randomGraph.getBlocks();
    m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
    m_edgeCountsPtr = &randomGraph.getEdgeCountsInBlocks();
    m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
}

const double BlockUniformProposer::getLogProposalProbRatio(const BlockMove& move) const {
    return (move.nextBlockIdx != m_vertexCountsPtr->size()) ? 0 : -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(m_vertexCountsPtr->size());
}

} // namespace FastMIDyNet
