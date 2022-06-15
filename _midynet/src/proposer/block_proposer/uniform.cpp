#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/utility/functions.h"
#include <random>


namespace FastMIDyNet {

BlockMove BlockUniformProposer::proposeMove(BaseGraph::VertexIndex movedVertex) const {
    size_t B = m_vertexCountsPtr->size();
    if (B == 1 && m_blockCreationProbability == 0)
        return {movedVertex, (*m_blocksPtr)[movedVertex], (*m_blocksPtr)[movedVertex], 0};


    const BlockIndex& currentBlock = (*m_blocksPtr)[movedVertex];

    BlockIndex newBlock;
    int addedBlocks = 0;
    if (m_createNewBlockDistribution(rng)){
        newBlock = B;
        addedBlocks = 1;
    }
    else if (B > 1) {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, B - 1)(rng);
    } else {
        return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], 0};
    }
    BlockMove move = {movedVertex, currentBlock, newBlock, addedBlocks};
    return {movedVertex, currentBlock, newBlock, addedBlocks};
}

void BlockUniformProposer::setUp(const RandomGraph& randomGraph) {
    m_blocksPtr = &randomGraph.getBlocks();
    m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
    m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
}

const double BlockUniformProposer::getLogProposalProbRatio(const BlockMove& move) const {
    size_t B = m_vertexCountsPtr->size();
    if (creatingNewBlock(move))
        return -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(B);
    else if (destroyingBlock(move))
        return log(B-1) - log(1-m_blockCreationProbability) + log(m_blockCreationProbability);
    return 0;
}

} // namespace FastMIDyNet
