#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/block_proposer/uniform.h"
#include "FastMIDyNet/utility/functions.h"
#include <random>


namespace FastMIDyNet {

BlockMove BlockUniformProposer::proposeMove(BaseGraph::VertexIndex movedVertex) const {
    if (*m_blockCountPtr == 1 && m_blockCreationProbability == 0)
        return {movedVertex, (*m_blocksPtr)[movedVertex], (*m_blocksPtr)[movedVertex], 0};


    const BlockIndex& currentBlock = (*m_blocksPtr)[movedVertex];

    BlockIndex newBlock;
    int addedBlocks = 0;
    if (m_createNewBlockDistribution(rng)){
        newBlock = *m_blockCountPtr;
        addedBlocks = 1;
    }
    else if (*m_blockCountPtr > 1) {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, *m_blockCountPtr - 1)(rng);
    } else {
        return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], -1};
    }

    if (destroyingBlock(currentBlock, newBlock) && creatingNewBlock(newBlock)){
        return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], -1};
    }

    return {movedVertex, currentBlock, newBlock, addedBlocks};
}

void BlockUniformProposer::setUp(const RandomGraph& randomGraph) {
    m_blockCountPtr = &randomGraph.getBlockCount();
    m_blocksPtr = &randomGraph.getBlocks();
    m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
    m_vertexDistribution = std::uniform_int_distribution<BaseGraph::VertexIndex>(0, randomGraph.getSize() - 1);
}

const double BlockUniformProposer::getLogProposalProbRatio(const BlockMove& move) const {
    if (creatingNewBlock(move.nextBlockIdx))
        return -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(*m_blockCountPtr);
    else if (destroyingBlock(move.prevBlockIdx, move.nextBlockIdx))
        return log(*m_blockCountPtr-1) - log(1-m_blockCreationProbability) + log(m_blockCreationProbability);
    return 0;
}

} // namespace FastMIDyNet
