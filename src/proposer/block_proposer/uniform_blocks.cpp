#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/block_proposer/uniform_proposer.h"
#include <random>


namespace FastMIDyNet {


UniformBlockProposer::UniformBlockProposer(double createNewBlockProbability):
    m_createNewBlockDistribution(createNewBlockProbability),
    m_blockCreationProbability(createNewBlockProbability) {
    assertValidProbability(createNewBlockProbability);
}

BlockMove UniformBlockProposer::proposeMove(BaseGraph::VertexIndex movedVertex) {
    if (*m_blockCountPtr == 1 && m_blockCreationProbability == 0)
        return {movedVertex, (*m_blocksPtr)[movedVertex], (*m_blocksPtr)[movedVertex], 0};


    const BlockIndex& currentBlock = (*m_blocksPtr)[movedVertex];

    BlockIndex newBlock;
    int addedBlocks = 0;
    if (m_createNewBlockDistribution(rng)){
        newBlock = *m_blockCountPtr;
        addedBlocks = 1;
    }
    else {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, *m_blockCountPtr - 2)(rng);
        if (newBlock >= currentBlock)
            newBlock++;
    }
    if (destroyingBlock(currentBlock, newBlock) && creatingNewBlock(newBlock)){
        return {0, (*m_blocksPtr)[0], (*m_blocksPtr)[0], -1};
    }
    return {movedVertex, currentBlock, newBlock, addedBlocks};
}

void UniformBlockProposer::setUp(const StochasticBlockModelFamily& sbmGraph) {
    m_blockCountPtr = &sbmGraph.getBlockCount();
    m_blocksPtr = &sbmGraph.getBlocks();
    m_vertexCountsPtr = &sbmGraph.getVertexCountsInBlocks();
}

double UniformBlockProposer::getLogProposalProbRatio(const BlockMove& move) const {
    if (creatingNewBlock(move.nextBlockIdx))
        return -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(*m_blockCountPtr);
    else if (destroyingBlock(move.prevBlockIdx, move.nextBlockIdx))
        return log(*m_blockCountPtr-1) - log(1-m_blockCreationProbability) + log(m_blockCreationProbability);
    return 0;
}

} // namespace FastMIDyNet
