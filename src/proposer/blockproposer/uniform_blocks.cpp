#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/blockproposer/uniform_blocks.h"
#include <random>


namespace FastMIDyNet {


UniformBlockProposer::UniformBlockProposer(size_t graphSize, double createNewBlockProbability):
    m_createNewBlockDistribution(createNewBlockProbability), m_vertexDistribution(0, graphSize-1),
    m_blockCreationProbability(createNewBlockProbability) {
    assertValidProbability(createNewBlockProbability);
}

BlockMove UniformBlockProposer::proposeMove() {
    const size_t& blockCount = *m_blockCountPtr;
    if (blockCount == 1 && m_blockCreationProbability == 0)
        return {0, 0, 0};


    auto movedVertex = m_vertexDistribution(rng);
    const BlockIndex& currentBlock = (*m_blockSequencePtr)[movedVertex];

    BlockIndex newBlock;
    if (m_createNewBlockDistribution(rng))
        newBlock = blockCount;
    else {
        BlockIndex newBlock = std::uniform_int_distribution<BlockIndex>(0, blockCount-2)(rng);
        if (newBlock >= currentBlock)
            newBlock++;
    }
    if (destroyingBlock(currentBlock, newBlock) && creatingNewBlock(newBlock))
        return {0, 0, 0};
    return {movedVertex, currentBlock, newBlock};
}

void UniformBlockProposer::setup(const BlockSequence& blockSequence, const size_t& blockCount) {
    m_blockCountPtr = &blockCount;
    m_blockSequencePtr = &blockSequence;

    m_vertexCountInBlocks.clear();
    m_vertexCountInBlocks.resize(blockCount, 0);
    for (auto block: blockSequence)
        m_vertexCountInBlocks[block]++;
}

double UniformBlockProposer::getLogProposalProbRatio(const BlockMove& move) const {
    const size_t& blockCount = *m_blockCountPtr;

    if (creatingNewBlock(move.nextBlockIdx))
        return -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(blockCount);
    else if (destroyingBlock(move.prevBlockIdx, move.nextBlockIdx))
        return log(blockCount-1) - log(1-m_blockCreationProbability) + log(m_blockCreationProbability);
    return 0;
}

void UniformBlockProposer::updateProbabilities(const BlockMove& move) {
    if (creatingNewBlock(move.nextBlockIdx))
        m_vertexCountInBlocks.push_back(0);

    m_vertexCountInBlocks[move.prevBlockIdx]--;
    m_vertexCountInBlocks[move.nextBlockIdx]++;

    if (destroyingBlock(move.prevBlockIdx, move.nextBlockIdx))
        m_vertexCountInBlocks.pop_back();
}

void UniformBlockProposer::checkConsistency() {
    std::vector<size_t> actualVertexCountInBlocks;

    for (auto vertexBlock: *m_blockSequencePtr) {
        if (vertexBlock >= actualVertexCountInBlocks.size())
            actualVertexCountInBlocks.resize(vertexBlock+1, 0);
        actualVertexCountInBlocks[vertexBlock]++;
    }

    if (actualVertexCountInBlocks.size() != m_vertexCountInBlocks.size())
        throw ConsistencyError("UniformBlockProposer: Incorrect size of vertex count in blocks.");

    size_t actualBlockCount = 0;
    for (size_t i=0; i<actualVertexCountInBlocks.size(); i++) {
        if (actualVertexCountInBlocks[i] != m_vertexCountInBlocks[i])
            throw ConsistencyError("UniformBlockProposer: Invalid vertex count in block " + std::to_string(i) + ".");

        if (actualVertexCountInBlocks[i] > 0)
            actualBlockCount++;
    }
    if (actualBlockCount != *m_blockCountPtr)
        throw ConsistencyError("UniformBlockProposer: Block count doesn't match with the number of "
                "non zero entries in vertexCountInBlocks.");
}

} // namespace FastMIDyNet
