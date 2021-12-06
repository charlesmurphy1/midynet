#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/blockproposer/uniform_blocks.h"
#include <random>


namespace FastMIDyNet {


UniformBlockProposer::UniformBlockProposer(size_t graphSize, double createNewBlockProbability):
    m_createNewBlockDistribution(createNewBlockProbability),
    m_vertexDistribution(0, graphSize-1),
    m_blockCreationProbability(createNewBlockProbability) {
    assertValidProbability(createNewBlockProbability);
}

BlockMove UniformBlockProposer::proposeMove() {
    if (m_blockCount == 1 && m_blockCreationProbability == 0)
        return {0, (*m_blockSequencePtr)[0], (*m_blockSequencePtr)[0]};


    auto movedVertex = m_vertexDistribution(rng);
    const BlockIndex& currentBlock = (*m_blockSequencePtr)[movedVertex];

    BlockIndex newBlock;
    int addedBlocks = 0;
    if (m_createNewBlockDistribution(rng)){
        newBlock = m_blockCount;
        addedBlocks = 1;
    }
    else {
        BlockIndex newBlock = std::uniform_int_distribution<BlockIndex>(0, m_blockCount-2)(rng);
        if (newBlock >= currentBlock)
            newBlock++;
    }
    if (destroyingBlock(currentBlock, newBlock) && creatingNewBlock(newBlock))
        return {0, (*m_blockSequencePtr)[0], (*m_blockSequencePtr)[0]};
    return {movedVertex, currentBlock, newBlock};
}

void UniformBlockProposer::setUp(const BlockSequence& blocks, size_t blockCount) {
    m_blockCount = blockCount;
    m_blockSequencePtr = &blocks;

    m_vertexCountInBlocks.clear();
    m_vertexCountInBlocks.resize(m_blockCount, 0);
    for (auto block : *m_blockSequencePtr)
        m_vertexCountInBlocks[block]++;
}

double UniformBlockProposer::getLogProposalProbRatio(const BlockMove& move) const {
    if (creatingNewBlock(move.nextBlockIdx))
        return -log(m_blockCreationProbability) + log(1-m_blockCreationProbability) - log(m_blockCount);
    else if (destroyingBlock(move.prevBlockIdx, move.nextBlockIdx))
        return log(m_blockCount-1) - log(1-m_blockCreationProbability) + log(m_blockCreationProbability);
    return 0;
}

void UniformBlockProposer::updateProbabilities(const BlockMove& move) {
    if (creatingNewBlock(move.nextBlockIdx)) {
        m_vertexCountInBlocks.push_back(0);
        m_blockCount++;
    }

    m_vertexCountInBlocks[move.prevBlockIdx]--;
    m_vertexCountInBlocks[move.nextBlockIdx]++;

    if (m_vertexCountInBlocks[move.prevBlockIdx] == 0) {
        m_vertexCountInBlocks.erase(m_vertexCountInBlocks.begin() + move.prevBlockIdx);
        m_blockCount--;
    }
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
    if (actualBlockCount != m_blockCount)
        throw ConsistencyError("UniformBlockProposer: Block count doesn't match with the number of "
                "non zero entries in vertexCountInBlocks.");
}

} // namespace FastMIDyNet
