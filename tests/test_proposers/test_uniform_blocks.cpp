#include "gtest/gtest.h"
#include "FastMIDyNet/proposer/blockproposer/uniform_blocks.h"
#include "FastMIDyNet/types.h"
#include "fixtures.hpp"


const double NEW_BLOCK_PROBABILITY = .1;
const FastMIDyNet::BlockIndex BLOCK_COUNT = 2;
const FastMIDyNet::BlockSequence BLOCK_SEQUENCE = {0, 0, 1, 0, 0};

class TestUniformBlockProposer: public::testing::Test {
    public:
        FastMIDyNet::UniformBlockProposer blockProposer = FastMIDyNet::UniformBlockProposer(5, NEW_BLOCK_PROBABILITY);
        void SetUp() {
            blockProposer.setup(BLOCK_SEQUENCE, BLOCK_COUNT);
        }
};


TEST_F(TestUniformBlockProposer, setup_anyBlockSequence_correctCountOfVerticesInBlocks) {
    auto vertexCountInBlocks = blockProposer.getVertexCountInBlocks();
    EXPECT_EQ(vertexCountInBlocks[0], 4);
    EXPECT_EQ(vertexCountInBlocks[1], 1);
    EXPECT_EQ(vertexCountInBlocks.size(), BLOCK_COUNT);
}

TEST_F(TestUniformBlockProposer, getLogProposalProbRatio_sameBlockmove_return0) {
    FastMIDyNet::BlockMove move = {0, 0, 0};
    EXPECT_EQ(blockProposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestUniformBlockProposer, getLogProposalProbRatio_moveBetweenExistingAndNonEmptyBlocks_return0) {
    FastMIDyNet::BlockMove move = {0, 0, 1};
    EXPECT_EQ(blockProposer.getLogProposalProbRatio(move), 0);
}

TEST_F(TestUniformBlockProposer, getLogProposalProbRatio_createNewBlock_returnCorrectRatio) {
    FastMIDyNet::BlockMove move = {0, 0, 2};
    EXPECT_EQ(blockProposer.getLogProposalProbRatio(move),
            -log(NEW_BLOCK_PROBABILITY)+log(1-NEW_BLOCK_PROBABILITY)-log(BLOCK_COUNT));
}

TEST_F(TestUniformBlockProposer, getLogProposalProbRatio_destroyBlock_returnCorrectRatio) {
    FastMIDyNet::BlockMove move = {2, 1, 0};
    EXPECT_EQ(blockProposer.getLogProposalProbRatio(move),
            log(BLOCK_COUNT-1)-log(1-NEW_BLOCK_PROBABILITY)+log(NEW_BLOCK_PROBABILITY));
}

TEST_F(TestUniformBlockProposer, updateProbabilities_moveBetweenExistsingAndNonEmptyBlocks_incrementNewAndDecrementPreviousBlocks) {
    FastMIDyNet::BlockMove move = {0, 0, 1};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(blockProposer.getVertexCountInBlocks(), std::vector<size_t>({3, 2}));
}

TEST_F(TestUniformBlockProposer, updateProbabilities_creatingNewBlock_createNewEntry) {
    FastMIDyNet::BlockMove move = {0, 0, 2};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(blockProposer.getVertexCountInBlocks(), std::vector<size_t>({3, 1, 1}));
}

TEST_F(TestUniformBlockProposer, updateProbabilities_destroyingMiddleBlock_blockRemovedInVertexCounts) {
    size_t blockCount = 3;
    FastMIDyNet::BlockSequence blockSequence = {0, 0, 1, 2, 2};
    blockProposer.setup(blockSequence, blockCount);

    FastMIDyNet::BlockMove move = {2, 1, 2};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(blockProposer.getVertexCountInBlocks(), std::vector<size_t>({2, 3}));
}

TEST_F(TestUniformBlockProposer, updateProbabilities_noBlockCreatedOrDestroyed_blockCountConstant) {
    FastMIDyNet::BlockMove move = {0, 0, 1};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(BLOCK_COUNT, blockProposer.getInternalBlockCount());
}

TEST_F(TestUniformBlockProposer, updateProbabilities_creatingNewBlock_blockCountIncrement) {
    FastMIDyNet::BlockMove move = {0, 0, 2};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(BLOCK_COUNT+1, blockProposer.getInternalBlockCount());
}

TEST_F(TestUniformBlockProposer, updateProbabilities_destroyingMiddleBlock_blockCountDecrement) {
    size_t blockCount = 3;
    FastMIDyNet::BlockSequence blockSequence = {0, 0, 1, 2, 2};
    blockProposer.setup(blockSequence, blockCount);

    FastMIDyNet::BlockMove move = {2, 1, 2};
    blockProposer.updateProbabilities(move);

    EXPECT_EQ(blockCount-1, blockProposer.getInternalBlockCount());
}

TEST_F(TestUniformBlockProposer, checkConsistency_lowerBlockCount_throwConsistencyError) {
    size_t wrongCount = 1;
    blockProposer.setup(BLOCK_SEQUENCE, wrongCount);
    EXPECT_THROW(blockProposer.checkConsistency(), FastMIDyNet::ConsistencyError);
}

TEST_F(TestUniformBlockProposer, checkConsistency_higherBlockCount_throwConsistencyError) {
    FastMIDyNet::BlockSequence wrongBlocks = {0, 0, 0, 0, 0};
    blockProposer.setup(wrongBlocks, BLOCK_COUNT);
    EXPECT_THROW(blockProposer.checkConsistency(), FastMIDyNet::ConsistencyError);
}
