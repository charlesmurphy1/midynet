#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

const double GRAPH_SIZE=10;
const double BLOCK_COUNT=5;
const double POISSON_MEAN=5;
// const std::vector<size_t> TESTED_INTEGERS;
using namespace FastMIDyNet;

class DummyBlockPrior: public BlockPrior {
    public:
        DummyBlockPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize, blockCountPrior) {};
        BlockSequence sample() {
            BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
            blockSeq[BLOCK_COUNT - 1];
            return blockSeq;
        }
        double getLogLikelihood(const BlockSequence& state) const { return 0.5; }
        double getLogLikelihoodRatio(const MultiBlockMove& move) const { return 3;};
        double getLogPriorRatio(const MultiBlockMove& move) {
            if (!m_isProcessed) return 2.;
            else return 0.;
        };
        double getLogJointRatio(const MultiBlockMove& move) {
            if (!m_isProcessed) return getLogPriorRatio(move) + getLogLikelihoodRatio(move);
            else return 0.;
        };
        double getLogPrior() { return 0.1; }
        void applyMove(const BlockMove& move) {
            if (!m_isProcessed)
                m_state[move.vertexIdx] = move.nextBlockIdx;
            m_isProcessed = true;
        };

        void checkSelfConsistency() const { }
        bool getIsProcessed() { return m_isProcessed; }

        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = FastMIDyNet::BlockCountPoissonPrior(POISSON_MEAN);

};

class TestBlockPrior: public ::testing::Test {
    public:

        FastMIDyNet::BlockCountPoissonPrior blockMovePrior = FastMIDyNet::BlockCountPoissonPrior(POISSON_MEAN);
        DummyBlockPrior prior = DummyBlockPrior(GRAPH_SIZE, blockMovePrior);
        void SetUp() {
            BlockSequence blockSeq;
            for (size_t idx = 0; idx < GRAPH_SIZE; idx++) {
                blockSeq.push_back(0);
            }
            blockSeq[0] = BLOCK_COUNT - 1;
            prior.setState(blockSeq);
            prior.computationFinished();
        }
};

class TestBlockUniformPrior: public::testing::Test{
    public:
        FastMIDyNet::BlockCountPoissonPrior blockCountPrior = FastMIDyNet::BlockCountPoissonPrior(POISSON_MEAN);
        FastMIDyNet::BlockUniformPrior prior = FastMIDyNet::BlockUniformPrior(GRAPH_SIZE, blockCountPrior);
        void SetUp() {
            BlockSequence blockSeq;
            for (size_t idx = 0; idx < GRAPH_SIZE; idx++) {
                blockSeq.push_back(0);
            }
            blockSeq[0] = BLOCK_COUNT - 1;
            prior.setState(blockSeq);
        }
};

TEST_F(TestBlockPrior, getBlockCount_returnCorrectBlockCount){
    size_t numBlocks = prior.getBlockCount();
    EXPECT_EQ(numBlocks, BLOCK_COUNT);
}

TEST_F(TestBlockPrior, getVertexCount_forSomeBlockSeq_returnCorrectVertexCount){
    BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
    blockSeq[0] = BLOCK_COUNT - 1;

    std::vector<size_t> actualVertexCount = prior.getVertexCount(blockSeq);
    EXPECT_EQ(actualVertexCount[0], GRAPH_SIZE - 1);
    EXPECT_EQ(actualVertexCount[BLOCK_COUNT - 1], 1);
}

TEST_F(TestBlockPrior, getSize_returnGraphSize){
    EXPECT_EQ(prior.getSize(), GRAPH_SIZE);
}

TEST_F(TestBlockPrior, getLogLikelihoodRatio_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.BlockPrior::getLogLikelihoodRatio(move), 0);
}

TEST_F(TestBlockPrior, getLogLikelihoodRatio_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.BlockPrior::getLogLikelihoodRatio(move), 3);
}

TEST_F(TestBlockPrior, getLogPrior_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.BlockPrior::getLogPriorRatio(move), 0);
}

TEST_F(TestBlockPrior, getLogPrior_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.BlockPrior::getLogPriorRatio(move), 2);
}


TEST_F(TestBlockPrior, getLogJoint_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.BlockPrior::getLogJointRatio(move), 0);
}

TEST_F(TestBlockPrior, getLogJoint_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.BlockPrior::getLogJointRatio(move), 2);
}

TEST_F(TestBlockPrior, applyMove_forSomeGraphMove_doNothing){
    GraphMove move({{0,0}}, {});
    prior.BlockPrior::applyMove(move);
}

TEST_F(TestBlockPrior, applyMove_forSomeBlockMove_changeBlockOfNode0From0To1){
    BlockMove move = {0, 0, 1};
    prior.applyMove(move);
}

TEST_F(TestBlockUniformPrior, sample_returnBlockSeqWithExpectedSizeAndBlockCount){
    BlockSequence blockSeq = prior.sample();
    EXPECT_EQ(blockSeq.size(), GRAPH_SIZE);
    EXPECT_TRUE(*max_element(blockSeq.begin(), blockSeq.end()) <= BLOCK_COUNT - 1);
}

TEST_F(TestBlockUniformPrior, getLogLikelihood_fromSomeRandomBlockSeq_returnCorrectLogLikelihood){
    for (size_t i = 0; i < 10; i++) {
        BlockSequence blockSeq = prior.sample();
        double logLikelihood = prior.getLogLikelihood(blockSeq);
        double expectedLogLikelihood = -logMultisetCoefficient(GRAPH_SIZE, prior.getBlockCount());
        EXPECT_EQ(expectedLogLikelihood, logLikelihood);
    }
}

TEST_F(TestBlockUniformPrior, getLogLikelihoodRatio_fromSomeMultiBlockMove_returnCorrectLogLikelihoodRatio){
    MultiBlockMove move(1, {2, 0, 1});

    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatio(move);
    double expectedLogLikelihoodRatio = -prior.getLogLikelihood(prior.getState());
    prior.BlockPrior::applyMove(move);
    expectedLogLikelihoodRatio += prior.getLogLikelihood(prior.getState());
    EXPECT_EQ(expectedLogLikelihoodRatio, actualLogLikelihoodRatio);
}

TEST_F(TestBlockUniformPrior, applyMove_forSomeBlockMove_changeBlockOfNode2From0To1){
    BlockMove move = {2, 0, 1};
    EXPECT_EQ(prior.getState()[2], 0);
    prior.applyMove(move);
    EXPECT_EQ(prior.getState()[2], 1);
}

TEST_F(TestBlockUniformPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockUniformPrior, checkSelfConsistency_inconsistenBlockSeqWithBlockCOunt_ThrowConsistencyError){
    MultiBlockMove move(1, {0,0,20});
    prior.BlockPrior::applyMove(move);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
}
