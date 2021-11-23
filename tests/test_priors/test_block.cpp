#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"

using namespace FastMIDyNet;

const double GRAPH_SIZE=10;
const double BLOCK_COUNT=5;
const double POISSON_MEAN=5;
const BlockSequence BLOCK_SEQ={0,0,0,0,0,1,1,1,1,1};


class DummyBlockPrior: public BlockPrior {
    public:
        DummyBlockPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize, blockCountPrior) {};
        void sampleState() {
            BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
            blockSeq[BLOCK_COUNT - 1];
            setState( blockSeq );
        }
        void samplePriors() {};
        double getLogLikelihood() const { return 0.5; }
        double getLogPrior() { return 0.1; }
        void applyGraphMove(const GraphMove& move) { }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction( [&](){ m_state[move.vertexIdx] = move.nextBlockIdx; } );
        }

        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const { return 0; }

        double getLogPriorRatioFromGraphMove(const GraphMove& move) { return 0; }
        double getLogPriorRatioFromBlockMove(const BlockMove& move) { return 0; }

        double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; }
        double getLogJointRatioFromBlockMove(const BlockMove& move) { return 0; }


        void checkSelfConsistency() const { }
        bool getIsProcessed() { return m_isProcessed; }
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

TEST_F(TestBlockPrior, getBlockCount_returnCorrectBlockCount){
    size_t numBlocks = prior.getBlockCount();
    EXPECT_EQ(numBlocks, BLOCK_COUNT);
}

TEST_F(TestBlockPrior, computeVertexCountsInBlock_forSomeBlockSeq_returnCorrectVertexCount){
    BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
    blockSeq[0] = BLOCK_COUNT - 1;

    std::vector<size_t> actualVertexCount = prior.computeVertexCountsInBlock(blockSeq);
    EXPECT_EQ(actualVertexCount[0], GRAPH_SIZE - 1);
    EXPECT_EQ(actualVertexCount[BLOCK_COUNT - 1], 1);
}

TEST_F(TestBlockPrior, getSize_returnGraphSize){
    EXPECT_EQ(prior.getSize(), GRAPH_SIZE);
}

TEST_F(TestBlockPrior, getLogLikelihoodRatio_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0.);
}

TEST_F(TestBlockPrior, getLogLikelihoodRatio_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), 0.);
}

TEST_F(TestBlockPrior, getLogPrior_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogPriorRatioFromGraphMove(move), 0);
}

TEST_F(TestBlockPrior, getLogPrior_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogPriorRatioFromBlockMove(move), 0.);
}


TEST_F(TestBlockPrior, getLogJoint_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogJointRatioFromGraphMove(move), 0.);
}

TEST_F(TestBlockPrior, getLogJoint_forSomeBlockMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogJointRatioFromBlockMove(move), 0.);
}

TEST_F(TestBlockPrior, applyMove_forSomeGraphMove_doNothing){
    GraphMove move({{0,0}}, {});
    prior.applyGraphMove(move);
}

TEST_F(TestBlockPrior, applyMove_forSomeBlockMove_changeBlockOfNode0From0To1){
    BlockMove move = {0, 0, 1};
    prior.applyBlockMove(move);
}

// void sampleState() {  };
// void samplePriors() { };
// double getLogLikelihood() const { return 0.; }
// double getLogPrior() { return 0.; };
// double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const { if (move.prevBlockIdx != move.nextBlockIdx) return -INFINITY; else return 0.;}
// double getLogPriorRatioFromBlockMove(const BlockMove& move) { return 0; }


class TestBlockDeltaPrior: public::testing::Test{
    public:
        FastMIDyNet::BlockDeltaPrior prior = FastMIDyNet::BlockDeltaPrior(BLOCK_SEQ);
        void SetUp() { }

        bool isCorrectBlockSequence(const FastMIDyNet::BlockSequence& blockSeq){
            if (blockSeq.size() != BLOCK_SEQ.size()) return false;
            for (size_t i=0; i<blockSeq.size(); ++i){
                if (blockSeq[i] != BLOCK_SEQ[i]) return false;
            }
            return true;
        }
};

TEST_F(TestBlockDeltaPrior, sampleState_doNothing){
    prior.sampleState();
    EXPECT_TRUE( isCorrectBlockSequence( prior.getState() ) );
}

TEST_F(TestBlockDeltaPrior, samplePriors_doNothing){
    prior.samplePriors();
    EXPECT_TRUE( isCorrectBlockSequence( prior.getState() ) );
}

TEST_F(TestBlockDeltaPrior, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0);
}

TEST_F(TestBlockDeltaPrior, getLogPrior_return0){
    EXPECT_EQ(prior.getLogPrior(), 0);
}

TEST_F(TestBlockDeltaPrior, getLogLikelihoodRatioFromBlockMove_forSomePreservingBlockMove_return0){
    FastMIDyNet::BlockMove move = {0, 0, 0, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), 0);
}

TEST_F(TestBlockDeltaPrior, getLogLikelihoodRatioFromBlockMove_forSomeNonPreservingBlockMove_returnMinusInf){
    FastMIDyNet::BlockMove move = {0, 0, 1, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), -INFINITY);
}

TEST_F(TestBlockDeltaPrior, getLogPriorRatioFromBlockMove_forSomePreservingBlockMove_return0){
    FastMIDyNet::BlockMove move = {0, 0, 0, 0};
    EXPECT_EQ(prior.getLogPriorRatioFromBlockMove(move), 0);
}

TEST_F(TestBlockDeltaPrior, getLogPriorRatioFromBlockMove_forSomeNonPreservingBlockMove_return0){
    FastMIDyNet::BlockMove move = {0, 0, 1, 0};
    EXPECT_EQ(prior.getLogPriorRatioFromBlockMove(move), 0);
}

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

TEST_F(TestBlockUniformPrior, sample_returnBlockSeqWithExpectedSizeAndBlockCount){
    prior.sample();
    auto blockSeq = prior.getState();
    EXPECT_EQ(prior.getState().size(), GRAPH_SIZE);
    EXPECT_TRUE(*max_element(blockSeq.begin(), blockSeq.end()) <= BLOCK_COUNT - 1);
}

TEST_F(TestBlockUniformPrior, getLogLikelihood_fromSomeRandomBlockSeq_returnCorrectLogLikelihood){
    for (size_t i = 0; i < 10; i++) {
        prior.sample();
        double logLikelihood = prior.getLogLikelihood();
        double expectedLogLikelihood = -logMultisetCoefficient(GRAPH_SIZE, prior.getBlockCount());
        EXPECT_EQ(expectedLogLikelihood, logLikelihood);
    }
}

TEST_F(TestBlockUniformPrior, getLogLikelihoodRatio_fromSomeBlockMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {2, 0, 1};

    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromBlockMove(move);
    double expectedLogLikelihoodRatio = -prior.getLogLikelihood();
    prior.applyBlockMove(move);
    expectedLogLikelihoodRatio += prior.getLogLikelihood();
    EXPECT_EQ(expectedLogLikelihoodRatio, actualLogLikelihoodRatio);
}

TEST_F(TestBlockUniformPrior, applyMove_forSomeBlockMove_changeBlockOfNode2From0To1){
    BlockMove move = {2, 0, 1};
    EXPECT_EQ(prior.getState()[2], 0);
    prior.applyBlockMove(move);
    EXPECT_EQ(prior.getState()[2], 1);
}

TEST_F(TestBlockUniformPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockUniformPrior, checkSelfConsistency_inconsistenBlockSeqWithBlockCOunt_ThrowConsistencyError){
    blockCountPrior.setState(20); // expected to be 5 in prior.
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
}
