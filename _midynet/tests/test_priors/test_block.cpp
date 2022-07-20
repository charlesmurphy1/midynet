#include "gtest/gtest.h"
#include <vector>
#include <iostream>

#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/block.h"
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
    private:
        BlockCountDeltaPrior m_blockCountDeltaPrior = BlockCountDeltaPrior();
        void _samplePriors() override {};
        const double _getLogPrior() const override { return 0.1; }
    public:
        DummyBlockPrior(size_t size, size_t blockCount):
            BlockPrior() {
                setSize(size);
                m_blockCountDeltaPrior.setState(blockCount);
                setBlockCountPrior(m_blockCountDeltaPrior);
            }
        void sampleState() {
            BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
            blockSeq[BLOCK_COUNT - 1] = 1;
            setState( blockSeq );
        }
        const double getLogLikelihood() const override { return 0.5; }
        void _applyLabelMove(const BlockMove& move) override {
            m_state[move.vertexIndex] = move.nextLabel;
        }

        const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const { return 0; }

        void checkSelfConsistency() const override { }
        bool getIsProcessed() { return m_isProcessed; }
};

class TestBlockPrior: public ::testing::Test {
    public:

        DummyBlockPrior prior = DummyBlockPrior(GRAPH_SIZE, BLOCK_COUNT);
        void SetUp() {
            BlockSequence blockSeq;
            for (size_t idx = 0; idx < GRAPH_SIZE; idx++) {
                blockSeq.push_back(0);
            }
            blockSeq[0] = BLOCK_COUNT - 1;
            prior.setState(blockSeq);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(TestBlockPrior, getBlockCount_returnCorrectBlockCount){
    size_t numBlocks = prior.getBlockCount();
    EXPECT_EQ(numBlocks, BLOCK_COUNT);
}

TEST_F(TestBlockPrior, computeVertexCountsInBlock_forSomeBlockSeq_returnCorrectVertexCount){
    BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
    blockSeq[0] = BLOCK_COUNT - 1;

    CounterMap<size_t> actualVertexCount = prior.computeVertexCounts(blockSeq);
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

TEST_F(TestBlockPrior, getLogLikelihoodRatio_forSomeLabelMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0.);
}

// TEST_F(TestBlockPrior, getLogPrior_forSomeLabelMove_return0){
//     BlockMove move = {0, 0, 1};
//     displayVector(prior.getState());
//     EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0.);
// }

TEST_F(TestBlockPrior, getLogJoint_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogJointRatioFromGraphMove(move), 0.);
}

// TEST_F(TestBlockPrior, getLogJoint_forSomeLabelMove_return0){
//     BlockMove move = {0, 0, 1};
//     EXPECT_EQ(prior.getLogJointRatioFromLabelMove(move), 0.);
// }

TEST_F(TestBlockPrior, applyMove_forSomeGraphMove_doNothing){
    GraphMove move({{0,0}}, {});
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockPrior, applyMove_forSomeLabelMove_changeBlockOfNode0From0To1){
    BlockMove move = {0, 0, 1};
    prior.applyLabelMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

class TestBlockDeltaPrior: public::testing::Test{
    public:
        BlockDeltaPrior prior = BlockDeltaPrior(BLOCK_SEQ);
        void SetUp() {
        prior.checkSafety();
    }
    void TearDown(){
        prior.checkConsistency();
    }

        bool isCorrectBlockSequence(const BlockSequence& blockSeq){
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

TEST_F(TestBlockDeltaPrior, getLogLikelihoodRatioFromLabelMove_forSomePreservingLabelMove_return0){
    BlockMove move = {0, 0, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0);
}

TEST_F(TestBlockDeltaPrior, getLogLikelihoodRatioFromLabelMove_forSomeNonPreservingLabelMove_returnMinusInf){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), -INFINITY);
}

TEST_F(TestBlockDeltaPrior, getLogPriorRatioFromLabelMove_forSomePreservingLabelMove_return0){
    BlockMove move = {0, 0, 0};
    EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0);
}

TEST_F(TestBlockDeltaPrior, getLogPriorRatioFromLabelMove_forSomeNonPreservingLabelMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0);
}

class TestBlockUniformPrior: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
        BlockUniformPrior prior = BlockUniformPrior(GRAPH_SIZE, blockCountPrior);
        void SetUp() {
            BlockSequence blockSeq;
            for (size_t idx = 0; idx < GRAPH_SIZE; idx++) {
                blockSeq.push_back(0);
            }
            blockSeq[0] = BLOCK_COUNT - 1;
            prior.setState(blockSeq);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
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
        std::cout << "getSize: " << prior.getSize() << std::endl;
        std::cout << "getBlockCount: " << prior.getBlockCount() << std::endl;
        double expectedLogLikelihood = -GRAPH_SIZE * log(prior.getBlockCount());
        EXPECT_FLOAT_EQ(expectedLogLikelihood, logLikelihood);
    }
}

TEST_F(TestBlockUniformPrior, getLogLikelihoodRatio_fromSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {2, 0, 1};

    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double expectedLogLikelihoodRatio = -prior.getLogLikelihood();
    prior.applyLabelMove(move);
    expectedLogLikelihoodRatio += prior.getLogLikelihood();
    EXPECT_FLOAT_EQ(expectedLogLikelihoodRatio, actualLogLikelihoodRatio);
}

TEST_F(TestBlockUniformPrior, applyMove_forSomeLabelMove_changeBlockOfNode2From0To1){
    BlockMove move = {2, 0, 1};
    EXPECT_EQ(prior.getState()[2], 0);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getState()[2], 1);
}

TEST_F(TestBlockUniformPrior, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}


class TestBlockUniformHyperPrior: public::testing::Test{
    public:
        BlockCountPoissonPrior blockCountPrior = BlockCountPoissonPrior(POISSON_MEAN);
        BlockUniformHyperPrior prior = BlockUniformHyperPrior(GRAPH_SIZE, blockCountPrior);
        void SetUp() {
            prior.sample();
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
        BlockIndex findLabelMove(BaseGraph::VertexIndex idx){
            BlockIndex blockIdx = prior.getBlockOfIdx(idx);
            if (blockIdx == prior.getBlockCount() - 1) return blockIdx - 1;
            else return blockIdx + 1;
        }
};

TEST_F(TestBlockUniformHyperPrior, sampleState_generateConsistentState){
    prior.sampleState();
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockUniformHyperPrior, getLogLikelihood_returnCorrectLogLikehood){
    const auto& nr = prior.getVertexCounts();
    EXPECT_LE( prior.getLogLikelihood(), 0 );
    EXPECT_FLOAT_EQ( prior.getLogLikelihood(), -logMultinomialCoefficient( nr.getValues() ) - logBinomialCoefficient(prior.getSize() - 1, prior.getBlockCount() - 1) );
}

TEST_F(TestBlockUniformHyperPrior, applyLabelMove_ForSomeLabelMove_getConsistentState){
    BlockMove move = {0, prior.getBlockOfIdx(0), findLabelMove(0)};
    prior.applyLabelMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockUniformHyperPrior, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, prior.getBlockOfIdx(0), findLabelMove(0)};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();

    prior.applyLabelMove(move);

    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_FLOAT_EQ(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore);
}
