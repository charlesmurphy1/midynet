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

class BlockPriorTest: public ::testing::Test {
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

TEST_F(BlockPriorTest, getBlockCount_returnCorrectBlockCount){
    size_t numBlocks = prior.getBlockCount();
    EXPECT_EQ(numBlocks, BLOCK_COUNT);
}

TEST_F(BlockPriorTest, computeVertexCountsInBlock_forSomeBlockSeq_returnCorrectVertexCount){
    BlockSequence blockSeq = BlockSequence(GRAPH_SIZE, 0);
    blockSeq[0] = BLOCK_COUNT - 1;

    CounterMap<BlockIndex> actualVertexCount = prior.computeVertexCounts(blockSeq);
    EXPECT_EQ(actualVertexCount[0], GRAPH_SIZE - 1);
    EXPECT_EQ(actualVertexCount[BLOCK_COUNT - 1], 1);
}

TEST_F(BlockPriorTest, getSize_returnGraphSize){
    EXPECT_EQ(prior.getSize(), GRAPH_SIZE);
}

TEST_F(BlockPriorTest, getLogLikelihoodRatio_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0.);
}

TEST_F(BlockPriorTest, getLogLikelihoodRatio_forSomeLabelMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0.);
}

// TEST_F(BlockPriorTest, getLogPrior_forSomeLabelMove_return0){
//     BlockMove move = {0, 0, 1};
//     displayVector(prior.getState());
//     EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0.);
// }

TEST_F(BlockPriorTest, getLogJoint_forSomeGraphMove_return0){
    GraphMove move({{0,0}}, {});
    EXPECT_EQ(prior.getLogJointRatioFromGraphMove(move), 0.);
}

// TEST_F(BlockPriorTest, getLogJoint_forSomeLabelMove_return0){
//     BlockMove move = {0, 0, 1};
//     EXPECT_EQ(prior.getLogJointRatioFromLabelMove(move), 0.);
// }

TEST_F(BlockPriorTest, applyMove_forSomeGraphMove_doNothing){
    GraphMove move({{0,0}}, {});
    prior.applyGraphMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(BlockPriorTest, applyMove_forSomeLabelMove_changeBlockOfNode0From0To1){
    BlockMove move = {0, 0, 1};
    prior.applyLabelMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(BlockPriorTest, reducePartition_forSomeIrreduciblePartition_returnSame){
    BlockSequence partition = {0, 0, 1, 1, 1, 2};
    BlockSequence reduced = prior.reducePartition(partition);
    EXPECT_EQ(partition, reduced);
}

TEST_F(BlockPriorTest, reducePartition_forSomeUnsortedPartition_returnSorted){
    BlockSequence partition = {0, 0, 2, 1, 1, 1}, expected = {0, 0, 1, 2, 2, 2};
    BlockSequence reduced = prior.reducePartition(partition);

    EXPECT_EQ(expected, reduced);
}

TEST_F(BlockPriorTest, reducePartition_forSomeReduciblePartition_returnReduced){
    BlockSequence partition = {0, 0, 2, 2, 2, 2}, expected = {0, 0, 1, 1, 1, 1};
    BlockSequence reduced = prior.reducePartition(partition);

    EXPECT_EQ(expected, reduced);
}

TEST_F(BlockPriorTest, reducePartition_forSomeUnsortedAndReduciblePartition_returnSortedAndReduced){
    BlockSequence partition = {2, 2, 0, 0, 0, 0}, expected = {0, 0, 1, 1, 1, 1};
    BlockSequence reduced = prior.reducePartition(partition);

    EXPECT_EQ(expected, reduced);
}

class BlockDeltaPriorTest: public::testing::Test{
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
            for (BlockIndex i=0; i<blockSeq.size(); ++i){
                if (blockSeq[i] != BLOCK_SEQ[i]) return false;
            }
            return true;
        }
};

TEST_F(BlockDeltaPriorTest, sampleState_doNothing){
    prior.sampleState();
    EXPECT_TRUE( isCorrectBlockSequence( prior.getState() ) );
}

TEST_F(BlockDeltaPriorTest, samplePriors_doNothing){
    prior.samplePriors();
    EXPECT_TRUE( isCorrectBlockSequence( prior.getState() ) );
}

TEST_F(BlockDeltaPriorTest, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0);
}

TEST_F(BlockDeltaPriorTest, getLogPrior_return0){
    EXPECT_EQ(prior.getLogPrior(), 0);
}

TEST_F(BlockDeltaPriorTest, getLogLikelihoodRatioFromLabelMove_forSomePreservingLabelMove_return0){
    BlockMove move = {0, 0, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), 0);
}

TEST_F(BlockDeltaPriorTest, getLogLikelihoodRatioFromLabelMove_forSomeNonPreservingLabelMove_returnMinusInf){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromLabelMove(move), -INFINITY);
}

TEST_F(BlockDeltaPriorTest, getLogPriorRatioFromLabelMove_forSomePreservingLabelMove_return0){
    BlockMove move = {0, 0, 0};
    EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0);
}

TEST_F(BlockDeltaPriorTest, getLogPriorRatioFromLabelMove_forSomeNonPreservingLabelMove_return0){
    BlockMove move = {0, 0, 1};
    EXPECT_EQ(prior.getLogPriorRatioFromLabelMove(move), 0);
}

class BlockUniformPriorTest: public::testing::Test{
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

TEST_F(BlockUniformPriorTest, sample_returnBlockSeqWithExpectedSizeAndBlockCount){
    prior.sample();
    auto blockSeq = prior.getState();
    EXPECT_EQ(prior.getState().size(), GRAPH_SIZE);
    EXPECT_TRUE(*max_element(blockSeq.begin(), blockSeq.end()) <= BLOCK_COUNT - 1);
}

TEST_F(BlockUniformPriorTest, getLogLikelihood_fromSomeRandomBlockSeq_returnCorrectLogLikelihood){
    for (size_t i = 0; i < 10; i++) {
        prior.sample();
        double logLikelihood = prior.getLogLikelihood();
        double expectedLogLikelihood = -GRAPH_SIZE * log(prior.getBlockCount());
        EXPECT_FLOAT_EQ(expectedLogLikelihood, logLikelihood);
    }
}

TEST_F(BlockUniformPriorTest, getLogLikelihoodRatio_fromSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {2, 0, 1};

    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double expectedLogLikelihoodRatio = -prior.getLogLikelihood();
    prior.applyLabelMove(move);
    expectedLogLikelihoodRatio += prior.getLogLikelihood();
    EXPECT_FLOAT_EQ(expectedLogLikelihoodRatio, actualLogLikelihoodRatio);
}

TEST_F(BlockUniformPriorTest, applyMove_forSomeLabelMove_changeBlockOfNode2From0To1){
    BlockMove move = {2, 0, 1};
    EXPECT_EQ(prior.getState()[2], 0);
    prior.applyLabelMove(move);
    EXPECT_EQ(prior.getState()[2], 1);
}

TEST_F(BlockUniformPriorTest, checkSelfConsistency_noError_noThrow){
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}


class BlockUniformHyperPriorTest: public::testing::Test{
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

TEST_F(BlockUniformHyperPriorTest, sampleState_generateConsistentState){
    prior.sampleState();
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(BlockUniformHyperPriorTest, getLogLikelihood_returnCorrectLogLikehood){
    const auto& nr = prior.getVertexCounts();
    EXPECT_LE( prior.getLogLikelihood(), 0 );
    EXPECT_FLOAT_EQ( prior.getLogLikelihood(), -logMultinomialCoefficient( nr.getValues() ) - logBinomialCoefficient(prior.getSize() - 1, prior.getBlockCount() - 1) );
}

TEST_F(BlockUniformHyperPriorTest, applyLabelMove_ForSomeLabelMove_getConsistentState){
    BlockMove move = {0, prior.getBlockOfIdx(0), findLabelMove(0)};
    prior.applyLabelMove(move);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(BlockUniformHyperPriorTest, getLogLikelihoodRatioFromLabelMove_forSomeLabelMove_returnCorrectLogLikelihoodRatio){
    BlockMove move = {0, prior.getBlockOfIdx(0), findLabelMove(0)};
    double actualLogLikelihoodRatio = prior.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = prior.getLogLikelihood();

    prior.applyLabelMove(move);

    double logLikelihoodAfter = prior.getLogLikelihood();

    EXPECT_FLOAT_EQ(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore);
}
