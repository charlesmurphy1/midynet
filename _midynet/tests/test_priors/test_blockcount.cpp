#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class DummyBlockCountPrior: public FastMIDyNet::BlockCountPrior {
    public:
        void sampleState() {}
        const double getLogLikelihoodFromState(const size_t& state) const { return state; }
        const double getLogPrior() { return 0; }

        void _checkSelfConsistency() const override {}
        bool getIsProcessed() { return m_isProcessed; }
};

class TestBlockCountPrior: public ::testing::Test {
    public:
        DummyBlockCountPrior prior;
        void SetUp() { prior.setState(0); prior.computationFinished(); }
};


TEST_F(TestBlockCountPrior, getStateAfterMove_blockMove_returnCorrectBlockNumberIfVertexInNewBlock) {
    for (auto currentBlockNumber: {1, 2, 5, 10}){
        prior.setState(currentBlockNumber);
        for (FastMIDyNet::BlockIndex nextBlockIdx : {0, 1, 2, 6}){
            FastMIDyNet::BlockMove blockMove = {0, 0, nextBlockIdx};
            if (nextBlockIdx >= currentBlockNumber) blockMove.addedBlocks = 1;


            if (currentBlockNumber > nextBlockIdx)
                EXPECT_EQ(prior.getStateAfterBlockMove(blockMove), currentBlockNumber);
            else
                EXPECT_EQ(prior.getStateAfterBlockMove(blockMove), currentBlockNumber + 1);
        }
    }

}

TEST_F(TestBlockCountPrior, getLogLikelihoodRatio_noNewblockMove_return0) {
    prior.setState(5);
    FastMIDyNet::BlockMove blockMove = {0, 0, 1, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(blockMove), 0);
}

TEST_F(TestBlockCountPrior, getLogLikelihoodRatio_newblockMove_returnCorrectRatio) {
    prior.setState(5);
    FastMIDyNet::BlockMove blockMove = {0, 0, 7, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(blockMove), 1);
}

TEST_F(TestBlockCountPrior, applyMove_noNewblockMove_blockNumberUnchangedIsProcessedIsTrue) {
    prior.setState(5);
    FastMIDyNet::BlockMove blockMove = {0, 0, 2, 0};
    prior.applyBlockMove(blockMove);
    EXPECT_EQ(prior.getState(), 5);
}

TEST_F(TestBlockCountPrior, applyMove_newblockMove_blockNumberIncrementsIsProcessedIsTrue) {
    prior.setState(5);
    FastMIDyNet::BlockMove blockMove = {0, 0, 7, 1};
    prior.applyBlockMove(blockMove);
    EXPECT_EQ(prior.getState(), 6);
}

/* BLOCK COUNT DELTA PRIOR TEST: BEGIN */
class TestBlockCountDeltaPrior: public::testing::Test{
    public:
        size_t blockCount = 5;
        FastMIDyNet::BlockCountDeltaPrior prior={blockCount};
};

TEST_F(TestBlockCountDeltaPrior, sampleState_doNothing){
    EXPECT_EQ(prior.getState(), blockCount);
    prior.sampleState();
    EXPECT_EQ(prior.getState(), blockCount);
}

TEST_F(TestBlockCountDeltaPrior, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0.);
}

TEST_F(TestBlockCountDeltaPrior, getLogLikelihoodFromState_forSomeStateDifferentThan5_returnMinusInf){
    EXPECT_EQ(prior.getLogLikelihoodFromState(10), -INFINITY);
}

TEST_F(TestBlockCountDeltaPrior, getLogLikelihoodRatio_forSomeBlockMovePreservingBlockCount_return0){
    FastMIDyNet::BlockMove move = {0, 0, 2, 0};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), 0);
}

TEST_F(TestBlockCountDeltaPrior, getLogLikelihoodRatio_forSomeGraphMoveNotPreservingBlockCount_return0){
    FastMIDyNet::BlockMove move = {0, 0, 6, 1};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromBlockMove(move), -INFINITY);
}
/* BLOCK COUNT DELTA PRIOR TEST: END */

/* BLOCK COUNT POISSON PRIOR TEST */
class TestBlockCountPoissonPrior: public::testing::Test{
    public:
        FastMIDyNet::BlockCountPoissonPrior prior={POISSON_MEAN};
};

TEST_F(TestBlockCountPoissonPrior, getLogLikelihood_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihoodFromState(x),
                    FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(TestBlockCountPoissonPrior, getLogPrior_returns0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

TEST_F(TestBlockCountPoissonPrior, checkSelfConsistency_noError_noThrow) {
    prior.setState(1);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
    prior.setState(2);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestBlockCountPoissonPrior, checkSelfConsistency_negativeMean_throwConsistencyError) {
    prior={-2};
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);

    prior={1};
    prior.setState(0);
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}
