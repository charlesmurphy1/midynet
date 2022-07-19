#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS={0, 5, 10};

namespace FastMIDyNet{

class DummyBlockCountPrior: public BlockCountPrior {
public:
    void sampleState() {}
    const double getLogLikelihoodFromState(const size_t& state) const { return state; }

    void checkSelfConsistency() const {}
};

class TestBlockCountPrior: public ::testing::Test {
public:
    DummyBlockCountPrior prior;
    bool expectConsistencyError = false;
    void SetUp() {
        prior.setState(0);
        prior.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }
};


// TEST_F(TestBlockCountPrior, getStateAfterMove_SomLabelMove_returnCorrectBlockNumberIfVertexInNewBlock) {
//     for (auto currentBlockNumber: {1, 2, 5, 10}){
//         prior.setState(currentBlockNumber);
//         for (BlockIndex nextBlockIdx : {0, 1, 2, 6}){
//             BlockMove blockMove = {0, 0, nextBlockIdx};
//             if (nextBlockIdx >= currentBlockNumber) blockMove.addedBlocks = 1;
//
//
//             if (currentBlockNumber > nextBlockIdx)
//                 EXPECT_EQ(prior.getStateAfterLabelMove(blockMove), currentBlockNumber);
//             else
//                 EXPECT_EQ(prior.getStateAfterLabelMove(blockMove), currentBlockNumber + 1);
//         }
//     }
//
// }

TEST_F(TestBlockCountPrior, getLogLikelihoodRatio_throwLogicError) {
    prior.setState(5);
    BlockMove blockMove = {0, 0, 1};
    EXPECT_THROW(prior.getLogLikelihoodRatioFromLabelMove(blockMove), std::logic_error);
}


TEST_F(TestBlockCountPrior, applyLabelMove_noNewblockMove_blockNumberUnchangedIsProcessedIsTrue) {
    prior.setState(5);
    BlockMove blockMove = {0, 0, 2};
    EXPECT_THROW(prior.applyLabelMove(blockMove), std::logic_error);
}

/* BLOCK COUNT DELTA PRIOR TEST: BEGIN */
class TestBlockCountDeltaPrior: public::testing::Test{
public:
    size_t blockCount = 5;
    BlockCountDeltaPrior prior={blockCount};
    bool expectConsistencyError = false;
    void SetUp(){
        prior.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }
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

/* BLOCK COUNT DELTA PRIOR TEST: END */

/* BLOCK COUNT POISSON PRIOR TEST */
class TestBlockCountPoissonPrior: public::testing::Test{

public:
    BlockCountPoissonPrior prior={POISSON_MEAN};
    bool expectConsistencyError = false;
    void SetUp(){
        prior.sample();
        prior.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }

};

TEST_F(TestBlockCountPoissonPrior, getLogLikelihood_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihoodFromState(x), logZeroTruncatedPoissonPMF(x, POISSON_MEAN));
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
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);

    prior={1};
    prior.setState(0);
    EXPECT_THROW(prior.checkSelfConsistency(), ConsistencyError);
    expectConsistencyError = true;
}

}
