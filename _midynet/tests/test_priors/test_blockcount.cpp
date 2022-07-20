#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/random_graph/prior/block_count.h"
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


class TestNestedBlockCountUniformPrior: public::testing::Test{
public:
    size_t N = 10;
    NestedBlockCountUniformPrior prior = {N};
    bool expectConsistencyError = false;
    void SetUp(){
        seedWithTime();
    }
    void TearDown(){
        if (not expectConsistencyError)
            prior.checkConsistency();
    }
};

TEST_F(TestNestedBlockCountUniformPrior, sampleState_returnConsistentState){
    prior.sample();
    EXPECT_NO_THROW(prior.checkConsistency());
}

TEST_F(TestNestedBlockCountUniformPrior, getDepth){
    std::vector<size_t> nestedState = {7, 3, 2, 1};
    prior.setNestedState(nestedState);
    EXPECT_EQ(4, prior.getDepth());
}

TEST_F(TestNestedBlockCountUniformPrior, setNestedState){
    std::vector<size_t> nestedState = {7, 3, 2, 1};
    prior.setNestedState(nestedState);
    EXPECT_EQ(nestedState, prior.getNestedState());
}

TEST_F(TestNestedBlockCountUniformPrior, getLogLikelihood_returnCorrectValue){
    std::vector<size_t> nestedState = {7, 3, 2, 1};
    prior.setNestedState(nestedState);
    EXPECT_EQ(-log(N - 1) -log(7 - 1) - log(3 - 1) - log(2 - 1), prior.getLogLikelihood());
}

TEST_F(TestNestedBlockCountUniformPrior, getLogLikelihoodFromNestedState_forValidState_returnCorrectValue){
    std::vector<size_t> nestedState = {7, 3, 2, 1};
    double logP = prior.getLogLikelihoodFromNestedState(nestedState);
    EXPECT_EQ(-log(N - 1) -log(7 - 1) - log(3 - 1) - log(2 - 1), logP);
    expectConsistencyError = true;
}

TEST_F(TestNestedBlockCountUniformPrior, getLogLikelihoodFromNestedState_forIncorrectState_returnMinusInfinity){
    std::vector<size_t> nestedState = {7, 8, 2, 1};
    double logP = prior.getLogLikelihoodFromNestedState(nestedState);
    EXPECT_EQ(-INFINITY, logP);
    expectConsistencyError = true;
}



}
