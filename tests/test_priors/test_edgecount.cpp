#include "gtest/gtest.h"

#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class DummyEdgeCountPrior: public FastMIDyNet::EdgeCountPrior {
    public:
        size_t sample() { return 0; }
        double getLogLikelihood(size_t state) const { return state; }
        double getLogPrior() const { return 0; }

        void checkSelfConsistency() const {}
};

class TestEdgeCountPrior: public ::testing::Test {
    public:
        DummyEdgeCountPrior prior;
        void SetUp() { prior.setState(0); }
};

class TestEdgeCountPoissonPrior: public::testing::Test{
    public:
        FastMIDyNet::EdgeCountPoissonPrior prior={POISSON_MEAN};
};


TEST_F(TestEdgeCountPrior, getStateAfterMove_addEdges_returnCorrectEdgeNumber) {
    for (auto currentEdgeNumber: {0, 1, 2, 10}) {
        prior.setState(currentEdgeNumber);
        for (auto addedNumber: {0, 1, 2, 10}) {
            FastMIDyNet::EdgeMove edgeMove(addedNumber, {0, 0});
            EXPECT_EQ(prior.getStateAfterMove({{}, edgeMove}), currentEdgeNumber+addedNumber);
        }
    }
}

TEST_F(TestEdgeCountPrior, getStateAftermove_removeEdges_returnCorrectEdgeNumber) {
    size_t currentEdgeNumber=10;
    prior.setState(currentEdgeNumber);

    for (auto removedNumber: {0, 1, 2, 10}) {
        FastMIDyNet::EdgeMove edgeMove(removedNumber, {0, 0});
        EXPECT_EQ(prior.getStateAfterMove({edgeMove, {}}), currentEdgeNumber-removedNumber);
    }
}

TEST_F(TestEdgeCountPrior, getLogLikelihoodRatio_addEdges_returnCorrectRatio) {
    prior.setState(5);
    FastMIDyNet::EdgeMove edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatio({{}, edgeMove}), 2);
}

TEST_F(TestEdgeCountPrior, getLogLikelihoodRatio_removeEdges_returnCorrectRatio) {
    prior.setState(5);
    FastMIDyNet::EdgeMove edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatio({edgeMove, {}}), -2);
}

TEST_F(TestEdgeCountPrior, applyMove_addEdges_edgeNumberIncrements) {
    prior.setState(5);
    FastMIDyNet::EdgeMove edgeMove(2, {0, 0});

    prior.applyMove({{}, edgeMove});
    EXPECT_EQ(prior.getState(), 7);
}

TEST_F(TestEdgeCountPrior, applyMove_removeEdges_edgeNumberDecrements) {
    prior.setState(5);
    FastMIDyNet::EdgeMove edgeMove(2, {0, 0});

    prior.applyMove({edgeMove, {}});
    EXPECT_EQ(prior.getState(), 3);
}




TEST_F(TestEdgeCountPoissonPrior, getLogLikelihood_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihood(x),
                    FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(TestEdgeCountPoissonPrior, getLogPrior_returns0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

TEST_F(TestEdgeCountPoissonPrior, applyMove_noMove_edgeNumberUnchanged) {
    prior.setState(1);
    prior.applyMove({});
    EXPECT_EQ(prior.getState(), 1);
}

TEST_F(TestEdgeCountPoissonPrior, applyMove_addEdges_edgeNumberIncremented) {
    FastMIDyNet::EdgeMove edgeMove(1, {0, 0});
    prior.applyMove({{}, edgeMove});
    EXPECT_EQ(prior.getState(), 2);

    edgeMove = {3, {0, 0}};
    prior.applyMove({{}, edgeMove});
    EXPECT_EQ(prior.getState(), 5);
}

TEST_F(TestEdgeCountPoissonPrior, applyMove_removeEdges_edgeNumberIncremented) {
    prior.setState(4);
    FastMIDyNet::EdgeMove edgeMove(1, {0, 0});
    prior.applyMove({edgeMove, {}});
    EXPECT_EQ(prior.getState(), 3);

    edgeMove = {3, {0, 0}};
    prior.applyMove({edgeMove, {}});
    EXPECT_EQ(prior.getState(), 0);
}

TEST_F(TestEdgeCountPoissonPrior, checkSelfConsistency_noError_noThrow) {
    prior.setState(0);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
    prior.setState(2);
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeCountPoissonPrior, checkSelfConsistency_negativeMean_throwConsistencyError) {
    prior={-2};
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}
