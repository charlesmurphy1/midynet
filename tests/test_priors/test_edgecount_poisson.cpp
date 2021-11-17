#include "gtest/gtest.h"

#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class TestEdgeCountPoissonPrior: public::testing::Test{
    public:
        FastMIDyNet::EdgeCountPoissonPrior prior={POISSON_MEAN};
};


TEST_F(TestEdgeCountPoissonPrior, getLogLikelihood_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihood(x),
                    FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(TestEdgeCountPoissonPrior, getLogPrior_returns0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

TEST_F(TestEdgeCountPoissonPrior, getLogLikelihoodRatio_addEdgesWithExistingEdges_returnCorrectLikelihoodRatio) {
    for (auto currentEdgeNumber: {0, 1, 2, 10}) {
        prior.setState(currentEdgeNumber);
        for (auto addedNumber: {0, 1, 2, 10}) {
            FastMIDyNet::EdgeMove edgeMove(addedNumber, {0, 0});
            EXPECT_EQ(FastMIDyNet::logPoissonPMF(currentEdgeNumber+addedNumber, POISSON_MEAN)-
                            FastMIDyNet::logPoissonPMF(currentEdgeNumber, POISSON_MEAN),
                      prior.getLogLikelihoodRatio({{}, edgeMove}));
        }
    }
}

TEST_F(TestEdgeCountPoissonPrior, getLogLikelihoodRatio_removeEdgesWithExistingEdges_returnCorrectLikelihoodRatio) {
    size_t currentEdgeNumber=10;
    prior.setState(currentEdgeNumber);

    for (auto removedNumber: {0, 1, 2, 10}) {
        FastMIDyNet::EdgeMove edgeMove(removedNumber, {0, 0});
        EXPECT_EQ(FastMIDyNet::logPoissonPMF(currentEdgeNumber-removedNumber, POISSON_MEAN)-
                        FastMIDyNet::logPoissonPMF(currentEdgeNumber, POISSON_MEAN),
                  prior.getLogLikelihoodRatio({edgeMove, {}}));
    }
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
