#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS;


class DummyEdgeCountPrior: public FastMIDyNet::EdgeCountPrior {
    public:
        void sampleState() {}
        const double getLogLikelihoodFromState(const size_t& state) const override { return state; }
        const double getLogPrior() const override { return 0; }
        void _checkSelfConsistency() const override { }
};

class TestEdgeCountPrior: public ::testing::Test {
    public:
        DummyEdgeCountPrior prior;
        void SetUp() { prior.setState(0); prior.computationFinished(); }
};

TEST_F(TestEdgeCountPrior, getStateAfterGraphMove_addEdges_returnCorrectEdgeNumber) {
    for (auto currentEdgeNumber: {0, 1, 2, 10}) {
        prior.setState(currentEdgeNumber);
        for (auto addedNumber: {0, 1, 2, 10}) {
            std::vector<BaseGraph::Edge> edgeMove(addedNumber, {0, 0});
            EXPECT_EQ(prior.getStateAfterGraphMove({{}, edgeMove}), currentEdgeNumber+addedNumber);
        }
    }
}

TEST_F(TestEdgeCountPrior, getStateAfterGraphMove_removeEdges_returnCorrectEdgeNumber) {
    size_t currentEdgeNumber=10;
    prior.setState(currentEdgeNumber);

    for (auto removedNumber: {0, 1, 2, 10}) {
        std::vector<BaseGraph::Edge> edgeMove(removedNumber, {0, 0});
        EXPECT_EQ(prior.getStateAfterGraphMove({edgeMove, {}}), currentEdgeNumber-removedNumber);
    }
}

TEST_F(TestEdgeCountPrior, getLogLikelihoodRatio_addEdges_returnCorrectRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove({{}, edgeMove}), 2);
}

TEST_F(TestEdgeCountPrior, getLogLikelihoodRatio_removeEdges_returnCorrectRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove({edgeMove, {}}), -2);
}

TEST_F(TestEdgeCountPrior, getLogJointRatio_graphMove_returnLogLikelihoodRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});
    FastMIDyNet::GraphMove move = {edgeMove, {}};

    EXPECT_EQ(prior.getLogJointRatioFromGraphMove(move), prior.getLogLikelihoodRatioFromGraphMove(move));
}

TEST_F(TestEdgeCountPrior, getLogJointRatio_blockMove_return0) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});
    FastMIDyNet::BlockMove move = {0, 0, 0};

    EXPECT_EQ(prior.getLogJointRatioFromBlockMove(move), 0);
}

TEST_F(TestEdgeCountPrior, applyMove_addEdges_edgeNumberIncrements) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    prior.applyGraphMove({{}, edgeMove});
    EXPECT_EQ(prior.getState(), 7);
}

TEST_F(TestEdgeCountPrior, applyMove_removeEdges_edgeNumberDecrements) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    prior.applyGraphMove({edgeMove, {}});
    EXPECT_EQ(prior.getState(), 3);
}

TEST_F(TestEdgeCountPrior, getLogPrior_return0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

class TestEdgeCountDeltaPrior: public::testing::Test{
    public:
        size_t edgeCount = 5;
        FastMIDyNet::EdgeCountDeltaPrior prior={edgeCount};
};

TEST_F(TestEdgeCountDeltaPrior, sampleState_doNothing){
    EXPECT_EQ(prior.getState(), edgeCount);
    prior.sampleState();
    EXPECT_EQ(prior.getState(), edgeCount);
}

TEST_F(TestEdgeCountDeltaPrior, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0.);
}

TEST_F(TestEdgeCountDeltaPrior, getLogLikelihoodFromState_forSomeStateDifferentThan5_returnMinusInf){
    EXPECT_EQ(prior.getLogLikelihoodFromState(10), -INFINITY);
}

TEST_F(TestEdgeCountDeltaPrior, getLogLikelihoodRatio_forSomeGraphMovePreservingEdgeCount_return0){
    FastMIDyNet::GraphMove move = {{{0,0}}, {{0,2}}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0);
}

TEST_F(TestEdgeCountDeltaPrior, getLogLikelihoodRatio_forSomeGraphMoveNotPreservingEdgeCount_return0){
    FastMIDyNet::GraphMove move = {{{0,0}}, {}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), -INFINITY);
}


class TestEdgeCountPoissonPrior: public::testing::Test{
    public:
        FastMIDyNet::EdgeCountPoissonPrior prior={POISSON_MEAN};
};

TEST_F(TestEdgeCountPoissonPrior, getLogLikelihoodFromState_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihoodFromState(x),
                    FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(TestEdgeCountPoissonPrior, checkSelfConsistency_validMean_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(TestEdgeCountPoissonPrior, checkSelfConsistency_nonPositiveMean_throwConsistencyError) {
    prior={-2};
    EXPECT_THROW(prior.checkSelfConsistency(), FastMIDyNet::ConsistencyError);
}


// class TestEdgeCountMultisetPrior: public::testing::Test{
// public:
//     size_t maxE = 10;
//     FastMIDyNet::EdgeCountMultisetPrior prior = {maxE};
//     void SetUp(){ prior.sample(); }
// };
//
// TEST_F(TestEdgeCountMultisetPrior, sample_returnASample){
//     prior.sample();
// }
//
// TEST_F(TestEdgeCountMultisetPrior, getWeight_forSomeEdgeCount_returnMultisetCoefficient){
//     EXPECT_EQ(prior.getWeight(5), FastMIDyNet::logMultisetCoefficient(10, 5));
// }
//
// class TestEdgeCountBinomialPrior: public::testing::Test{
// public:
//     size_t maxE = 10;
//     FastMIDyNet::EdgeCountBinomialPrior prior = {maxE};
//     void SetUp(){ prior.sample(); }
// };
//
// TEST_F(TestEdgeCountBinomialPrior, sample_returnASample){
//     prior.sample();
// }
//
// TEST_F(TestEdgeCountBinomialPrior, getWeight_forSomeEdgeCount_returnBinomialCoefficient){
//     EXPECT_EQ(prior.getWeight(5), FastMIDyNet::logBinomialCoefficient(maxE, 5));
// }
//
// TEST_F(TestEdgeCountBinomialPrior, getLogNormalization){
//     EXPECT_TRUE( prior.getLogNormalization() > 0 );
// }
