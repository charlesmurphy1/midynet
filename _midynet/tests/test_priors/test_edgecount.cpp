#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/exceptions.h"


const double POISSON_MEAN=5;
const std::vector<size_t> TESTED_INTEGERS={0,5,10};


class DummyEdgeCountPrior: public FastMIDyNet::EdgeCountPrior {
    public:
        void sampleState() override {}
        const double getLogLikelihoodFromState(const size_t& state) const override { return state; }
        void checkSelfConsistency() const override { }
};

class EdgeCountPriorTest: public ::testing::Test {
    public:
        DummyEdgeCountPrior prior;
        void SetUp() {
            prior.setState(0);
            prior.checkSafety();
        }
        void TearDown(){
            prior.checkConsistency();
        }
};

TEST_F(EdgeCountPriorTest, getStateAfterGraphMove_addEdges_returnCorrectEdgeNumber) {
    for (auto currentEdgeNumber: {0, 1, 2, 10}) {
        prior.setState(currentEdgeNumber);
        for (auto addedNumber: {0, 1, 2, 10}) {
            std::vector<BaseGraph::Edge> edgeMove(addedNumber, {0, 0});
            EXPECT_EQ(prior.getStateAfterGraphMove({{}, edgeMove}), currentEdgeNumber+addedNumber);
        }
    }
}

TEST_F(EdgeCountPriorTest, getStateAfterGraphMove_removeEdges_returnCorrectEdgeNumber) {
    size_t currentEdgeNumber=10;
    prior.setState(currentEdgeNumber);

    for (auto removedNumber: {0, 1, 2, 10}) {
        std::vector<BaseGraph::Edge> edgeMove(removedNumber, {0, 0});
        EXPECT_EQ(prior.getStateAfterGraphMove({edgeMove, {}}), currentEdgeNumber-removedNumber);
    }
}

TEST_F(EdgeCountPriorTest, getLogLikelihoodRatio_addEdges_returnCorrectRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove({{}, edgeMove}), 2);
}

TEST_F(EdgeCountPriorTest, getLogLikelihoodRatio_removeEdges_returnCorrectRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove({edgeMove, {}}), -2);
}

TEST_F(EdgeCountPriorTest, getLogJointRatio_graphMove_returnLogLikelihoodRatio) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});
    FastMIDyNet::GraphMove move = {edgeMove, {}};

    EXPECT_EQ(prior.getLogJointRatioFromGraphMove(move), prior.getLogLikelihoodRatioFromGraphMove(move));
}

TEST_F(EdgeCountPriorTest, applyGraphMove_addEdges_edgeNumberIncrements) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    prior.applyGraphMove({{}, edgeMove});
    EXPECT_EQ(prior.getState(), 7);
}

TEST_F(EdgeCountPriorTest, applyGraphMove_removeEdges_edgeNumberDecrements) {
    prior.setState(5);
    std::vector<BaseGraph::Edge> edgeMove(2, {0, 0});

    prior.applyGraphMove({edgeMove, {}});
    EXPECT_EQ(prior.getState(), 3);
}

TEST_F(EdgeCountPriorTest, getLogPrior_return0) {
    EXPECT_DOUBLE_EQ(prior.getLogPrior(), 0);
}

class EdgeCountDeltaPriorTest: public::testing::Test{
public:
    size_t edgeCount = 5;
    FastMIDyNet::EdgeCountDeltaPrior prior={edgeCount};
    void SetUp(){
        prior.checkSafety();
    }
    void TearDown(){
        prior.checkConsistency();
    }
};

TEST_F(EdgeCountDeltaPriorTest, sampleState_doNothing){
    EXPECT_EQ(prior.getState(), edgeCount);
    prior.sampleState();
    EXPECT_EQ(prior.getState(), edgeCount);
}

TEST_F(EdgeCountDeltaPriorTest, getLogLikelihood_return0){
    EXPECT_EQ(prior.getLogLikelihood(), 0.);
}

TEST_F(EdgeCountDeltaPriorTest, getLogLikelihoodFromState_forSomeStateDifferentThan5_returnMinusInf){
    EXPECT_EQ(prior.getLogLikelihoodFromState(10), -INFINITY);
}

TEST_F(EdgeCountDeltaPriorTest, getLogLikelihoodRatio_forSomeGraphMovePreservingEdgeCount_return0){
    FastMIDyNet::GraphMove move = {{{0,0}}, {{0,2}}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), 0);
}

TEST_F(EdgeCountDeltaPriorTest, getLogLikelihoodRatio_forSomeGraphMoveNotPreservingEdgeCount_return0){
    FastMIDyNet::GraphMove move = {{{0,0}}, {}};
    EXPECT_EQ(prior.getLogLikelihoodRatioFromGraphMove(move), -INFINITY);
}


class EdgeCountPoissonPriorTest: public::testing::Test{
public:
    FastMIDyNet::EdgeCountPoissonPrior prior={POISSON_MEAN};
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

TEST_F(EdgeCountPoissonPriorTest, getLogLikelihoodFromState_differentIntegers_returnPoissonPMF) {
    for (auto x: TESTED_INTEGERS)
        EXPECT_DOUBLE_EQ(prior.getLogLikelihoodFromState(x), FastMIDyNet::logPoissonPMF(x, POISSON_MEAN));
}

TEST_F(EdgeCountPoissonPriorTest, checkSelfConsistency_validMean_noThrow) {
    EXPECT_NO_THROW(prior.checkSelfConsistency());
}

TEST_F(EdgeCountPoissonPriorTest, checkSelfConsistency_nonPositiveMean_throwConsistencyError) {
    prior={-2};
    EXPECT_THROW(prior.checkSafety(), FastMIDyNet::SafetyError);
    expectConsistencyError = true;
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
