#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/cowan.hpp"
#include "FastMIDyNet/dynamics/types.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge/hinge_flip.h"
#include "../fixtures.hpp"


namespace FastMIDyNet {


class TestWilsonCowan: public::testing::Test{
public:
    const double A = 1., NU = 7., MU = 1., ETA = 0.5;
    const size_t NUM_STEPS=20;
    const std::list<std::vector<VertexState>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};
    DummyErdosRenyiGraph randomGraph = DummyErdosRenyiGraph(10, 10);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::CowanDynamics<RandomGraph> dynamics = FastMIDyNet::CowanDynamics<RandomGraph>(randomGraph, NUM_STEPS, NU, A, MU, ETA, 0, 0, false, -1);
};


TEST_F(TestWilsonCowan, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (const auto& neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(sigmoid(A*(NU*neighbor_state[1] - MU)), dynamics.getActivationProb(neighbor_state));
}

TEST_F(TestWilsonCowan, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(ETA, dynamics.getDeactivationProb(neighbor_state));
}

TEST_F(TestWilsonCowan, afterSample_getCorrectNeighborState){
    dynamics.sample();
    dynamics.checkConsistency();
}

TEST_F(TestWilsonCowan, getLogLikelihood_returnCorrectLogLikelikehood){
    dynamics.sample();
    auto past = dynamics.getPastStates();
    auto future = dynamics.getFutureStates();
    auto neighborState = dynamics.getNeighborsPastStates();

    double expected = dynamics.getLogLikelihood();
    double actual = 0;
    for(size_t t=0; t<dynamics.getNumSteps(); ++t){
        for (auto vertex : dynamics.getGraph()){
            actual += log(dynamics.getTransitionProb(past[vertex][t], future[vertex][t], neighborState[vertex][t]));
        }
    }
    EXPECT_NEAR(expected, actual, 1E-6);
}

TEST_F(TestWilsonCowan, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    edgeProposer.setUp(randomGraph.getGraph());
    auto graphMove = edgeProposer.proposeMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();

    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

} /* FastMIDyNet */
