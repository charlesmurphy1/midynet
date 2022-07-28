#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/glauber.hpp"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge/hinge_flip.h"
#include "../fixtures.hpp"

namespace FastMIDyNet{


class TestGlauberDynamics: public::testing::Test{
public:
    const double COUPLING_CONSTANT = 2.;
    const std::list<std::vector<VertexState>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}};
    const size_t NUM_STEPS=20;
    DummyErdosRenyiGraph randomGraph = DummyErdosRenyiGraph(10, 10);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::GlauberDynamics<RandomGraph> dynamics = FastMIDyNet::GlauberDynamics<RandomGraph>(randomGraph, NUM_STEPS, COUPLING_CONSTANT, 0, 0, false, -1);
};


TEST_F(TestGlauberDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(
            sigmoid( 2 * COUPLING_CONSTANT * (neighborState[0]-neighborState[1]) ),
            dynamics.getActivationProb(neighborState)
        );
    }
}

TEST_F(TestGlauberDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(sigmoid(
            -2*COUPLING_CONSTANT*(neighborState[0]-neighborState[1])),
            dynamics.getDeactivationProb(neighborState)
        );
    }
}

TEST_F(TestGlauberDynamics, afterSample_getCorrectNeighborState){
    dynamics.sample();
    dynamics.checkConsistency();
}

TEST_F(TestGlauberDynamics, getLogLikelihood_returnCorrectLogLikelikehood){
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

TEST_F(TestGlauberDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    edgeProposer.setUp(randomGraph.getGraph());
    auto graphMove = edgeProposer.proposeMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();

    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

}
