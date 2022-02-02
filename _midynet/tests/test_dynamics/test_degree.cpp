#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/degree.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

const double C = 10.;
const double EPSILON = 0.05;
const std::list<std::vector<int>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};


class TestDegreeDynamics: public::testing::Test{
public:
    EdgeCountDeltaPrior edgeCountPrior = {10};
    ErdosRenyiFamily graph = ErdosRenyiFamily(10, edgeCountPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::DegreeDynamics dynamics = FastMIDyNet::DegreeDynamics(graph, NUM_STEPS, C, EPSILON);
};


TEST_F(TestDegreeDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(
            (1 - EPSILON) * (neighbor_state[0] + neighbor_state[1])/C + EPSILON,
            dynamics.getActivationProb(neighbor_state)
        );
}

TEST_F(TestDegreeDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(
            1 - dynamics.getActivationProb(neighbor_state), dynamics.getDeactivationProb(neighbor_state)
        );
}

TEST_F(TestDegreeDynamics, afterSample_getCorrectNeighborState){
    dynamics.sample();
    auto past = dynamics.getPastStates();
    auto expectedNeighborState = dynamics.getNeighborStates();

    for(size_t t=0; t<dynamics.getNumSteps(); ++t){
        for (auto vertex : dynamics.getGraph()){
            std::vector<size_t> actualNeighborState(dynamics.getNumStates(), 0);
            for (auto neighbor : dynamics.getGraph().getNeighboursOfIdx(vertex)){
                actualNeighborState[past[t][neighbor.vertexIndex]]+= neighbor.label;
            }
            for (size_t s=0 ; s< dynamics.getNumStates(); ++s)
                EXPECT_EQ(actualNeighborState[s], expectedNeighborState[t][vertex][s]);
        }
    }
}

TEST_F(TestDegreeDynamics, getLogLikelihood_returnCorrectLogLikelikehood){
    dynamics.sample();
    auto past = dynamics.getPastStates();
    auto future = dynamics.getFutureStates();
    auto neighborState = dynamics.getNeighborStates();

    double expected = dynamics.getLogLikelihood();
    double actual = 0;
    for(size_t t=0; t<dynamics.getNumSteps(); ++t){
        for (auto vertex : dynamics.getGraph()){
            actual += log(dynamics.getTransitionProb(past[t][vertex], future[t][vertex], neighborState[t][vertex]));
        }
    }
    EXPECT_NEAR(expected, actual, 1E-6);
}

TEST_F(TestDegreeDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    edgeProposer.setUp(graph);
    auto graphMove = edgeProposer.proposeMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();
    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

}
