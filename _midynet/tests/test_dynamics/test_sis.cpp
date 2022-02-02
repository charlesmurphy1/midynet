#include "gtest/gtest.h"
#include <list>
#include <cmath>

#include "FastMIDyNet/dynamics/sis.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "fixtures.hpp"

namespace FastMIDyNet{

const double INFECTION_PROB(0.7), RECOVERY_PROB(0.3), AUTO_INFECTION_PROB(1e-6);
const std::list<std::vector<int>> neighbor_states = {{1, 3}, {2, 2}, {3, 1}};


class TestSISDynamics: public::testing::Test{
public:
    EdgeCountDeltaPrior edgeCountPrior = {10};
    ErdosRenyiFamily graph = ErdosRenyiFamily(10, edgeCountPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::SISDynamics dynamics = FastMIDyNet::SISDynamics(graph, NUM_STEPS, INFECTION_PROB, RECOVERY_PROB, AUTO_INFECTION_PROB, false);
};


TEST_F(TestSISDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {

    for (auto neighbor_state: neighbor_states)
    EXPECT_EQ( (1-AUTO_INFECTION_PROB) * ( 1 - std::pow(1-INFECTION_PROB, neighbor_state[1])) + AUTO_INFECTION_PROB,
    dynamics.getActivationProb(neighbor_state));
}

TEST_F(TestSISDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {

    for (auto neighbor_state: neighbor_states)
    EXPECT_EQ(RECOVERY_PROB, dynamics.getDeactivationProb(neighbor_state));
}

TEST_F(TestSISDynamics, afterSample_getCorrectNeighborState){
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

TEST_F(TestSISDynamics, getLogLikelihood_returnCorrectLogLikelikehood){
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

TEST_F(TestSISDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
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
