#include "gtest/gtest.h"
#include <list>

#include "FastMIDyNet/dynamics/cowan.h"
#include "FastMIDyNet/random_graph/erdosrenyi.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "fixtures.hpp"


namespace FastMIDyNet {

const double A = 1., NU = 7., MU = 1., ETA = 0.5;
const std::list<std::vector<int>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};

inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}

class TestWilsonCowan: public::testing::Test{
public:
    EdgeCountDeltaPrior edgeCountPrior = {10};
    ErdosRenyiFamily graph = ErdosRenyiFamily(10, edgeCountPrior);
    HingeFlipUniformProposer edgeProposer = HingeFlipUniformProposer();
    FastMIDyNet::CowanDynamics dynamics = FastMIDyNet::CowanDynamics(graph, NUM_STEPS, NU, A, MU, ETA, false);
};


TEST_F(TestWilsonCowan, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(sigmoid(A*(NU*neighbor_state[1] - MU)), dynamics.getActivationProb(neighbor_state));
}

TEST_F(TestWilsonCowan, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighbor_state: NEIGHBOR_STATES)
        EXPECT_EQ(ETA, dynamics.getDeactivationProb(neighbor_state));
}

TEST_F(TestWilsonCowan, afterSample_getCorrectNeighborState){
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

TEST_F(TestWilsonCowan, getLogLikelihood_returnCorrectLogLikelikehood){
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

TEST_F(TestWilsonCowan, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    edgeProposer.setUp(graph);
    auto graphMove = edgeProposer.proposeMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();

    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

} /* FastMIDyNet */
