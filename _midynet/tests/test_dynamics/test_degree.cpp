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
