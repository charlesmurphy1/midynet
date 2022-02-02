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
